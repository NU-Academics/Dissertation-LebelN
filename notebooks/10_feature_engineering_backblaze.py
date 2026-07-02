# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 10. Backblaze Feature Engineering
#
# **Purpose.** Turn the preprocessed HDD dataset (notebook 09) into the tiered
# SMART feature matrix the failure and drift models train on, and undersample
# the abundant healthy observations to a manageable class ratio.
#
# **Feature tiers** come from `src/features/backblaze_smart.py`: Tier 1
# pre-event signals on the primary SMART attributes plus zero-inflation
# indicators, degradation-onset timing, manufacturer identity, and capacity
# (V14, V18, V19); Tier 2 rolling location and upper-quantile statistics and
# rate-of-change on primary attributes and a reduced set on secondary
# attributes (V15); Tier 3 drift-aware cohort features and the era-gated SMART
# 187 / 188 handling (V16, the era census). Multi-horizon 7 / 14 / 30-day
# failure targets (P07) are added at feature time.
#
# **Ordering and leakage.** Rolling and lag features are computed within each
# drive ordered by date, so no future row enters a past window. The temporal
# split (test period 2023-2025) is preserved downstream; drive-model target
# encoding is deferred to modeling (prior years only, V18 / O07).
#
# **Scale.** Features cannot be materialized for all 676M rows at once, and a
# drive's whole history must sit together for rolling windows. Processing is
# therefore partitioned by a hash of the drive serial (each partition holds
# complete drive histories), and each partition runs in a fresh subprocess that
# sorts by `(serial_number, date)`, computes the features, undersamples the
# healthy observations, and writes only the retained rows. Peak memory is
# bounded to one partition and reclaimed on subprocess exit.
#
# **Working set.** The primary working set keeps every row that is positive for
# the 30-day horizon (the union of the pre-failure windows) and a
# proportionally stratified sample of the remaining healthy observations at the
# target healthy-to-positive ratio (default 100:1). The formal sampler and the
# 50:1 / 200:1 sensitivity branches follow in the working-set construction step.
#
# **Outputs.**
# - GCS Parquet working set under
#   `gs://{PROJECT}-dissertation-data/backblaze_features/working_set_100x/`.
# - `outputs/features/backblaze_feature_schema.json`.
# - `outputs/tables/backblaze_feature_engineering_verification.csv`.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars pandas pyarrow google-cloud-storage

# %%
import os
import sys
from pathlib import Path

from google.colab import userdata

GITHUB_PAT = userdata.get('GITHUB_PAT')
PROJECT_ID = userdata.get('GCP_PROJECT_ID')
REPO_OWNER = 'NU-Academics'
REPO_NAME = 'Dissertation-LebelN'
REPO_DIR = f'/content/{REPO_NAME}'
REPO_URL = f'https://{GITHUB_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git'

if os.path.exists(REPO_DIR):
    # !cd {REPO_DIR} && git pull --quiet
    print(f"Repo already present at {REPO_DIR}; pulled latest.")
else:
    # !git clone --quiet {REPO_URL} {REPO_DIR}
    print(f"Cloned repo to {REPO_DIR}.")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from google.colab import auth
auth.authenticate_user()

# %%
import json
import subprocess

import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR

setup_drive()

FEATURES_DIR = OUTPUT_DIR / 'features'
TABLES_DIR = OUTPUT_DIR / 'tables'
CLEANED_DIR = Path('/content/backblaze_cleaned')
WORKING_DIR = Path('/content/backblaze_features')
for d in [FEATURES_DIR, TABLES_DIR, WORKING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_CLEANED_PREFIX = 'backblaze_preprocessed/cleaned'
GCS_FEATURES_PREFIX = 'backblaze_features/working_set_100x'

N_BUCKETS = 40          # drive-hash partitions; each holds complete drive histories
RATIO = 100             # target healthy-to-positive observation ratio (primary set)
POSITIVE_HORIZON = 30   # keep every row positive for this horizon in full
SEED = 42

# %% [markdown]
# ### Fetch the cleaned dataset from GCS
#
# The cleaned per-file Parquet (notebook 09) is pulled to local NVMe. If the
# cleaning notebook ran in this same session the files are already present.

# %%
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)

for blob in bucket.list_blobs(prefix=GCS_CLEANED_PREFIX):
    if not blob.name.endswith('.parquet'):
        continue
    name = blob.name.split('/')[-1]
    local_path = CLEANED_DIR / name
    if not (local_path.exists() and local_path.stat().st_size == blob.size):
        blob.download_to_filename(str(local_path))

cleaned_files = sorted(CLEANED_DIR.glob('*_cleaned.parquet'))
print(f"{len(cleaned_files)} cleaned Parquet files at {CLEANED_DIR}")

# %% [markdown]
# ## 1. Dataset start date
#
# ``fleet_age_days`` counts from the earliest observation date across the whole
# dataset, so the start date is computed once (a cheap streaming min) and passed
# to every partition.

# %%
dataset_start = (
    pl.scan_parquet(str(CLEANED_DIR / '*_cleaned.parquet'))
    .select(pl.col('date').min().alias('m'))
    .collect(engine='streaming')['m'][0]
)
print(f"Dataset start date: {dataset_start}")

# %% [markdown]
# ## 2. Per-partition feature engineering worker
#
# The worker script is written to disk and invoked once per drive-hash
# partition. Each run filters the cleaned data to its partition, sorts by
# `(serial_number, date)`, applies the four feature builders, undersamples the
# healthy observations, and writes the partition's working-set rows. Running in
# a subprocess guarantees the partition's memory is reclaimed on exit.

# %%
WORKER_PATH = Path('/content/_bb_feature_worker.py')
WORKER_SRC = '''
import sys
import glob
import polars as pl

repo_dir, cleaned_dir, out_dir, bucket_idx, n_buckets, ratio, horizon, seed, start_iso = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9],
)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
from datetime import date
from src.features.backblaze_smart import (
    add_tier1_smart_features, add_tier2_rolling_features,
    add_tier3_drift_features, add_multi_horizon_targets,
)

start_date = date.fromisoformat(start_iso)
files = sorted(glob.glob(cleaned_dir + "/*_cleaned.parquet"))
lf = (
    pl.scan_parquet(files)
    .filter((pl.col("serial_number").hash(seed) % n_buckets) == bucket_idx)
    .sort(["serial_number", "date"])
)
lf = add_tier1_smart_features(lf)
lf = add_tier2_rolling_features(lf)
lf = add_tier3_drift_features(lf, dataset_start_date=start_date)
lf = add_multi_horizon_targets(lf)
df = lf.collect()

pos_col = "failure_within_" + str(horizon) + "d"
positives = df.filter(pl.col(pos_col) == 1)
healthy = df.filter(pl.col(pos_col) == 0)
n_pos = positives.height
n_healthy = healthy.height
target_healthy = ratio * n_pos
if n_healthy > target_healthy and n_healthy > 0:
    keep = int(target_healthy / n_healthy * 1_000_000)
    healthy = healthy.filter(
        (pl.struct(["serial_number", "date"]).hash(seed) % 1_000_000) < keep
    )
out = pl.concat([positives, healthy])
out.write_parquet(out_dir + "/bucket_" + str(bucket_idx) + ".parquet")
print(str(out.height) + " " + str(n_pos) + " " + str(healthy.height))
'''
WORKER_PATH.write_text(WORKER_SRC)
print(f"Wrote worker: {WORKER_PATH}")

# %%
total_rows = 0
total_pos = 0
total_healthy = 0
for b in range(N_BUCKETS):
    result = subprocess.run(
        [sys.executable, str(WORKER_PATH), REPO_DIR, str(CLEANED_DIR), str(WORKING_DIR),
         str(b), str(N_BUCKETS), str(RATIO), str(POSITIVE_HORIZON), str(SEED),
         dataset_start.isoformat()],
        capture_output=True, text=True, check=True,
    )
    rows, pos, healthy = (int(x) for x in result.stdout.strip().splitlines()[-1].split())
    total_rows += rows
    total_pos += pos
    total_healthy += healthy
    print(f"  bucket {b:>2}/{N_BUCKETS}: {rows:,} rows ({pos:,} positive, {healthy:,} healthy)")

print(f"\nWorking set built: {total_rows:,} rows "
      f"({total_pos:,} positive, {total_healthy:,} healthy, "
      f"ratio {total_healthy / max(total_pos, 1):.1f}:1)")

# %% [markdown]
# ## 3. Feature schema and verification
#
# Load the working set lazily to record the feature schema and per-era and
# per-year positive rates for the verification table.

# %%
working_lf = pl.scan_parquet(str(WORKING_DIR / 'bucket_*.parquet'))
schema = working_lf.collect_schema()
feature_cols = [c for c in schema.names()]
print(f"Working set columns: {len(feature_cols)}")

FEATURE_SCHEMA_PATH = FEATURES_DIR / 'backblaze_feature_schema.json'
with open(FEATURE_SCHEMA_PATH, 'w') as f:
    json.dump(
        {
            "n_columns": len(feature_cols),
            "columns": feature_cols,
            "dtypes": {c: str(schema[c]) for c in feature_cols},
            "n_buckets": N_BUCKETS,
            "ratio": RATIO,
            "positive_horizon": POSITIVE_HORIZON,
            "seed": SEED,
            "dataset_start": dataset_start.isoformat(),
        },
        f, indent=2,
    )
print(f"Wrote feature schema: {FEATURE_SCHEMA_PATH}")

# %%
verification = (
    working_lf.group_by('year')
    .agg(
        pl.len().alias('rows'),
        pl.col('failure_within_30d').sum().alias('pos_30d'),
        pl.col('failure').sum().alias('failures'),
    )
    .sort('year')
    .collect(engine='streaming')
)
print(verification.to_pandas().to_string(index=False))
verification.write_csv(TABLES_DIR / 'backblaze_feature_engineering_verification.csv')
print(f"Saved {TABLES_DIR / 'backblaze_feature_engineering_verification.csv'}")

# %% [markdown]
# ## 4. Export to GCS
#
# Upload the working-set partitions to GCS for the modeling notebooks.

# %%
bucket_files = sorted(WORKING_DIR.glob('bucket_*.parquet'))
for bf in bucket_files:
    bucket.blob(f'{GCS_FEATURES_PREFIX}/{bf.name}').upload_from_filename(str(bf))
print(f"Uploaded {len(bucket_files)} working-set partitions to "
      f"gs://{GCS_BUCKET}/{GCS_FEATURES_PREFIX}/")

# %% [markdown]
# ## 5. Summary

# %%
print("BACKBLAZE FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(f"Working-set rows:   {total_rows:,}")
print(f"Positive (30d):     {total_pos:,}")
print(f"Healthy (sampled):  {total_healthy:,}")
print(f"Feature columns:    {len(feature_cols)}")
print(f"GCS: gs://{GCS_BUCKET}/{GCS_FEATURES_PREFIX}/")
print("=" * 60)
