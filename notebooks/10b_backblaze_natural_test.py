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
# # 10b. Backblaze Natural-Prevalence Test Set (2023-2025)
#
# **Purpose.** Build the held-out evaluation set for the Backblaze failure
# models at the true class prevalence. The training working set (notebook 10) is
# undersampled to a fixed healthy-to-positive ratio for both its train and test
# rows, so scoring on those rows overstates performance: MCC and PR-AUC are
# prevalence-sensitive. The honest reported numbers must come from a set that
# carries the natural failure rate. This notebook produces that set for the
# 2023-2025 test period and does not undersample it. The set is scored, never
# trained on.
#
# **Method.** The same four feature builders from `src/features/backblaze_smart.py`
# used to build the training matrix are applied here, so the columns align
# exactly with the working set. Features are computed over each drive's full
# history so the rolling and lag windows for a 2023-2025 row correctly include
# that drive's own earlier observations, then rows are filtered to the test
# period. A 2023-2025 row using the same drive's prior-year history is a
# legitimate at-prediction-time observable, not leakage: the models never see the
# label of any test row.
#
# **Scale.** As in notebook 10, features cannot be materialized for all rows at
# once and a drive's whole history must sit together for the windows. Processing
# is partitioned by a hash of the drive serial (each partition holds complete
# drive histories) and each partition runs in a fresh subprocess. The difference
# from notebook 10 is that no healthy undersampling is applied: every test-period
# row is retained at natural prevalence. Partitions are written and later scored
# one at a time, so the full set is never held in memory at once.
#
# **Outputs.**
# - GCS Parquet under
#   `gs://{PROJECT}-dissertation-data/backblaze_features/natural_test_2023_2025/`
#   (one partition per drive-hash bucket, columns matching the working set).
# - `outputs/tables/backblaze_natural_test_verification.csv` (per-year row counts
#   and per-horizon positive rates confirming natural prevalence).

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

# Purge cached repo modules so a git pull actually takes effect in a warm runtime.
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

TABLES_DIR = OUTPUT_DIR / 'tables'
FEATURES_DIR = OUTPUT_DIR / 'features'
CLEANED_DIR = Path('/content/backblaze_cleaned')
TEST_DIR = Path('/content/backblaze_natural_test')
for d in [TABLES_DIR, FEATURES_DIR, CLEANED_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_CLEANED_PREFIX = 'backblaze_preprocessed/cleaned'
GCS_TEST_PREFIX = 'backblaze_features/natural_test_2023_2025'

N_BUCKETS = 40          # drive-hash partitions; must match notebook 10 so features align
TEST_START_YEAR = 2023  # first year of the held-out test period
SEED = 42               # must match notebook 10 so the hash partitioning is identical

# Assert the partitioning constants match the working-set schema so the test set
# aligns with the trained matrix rather than silently diverging.
_ws_schema = json.loads((FEATURES_DIR / 'backblaze_feature_schema.json').read_text())
assert _ws_schema['n_buckets'] == N_BUCKETS, "bucket count differs from the working set"
assert _ws_schema['seed'] == SEED, "seed differs from the working set"
DATASET_START_ISO = _ws_schema['dataset_start']
print(f"Working-set dataset start: {DATASET_START_ISO}; buckets: {N_BUCKETS}; seed: {SEED}")

# %% [markdown]
# ### Fetch the cleaned dataset from GCS
#
# The cleaned per-file Parquet (notebook 09) is pulled to local NVMe. If the
# cleaning or feature notebook ran in this same session the files are already
# present.

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
# `fleet_age_days` counts from the earliest observation date across the whole
# dataset. The value is taken from the working-set schema so the test features
# are computed on the same origin as the training features, and verified against
# a cheap streaming min over the cleaned data.

# %%
observed_start = (
    pl.scan_parquet(str(CLEANED_DIR / '*_cleaned.parquet'))
    .select(pl.col('date').min().alias('m'))
    .collect(engine='streaming')['m'][0]
)
assert observed_start.isoformat() == DATASET_START_ISO, (
    f"cleaned-data start {observed_start} differs from working-set start {DATASET_START_ISO}"
)
print(f"Dataset start date confirmed: {observed_start}")

# %% [markdown]
# ## 2. Per-partition feature worker (natural prevalence, no undersampling)
#
# The worker filters the cleaned data to its drive-hash partition, sorts by
# `(serial_number, date)`, applies the four feature builders over each drive's
# full history, then keeps only the test-period rows. No healthy undersampling is
# applied, so the partition carries the true failure prevalence. Running in a
# subprocess reclaims the partition's memory on exit.

# %%
WORKER_PATH = Path('/content/_bb_natural_test_worker.py')
WORKER_SRC = '''
import sys
import glob
import polars as pl

repo_dir, cleaned_dir, out_dir, bucket_idx, n_buckets, seed, start_iso, test_start_year = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    int(sys.argv[6]), sys.argv[7], int(sys.argv[8]),
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
# Full-history features are computed above; keep only the test-period rows.
df = lf.collect().filter(pl.col("year") >= test_start_year)
df.write_parquet(out_dir + "/bucket_" + str(bucket_idx) + ".parquet")

pos7 = int(df["failure_within_7d"].sum())
pos14 = int(df["failure_within_14d"].sum())
pos30 = int(df["failure_within_30d"].sum())
fail = int(df["failure"].sum())
print(str(df.height) + ":" + str(fail) + ":" + str(pos7) + ":" + str(pos14) + ":" + str(pos30))
'''
WORKER_PATH.write_text(WORKER_SRC)
print(f"Wrote worker: {WORKER_PATH}")

# %%
totals = {"rows": 0, "fail": 0, "pos7": 0, "pos14": 0, "pos30": 0}
for b in range(N_BUCKETS):
    result = subprocess.run(
        [sys.executable, str(WORKER_PATH), REPO_DIR, str(CLEANED_DIR), str(TEST_DIR),
         str(b), str(N_BUCKETS), str(SEED), DATASET_START_ISO, str(TEST_START_YEAR)],
        capture_output=True, text=True, check=True,
    )
    rows, fail, pos7, pos14, pos30 = (int(x) for x in result.stdout.strip().splitlines()[-1].split(":"))
    totals["rows"] += rows
    totals["fail"] += fail
    totals["pos7"] += pos7
    totals["pos14"] += pos14
    totals["pos30"] += pos30
    print(f"  bucket {b:>2}/{N_BUCKETS} done: {rows:,} rows")

print()
print(f"  natural test rows:       {totals['rows']:,}")
print(f"  failure drive-days:      {totals['fail']:,} "
      f"({100 * totals['fail'] / max(totals['rows'], 1):.4f}%)")
for h in (7, 14, 30):
    p = totals[f"pos{h}"]
    print(f"  failure_within_{h}d:      {p:,} "
          f"({100 * p / max(totals['rows'], 1):.4f}% positive)")

# %% [markdown]
# ## 3. Verification and schema check
#
# Confirm the test set columns match the working set exactly (so a model trained
# on the working set scores without column realignment), and record per-year row
# counts and per-horizon positive rates.

# %%
test_lf = pl.scan_parquet(str(TEST_DIR / 'bucket_*.parquet'))
test_cols = list(test_lf.collect_schema().names())
ws_cols = _ws_schema['columns']
assert test_cols == ws_cols, (
    "test columns differ from the working set: "
    f"missing {set(ws_cols) - set(test_cols)}, extra {set(test_cols) - set(ws_cols)}"
)
print(f"Column check passed: {len(test_cols)} columns match the working set")

# %%
verification = (
    test_lf.group_by('year')
    .agg(
        pl.len().alias('rows'),
        pl.col('failure').sum().alias('failure_drive_days'),
        pl.col('failure_within_7d').sum().alias('pos_7d'),
        pl.col('failure_within_14d').sum().alias('pos_14d'),
        pl.col('failure_within_30d').sum().alias('pos_30d'),
    )
    .with_columns(
        (pl.col('failure_drive_days') / pl.col('rows')).alias('failure_rate'),
        (pl.col('pos_30d') / pl.col('rows')).alias('pos_30d_rate'),
    )
    .sort('year')
    .collect(engine='streaming')
)
print(verification.to_pandas().to_string(index=False))
verification.write_csv(TABLES_DIR / 'backblaze_natural_test_verification.csv')
print(f"Saved {TABLES_DIR / 'backblaze_natural_test_verification.csv'}")

# %% [markdown]
# ## 4. Export to GCS
#
# Upload each partition to the natural-test prefix for the modeling notebooks to
# score. Partitions stay separate so scoring can proceed one at a time.

# %%
test_files = sorted(TEST_DIR.glob('bucket_*.parquet'))
for tf in test_files:
    bucket.blob(f'{GCS_TEST_PREFIX}/{tf.name}').upload_from_filename(str(tf))
print(f"Uploaded {len(test_files)} partitions to gs://{GCS_BUCKET}/{GCS_TEST_PREFIX}/")

# %% [markdown]
# ## 5. Summary

# %%
print("BACKBLAZE NATURAL-PREVALENCE TEST SET (2023-2025)")
print("=" * 60)
print(f"  rows:               {totals['rows']:,}")
print(f"  failure drive-days: {totals['fail']:,} "
      f"({100 * totals['fail'] / max(totals['rows'], 1):.4f}%)")
print(f"  columns:            {len(test_cols)} (match working set)")
print(f"  partitions:         {len(test_files)}")
print(f"  GCS: gs://{GCS_BUCKET}/{GCS_TEST_PREFIX}/")
print("=" * 60)
