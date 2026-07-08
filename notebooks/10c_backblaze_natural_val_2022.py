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
# # 10c. Backblaze Natural-Prevalence Validation Set (2022)
#
# **Purpose.** Provide a validation set at the true class prevalence for the year
# immediately before the 2023-2025 test period. The failure models are trained on
# 2021-and-earlier data; their operating threshold and probability calibration
# must be fit on held-out data at the deployment prevalence, not on the
# undersampled working set. Fitting them on an undersampled slice leaves the
# threshold mis-set and the probabilities calibrated to the wrong base rate, which
# also corrupts any checkpoint that downstream drift work consumes. This notebook
# supplies a 2022 slice at natural prevalence for that purpose.
#
# **Method.** Identical feature engineering to the natural-prevalence test
# (notebook 10b), applied to the 2022 rows only, then uniformly sampled. Uniform
# fraction sampling preserves the natural class balance, so the fitted threshold
# and calibrator see the true prevalence while the set stays small enough to hold
# in memory. Features are computed over each drive's full history so the rolling
# windows for a 2022 row correctly include that drive's earlier observations.
#
# **Temporal placement.** Train on 2021-and-earlier, fit the operating point and
# calibration on this 2022 set, evaluate on 2023-2025. The three periods do not
# overlap, so the operating point is chosen without touching the test.
#
# **Output.** GCS Parquet under
# `gs://{PROJECT}-dissertation-data/backblaze_features/natural_val_2022/`, columns
# matching the working set, at natural prevalence, uniformly subsampled.

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
VAL_DIR = Path('/content/backblaze_natural_val_2022')
for d in [FEATURES_DIR, TABLES_DIR, CLEANED_DIR, VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_CLEANED_PREFIX = 'backblaze_preprocessed/cleaned'
GCS_VAL_PREFIX = 'backblaze_features/natural_val_2022'

N_BUCKETS = 40
VAL_YEAR = 2022
TARGET_VAL_ROWS = 6_000_000   # uniform prevalence-preserving sample target
SEED = 42

_ws_schema = json.loads((FEATURES_DIR / 'backblaze_feature_schema.json').read_text())
assert _ws_schema['n_buckets'] == N_BUCKETS, "bucket count differs from the working set"
assert _ws_schema['seed'] == SEED, "seed differs from the working set"
DATASET_START_ISO = _ws_schema['dataset_start']
print(f"Working-set dataset start: {DATASET_START_ISO}; buckets: {N_BUCKETS}; seed: {SEED}")

# %% [markdown]
# ### Fetch the cleaned dataset from GCS

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
# ## 1. Sampling fraction for the target year
#
# Count the 2022 rows once (cheap streaming filter) to set a uniform sampling
# fraction that lands near the target size.

# %%
val_year_rows = (
    pl.scan_parquet(str(CLEANED_DIR / '*_cleaned.parquet'))
    .filter(pl.col('date').dt.year() == VAL_YEAR)
    .select(pl.len()).collect(engine='streaming').item()
)
sample_frac = min(1.0, TARGET_VAL_ROWS / val_year_rows)
print(f"{VAL_YEAR} rows: {val_year_rows:,}; sampling fraction: {sample_frac:.5f}")

# %% [markdown]
# ## 2. Per-partition worker (feature engineer, filter to the year, uniform sample)

# %%
WORKER_PATH = Path('/content/_bb_natural_val_worker.py')
WORKER_SRC = '''
import sys
import glob
import polars as pl

repo_dir, cleaned_dir, out_dir, bucket_idx, n_buckets, seed, start_iso, val_year, frac = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    int(sys.argv[6]), sys.argv[7], int(sys.argv[8]), float(sys.argv[9]),
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
df = lf.collect().filter(pl.col("year") == val_year)
df = df.sample(fraction=frac, seed=seed)
df.write_parquet(out_dir + "/bucket_" + str(bucket_idx) + ".parquet")
print(str(df.height) + ":" + str(int(df["failure_within_30d"].sum())))
'''
WORKER_PATH.write_text(WORKER_SRC)
print(f"Wrote worker: {WORKER_PATH}")

# %%
total_rows, total_pos30 = 0, 0
for b in range(N_BUCKETS):
    result = subprocess.run(
        [sys.executable, str(WORKER_PATH), REPO_DIR, str(CLEANED_DIR), str(VAL_DIR),
         str(b), str(N_BUCKETS), str(SEED), DATASET_START_ISO, str(VAL_YEAR), str(sample_frac)],
        capture_output=True, text=True, check=True,
    )
    rows, pos30 = (int(x) for x in result.stdout.strip().splitlines()[-1].split(":"))
    total_rows += rows
    total_pos30 += pos30
    print(f"  bucket {b:>2}/{N_BUCKETS} done: {rows:,} rows")

print()
print(f"  validation sample rows: {total_rows:,}")
print(f"  failure_within_30d:     {total_pos30:,} "
      f"({100 * total_pos30 / max(total_rows, 1):.4f}% positive)")

# %% [markdown]
# ## 3. Column check and export

# %%
val_lf = pl.scan_parquet(str(VAL_DIR / 'bucket_*.parquet'))
val_cols = list(val_lf.collect_schema().names())
assert val_cols == _ws_schema['columns'], "validation columns differ from the working set"
print(f"Column check passed: {len(val_cols)} columns match the working set")

val_files = sorted(VAL_DIR.glob('bucket_*.parquet'))
for vf in val_files:
    bucket.blob(f'{GCS_VAL_PREFIX}/{vf.name}').upload_from_filename(str(vf))
print(f"Uploaded {len(val_files)} partitions to gs://{GCS_BUCKET}/{GCS_VAL_PREFIX}/")

# %%
print("BACKBLAZE NATURAL-PREVALENCE VALIDATION SET (2022)")
print("=" * 58)
print(f"  rows:            {total_rows:,}")
print(f"  30d positives:   {total_pos30:,} ({100 * total_pos30 / max(total_rows, 1):.4f}%)")
print(f"  columns:         {len(val_cols)} (match working set)")
print(f"  GCS: gs://{GCS_BUCKET}/{GCS_VAL_PREFIX}/")
print("=" * 58)
