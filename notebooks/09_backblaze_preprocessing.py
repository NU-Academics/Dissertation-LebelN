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
# # 09. Backblaze Preprocessing
#
# **Purpose.** Turn the raw Backblaze daily Parquet (notebook 04) into a clean,
# labeled, schema-reconciled HDD table ready for feature engineering. The pass
# applies, in order: SSD exclusion, schema-era assignment, SMART schema
# reconciliation, availability-indicator encoding, SMART cleaning, drive-model
# canonicalization, drive-day deduplication, a per-drive temporal sort, and a
# censoring marker. A post-preprocessing assertion suite then confirms the
# cleaned table against the Phase 2 statistics before export.
#
# **Decisions operationalized** (`outputs/tables/eda_decisions.csv`): V14 / V15
# (primary and secondary SMART attributes), V16 (SMART 187 / 188 conditional
# inclusion via availability indicators), V18 (drive-model identity and
# manufacturer), and the era-census row (three SMART schema eras from notebook
# 07c, materialized as `src.data.schemas.BACKBLAZE_ERAS`).
#
# **Validated logic** is extracted into `src/preprocessing/backblaze.py` (the
# per-row transforms) and `src/data/validation.py` (the assertion suite); this
# notebook composes them and owns all reads and writes.
#
# **Outputs.**
# - GCS Parquet: `gs://{PROJECT}-dissertation-data/backblaze_preprocessed/`.
# - Drive manifest: `{OUTPUT_DIR}/preprocessed/backblaze/manifest.json`.
# - `outputs/tables/backblaze_preprocessing_verification.csv`.
#
# **Scale discipline.** The daily data is about 682M drive-day rows across 43
# Parquet files with an evolving schema. Files are scanned lazily, reconciled to
# a common column set, concatenated, and streamed to Parquet, so the row-level
# data is not materialized in memory in one piece. The per-drive sort and the
# censoring window are the memory-heavy steps and use the streaming engine.

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

# Colab caches imported modules in sys.modules, so a later `git pull` has no
# effect until the runtime restarts. Drop any previously imported repo modules
# here so the freshly pulled source is what gets imported in the cells below.
for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from google.colab import auth
auth.authenticate_user()

# %%
import json
import re

import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR
from src.data.schemas import BACKBLAZE_ERAS
from src.data.validation import (
    assert_era_assignment_complete,
    assert_failure_event_count,
    assert_fleet_expansion,
    assert_one_row_per_drive_day,
)
from src.preprocessing.backblaze import (
    assign_era,
    canonicalize_drive_model,
    encode_smart_availability_indicators,
    filter_hdds_only,
    mark_censoring,
    reconcile_smart_schema,
)

setup_drive()

PREPROCESSED_DIR = OUTPUT_DIR / 'preprocessed' / 'backblaze'
TABLES_DIR = OUTPUT_DIR / 'tables'
LOCAL_SINK = Path('/content/backblaze_preprocessed')
BACKBLAZE_DIR = Path('/content/backblaze_parquet')
for d in [PREPROCESSED_DIR, TABLES_DIR, LOCAL_SINK, BACKBLAZE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_RAW_PREFIX = 'backblaze_parquet/'
GCS_PREPROCESSED_PREFIX = 'backblaze_preprocessed'
MANIFEST_PATH = PREPROCESSED_DIR / 'manifest.json'

# Analytically relevant SMART IDs: the union of the per-era available sets from
# the schema-evolution census. The long tail of near-empty IDs is not modeled,
# so reconciliation targets this set (keeps dimensionality bounded per V15).
MODELED_SMART_IDS = tuple(sorted(set().union(*[set(ids) for *_, ids in BACKBLAZE_ERAS])))
# Cross-era universal IDs (present in every era) versus the era-gated remainder
# that needs availability indicators (V16). 187 / 188 fall in the gated set.
UNIVERSAL_SMART_IDS = tuple(sorted(set.intersection(*[set(ids) for *_, ids in BACKBLAZE_ERAS])))
ERA_GATED_SMART_IDS = tuple(sorted(set(MODELED_SMART_IDS) - set(UNIVERSAL_SMART_IDS)))

print(f"Modeled SMART IDs ({len(MODELED_SMART_IDS)}): {MODELED_SMART_IDS}")
print(f"Universal across eras ({len(UNIVERSAL_SMART_IDS)}): {UNIVERSAL_SMART_IDS}")
print(f"Era-gated, need indicators ({len(ERA_GATED_SMART_IDS)}): {ERA_GATED_SMART_IDS}")

# %% [markdown]
# ### Download raw Parquet from GCS

# %%
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)

raw_blobs = sorted(
    [b for b in bucket.list_blobs(prefix=GCS_RAW_PREFIX) if b.name.endswith('.parquet')],
    key=lambda b: b.name,
)
for b in raw_blobs:
    name = b.name.split('/')[-1]
    local_path = BACKBLAZE_DIR / name
    if not (local_path.exists() and local_path.stat().st_size == b.size):
        b.download_to_filename(str(local_path))

parquet_files = sorted(BACKBLAZE_DIR.glob('*.parquet'))
print(f"{len(parquet_files)} raw Parquet files available at {BACKBLAZE_DIR}")

# %% [markdown]
# ## 1. SSD exclusion
#
# The daily schema carries no drive-type flag, so SSDs are identified by model
# name. Candidate SSD models are flagged by the keyword and prefix conventions
# surfaced in notebook 05 (an "SSD" token or a known consumer-SSD family), then
# printed for verification before exclusion. The verified set is passed to
# `filter_hdds_only`; the failure-count assertion downstream confirms no HDD
# failures were lost.

# %%
# Inventory distinct models and their row counts across all files.
model_counts: dict[str, int] = {}
for pf in parquet_files:
    out = (
        pl.scan_parquet(pf)
        .group_by('model')
        .agg(pl.len().alias('n'))
        .collect()
    )
    for row in out.iter_rows(named=True):
        model_counts[row['model']] = model_counts.get(row['model'], 0) + int(row['n'])

_SSD_TOKENS = re.compile(
    r'(SSD|Samsung|Crucial|Micron|DELLBOSS|WDC WDS|TOSHIBA-?(TR|HK|KSG)|'
    r'Seagate SSD|SanDisk|Intel SSD|PHISON|HP SSD)',
    flags=re.IGNORECASE,
)
ssd_models = {m for m in model_counts if _SSD_TOKENS.search(m or '')}

print(f"Flagged {len(ssd_models)} SSD models for exclusion:")
for m in sorted(ssd_models):
    print(f"  {m:40s}  {model_counts[m]:>12,} rows")
print("Review this list against notebook 05 before trusting the exclusion.")

# %% [markdown]
# ## 2. Build the reconciled, labeled HDD pipeline
#
# Each file is scanned, reconciled to the modeled SMART column set so all files
# share one schema, and then concatenated. The transforms are applied to the
# combined lazy frame: SSD exclusion, era assignment, availability indicators,
# SMART cleaning (drop the `_normalized` siblings, since the EDA models on raw
# values per notebook 05 Section 3.1), and drive-model canonicalization.

# %%
KEEP_BASE_COLS = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']

lazy_frames = []
for pf in parquet_files:
    schema = pl.read_parquet_schema(pf)
    base_present = [c for c in KEEP_BASE_COLS if c in schema]
    raw_present = [
        f'smart_{sid}_raw' for sid in MODELED_SMART_IDS
        if f'smart_{sid}_raw' in schema
    ]
    lf_file = pl.scan_parquet(pf).select(base_present + raw_present)
    # Reconcile to the full modeled set (raw only; normalized is dropped next).
    lf_file = reconcile_smart_schema(lf_file, MODELED_SMART_IDS, keep_normalized=False)
    lazy_frames.append(lf_file)

raw_cols = [f'smart_{sid}_raw' for sid in MODELED_SMART_IDS]
combined = pl.concat(lazy_frames, how='vertical_relaxed').select(KEEP_BASE_COLS + raw_cols)

# Fail-loud column guard right after the first build step: a stale module or a
# bad reconcile would surface here before any expensive downstream work.
built_cols = set(combined.collect_schema().names())
required = set(KEEP_BASE_COLS) | set(raw_cols)
missing = required - built_cols
assert not missing, f"combined frame missing expected columns: {sorted(missing)}"
print(f"Combined lazy frame columns: {len(built_cols)} (base + {len(raw_cols)} SMART raw)")

# %%
pipeline = filter_hdds_only(combined, ssd_models=ssd_models)
pipeline = assign_era(pipeline)
pipeline = encode_smart_availability_indicators(pipeline, ERA_GATED_SMART_IDS)
pipeline = canonicalize_drive_model(pipeline)

# %% [markdown]
# ## 3. Drive-day deduplication and per-drive temporal sort
#
# There should be exactly one observation per `(serial_number, date)`. Duplicates
# are dropped, keeping the first. The single sort by `(serial_number, date)`
# established here is what every downstream rolling and lag feature relies on.

# %%
pipeline = pipeline.unique(subset=['serial_number', 'date'], keep='first')
pipeline = pipeline.sort(['serial_number', 'date'])

# %% [markdown]
# ## 4. Censoring marker
#
# Each drive's final observation is classified as an observed failure
# (`failure_observed`) or a right-censoring event (`censored`), required for the
# survival framing of the lead-time analysis.

# %%
pipeline = mark_censoring(pipeline)

# %% [markdown]
# ## 5. Materialize and post-preprocessing assertions
#
# Stream the pipeline to local Parquet, then run the assertion suite on a lazy
# scan of the result. The suite confirms the failure-event count survived, the
# grain is one row per drive-day, every row received a known era, and the
# multi-year fleet expansion is preserved.

# %%
LOCAL_PARQUET = LOCAL_SINK / 'backblaze_preprocessed.parquet'
pipeline.sink_parquet(LOCAL_PARQUET, engine='streaming')
print(f"Streamed preprocessed table to {LOCAL_PARQUET}")

result_lf = pl.scan_parquet(LOCAL_PARQUET)

n_failures = assert_failure_event_count(result_lf, expected=31_062, tolerance=200)
n_dupe = assert_one_row_per_drive_day(result_lf)
n_unknown_era = assert_era_assignment_complete(result_lf)
fleet_before, fleet_after = assert_fleet_expansion(result_lf, cutoff_date='2020-01-01')

total_rows = int(result_lf.select(pl.len()).collect().item())
print(f"Rows:                 {total_rows:,}")
print(f"Failure events:       {n_failures:,}")
print(f"Duplicate drive-days: {n_dupe}")
print(f"Unknown-era rows:     {n_unknown_era}")
print(f"Distinct drives pre/post 2020: {fleet_before:,} / {fleet_after:,}")

# %% [markdown]
# ### Per-era verification table
#
# Row counts, distinct drives, failure events, and SMART 187 / 188 availability
# by era, to confirm the cleaned table matches the schema-evolution census.

# %%
verification = (
    result_lf.group_by('era')
    .agg(
        pl.len().alias('rows'),
        pl.col('serial_number').n_unique().alias('drives'),
        (pl.col('failure') == 1).sum().alias('failures'),
        pl.col('has_smart_187').mean().alias('smart_187_avail'),
        pl.col('has_smart_188').mean().alias('smart_188_avail'),
    )
    .sort('era')
    .collect()
)
print(verification.to_pandas().to_string(index=False))
verification.write_csv(TABLES_DIR / 'backblaze_preprocessing_verification.csv')
print(f"Saved {TABLES_DIR / 'backblaze_preprocessing_verification.csv'}")

# %% [markdown]
# ## 6. Export to GCS Parquet plus Drive manifest
#
# Upload the preprocessed Parquet to GCS (the modeling notebooks read it from
# there) and write a small manifest to Drive describing the produced artifact.

# %%
blob_name = f'{GCS_PREPROCESSED_PREFIX}/backblaze_preprocessed.parquet'
bucket.blob(blob_name).upload_from_filename(str(LOCAL_PARQUET))
gcs_uri = f'gs://{GCS_BUCKET}/{blob_name}'
print(f"Uploaded to {gcs_uri}")

manifest = {
    "dataset": "backblaze",
    "stage": "preprocessed",
    "source_notebook": "09_backblaze_preprocessing.py",
    "gcs_uri": gcs_uri,
    "local_parquet": str(LOCAL_PARQUET),
    "rows": total_rows,
    "failure_events": n_failures,
    "duplicate_drive_days": n_dupe,
    "unknown_era_rows": n_unknown_era,
    "distinct_drives_pre_2020": fleet_before,
    "distinct_drives_post_2020": fleet_after,
    "modeled_smart_ids": list(MODELED_SMART_IDS),
    "universal_smart_ids": list(UNIVERSAL_SMART_IDS),
    "era_gated_smart_ids": list(ERA_GATED_SMART_IDS),
    "excluded_ssd_models": sorted(ssd_models),
    "columns": result_lf.collect_schema().names(),
}
with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote manifest: {MANIFEST_PATH}")

# %% [markdown]
# ## 7. Summary
#
# Report back: the row count and failure-event count after cleaning, the
# per-era verification table (especially the SMART 187 / 188 availability by
# era), and the excluded SSD model list. These feed the feature-engineering
# notebook and the working-set construction.

# %%
print("BACKBLAZE PREPROCESSING SUMMARY")
print("=" * 60)
print(f"Rows:            {total_rows:,}")
print(f"Failure events:  {n_failures:,}")
print(f"Excluded SSDs:   {len(ssd_models)} models")
print(f"Modeled SMART:   {len(MODELED_SMART_IDS)} IDs ({len(ERA_GATED_SMART_IDS)} era-gated)")
print(f"GCS:             {gcs_uri}")
print("=" * 60)
