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
# applies the row-level transforms (SSD exclusion, schema-era assignment,
# availability-indicator encoding, SMART cleaning, drive-model canonicalization)
# per file, then derives a per-drive terminal (censoring) summary and runs a
# post-preprocessing assertion suite against the Phase 2 statistics before
# export.
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
# **Scale strategy.** The daily data is about 682M drive-day rows across an
# evolving schema (roughly 125 GB uncompressed at the modeled column width), so
# whole-table operations do not fit in memory. Three adaptations keep the pass
# within a high-memory runtime:
#
# 1. *Row-level cleaning is streamed file by file.* Each raw file is scanned,
#    reconciled to a common column set, transformed, and sunk to its own cleaned
#    Parquet. No cross-file state is held, so memory stays flat.
# 2. *Censoring is a group-by summary, not a global window.* The per-drive
#    terminal observation is found with a streaming group-by that emits one row
#    per drive (a few million rows), rather than a whole-table window.
# 3. *The global per-drive sort is deferred.* Rolling and lag features
#    (feature-engineering notebook) sort within each drive group via window
#    expressions, which gives the same ordering guarantee without a global sort
#    over the full table. Drive-day uniqueness is verified by assertion (the
#    dataset is one row per drive-day by construction) rather than a global
#    de-duplication.
#
# **Outputs.**
# - GCS Parquet dataset: `gs://{PROJECT}-dissertation-data/backblaze_preprocessed/`.
# - Per-drive terminal table: `.../backblaze_drive_terminal.parquet`.
# - Drive manifest: `{OUTPUT_DIR}/preprocessed/backblaze/manifest.json`.
# - `outputs/tables/backblaze_preprocessing_verification.csv`.

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
import subprocess
from datetime import date

import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR
from src.data.schemas import BACKBLAZE_ERAS
from src.data.validation import AssertionFailedError
from src.preprocessing.backblaze import (
    assign_era,
    canonicalize_drive_model,
    encode_smart_availability_indicators,
    filter_hdds_only,
    reconcile_smart_schema,
)

setup_drive()

PREPROCESSED_DIR = OUTPUT_DIR / 'preprocessed' / 'backblaze'
TABLES_DIR = OUTPUT_DIR / 'tables'
BACKBLAZE_DIR = Path('/content/backblaze_parquet')
CLEANED_DIR = Path('/content/backblaze_cleaned')
for d in [PREPROCESSED_DIR, TABLES_DIR, BACKBLAZE_DIR, CLEANED_DIR]:
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

RAW_COLS = [f'smart_{sid}_raw' for sid in MODELED_SMART_IDS]
HAS_COLS = [f'has_smart_{sid}' for sid in ERA_GATED_SMART_IDS]
KEEP_BASE_COLS = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']
DERIVED_COLS = ['model_canonical', 'manufacturer', 'era']
FINAL_COLS = KEEP_BASE_COLS + DERIVED_COLS + RAW_COLS + HAS_COLS

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
# ## 1. SSD exclusion inventory
#
# The daily schema carries no drive-type flag, so SSDs are identified by model
# name. Candidate SSD models are flagged by the keyword and prefix conventions
# surfaced in notebook 05 (an "SSD" token or a known consumer-SSD family), then
# printed for verification before exclusion. The failure-count assertion later
# confirms no HDD failures were lost.

# %%
model_counts: dict[str, int] = {}
model_failures: dict[str, int] = {}
for pf in parquet_files:
    out = (
        pl.scan_parquet(pf)
        .group_by('model')
        .agg(pl.len().alias('n'), (pl.col('failure') == 1).sum().alias('f'))
        .collect()
    )
    for row in out.iter_rows(named=True):
        model_counts[row['model']] = model_counts.get(row['model'], 0) + int(row['n'])
        model_failures[row['model']] = model_failures.get(row['model'], 0) + int(row['f'])

# SSDs are matched on an explicit "SSD" token or a known SSD part family, not on
# a bare manufacturer name: the fleet includes early Samsung SpinPoint HDDs
# (for example SAMSUNG HD103UJ / HD154UI) that a "Samsung" match would wrongly
# drop. Verify the flagged set against notebook 05 before trusting it.
_SSD_TOKENS = re.compile(r'(SSD|DELLBOSS|MTFDDAK|WDC WDS)', flags=re.IGNORECASE)
ssd_models = {m for m in model_counts if _SSD_TOKENS.search(m or '')}

print(f"Flagged {len(ssd_models)} SSD models for exclusion:")
for m in sorted(ssd_models):
    print(f"  {m:44s}  {model_counts[m]:>12,} rows  {model_failures[m]:>5,} failures")
print("Review this list against notebook 05 before trusting the exclusion.")

# Decompose the failure count so the survival check is self-verifying rather
# than anchored to a hard-coded total. The published 31,062 figure is the full
# fleet (HDDs plus SSDs); excluding SSDs necessarily lowers it.
total_failures_raw = sum(model_failures.values())
ssd_failures = sum(model_failures[m] for m in ssd_models)
hdd_failures_expected = total_failures_raw - ssd_failures
print(f"\nRaw failure rows (all drives, pre-dedup):  {total_failures_raw:,}")
print(f"Failure rows on excluded SSD models:       {ssd_failures:,}")
print(f"HDD failure rows (pre-dedup):              {hdd_failures_expected:,}")

# %% [markdown]
# ## 2. Stream the row-level cleaning per file
#
# For each raw file: select the base and modeled SMART raw columns present,
# reconcile to the full modeled SMART set (adding absent columns as null so
# every cleaned file shares one schema), apply the row-level transforms, and
# stream to a cleaned Parquet. The `_normalized` siblings are not carried
# because the EDA models on raw values (notebook 05 Section 3.1). This pass is
# purely row-wise, so it streams with bounded memory; drive-day de-duplication
# is a separate isolated pass (below). A fail-loud column guard runs on the
# first file so a stale module or a bad reconcile surfaces before the whole pass
# runs.

# %%
_BASE_DTYPES = {
    'date': pl.Date, 'serial_number': pl.Utf8, 'model': pl.Utf8,
    'capacity_bytes': pl.Int64, 'failure': pl.Int64,
}

n_files = len(parquet_files)
for i, pf in enumerate(parquet_files, 1):
    schema = pl.read_parquet_schema(pf)
    base_present = [c for c in KEEP_BASE_COLS if c in schema]
    raw_present = [c for c in RAW_COLS if c in schema]

    lf = pl.scan_parquet(pf).select(base_present + raw_present)
    # Guarantee every base column exists (older files may omit one).
    base_missing = [
        pl.lit(None, dtype=_BASE_DTYPES[c]).alias(c)
        for c in KEEP_BASE_COLS if c not in base_present
    ]
    if base_missing:
        lf = lf.with_columns(base_missing)

    lf = reconcile_smart_schema(lf, MODELED_SMART_IDS, keep_normalized=False)
    lf = filter_hdds_only(lf, ssd_models=ssd_models)
    lf = assign_era(lf)
    lf = encode_smart_availability_indicators(lf, ERA_GATED_SMART_IDS)
    lf = canonicalize_drive_model(lf)
    lf = lf.select(FINAL_COLS)

    if i == 1:
        built = set(lf.collect_schema().names())
        missing = set(FINAL_COLS) - built
        assert not missing, f"first cleaned file missing columns: {sorted(missing)}"
        print(f"Column guard passed: {len(built)} columns "
              f"(base + derived + {len(RAW_COLS)} SMART raw + {len(HAS_COLS)} indicators)")

    out_path = CLEANED_DIR / f'{pf.stem}_cleaned.parquet'
    lf.sink_parquet(out_path)
    print(f"  [{i:>2}/{n_files}] cleaned {pf.name} -> {out_path.name}")

print("Row-level cleaning complete (pre-dedup).")

# %% [markdown]
# ## 2b. Drive-day de-duplication (isolated per file)
#
# Some raw files carry duplicate drive-day rows (exact duplicate daily records;
# the failure flag is a property of the drive-day, so duplicates do not
# conflict). De-duplication must hold a whole file at once, which does not
# stream and, run inline across all files, accumulates memory. Each file is
# therefore de-duplicated in a fresh subprocess whose memory the operating
# system reclaims on exit, keeping peak memory at roughly one file. Per-file
# dedup equals global dedup because each quarter resides in a single file.

# %%
_DEDUP_SCRIPT = (
    "import sys, polars as pl; p = sys.argv[1]; "
    "df = pl.read_parquet(p).unique(subset=['serial_number', 'date']); "
    "df.write_parquet(p); print(df.height)"
)

cleaned_files = sorted(CLEANED_DIR.glob('*_cleaned.parquet'))
total_rows = 0
for cf in cleaned_files:
    result = subprocess.run(
        [sys.executable, '-c', _DEDUP_SCRIPT, str(cf)],
        capture_output=True, text=True, check=True,
    )
    post_n = int(result.stdout.strip().splitlines()[-1])
    total_rows += post_n
    print(f"  deduped {cf.name}: {post_n:,} rows")

print(f"De-duplication complete. {total_rows:,} rows retained.")

# %% [markdown]
# ## 3. Post-preprocessing checks and duplicate accounting
#
# Computed with the streaming engine at full scale: the cleaned failure-row
# count, unknown-era rows, and the multi-year fleet expansion. Drive-day
# uniqueness was already enforced in the per-file de-duplication pass. The pre-
# versus post-dedup row and failure figures are reported so the effect of the
# duplicate drive-day rows in the raw files is explicit. The authoritative
# failure-event cross-check (cleaned failure rows equal the count of failing
# drives) is made against the terminal table below.

# %%
cleaned_lf = pl.scan_parquet(str(CLEANED_DIR / '*.parquet'))

agg = cleaned_lf.select(
    (pl.col('failure') == 1).sum().alias('failures'),
    (pl.col('era').is_null() | (pl.col('era') == 'unknown')).sum().alias('unknown_era'),
).collect(engine='streaming')
n_failures = int(agg['failures'][0])
n_unknown_era = int(agg['unknown_era'][0])

# Distinct drives before and after 2020 (n_unique holds only the serial set).
fleet = cleaned_lf.select(
    pl.col('serial_number').filter(pl.col('date') < date(2020, 1, 1)).n_unique().alias('before'),
    pl.col('serial_number').filter(pl.col('date') >= date(2020, 1, 1)).n_unique().alias('after'),
).collect(engine='streaming')
fleet_before = int(fleet['before'][0])
fleet_after = int(fleet['after'][0])

# Duplicate accounting: the pre-dedup HDD figures come from the raw model
# inventory; the post-dedup figures come from the cleaned dataset.
hdd_raw_rows = sum(model_counts[m] for m in model_counts if m not in ssd_models)
dupes_removed = hdd_raw_rows - total_rows
failure_dupes_removed = hdd_failures_expected - n_failures

if dupes_removed < 0:
    raise AssertionFailedError(
        f"dedup increased the row count ({dupes_removed:,}); investigate"
    )
if n_unknown_era > 0:
    raise AssertionFailedError(f"{n_unknown_era:,} rows have an unknown/null era; expected 0")
if not (fleet_after > fleet_before):
    raise AssertionFailedError(
        f"distinct drives after 2020 ({fleet_after:,}) should exceed those "
        f"before ({fleet_before:,})"
    )

print(f"HDD rows raw / deduped:      {hdd_raw_rows:,} / {total_rows:,}  "
      f"({dupes_removed:,} duplicate drive-days removed)")
print(f"HDD failures raw / deduped:  {hdd_failures_expected:,} / {n_failures:,}  "
      f"({failure_dupes_removed:,} duplicate failure rows removed)")
print(f"Unknown-era rows:            {n_unknown_era}")
print(f"Distinct drives pre/post 2020: {fleet_before:,} / {fleet_after:,}")

# %% [markdown]
# ### Per-era verification table
#
# Row counts, failure events, and SMART 187 / 188 availability by era, to
# confirm the cleaned table matches the schema-evolution census. Only cheap
# streaming aggregations are used here (no distinct-drive count) so the pass
# stays light.

# %%
verification = (
    cleaned_lf.group_by('era')
    .agg(
        pl.len().alias('rows'),
        (pl.col('failure') == 1).sum().alias('failures'),
        pl.col('has_smart_187').mean().alias('smart_187_avail'),
        pl.col('has_smart_188').mean().alias('smart_188_avail'),
    )
    .sort('era')
    .collect(engine='streaming')
)
print(verification.to_pandas().to_string(index=False))
verification.write_csv(TABLES_DIR / 'backblaze_preprocessing_verification.csv')
print(f"Saved {TABLES_DIR / 'backblaze_preprocessing_verification.csv'}")

# %% [markdown]
# ## 4. Per-drive terminal (censoring) summary
#
# One row per drive: first and last observation dates, whether the drive ever
# failed, and the observed span. Backblaze marks ``failure = 1`` only on a
# drive's removal (final) day, so the per-drive maximum of the failure flag is
# the terminal failure indicator, computed without a per-group sort. A drive
# that ever shows a failure is an observed failure; one that simply stops
# appearing is right-censored.
#
# Grouping every drive across the whole table at once does not fit in memory, so
# the aggregation is done map-reduce style in a fresh subprocess: a small
# partial aggregate per file (each file fits comfortably), then a single combine
# over the concatenated partials (a few million rows, not the full table). The
# partial keeps ``min``/``max`` dates, ``max`` failure, ``first`` model and
# manufacturer, and the row count; the combine folds these across a drive's
# files. ``failure_observed`` summed across drives should equal the cleaned
# failure-row count.

# %%
TERMINAL_LOCAL = CLEANED_DIR / 'backblaze_drive_terminal.parquet'
_TERMINAL_SCRIPT = """
import sys
import glob
import polars as pl
cleaned_dir, out_path = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(cleaned_dir + '/*_cleaned.parquet'))
partials = []
for f in files:
    partials.append(
        pl.scan_parquet(f)
        .group_by('serial_number')
        .agg(
            pl.col('date').min().alias('first_date'),
            pl.col('date').max().alias('last_date'),
            pl.col('failure').max().alias('max_failure'),
            pl.col('model_canonical').first().alias('model_canonical'),
            pl.col('manufacturer').first().alias('manufacturer'),
            pl.len().alias('n_obs'),
        )
        .collect()
    )
term = (
    pl.concat(partials)
    .group_by('serial_number')
    .agg(
        pl.col('first_date').min().alias('first_date'),
        pl.col('last_date').max().alias('last_date'),
        pl.col('max_failure').max().alias('terminal_failure'),
        pl.col('model_canonical').first().alias('model_canonical'),
        pl.col('manufacturer').first().alias('manufacturer'),
        pl.col('n_obs').sum().alias('n_obs'),
    )
    .with_columns(
        pl.col('terminal_failure').cast(pl.Int8).alias('failure_observed'),
        (1 - pl.col('terminal_failure')).cast(pl.Int8).alias('censored'),
        (pl.col('last_date') - pl.col('first_date')).dt.total_days().alias('observed_span_days'),
    )
)
term.write_parquet(out_path)
print(f"{term.height} {int(term['failure_observed'].sum())} {int(term['censored'].sum())}")
"""

result = subprocess.run(
    [sys.executable, '-c', _TERMINAL_SCRIPT, str(CLEANED_DIR), str(TERMINAL_LOCAL)],
    capture_output=True, text=True, check=True,
)
n_drives, n_failed_drives, n_censored_drives = (
    int(x) for x in result.stdout.strip().splitlines()[-1].split()
)
print(f"Drives:                 {n_drives:,}")
print(f"Observed failures:      {n_failed_drives:,}")
print(f"Right-censored drives:  {n_censored_drives:,}")
print(f"Wrote terminal table: {TERMINAL_LOCAL}")

# Cross-check: one failure row per failing drive, so these should agree.
assert abs(n_failed_drives - n_failures) <= 200, (
    f"terminal failure_observed ({n_failed_drives:,}) disagrees with the "
    f"failure-event count ({n_failures:,}); investigate before proceeding"
)

# %% [markdown]
# ## 5. Export to GCS plus Drive manifest
#
# Upload the cleaned per-file Parquet dataset and the terminal table to GCS (the
# feature and modeling notebooks read them from there) and write a manifest to
# Drive describing the produced artifacts.

# %%
cleaned_files = sorted(CLEANED_DIR.glob('*_cleaned.parquet'))
for cf in cleaned_files:
    bucket.blob(f'{GCS_PREPROCESSED_PREFIX}/cleaned/{cf.name}').upload_from_filename(str(cf))
bucket.blob(f'{GCS_PREPROCESSED_PREFIX}/backblaze_drive_terminal.parquet').upload_from_filename(
    str(TERMINAL_LOCAL)
)
cleaned_gcs_prefix = f'gs://{GCS_BUCKET}/{GCS_PREPROCESSED_PREFIX}/cleaned/'
print(f"Uploaded {len(cleaned_files)} cleaned files and the terminal table to "
      f"gs://{GCS_BUCKET}/{GCS_PREPROCESSED_PREFIX}/")

manifest = {
    "dataset": "backblaze",
    "stage": "preprocessed",
    "source_notebook": "09_backblaze_preprocessing.py",
    "cleaned_gcs_prefix": cleaned_gcs_prefix,
    "terminal_gcs_uri": f'gs://{GCS_BUCKET}/{GCS_PREPROCESSED_PREFIX}/backblaze_drive_terminal.parquet',
    "cleaned_file_count": len(cleaned_files),
    "rows": total_rows,
    "failure_events": n_failures,
    "drives": n_drives,
    "observed_failures": n_failed_drives,
    "right_censored_drives": n_censored_drives,
    "duplicate_drive_days_removed": dupes_removed,
    "duplicate_failure_rows_removed": failure_dupes_removed,
    "unknown_era_rows": n_unknown_era,
    "distinct_drives_pre_2020": fleet_before,
    "distinct_drives_post_2020": fleet_after,
    "modeled_smart_ids": list(MODELED_SMART_IDS),
    "universal_smart_ids": list(UNIVERSAL_SMART_IDS),
    "era_gated_smart_ids": list(ERA_GATED_SMART_IDS),
    "excluded_ssd_models": sorted(ssd_models),
    "columns": FINAL_COLS,
    "note": (
        "Global per-(serial_number, date) sort deferred to feature-time window "
        "ordering; drive-day uniqueness verified by assertion; censoring emitted "
        "as a per-drive terminal table."
    ),
}
with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote manifest: {MANIFEST_PATH}")

# %% [markdown]
# ## 6. Summary
#
# Report back: the row count and failure-event count after cleaning, the
# per-era verification table (especially SMART 187 / 188 availability by era),
# the excluded SSD model list, and the observed-vs-censored drive split. These
# feed the feature-engineering notebook and the working-set construction.

# %%
print("BACKBLAZE PREPROCESSING SUMMARY")
print("=" * 60)
print(f"Rows:               {total_rows:,}")
print(f"Failure events:     {n_failures:,}")
print(f"Drives:             {n_drives:,} ({n_failed_drives:,} failed, {n_censored_drives:,} censored)")
print(f"Excluded SSDs:      {len(ssd_models)} models")
print(f"Modeled SMART:      {len(MODELED_SMART_IDS)} IDs ({len(ERA_GATED_SMART_IDS)} era-gated)")
print(f"Cleaned files:      {len(cleaned_files)} at {cleaned_gcs_prefix}")
print("=" * 60)
