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
# # 06 — Data Quality Summary for Chapter 3
#
# **Purpose:** Consolidate data quality assessments from both datasets into the
# formal **Data Quality Summary Table** required by Chapter 3.  This is a key
# deliverable — it documents every cleaning and handling decision with evidence
# from the EDA notebooks (02, 03, 05).
#
# **Inputs:**
# - Google Cluster Traces: BigQuery cached tables (`dissertation_lebel.*`)
# - Backblaze Hard Drive Data: Parquet files on GCS (produced by notebook 04)
# - EDA outputs from notebooks 02, 03, 05 (for cross-reference)
#
# **Outputs (saved to Drive → also committed to `outputs/tables/`):**
# - `data_quality_summary.csv` — high-level table (one row per dataset×table)
# - `google_column_profile.csv` — per-column detail for Google Traces
# - `backblaze_column_profile.csv` — per-column detail for Backblaze
# - `class_imbalance.csv` — class distribution details for both datasets
#
# **Sections:**
# 1. Setup & Helpers
# 2. Google Cluster Traces — Quality Profile
# 3. Backblaze Hard Drive Data — Quality Profile
# 4. High-Level Data Quality Summary Table
# 5. Class Imbalance & Distribution Details
# 6. Key Findings Summary
#
# **Prerequisites:**
# - Notebooks 01–05 completed
# - Colab Secrets: `GCP_PROJECT_ID`

# %% [markdown]
# ---
# ## 0. Colab Session Setup

# %%
# !pip install -q polars google-cloud-bigquery google-cloud-storage

# %%
from google.colab import userdata

PROJECT_ID = userdata.get('GCP_PROJECT_ID')
DATASET = f"{PROJECT_ID}.dissertation_lebel"
print(f"GCP Project: {PROJECT_ID}")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from google.colab import auth
auth.authenticate_user()

# %%
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
TABLES_DIR = DRIVE_PATH / 'outputs' / 'tables'
FIGURES_DIR = DRIVE_PATH / 'outputs' / 'figures'
CHECKPOINT_DIR = DRIVE_PATH / 'checkpoints'
BACKBLAZE_DIR = Path('/content/backblaze_parquet')

for dir_path in [TABLES_DIR, FIGURES_DIR, CHECKPOINT_DIR, BACKBLAZE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# %%
from google.cloud import bigquery
bq_client = bigquery.Client(project=PROJECT_ID)

# %%
import polars as pl
import numpy as np
import gc

# %% [markdown]
# ### Helper Functions

# %%
def fqn(table: str) -> str:
    """Return fully-qualified BigQuery table name."""
    return f"`{DATASET}.{table}`"


def run_query(sql: str) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame."""
    return pl.from_pandas(bq_client.query(sql).to_dataframe())


def save_table(df: pl.DataFrame, name: str) -> None:
    """Save a Polars DataFrame as CSV to the Drive tables directory."""
    path = TABLES_DIR / f"{name}.csv"
    df.write_csv(str(path))
    print(f"  Saved: {path}  ({len(df)} rows)")


def save_checkpoint(obj, name: str) -> None:
    """Pickle an object to Drive checkpoints directory."""
    import pickle
    path = CHECKPOINT_DIR / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(name: str):
    """Load a pickled checkpoint, or return None if absent."""
    import pickle
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# %% [markdown]
# ---
# # Section 1: Google Cluster Traces — Quality Profile
#
# Query each cached BigQuery table for:
# - Row/column counts
# - Per-column null counts, data types, distinct values
# - Numeric min/max/mean
# - Top frequent values for categoricals
# - Duplicate row check
# - Temporal coverage

# %% [markdown]
# ### 1.1 Table-Level Dimensions & Temporal Coverage
#
# These are already confirmed in notebooks 02/03, but we re-query here to
# produce a single self-contained quality report.

# %%
GOOGLE_TABLES = [
    'instance_events_full',
    'machine_events_full',
    'instance_usage_full',
    'collection_events_full',
    'machine_attributes_full',
]

# Time columns per table (for temporal coverage)
TIME_COLS = {
    'instance_events_full': 'time',
    'machine_events_full': 'time',
    'instance_usage_full': 'start_time',
    'collection_events_full': 'time',
    'machine_attributes_full': 'time',
}

# %%
google_dimensions = load_checkpoint('dq_google_dimensions')

if google_dimensions is not None:
    print("Loaded Google dimensions from checkpoint")
else:
    dim_rows = []
    for table in GOOGLE_TABLES:
        time_col = TIME_COLS[table]
        sql = f"""
        SELECT
            '{table}' AS table_name,
            COUNT(*) AS total_rows,
            MIN({time_col}) AS min_time_us,
            MAX(CASE WHEN {time_col} < 9000000000000000000 THEN {time_col} END) AS max_time_us
        FROM {fqn(table)}
        """
        row = run_query(sql).to_dicts()[0]

        # Column count from INFORMATION_SCHEMA
        col_sql = f"""
        SELECT COUNT(*) AS col_count
        FROM `{DATASET}`.INFORMATION_SCHEMA.COLUMNS
        WHERE table_name = '{table}'
        """
        row['total_columns'] = run_query(col_sql).to_dicts()[0]['col_count']
        dim_rows.append(row)

    google_dimensions = pl.DataFrame(dim_rows)
    save_checkpoint(google_dimensions, 'dq_google_dimensions')

print(google_dimensions.to_pandas().to_string(index=False))

# %% [markdown]
# *Temporal coverage for all Google tables spans May 2019 (31 days).
# `instance_events_full` contains sentinel values (0 and 2^63-1) which are
# excluded from the max_time_us calculation.*

# %% [markdown]
# ### 1.2 Per-Column Quality Profile (Google)
#
# For each table, collect: column name, BigQuery type, null count, null %,
# distinct count, and (for numerics) min/max/mean.  For categoricals, we also
# grab the top-5 most frequent values.

# %%
google_col_profile = load_checkpoint('dq_google_col_profile')

if google_col_profile is not None:
    print("Loaded Google column profile from checkpoint")
else:
    all_profiles = []

    for table in GOOGLE_TABLES:
        # --- Get column metadata from INFORMATION_SCHEMA ---
        meta_sql = f"""
        SELECT column_name, data_type, is_nullable
        FROM `{DATASET}`.INFORMATION_SCHEMA.COLUMNS
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
        """
        meta = run_query(meta_sql)

        for row in meta.to_dicts():
            col = row['column_name']
            dtype = row['data_type']

            # Skip ARRAY/STRUCT columns — cannot COUNT(DISTINCT) or aggregate
            if 'ARRAY' in dtype or 'STRUCT' in dtype:
                all_profiles.append({
                    'table': table,
                    'column': col,
                    'bq_type': dtype,
                    'total_rows': None,
                    'non_null_count': None,
                    'null_count': None,
                    'null_pct': None,
                    'distinct_count': None,
                    'min_val': None,
                    'max_val': None,
                    'mean_val': None,
                    'top5_values': 'ARRAY/STRUCT — see raw schema',
                })
                continue

            # --- Null count and distinct values ---
            stats_sql = f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNTIF({col} IS NOT NULL) AS non_null_count,
                COUNTIF({col} IS NULL) AS null_count,
                ROUND(COUNTIF({col} IS NULL) * 100.0 / COUNT(*), 4) AS null_pct,
                COUNT(DISTINCT {col}) AS distinct_count
            FROM {fqn(table)}
            """
            stats = run_query(stats_sql).to_dicts()[0]

            # --- Numeric stats (for INT64, FLOAT64, NUMERIC, BIGNUMERIC) ---
            min_val = None
            max_val = None
            mean_val = None
            if dtype in ('INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC'):
                num_sql = f"""
                SELECT
                    CAST(MIN({col}) AS FLOAT64) AS min_val,
                    CAST(MAX({col}) AS FLOAT64) AS max_val,
                    AVG(CAST({col} AS FLOAT64)) AS mean_val
                FROM {fqn(table)}
                WHERE {col} IS NOT NULL
                """
                nstats = run_query(num_sql).to_dicts()[0]
                min_val = nstats['min_val']
                max_val = nstats['max_val']
                mean_val = nstats['mean_val']

            # --- Top-5 frequent values (for low-cardinality columns) ---
            top5_str = None
            if stats['distinct_count'] is not None and stats['distinct_count'] <= 500:
                top_sql = f"""
                SELECT CAST({col} AS STRING) AS val, COUNT(*) AS cnt
                FROM {fqn(table)}
                WHERE {col} IS NOT NULL
                GROUP BY val
                ORDER BY cnt DESC
                LIMIT 5
                """
                top5 = run_query(top_sql)
                if len(top5) > 0:
                    pairs = [
                        f"{r['val']} ({r['cnt']:,})"
                        for r in top5.to_dicts()
                    ]
                    top5_str = '; '.join(pairs)

            all_profiles.append({
                'table': table,
                'column': col,
                'bq_type': dtype,
                'total_rows': stats['total_rows'],
                'non_null_count': stats['non_null_count'],
                'null_count': stats['null_count'],
                'null_pct': stats['null_pct'],
                'distinct_count': stats['distinct_count'],
                'min_val': min_val,
                'max_val': max_val,
                'mean_val': mean_val,
                'top5_values': top5_str,
            })

        print(f"  {table}: {len(meta)} columns profiled")

    google_col_profile = pl.DataFrame(all_profiles)
    save_checkpoint(google_col_profile, 'dq_google_col_profile')

print(f"\nGoogle column profile: {len(google_col_profile)} rows")
print(google_col_profile.select(['table', 'column', 'bq_type', 'null_pct', 'distinct_count'])
      .to_pandas().to_string(index=False))

# %%
save_table(google_col_profile, 'google_column_profile')

# %% [markdown]
# *The per-column profile captures every column across all five Google Traces
# tables.  Notable null patterns confirmed: machine_id (48% — structural,
# pre-scheduling events), sample_memory (100% — never collected), CPI/MAPI
# (20.5% — MNAR), max_per_machine/switch (99%+ — rarely set).*

# %% [markdown]
# ### 1.3 Duplicate Check (Google)
#
# For each table, check if there are exact duplicate rows.  Due to the table
# sizes we use an approximate approach: count rows vs. count distinct composite
# keys.

# %%
google_dup_results = load_checkpoint('dq_google_duplicates')

if google_dup_results is not None:
    print("Loaded Google duplicate check from checkpoint")
else:
    # Define the natural key columns for each table
    KEY_COLS = {
        'instance_events_full': 'time, type, collection_id, instance_index',
        'machine_events_full': 'time, type, machine_id',
        'instance_usage_full': 'start_time, end_time, collection_id, instance_index, machine_id',
        'collection_events_full': 'time, type, collection_id',
        'machine_attributes_full': 'time, machine_id, name',
    }

    dup_rows = []
    for table in GOOGLE_TABLES:
        keys = KEY_COLS[table]
        # Use GROUP BY + HAVING to find duplicates — avoids COUNT(DISTINCT STRUCT)
        # which BigQuery does not support for ARRAY/STRUCT column types.
        sql = f"""
        WITH key_counts AS (
            SELECT {keys}, COUNT(*) AS n
            FROM {fqn(table)}
            GROUP BY {keys}
        )
        SELECT
            SUM(n) AS total_rows,
            COUNT(*) AS distinct_keys,
            SUM(n - 1) AS duplicate_rows
        FROM key_counts
        """
        result = run_query(sql).to_dicts()[0]
        dup_count = result['duplicate_rows']
        dup_rows.append({
            'table': table,
            'total_rows': result['total_rows'],
            'distinct_keys': result['distinct_keys'],
            'duplicate_rows': dup_count,
            'dup_pct': round(dup_count * 100.0 / result['total_rows'], 4)
                       if result['total_rows'] > 0 else 0.0,
        })
        print(f"  {table}: {dup_count:,} duplicates ({dup_rows[-1]['dup_pct']:.4f}%)")

    google_dup_results = pl.DataFrame(dup_rows)
    save_checkpoint(google_dup_results, 'dq_google_duplicates')

print(google_dup_results.to_pandas().to_string(index=False))

# %% [markdown]
# ### 1.4 Google Class Distribution
#
# Failure definition (EDA-confirmed in notebook 03):
# - **Failure:** FAIL (type 5) + LOST (type 8)
# - **Success:** FINISH (type 6)
# - **Excluded:** EVICT (4), KILL (7), lifecycle events (0-3, 9-10)
#
# Class distribution is computed over **terminal events only** (types 4–8).

# %%
google_class_dist = load_checkpoint('dq_google_class_dist')

if google_class_dist is not None:
    print("Loaded Google class distribution from checkpoint")
else:
    sql = f"""
    WITH terminal_events AS (
        SELECT
            type,
            CASE
                WHEN type IN (5, 8) THEN 'failure'
                WHEN type = 6 THEN 'success'
                WHEN type = 4 THEN 'evict_excluded'
                WHEN type = 7 THEN 'kill_excluded'
            END AS outcome
        FROM {fqn('instance_events_full')}
        WHERE type BETWEEN 4 AND 8
    )
    SELECT
        outcome,
        type,
        COUNT(*) AS event_count
    FROM terminal_events
    GROUP BY outcome, type
    ORDER BY type
    """
    google_class_dist = run_query(sql)
    save_checkpoint(google_class_dist, 'dq_google_class_dist')

print(google_class_dist.to_pandas().to_string(index=False))

# %%
# Compute failure vs success totals (excluding EVICT and KILL)
_g = google_class_dist.filter(pl.col('outcome').is_in(['failure', 'success']))
google_failure_total = _g.filter(pl.col('outcome') == 'failure')['event_count'].sum()
google_success_total = _g.filter(pl.col('outcome') == 'success')['event_count'].sum()
google_model_total = google_failure_total + google_success_total
google_imbalance = round(google_success_total / google_failure_total, 1)

print(f"\nGoogle Traces — Modeled Population (types 5, 6, 8):")
print(f"  Success (FINISH):     {google_success_total:>14,}  "
      f"({google_success_total/google_model_total*100:.2f}%)")
print(f"  Failure (FAIL+LOST):  {google_failure_total:>14,}  "
      f"({google_failure_total/google_model_total*100:.2f}%)")
print(f"  Total modeled:        {google_model_total:>14,}")
print(f"  Imbalance ratio:      {google_imbalance}:1 (success:failure)")

# %% [markdown]
# *Google class distribution confirms the 3.4:1 success-to-failure ratio
# (moderate imbalance). EVICT and KILL events are excluded from the modeling
# population per the EDA-validated failure definition.*

# %% [markdown]
# ---
# # Section 2: Backblaze Hard Drive Data — Quality Profile
#
# Load Parquet files from GCS, compute per-column quality metrics, duplicate
# checks, and class distribution.

# %% [markdown]
# ### 2.0 Download Parquet Files from GCS

# %%
from google.cloud import storage

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_PARQUET_PREFIX = 'backblaze_parquet/'

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)

parquet_blobs = sorted(
    [b for b in bucket.list_blobs(prefix=GCS_PARQUET_PREFIX)
     if b.name.endswith('.parquet')],
    key=lambda b: b.name,
)

print(f"Found {len(parquet_blobs)} Parquet files on "
      f"gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}\n")

for b in parquet_blobs:
    name = b.name.split('/')[-1]
    local_path = BACKBLAZE_DIR / name
    if local_path.exists() and local_path.stat().st_size == b.size:
        print(f"  {name:35s}  {b.size/1024**2:>8.1f} MB  (cached)")
    else:
        print(f"  {name:35s}  {b.size/1024**2:>8.1f} MB  downloading...",
              end='', flush=True)
        b.download_to_filename(str(local_path))
        print("  done")

print(f"\nAll Parquet files available at {BACKBLAZE_DIR}")
parquet_files = sorted(BACKBLAZE_DIR.glob('*.parquet'))

# %% [markdown]
# ### 2.1 Backblaze Grand Totals

# %%
bb_grand_stats = load_checkpoint('dq_bb_grand_stats')

if bb_grand_stats is not None:
    print("Loaded Backblaze grand stats from checkpoint")
else:
    total_rows = 0
    total_failures = 0
    all_cols = set()
    date_min = None
    date_max = None

    for pf in parquet_files:
        schema = pl.read_parquet_schema(pf)
        all_cols.update(schema.keys())

        stats = pl.scan_parquet(pf).select(
            pl.len().alias('n'),
            pl.col('failure').sum().alias('failures'),
            pl.col('date').min().alias('date_min'),
            pl.col('date').max().alias('date_max'),
        ).collect()

        row = stats.to_dicts()[0]
        total_rows += row['n']
        total_failures += row['failures']
        if date_min is None or row['date_min'] < date_min:
            date_min = row['date_min']
        if date_max is None or row['date_max'] > date_max:
            date_max = row['date_max']

    bb_grand_stats = {
        'total_rows': total_rows,
        'total_failures': total_failures,
        'total_columns_union': len(all_cols),
        'date_min': str(date_min),
        'date_max': str(date_max),
    }
    save_checkpoint(bb_grand_stats, 'dq_bb_grand_stats')

print(f"Total records:    {bb_grand_stats['total_rows']:>14,}")
print(f"Total failures:   {bb_grand_stats['total_failures']:>14,}")
print(f"Columns (union):  {bb_grand_stats['total_columns_union']:>14,}")
print(f"Date range:       {bb_grand_stats['date_min']} to {bb_grand_stats['date_max']}")

# %% [markdown]
# ### 2.2 Per-Column Quality Profile (Backblaze)
#
# Because Backblaze data has schema evolution (different SMART attributes across
# years), we compute column profiles across all files combined using lazy scans.
# We profile the 5 core columns plus all SMART attributes present in the union
# schema.

# %%
bb_col_profile = load_checkpoint('dq_bb_col_profile')

if bb_col_profile is not None:
    print("Loaded Backblaze column profile from checkpoint")
else:
    # Build the union schema (all columns that appear in any file)
    union_schema = {}
    for pf in parquet_files:
        schema = pl.read_parquet_schema(pf)
        for col_name, col_dtype in schema.items():
            if col_name not in union_schema:
                union_schema[col_name] = str(col_dtype)

    # Separate core columns and SMART columns
    core_cols = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']
    smart_cols = sorted(
        [c for c in union_schema if c.startswith('smart_')],
        key=lambda c: (int(c.split('_')[1]), c.split('_')[2]),
    )

    all_cols_ordered = core_cols + smart_cols
    total_rows = bb_grand_stats['total_rows']
    profiles = []

    # --- Core columns: profile across all files ---
    print("Profiling core columns...")
    for col in core_cols:
        # All files have core columns — safe to scan all
        non_null = 0
        null_count = 0
        distinct_approx = set()
        min_val = None
        max_val = None
        sum_val = 0.0
        count_for_mean = 0

        for pf in parquet_files:
            lf = pl.scan_parquet(pf)

            stats = lf.select(
                pl.col(col).is_not_null().sum().alias('non_null'),
                pl.col(col).is_null().sum().alias('null_count'),
            ).collect()

            s = stats.to_dicts()[0]
            non_null += s['non_null']
            null_count += s['null_count']

            # Numeric stats
            if col in ('capacity_bytes', 'failure'):
                nstats = lf.select(
                    pl.col(col).min().alias('mn'),
                    pl.col(col).max().alias('mx'),
                    pl.col(col).mean().alias('avg'),
                    pl.col(col).count().alias('cnt'),
                ).collect().to_dicts()[0]

                if nstats['mn'] is not None:
                    min_val = nstats['mn'] if min_val is None else min(min_val, nstats['mn'])
                if nstats['mx'] is not None:
                    max_val = nstats['mx'] if max_val is None else max(max_val, nstats['mx'])
                if nstats['avg'] is not None:
                    sum_val += nstats['avg'] * nstats['cnt']
                    count_for_mean += nstats['cnt']

        mean_val = sum_val / count_for_mean if count_for_mean > 0 else None

        profiles.append({
            'column': col,
            'polars_dtype': union_schema[col],
            'total_rows': total_rows,
            'non_null_count': non_null,
            'null_count': null_count,
            'null_pct': round(null_count * 100.0 / total_rows, 4) if total_rows > 0 else 0.0,
            'distinct_count': None,  # too expensive for full dataset
            'min_val': str(min_val) if min_val is not None else None,
            'max_val': str(max_val) if max_val is not None else None,
            'mean_val': round(mean_val, 6) if mean_val is not None else None,
            'top5_values': None,
        })
        print(f"  {col}: null_pct={profiles[-1]['null_pct']:.4f}%")

    # --- SMART columns: profile across files that contain them ---
    print(f"\nProfiling {len(smart_cols)} SMART columns...")
    for i, col in enumerate(smart_cols):
        non_null = 0
        null_count = 0
        files_present = 0
        min_val = None
        max_val = None
        sum_val = 0.0
        count_for_mean = 0

        for pf in parquet_files:
            file_schema = pl.read_parquet_schema(pf)
            if col not in file_schema:
                # Column not in this file — entire file is "null" for this column
                file_rows = pl.scan_parquet(pf).select(pl.len()).collect().item()
                null_count += file_rows
                continue

            files_present += 1
            lf = pl.scan_parquet(pf).select(col)

            stats = lf.select(
                pl.col(col).is_not_null().sum().alias('non_null'),
                pl.col(col).is_null().sum().alias('null_count'),
            ).collect().to_dicts()[0]

            non_null += stats['non_null']
            null_count += stats['null_count']

            # Numeric stats for SMART columns
            nstats = lf.select(
                pl.col(col).min().alias('mn'),
                pl.col(col).max().alias('mx'),
                pl.col(col).mean().alias('avg'),
                pl.col(col).count().alias('cnt'),
            ).collect().to_dicts()[0]

            if nstats['mn'] is not None:
                min_val = nstats['mn'] if min_val is None else min(min_val, nstats['mn'])
            if nstats['mx'] is not None:
                max_val = nstats['mx'] if max_val is None else max(max_val, nstats['mx'])
            if nstats['avg'] is not None and nstats['cnt'] > 0:
                sum_val += nstats['avg'] * nstats['cnt']
                count_for_mean += nstats['cnt']

        mean_val = sum_val / count_for_mean if count_for_mean > 0 else None

        profiles.append({
            'column': col,
            'polars_dtype': union_schema[col],
            'total_rows': total_rows,
            'non_null_count': non_null,
            'null_count': null_count,
            'null_pct': round(null_count * 100.0 / total_rows, 4) if total_rows > 0 else 0.0,
            'distinct_count': None,
            'min_val': str(min_val) if min_val is not None else None,
            'max_val': str(max_val) if max_val is not None else None,
            'mean_val': round(mean_val, 6) if mean_val is not None else None,
            'top5_values': f"present in {files_present}/{len(parquet_files)} files",
        })

        if (i + 1) % 20 == 0:
            print(f"  ...profiled {i + 1}/{len(smart_cols)} SMART columns")

    bb_col_profile = pl.DataFrame(profiles)
    save_checkpoint(bb_col_profile, 'dq_bb_col_profile')
    print(f"\nDone — {len(bb_col_profile)} columns profiled")

# %%
save_table(bb_col_profile, 'backblaze_column_profile')

# %%
# Display top-20 columns by null rate
print("Top 20 columns by null %:")
print(
    bb_col_profile
    .sort('null_pct', descending=True)
    .head(20)
    .select(['column', 'null_pct', 'non_null_count'])
    .to_pandas()
    .to_string(index=False)
)

# %% [markdown]
# *Backblaze schema evolution means many SMART columns exist only in subsets of
# files.  Columns present in all files have low null rates; columns introduced
# later or retired earlier have high structural null rates.  This is analogous
# to Google's structural nulls (machine_id pre-scheduling).*

# %% [markdown]
# ### 2.3 Duplicate Check (Backblaze)
#
# Natural key: (date, serial_number). Each drive should have at most one
# observation per day.

# %%
bb_dup_results = load_checkpoint('dq_bb_duplicates')

if bb_dup_results is not None:
    print("Loaded Backblaze duplicate check from checkpoint")
else:
    total_rows = 0
    total_distinct_keys = 0

    for pf in parquet_files:
        stats = pl.scan_parquet(pf).select(
            pl.len().alias('n'),
            pl.struct('date', 'serial_number').n_unique().alias('distinct_keys'),
        ).collect().to_dicts()[0]

        total_rows += stats['n']
        total_distinct_keys += stats['distinct_keys']

    dup_count = total_rows - total_distinct_keys
    bb_dup_results = {
        'total_rows': total_rows,
        'distinct_keys': total_distinct_keys,
        'duplicate_rows': dup_count,
        'dup_pct': round(dup_count * 100.0 / total_rows, 4) if total_rows > 0 else 0.0,
    }
    save_checkpoint(bb_dup_results, 'dq_bb_duplicates')

print(f"Total rows:       {bb_dup_results['total_rows']:>14,}")
print(f"Distinct keys:    {bb_dup_results['distinct_keys']:>14,}")
print(f"Duplicate rows:   {bb_dup_results['duplicate_rows']:>14,}")
print(f"Duplicate %:      {bb_dup_results['dup_pct']:.4f}%")

# %% [markdown]
# ### 2.4 Backblaze Class Distribution

# %%
bb_class_dist = load_checkpoint('dq_bb_class_dist')

if bb_class_dist is not None:
    print("Loaded Backblaze class distribution from checkpoint")
else:
    healthy_count = 0
    failure_count = 0

    for pf in parquet_files:
        stats = pl.scan_parquet(pf).select(
            (pl.col('failure') == 0).sum().alias('healthy'),
            (pl.col('failure') == 1).sum().alias('failed'),
        ).collect().to_dicts()[0]

        healthy_count += stats['healthy']
        failure_count += stats['failed']

    bb_class_dist = {
        'healthy_count': healthy_count,
        'failure_count': failure_count,
        'total': healthy_count + failure_count,
    }
    save_checkpoint(bb_class_dist, 'dq_bb_class_dist')

bb_total = bb_class_dist['total']
bb_failure = bb_class_dist['failure_count']
bb_healthy = bb_class_dist['healthy_count']
bb_imbalance = round(bb_healthy / bb_failure, 1) if bb_failure > 0 else float('inf')

print(f"Backblaze — Class Distribution:")
print(f"  Healthy (failure=0): {bb_healthy:>14,}  ({bb_healthy/bb_total*100:.4f}%)")
print(f"  Failed  (failure=1): {bb_failure:>14,}  ({bb_failure/bb_total*100:.4f}%)")
print(f"  Total:               {bb_total:>14,}")
print(f"  Imbalance ratio:     {bb_imbalance}:1 (healthy:failed)")

# %% [markdown]
# *Backblaze has extreme class imbalance — the daily failure rate is very low
# because each drive contributes one observation per day and failures are rare
# events across long lifespans.  This contrasts with Google's moderate 3.4:1
# ratio.*

# %% [markdown]
# ---
# # Section 3: High-Level Data Quality Summary Table
#
# One row per dataset×table with all key quality dimensions.  This is the
# primary Chapter 3 deliverable from this notebook.

# %%
# --- Build the summary rows ---

summary_rows = []

# --- Google Traces tables ---
for table in GOOGLE_TABLES:
    dim = google_dimensions.filter(pl.col('table_name') == table).to_dicts()[0]

    # Per-column null info from profile
    tbl_profile = google_col_profile.filter(
        (pl.col('table') == table) & (pl.col('null_pct').is_not_null())
    )

    # Overall missing rate (average null % across all columns)
    if len(tbl_profile) > 0:
        overall_null = round(tbl_profile['null_pct'].mean(), 4)
    else:
        overall_null = 0.0

    # Top 3 null columns
    top3_null = (
        tbl_profile
        .sort('null_pct', descending=True)
        .head(3)
    )
    top3_str = '; '.join(
        f"{r['column']} ({r['null_pct']:.2f}%)"
        for r in top3_null.to_dicts()
    )

    # Temporal coverage
    min_t = dim.get('min_time_us')
    max_t = dim.get('max_time_us')
    temporal_str = "May 2019 (31 days)"

    # Duplicate count for this table
    dup_row = google_dup_results.filter(pl.col('table') == table)
    dup_count = dup_row['duplicate_rows'].item() if len(dup_row) > 0 else 0

    # Class distribution (only relevant for instance_events_full)
    if table == 'instance_events_full':
        class_str = (f"success={google_success_total:,} "
                     f"({google_success_total/google_model_total*100:.1f}%); "
                     f"failure={google_failure_total:,} "
                     f"({google_failure_total/google_model_total*100:.1f}%)")
        imbalance_str = f"{google_imbalance}:1"
    else:
        class_str = "N/A (not target table)"
        imbalance_str = "N/A"

    summary_rows.append({
        'dataset': 'Google Cluster Traces',
        'table_name': table,
        'total_rows': dim['total_rows'],
        'total_columns': dim['total_columns'],
        'temporal_coverage': temporal_str,
        'overall_missing_rate_pct': overall_null,
        'top3_null_columns': top3_str,
        'duplicate_rows': dup_count,
        'class_distribution': class_str,
        'imbalance_ratio': imbalance_str,
    })

# --- Backblaze (single logical table) ---
bb_profile_sorted = (
    bb_col_profile
    .filter(pl.col('null_pct').is_not_null())
    .sort('null_pct', descending=True)
)

bb_overall_null = round(bb_profile_sorted['null_pct'].mean(), 4) if len(bb_profile_sorted) > 0 else 0.0

bb_top3_null = bb_profile_sorted.head(3)
bb_top3_str = '; '.join(
    f"{r['column']} ({r['null_pct']:.2f}%)"
    for r in bb_top3_null.to_dicts()
)

bb_class_str = (f"healthy={bb_healthy:,} "
                f"({bb_healthy/bb_total*100:.4f}%); "
                f"failed={bb_failure:,} "
                f"({bb_failure/bb_total*100:.4f}%)")

summary_rows.append({
    'dataset': 'Backblaze Hard Drive Data',
    'table_name': 'daily_drive_stats (all years)',
    'total_rows': bb_grand_stats['total_rows'],
    'total_columns': bb_grand_stats['total_columns_union'],
    'temporal_coverage': f"{bb_grand_stats['date_min']} to {bb_grand_stats['date_max']}",
    'overall_missing_rate_pct': bb_overall_null,
    'top3_null_columns': bb_top3_str,
    'duplicate_rows': bb_dup_results['duplicate_rows'],
    'class_distribution': bb_class_str,
    'imbalance_ratio': f"{bb_imbalance}:1",
})

# --- Build final DataFrame ---
dq_summary = pl.DataFrame(summary_rows)

# %%
print("=== DATA QUALITY SUMMARY TABLE (Chapter 3) ===\n")
print(dq_summary.to_pandas().to_string(index=False))

# %%
save_table(dq_summary, 'data_quality_summary')

# %% [markdown]
# *This table is the key Chapter 3 deliverable.  It provides a one-glance
# overview of both datasets' quality dimensions: size, coverage, missingness,
# duplicates, and class balance.  The Google Traces occupy five tables totaling
# ~9.3B rows; Backblaze is a single logical table spanning 12+ years.*

# %% [markdown]
# ---
# # Section 4: Class Imbalance & Distribution Details

# %%
# Build a unified class imbalance table for both datasets

class_rows = []

# Google — detailed breakdown by event type
for row in google_class_dist.to_dicts():
    class_rows.append({
        'dataset': 'Google Cluster Traces',
        'class_label': row['outcome'],
        'event_type': row['type'],
        'count': row['event_count'],
        'pct_of_modeled': round(row['event_count'] / google_model_total * 100, 4)
                          if row['outcome'] in ('failure', 'success') else None,
        'pct_of_all_terminal': None,  # filled below
    })

# Add terminal event percentages
_terminal_total = google_class_dist['event_count'].sum()
for r in class_rows:
    r['pct_of_all_terminal'] = round(r['count'] / _terminal_total * 100, 4)

# Google summary rows
class_rows.append({
    'dataset': 'Google Cluster Traces',
    'class_label': 'TOTAL_MODELED',
    'event_type': None,
    'count': google_model_total,
    'pct_of_modeled': 100.0,
    'pct_of_all_terminal': round(google_model_total / _terminal_total * 100, 4),
})
class_rows.append({
    'dataset': 'Google Cluster Traces',
    'class_label': 'IMBALANCE_RATIO',
    'event_type': None,
    'count': None,
    'pct_of_modeled': None,
    'pct_of_all_terminal': None,
})

# Backblaze
class_rows.append({
    'dataset': 'Backblaze Hard Drive Data',
    'class_label': 'healthy',
    'event_type': 0,
    'count': bb_healthy,
    'pct_of_modeled': round(bb_healthy / bb_total * 100, 4),
    'pct_of_all_terminal': round(bb_healthy / bb_total * 100, 4),
})
class_rows.append({
    'dataset': 'Backblaze Hard Drive Data',
    'class_label': 'failed',
    'event_type': 1,
    'count': bb_failure,
    'pct_of_modeled': round(bb_failure / bb_total * 100, 4),
    'pct_of_all_terminal': round(bb_failure / bb_total * 100, 4),
})
class_rows.append({
    'dataset': 'Backblaze Hard Drive Data',
    'class_label': 'TOTAL',
    'event_type': None,
    'count': bb_total,
    'pct_of_modeled': 100.0,
    'pct_of_all_terminal': 100.0,
})
class_rows.append({
    'dataset': 'Backblaze Hard Drive Data',
    'class_label': 'IMBALANCE_RATIO',
    'event_type': None,
    'count': None,
    'pct_of_modeled': None,
    'pct_of_all_terminal': None,
})

class_imbalance_df = pl.DataFrame(class_rows)

# %%
print("=== CLASS IMBALANCE DETAILS ===\n")
print(class_imbalance_df.to_pandas().to_string(index=False))

# %%
save_table(class_imbalance_df, 'class_imbalance')

# %% [markdown]
# *Google's 3.4:1 ratio (success:failure) is moderate and manageable with
# cost-sensitive learning + SMOTE.  Backblaze's extreme imbalance (daily
# observations with rare failure events) requires different strategies:
# undersampling healthy observations, or aggregating to drive-level labels
# with sliding windows.  Both imbalance levels are documented in the literature
# (Li et al., 2021; Yu et al., 2023).*

# %% [markdown]
# ---
# # Section 5: Key Findings Summary
#
# ### Google Cluster Traces — Data Quality
#
# 1. **Five tables**, ~9.3B rows total, 31-day trace (May 2019).
# 2. **Structural nulls** dominate missingness: `machine_id` (48%) is null for
#    pre-scheduling events (types 0–2) -  not missing data, but absent by design.
# 3. **100% null column:** `sample_memory` in `instance_usage_full`-  never collected.
#    Will be dropped in Phase 3.
# 4. **MNAR pattern:** `cycles_per_instruction` and `memory_accesses_per_instruction`
#    are 20.5% null — workload-type dependent (MNAR). Requires indicator encoding,
#    not standard imputation.
# 5. **Near-empty columns:** `max_per_machine` (99.3%) and `max_per_switch` (99.7%)
#    in `collection_events_full` — will be dropped.
# 6. **Class balance:** 3.4:1 success:failure — moderate imbalance.
#
# ### Backblaze Hard Drive Data — Data Quality
#
# 1. **Single logical table**, 12+ years of daily drive observations.
# 2. **Schema evolution:** SMART attribute availability varies across years —
#    columns added/retired over time create structural nulls.
# 3. **Core columns** (`date`, `serial_number`, `model`, `capacity_bytes`, `failure`)
#    are fully populated across all files.
# 4. **Class imbalance:** Extreme at the daily-observation level (failure is a
#    rare daily event). Drive-level aggregation in Phase 3 will reframe the ratio.
#
# ### Cross-Dataset Comparison
#
# | Dimension | Google Cluster Traces | Backblaze Hard Drive Data |
# |-----------|----------------------|--------------------------|
# | Scale | ~9.3B rows | ~700M+ rows |
# | Duration | 31 days | 12+ years |
# | Failure type | Rapid-onset crashes (22s median) | Gradual SMART degradation |
# | Imbalance | 3.4:1 (moderate) | Extreme at daily level |
# | Missingness | Structural + MNAR | Schema evolution |
# | Prediction point | At submission/scheduling | Sliding window over SMART |

# %% [markdown]
# ---
# *End of notebook 06.  All four output tables saved to
# `outputs/tables/`.  These feed directly into Chapter 3, Section 3.3–3.5.*
