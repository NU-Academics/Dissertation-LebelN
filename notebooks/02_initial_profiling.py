# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Initial Profiling: Quick Data Survey
#
# **Purpose:** A fast first pass over all 5 cached BigQuery tables to understand
# schemas, null patterns, value distributions, and temporal coverage. This is NOT
# deep EDA — it's a reconnaissance sweep that tells us where to focus in notebooks
# 03 (Google deep EDA) and 05 (Backblaze deep EDA).
#
# **Outputs (saved to Drive `outputs/tables/`):**
# - `schema_summary.csv` — column names, types, nullable status for all tables
# - `null_profile.csv` — per-column null counts and percentages for all tables
# - `numeric_stats.csv` — min, max, mean, stddev, percentiles for numeric columns
# - `categorical_cardinality.csv` — distinct value counts for categorical/ID columns
#
# **Cost:** All queries target our cached `dissertation_lebel.*` tables — zero or
# near-zero cost.
#
# **Prerequisites:**
# - `01_bigquery_caching.py` has been run (all 5 tables cached)
# - Colab Secrets: `GCP_PROJECT_ID`

# %% [markdown]
# ---
# ## 1. Colab Session Setup

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes

# %%
from google.colab import userdata

PROJECT_ID = userdata.get('GCP_PROJECT_ID')
DATASET = f"{PROJECT_ID}.dissertation_lebel"
print(f"GCP Project: {PROJECT_ID}")
print(f"Dataset:     {DATASET}")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
TABLES_DIR = DRIVE_PATH / 'outputs' / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# %%
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
bq_client = bigquery.Client(project=PROJECT_ID)

# %%
import polars as pl
import time

# %% [markdown]
# ---
# ## 2. Table Registry
#
# Central definition of all 5 cached tables and their column classifications.
# This drives every profiling query below — add a table here and the entire
# notebook profiles it automatically.

# %%
# Column classifications per table.
# - numeric: columns for summary stats (min/max/mean/stddev/percentiles)
# - categorical: columns for distinct-value counts
# - temporal: columns for time-range analysis
#
# Every column appears in the null profile automatically (derived from schema).

TABLES = {
    'instance_events_full': {
        'numeric': [
            'priority', 'cpu_request', 'memory_request',
        ],
        'categorical': [
            'type', 'scheduling_class', 'collection_type', 'constraint',
        ],
        'temporal': ['time'],
    },
    'machine_events_full': {
        'numeric': [
            'capacity_cpus', 'capacity_memory',
        ],
        'categorical': [
            'type', 'platform_id',
        ],
        'temporal': ['time'],
    },
    'instance_usage_full': {
        'numeric': [
            'avg_cpu', 'avg_memory', 'max_cpu', 'max_memory',
            'sample_cpu', 'sample_memory',
            'assigned_memory', 'page_cache_memory',
            'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate',
        ],
        'categorical': [],
        'temporal': ['start_time', 'end_time'],
    },
    'collection_events_full': {
        'numeric': [
            'priority', 'max_per_machine', 'max_per_switch',
        ],
        'categorical': [
            'type', 'scheduling_class', 'collection_type',
            'vertical_scaling', 'scheduler',
        ],
        'temporal': ['time'],
    },
    'machine_attributes_full': {
        'numeric': [],
        'categorical': [
            'name', 'deleted',
        ],
        'temporal': ['time'],
    },
}

TABLE_NAMES = list(TABLES.keys())
print(f"Tables to profile: {TABLE_NAMES}")

# %% [markdown]
# ---
# ## 3. Helper Functions

# %%
def fqn(table: str) -> str:
    """Return fully-qualified BigQuery table name."""
    return f"`{DATASET}.{table}`"


def run_query(sql: str) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame."""
    return pl.from_pandas(bq_client.query(sql).to_dataframe())


def print_section(title: str) -> None:
    """Print a visible section divider."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

# %% [markdown]
# ---
# ## 4. Schema Inspection
#
# For every cached table: column name, BigQuery data type, and nullable mode.
# This uses the BigQuery metadata API (no bytes scanned).

# %%
print_section("SCHEMA INSPECTION")

schema_rows = []
for table_name in TABLE_NAMES:
    table_ref = bq_client.get_table(f"{DATASET}.{table_name}")
    for field in table_ref.schema:
        schema_rows.append({
            'table': table_name,
            'column': field.name,
            'bq_type': field.field_type,
            'mode': field.mode,  # NULLABLE, REQUIRED, REPEATED
        })
    print(f"  {table_name}: {len(table_ref.schema)} columns")

schema_df = pl.DataFrame(schema_rows)
print(f"\nTotal columns across all tables: {len(schema_rows)}")
print(schema_df)

# %% [markdown]
# ---
# ## 5. Null Profile
#
# For every column in every table: total row count, null count, and null
# percentage. This is the single most important profiling output — it tells
# us which columns are reliable and which need imputation or exclusion.

# %%
print_section("NULL PROFILE")

null_rows = []

for table_name in TABLE_NAMES:
    # Get column names from schema_df (already collected above)
    columns = schema_df.filter(pl.col('table') == table_name)['column'].to_list()

    # Build a single query that counts nulls for every column at once
    count_expr = "COUNT(*) AS total_rows"
    null_exprs = [
        f"COUNTIF({col} IS NULL) AS null_{col}" for col in columns
    ]
    all_exprs = ",\n    ".join([count_expr] + null_exprs)

    sql = f"SELECT\n    {all_exprs}\nFROM {fqn(table_name)}"
    result = run_query(sql)

    total_rows = result['total_rows'][0]
    for col in columns:
        null_count = result[f'null_{col}'][0]
        null_pct = round(null_count / total_rows * 100, 4) if total_rows > 0 else 0.0
        null_rows.append({
            'table': table_name,
            'column': col,
            'total_rows': total_rows,
            'null_count': null_count,
            'null_pct': null_pct,
        })

    print(f"  {table_name}: {total_rows:>15,} rows")

null_df = pl.DataFrame(null_rows)

# Show columns with any nulls
has_nulls = null_df.filter(pl.col('null_count') > 0)
print(f"\nColumns with nulls: {len(has_nulls)} of {len(null_df)}")
print(has_nulls)

# %% [markdown]
# ---
# ## 6. Categorical / ID Column Cardinality
#
# For columns that represent categories, types, or IDs: how many distinct
# values exist? High cardinality (e.g., machine_id) vs. low cardinality
# (e.g., type with 11 values) informs encoding strategy later.
#
# We also profile key ID columns (machine_id, collection_id, etc.) even
# though they aren't "categorical" in the modeling sense — knowing their
# cardinality is essential for understanding the data.

# %%
print_section("CATEGORICAL CARDINALITY")

# Add high-cardinality ID columns that we want counts for,
# separate from the low-cardinality categoricals used in TABLES.
ID_COLUMNS = {
    'instance_events_full': ['machine_id', 'collection_id', 'alloc_collection_id'],
    'machine_events_full': ['machine_id', 'switch_id'],
    'instance_usage_full': ['machine_id', 'collection_id', 'alloc_collection_id'],
    'collection_events_full': [
        'collection_id', 'user', 'collection_name',
        'parent_collection_id',
    ],
    'machine_attributes_full': ['machine_id'],
}

cardinality_rows = []

for table_name in TABLE_NAMES:
    cat_cols = TABLES[table_name]['categorical']
    id_cols = ID_COLUMNS.get(table_name, [])
    all_cols = cat_cols + id_cols

    if not all_cols:
        continue

    # Get the detailed schema for the current table to identify ARRAY (REPEATED) types
    table_ref = bq_client.get_table(f"{DATASET}.{table_name}")
    table_schema_map = {field.name: field for field in table_ref.schema}

    non_array_cols = []
    array_cols = []
    for col in all_cols:
        field = table_schema_map.get(col)
        # If field.mode is 'REPEATED', it's an ARRAY type
        if field and field.mode == 'REPEATED':
            array_cols.append(col)
        else:
            non_array_cols.append(col)

    # Process non-ARRAY columns with a single query
    if non_array_cols:
        distinct_exprs = ",\n    ".join(
            f"COUNT(DISTINCT {col}) AS distinct_{col}" for col in non_array_cols
        )
        sql = f"SELECT\n    {distinct_exprs}\nFROM {fqn(table_name)}"
        result = run_query(sql)

        for col in non_array_cols:
            cardinality_rows.append({
                'table': table_name,
                'column': col,
                'distinct_count': result[f'distinct_{col}'][0],
                'column_kind': 'categorical' if col in cat_cols else 'id',
            })

    # Process ARRAY columns with individual UNNEST queries
    for col in array_cols:
        field = table_schema_map.get(col)
        if field and field.field_type == 'RECORD': # It's an array of structs
            # For ARRAY<STRUCT> columns, stringify the struct to count distinct values
            sql = f"""
            SELECT COUNT(DISTINCT TO_JSON_STRING(element)) AS distinct_{col}
            FROM {fqn(table_name)}, UNNEST({col}) AS element
            """
        else: # It's an array of primitive types (STRING, INTEGER, etc.)
            sql = f"""
            SELECT COUNT(DISTINCT element) AS distinct_{col}
            FROM {fqn(table_name)}, UNNEST({col}) AS element
            """
        result = run_query(sql)
        cardinality_rows.append({
            'table': table_name,
            'column': col,
            'distinct_count': result[f'distinct_{col}'][0],
            'column_kind': 'categorical' if col in cat_cols else 'id',
        })

cardinality_df = pl.DataFrame(cardinality_rows)
print(cardinality_df)

# %% [markdown]
# ---
# ## 7. Numeric Summary Statistics
#
# For every numeric column: min, max, mean, stddev, and approximate percentiles
# (25th, 50th, 75th, 95th, 99th). BigQuery's `APPROX_QUANTILES` is used for
# percentiles to keep costs low on large tables.

# %%
print_section("NUMERIC SUMMARY STATISTICS")

numeric_rows = []

for table_name in TABLE_NAMES:
    num_cols = TABLES[table_name]['numeric']
    if not num_cols:
        print(f"  {table_name}: no numeric columns — skipping")
        continue

    # Build per-column stat expressions
    stat_parts = []
    for col in num_cols:
        stat_parts.append(f"""
    STRUCT(
        '{col}' AS col,
        MIN({col}) AS min_val,
        MAX({col}) AS max_val,
        AVG({col}) AS mean_val,
        STDDEV({col}) AS stddev_val,
        COUNTIF({col} IS NOT NULL) AS non_null_count
    ) AS stats_{col}""")

    # Percentiles need a separate query per column (APPROX_QUANTILES returns an array)
    # We'll run a single query that unions per-column percentile subqueries.
    pctile_subqueries = []
    for col in num_cols:
        pctile_subqueries.append(f"""
    SELECT
        '{col}' AS col,
        pctiles[OFFSET(25)] AS p25,
        pctiles[OFFSET(50)] AS p50,
        pctiles[OFFSET(75)] AS p75,
        pctiles[OFFSET(95)] AS p95,
        pctiles[OFFSET(99)] AS p99
    FROM (
        SELECT APPROX_QUANTILES({col}, 100) AS pctiles
        FROM {fqn(table_name)}
        WHERE {col} IS NOT NULL
    )""")

    pctile_sql = "\nUNION ALL\n".join(pctile_subqueries)

    # Run basic stats query (one row with structs)
    stats_sql = f"SELECT\n{','.join(stat_parts)}\nFROM {fqn(table_name)}"
    stats_result = run_query(stats_sql)

    # Run percentile query (one row per column)
    pctile_result = run_query(pctile_sql)
    pctile_dict = {
        row['col']: row for row in pctile_result.to_dicts()
    }

    for col in num_cols:
        stats_col = f"stats_{col}"
        # BigQuery STRUCT comes back as a dict in pandas → Polars keeps it as a struct
        struct_val = stats_result[stats_col][0]
        # Handle both dict and struct access patterns
        if isinstance(struct_val, dict):
            sv = struct_val
        else:
            sv = stats_result[stats_col].struct.unnest().to_dicts()[0]

        p = pctile_dict.get(col, {})
        numeric_rows.append({
            'table': table_name,
            'column': col,
            'non_null_count': sv.get('non_null_count'),
            'min': sv.get('min_val'),
            'max': sv.get('max_val'),
            'mean': round(sv.get('mean_val', 0) or 0, 6),
            'stddev': round(sv.get('stddev_val', 0) or 0, 6),
            'p25': p.get('p25'),
            'p50': p.get('p50'),
            'p75': p.get('p75'),
            'p95': p.get('p95'),
            'p99': p.get('p99'),
        })

    print(f"  {table_name}: {len(num_cols)} numeric columns profiled")

numeric_df = pl.DataFrame(numeric_rows)
print(numeric_df)

# %% [markdown]
# ---
# ## 8. Temporal Coverage
#
# For each table with timestamp columns: min and max values, and the duration
# of the trace window. Google Cluster Traces v3 timestamps are in microseconds
# since a reference point (not Unix epoch). We report raw values here and
# convert in notebook 03.

# %%
print_section("TEMPORAL COVERAGE")

temporal_rows = []

for table_name in TABLE_NAMES:
    time_cols = TABLES[table_name]['temporal']
    if not time_cols:
        continue

    for col in time_cols:
        sql = f"""
        SELECT
            MIN({col}) AS min_time,
            MAX({col}) AS max_time,
            COUNT(DISTINCT {col}) AS distinct_timestamps,
            COUNT(*) AS total_rows
        FROM {fqn(table_name)}
        WHERE {col} IS NOT NULL
        """
        result = run_query(sql)
        row = result.to_dicts()[0]

        min_t = row['min_time']
        max_t = row['max_time']
        # Duration in seconds (timestamps are in microseconds)
        if min_t is not None and max_t is not None:
            duration_sec = (max_t - min_t) / 1_000_000
            duration_hours = duration_sec / 3600
            duration_days = duration_hours / 24
        else:
            duration_sec = None
            duration_hours = None
            duration_days = None

        temporal_rows.append({
            'table': table_name,
            'column': col,
            'min_time_us': min_t,
            'max_time_us': max_t,
            'duration_seconds': round(duration_sec, 1) if duration_sec else None,
            'duration_hours': round(duration_hours, 2) if duration_hours else None,
            'duration_days': round(duration_days, 2) if duration_days else None,
            'distinct_timestamps': row['distinct_timestamps'],
            'total_non_null': row['total_rows'],
        })

        print(f"  {table_name}.{col}:")
        print(f"    Range: {min_t} → {max_t}")
        if duration_days is not None:
            print(f"    Duration: {duration_days:.2f} days ({duration_hours:.1f} hours)")

temporal_df = pl.DataFrame(temporal_rows)
print()
print(temporal_df)

# %% [markdown]
# ---
# ## 9. Save All Outputs to Drive

# %%
outputs = {
    'schema_summary.csv': schema_df,
    'null_profile.csv': null_df,
    'numeric_stats.csv': numeric_df,
    'categorical_cardinality.csv': cardinality_df,
}

print_section("SAVING OUTPUTS")
for filename, df in outputs.items():
    path = TABLES_DIR / filename
    df.write_csv(str(path))
    print(f"  Saved: {path}  ({len(df)} rows)")

# Also save temporal coverage (useful reference but not in the main 4 deliverables)
temporal_path = TABLES_DIR / 'temporal_coverage.csv'
temporal_df.write_csv(str(temporal_path))
print(f"  Saved: {temporal_path}  ({len(temporal_df)} rows)")

print("\nAll profiling outputs saved.")

# %% [markdown]
# ---
# ## 10. Quick Summary
#
# High-level overview to orient the deep EDA notebooks.

# %%
print_section("PROFILING SUMMARY")

# Row counts per table
print("\n--- Row Counts ---")
row_counts = null_df.group_by('table').agg(
    pl.col('total_rows').first()
).sort('table')
print(row_counts)

# Most-null columns (top 15)
print("\n--- Top 15 Columns by Null % ---")
top_nulls = null_df.filter(
    pl.col('null_pct') > 0
).sort('null_pct', descending=True).head(15)
print(top_nulls)

# Lowest-cardinality categoricals (good for encoding)
print("\n--- Categorical Columns (sorted by cardinality) ---")
cat_only = cardinality_df.filter(
    pl.col('column_kind') == 'categorical'
).sort('distinct_count')
print(cat_only)

# Highest-cardinality IDs
print("\n--- ID Columns (sorted by cardinality) ---")
id_only = cardinality_df.filter(
    pl.col('column_kind') == 'id'
).sort('distinct_count', descending=True)
print(id_only)

# %% [markdown]
# ---
# ## Next Steps
#
# This survey gives the lay of the land. Key things to investigate in deep EDA:
#
# 1. **High-null columns** — decide on imputation vs. exclusion
# 2. **Event type distribution** — class balance for failure prediction
# 3. **Resource distribution shapes** — normalization strategy
# 4. **Temporal gaps** — are there missing windows in the trace?
# 5. **Cardinality of IDs** — embedding vs. aggregation strategy
#
