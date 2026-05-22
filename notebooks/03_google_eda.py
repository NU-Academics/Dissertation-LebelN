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
# # 03 — Google Cluster Traces: Deep Exploratory Data Analysis
#
# **Purpose:** Comprehensive EDA of the 5 cached Google Cluster Traces v3 (2019)
# tables. This notebook produces the tables and figures needed for Chapter 3
# (Research Methodology) and documents the analytical decisions that will shape
# preprocessing and modeling in later phases.
#
# **Outputs (saved to Drive):**
# - `outputs/tables/` — CSV summary tables for Chapter 3
# - `outputs/figures/` — PNG figures for Chapter 3
#
# **Cost:** All queries target our cached `dissertation_lebel.*` tables — zero or
# near-zero cost. The `instance_usage_full` table is large (~1.5 TB cached), so
# queries against it use TABLESAMPLE or LIMIT where noted.
#
# **Prerequisites:**
# - `01_bigquery_caching.py` has been run (all 5 tables cached)
# - `02_initial_profiling.py` has been run (schema/null reference available)
# - Colab Secrets: `GCP_PROJECT_ID`
#
# **Sections:**
# 1. Dataset Overview (dimensions, temporal coverage, join keys)
# 2. instance_events Deep Dive
# 3. machine_events Analysis
# 4. instance_usage Analysis (sampled where necessary)
# 5. collection_events Analysis
# 6. Cross-Table Analysis
# 7. Key Findings Summary

# %% [markdown]
# ---
# ## 0. Colab Session Setup

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes matplotlib seaborn

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
FIGURES_DIR = DRIVE_PATH / 'outputs' / 'figures'

for dir_path in [TABLES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# %%
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
bq_client = bigquery.Client(project=PROJECT_ID)

# %%
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 5)

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


def save_figure(fig, name: str) -> None:
    """Save a matplotlib figure as PNG to the Drive figures directory."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(str(path), bbox_inches='tight')
    print(f"  Saved: {path}")


# Google Cluster Traces timestamp conversion
TRACE_START = datetime(2019, 5, 1, 0, 0, 0)
OFFSET_SECONDS = 600

def us_to_datetime(time_us: int) -> datetime | None:
    """Convert Google trace microsecond timestamp to datetime (PDT)."""
    if time_us is None or time_us == 0:
        return None
    if time_us >= 2**63 - 1:
        return None
    seconds_since_start = (time_us / 1_000_000) - OFFSET_SECONDS
    return TRACE_START + timedelta(seconds=seconds_since_start)


# Event type labels for readability
EVENT_TYPE_LABELS = {
    0: 'SUBMIT',
    1: 'QUEUE',
    2: 'ENABLE',
    3: 'SCHEDULE',
    4: 'EVICT',
    5: 'FAIL',
    6: 'FINISH',
    7: 'KILL',
    8: 'LOST',
    9: 'UPDATE_PENDING',
    10: 'UPDATE_RUNNING',
}

# %% [markdown]
# ---
# # Section 1: Dataset Overview
#
# High-level dimensions, temporal coverage, and join relationships across all
# 5 cached tables.

# %% [markdown]
# ### 1.1 Table Dimensions
#
# Row counts and column counts for each cached table. These are metadata queries
# (no bytes scanned).

# %%
TABLE_NAMES = [
    'instance_events_full',
    'machine_events_full',
    'instance_usage_full',
    'collection_events_full',
    'machine_attributes_full',
]

dims_rows = []
for table_name in TABLE_NAMES:
    table_ref = bq_client.get_table(f"{DATASET}.{table_name}")
    dims_rows.append({
        'table': table_name,
        'rows': table_ref.num_rows,
        'columns': len(table_ref.schema),
        'size_gb': round(table_ref.num_bytes / (1024**3), 2),
    })

dims_df = pl.DataFrame(dims_rows)
print(dims_df)
save_table(dims_df, 'google_table_dimensions')

# %% [markdown]
# ### 1.2 Temporal Coverage
#
# For each table, query the minimum and maximum timestamp values. Google Cluster
# Traces v3 timestamps are in microseconds relative to a fixed offset. We convert
# to human-readable datetimes for context.

# %%
# Define which time column(s) each table has
TIME_COLUMNS = {
    'instance_events_full': ['time'],
    'machine_events_full': ['time'],
    'instance_usage_full': ['start_time', 'end_time'],
    'collection_events_full': ['time'],
    'machine_attributes_full': ['time'],
}

temporal_rows = []
for table_name, time_cols in TIME_COLUMNS.items():
    for col in time_cols:
        sql = f"""
        SELECT
            MIN({col}) AS min_time,
            MAX({col}) AS max_time
        FROM {fqn(table_name)}
        WHERE {col} IS NOT NULL
        """
        result = run_query(sql)
        row = result.to_dicts()[0]
        min_t = row['min_time']
        max_t = row['max_time']

        min_dt = us_to_datetime(min_t)
        max_dt = us_to_datetime(max_t)
        duration_days = (max_dt - min_dt).total_seconds() / 86400 if min_dt and max_dt else None

        temporal_rows.append({
            'table': table_name,
            'time_column': col,
            'min_timestamp_us': min_t,
            'max_timestamp_us': max_t,
            'min_datetime': str(min_dt) if min_dt else None,
            'max_datetime': str(max_dt) if max_dt else None,
            'duration_days': round(duration_days, 2) if duration_days else None,
        })
        print(f"{table_name}.{col}: {min_dt} -> {max_dt}  ({duration_days:.1f} days)" if duration_days else f"{table_name}.{col}: N/A")

temporal_df = pl.DataFrame(temporal_rows)
save_table(temporal_df, 'google_temporal_coverage')

# %% [markdown]
# ### 1.3 Join Relationships Between Tables
#
# Understanding which keys link the tables is essential for cross-table analysis
# in Section 6. Here we verify the shared keys and their cardinality overlap.

# %%
# Key columns shared between tables:
# - machine_id: instance_events, machine_events, instance_usage, machine_attributes
# - collection_id + instance_index: instance_events <-> instance_usage
# - collection_id: instance_events <-> collection_events

join_checks = [
    {
        'join': 'instance_events <-> machine_events',
        'key': 'machine_id',
        'sql': f"""
        SELECT
            (SELECT COUNT(DISTINCT machine_id) FROM {fqn('instance_events_full')} WHERE machine_id IS NOT NULL) AS ie_machines,
            (SELECT COUNT(DISTINCT machine_id) FROM {fqn('machine_events_full')} WHERE machine_id IS NOT NULL) AS me_machines
        """,
    },
    {
        'join': 'instance_events <-> instance_usage',
        'key': 'collection_id, instance_index, machine_id',
        'sql': f"""
        SELECT
            (SELECT COUNT(DISTINCT collection_id) FROM {fqn('instance_events_full')} WHERE collection_id IS NOT NULL) AS ie_collections,
            (SELECT COUNT(DISTINCT collection_id) FROM {fqn('instance_usage_full')} WHERE collection_id IS NOT NULL) AS iu_collections,
            (SELECT COUNT(DISTINCT machine_id) FROM {fqn('instance_usage_full')} WHERE machine_id IS NOT NULL) AS iu_machines
        """,
    },
    {
        'join': 'instance_events <-> collection_events',
        'key': 'collection_id',
        'sql': f"""
        SELECT
            (SELECT COUNT(DISTINCT collection_id) FROM {fqn('instance_events_full')} WHERE collection_id IS NOT NULL) AS ie_collections,
            (SELECT COUNT(DISTINCT collection_id) FROM {fqn('collection_events_full')} WHERE collection_id IS NOT NULL) AS ce_collections
        """,
    },
    {
        'join': 'machine_events <-> machine_attributes',
        'key': 'machine_id',
        'sql': f"""
        SELECT
            (SELECT COUNT(DISTINCT machine_id) FROM {fqn('machine_events_full')} WHERE machine_id IS NOT NULL) AS me_machines,
            (SELECT COUNT(DISTINCT machine_id) FROM {fqn('machine_attributes_full')} WHERE machine_id IS NOT NULL) AS ma_machines
        """,
    },
]

print("Join Key Cardinality Overlap")
print("=" * 60)
for check in join_checks:
    result = run_query(check['sql'])
    print(f"\n{check['join']} (key: {check['key']})")
    for col_name in result.columns:
        print(f"  {col_name}: {result[col_name][0]:,}")

# %% [markdown]
# ---
# # Section 2: instance_events Deep Dive
#
# This is the most critical table for failure prediction. It contains every
# lifecycle event for every instance — SUBMIT, QUEUE, SCHEDULE, EVICT, FAIL,
# FINISH, KILL, LOST, and UPDATE variants.

# %% [markdown]
# ### 2.1 Event Type Distribution
#
# How many events of each type exist? What percentage of all events does each
# type represent? This directly determines class imbalance for failure prediction.

# %%
sql_event_types = f"""
SELECT
    type,
    COUNT(*) AS event_count
FROM {fqn('instance_events_full')}
GROUP BY type
ORDER BY type
"""
event_types_df = run_query(sql_event_types)

# Add labels and percentages
total_events = event_types_df['event_count'].sum()
event_types_df = event_types_df.with_columns([
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label'),
    (pl.col('event_count') / total_events * 100).round(4).alias('pct_of_total'),
    pl.col('event_count').map_elements(lambda x: f"{x:,}", return_dtype=pl.Utf8).alias('count_formatted'),
])

print(f"Total instance events: {total_events:,}\n")
print(event_types_df.select(['type', 'type_label', 'event_count', 'pct_of_total']))
save_table(event_types_df, 'google_event_type_distribution')

# %% [markdown]
# **Figure 2.1a:** Event type distribution (bar chart, log scale for counts).

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of counts (log scale)
labels = event_types_df['type_label'].to_list()
counts = event_types_df['event_count'].to_list()

ax = axes[0]
bars = ax.barh(labels, counts, color=sns.color_palette("viridis", len(labels)))
ax.set_xscale('log')
ax.set_xlabel('Event Count (log scale)')
ax.set_title('Instance Event Type Counts')
ax.invert_yaxis()
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
            f'{count:,.0f}', va='center', fontsize=9)

# Percentage pie chart
ax = axes[1]
pcts = event_types_df['pct_of_total'].to_list()
# Only label slices > 1%
pie_labels = [l if p > 1 else '' for l, p in zip(labels, pcts)]
ax.pie(pcts, labels=pie_labels, autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
       startangle=90, colors=sns.color_palette("viridis", len(labels)))
ax.set_title('Event Type Proportions')

fig.suptitle('Figure 2.1a: Instance Event Type Distribution', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_event_type_distribution')
plt.show()

# %% [markdown]
# *Figure 2.1a shows the distribution of all 11 instance event types. The log-scale
# bar chart (left) reveals the absolute counts, while the pie chart (right) shows
# relative proportions. Key observation: the class balance between potential "failure"
# types (EVICT, FAIL, LOST) and "success" (FINISH) informs class imbalance strategy.*

# %% [markdown]
# ### 2.2 Per-Event-Type Resource and Scheduling Analysis
#
# For each event type: average and median CPU/memory requests, and the distribution
# of scheduling_class and priority. This reveals whether certain event types are
# associated with resource-heavy or low-priority tasks.

# %%
sql_per_type_resources = f"""
SELECT
    type,
    COUNT(*) AS n,
    AVG(cpu_request) AS avg_cpu,
    AVG(memory_request) AS avg_memory,
    -- Use APPROX_QUANTILES for median (index 50 of 100 quantiles)
    APPROX_QUANTILES(cpu_request, 100)[OFFSET(50)] AS median_cpu,
    APPROX_QUANTILES(memory_request, 100)[OFFSET(50)] AS median_memory,
    AVG(priority) AS avg_priority,
    APPROX_QUANTILES(priority, 100)[OFFSET(50)] AS median_priority
FROM {fqn('instance_events_full')}
GROUP BY type
ORDER BY type
"""
per_type_resources = run_query(sql_per_type_resources)
per_type_resources = per_type_resources.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)
print(per_type_resources)
save_table(per_type_resources, 'google_per_type_resource_stats')

# %% [markdown]
# **Figure 2.2a:** Average CPU and memory requests by event type.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

type_labels = per_type_resources['type_label'].to_list()
x = np.arange(len(type_labels))

# CPU requests
ax = axes[0]
avg_cpu = per_type_resources['avg_cpu'].to_list()
median_cpu = per_type_resources['median_cpu'].to_list()
ax.bar(x - 0.15, avg_cpu, 0.3, label='Mean', color='steelblue')
ax.bar(x + 0.15, median_cpu, 0.3, label='Median', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(type_labels, rotation=45, ha='right')
ax.set_ylabel('CPU Request (normalized)')
ax.set_title('CPU Request by Event Type')
ax.legend()

# Memory requests
ax = axes[1]
avg_mem = per_type_resources['avg_memory'].to_list()
median_mem = per_type_resources['median_memory'].to_list()
ax.bar(x - 0.15, avg_mem, 0.3, label='Mean', color='steelblue')
ax.bar(x + 0.15, median_mem, 0.3, label='Median', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(type_labels, rotation=45, ha='right')
ax.set_ylabel('Memory Request (normalized)')
ax.set_title('Memory Request by Event Type')
ax.legend()

fig.suptitle('Figure 2.2a: Resource Requests by Event Type', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_resource_by_event_type')
plt.show()

# %% [markdown]
# *Figure 2.2a compares mean vs. median CPU and memory requests for each event type.
# A large gap between mean and median indicates skewed distributions (heavy tail).
# Event types with higher resource requests may be more prone to failures due to
# resource contention.*

# %% [markdown]
# Scheduling class distribution per event type.

# %%
sql_sched_by_type = f"""
SELECT
    type,
    scheduling_class,
    COUNT(*) AS n
FROM {fqn('instance_events_full')}
WHERE scheduling_class IS NOT NULL
GROUP BY type, scheduling_class
ORDER BY type, scheduling_class
"""
sched_by_type = run_query(sql_sched_by_type)
sched_by_type = sched_by_type.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)

# Pivot for display
sched_pivot = sched_by_type.pivot(
    on='scheduling_class',
    index=['type', 'type_label'],
    values='n',
).sort('type')
print("Scheduling class counts by event type:")
print(sched_pivot)
save_table(sched_by_type, 'google_scheduling_class_by_event_type')

# %% [markdown]
# Priority tier distribution per event type. Priority ranges are grouped into
# meaningful tiers per the Google documentation.

# %%
sql_priority_by_type = f"""
SELECT
    type,
    CASE
        WHEN priority <= 99 THEN 'Free (<=99)'
        WHEN priority BETWEEN 100 AND 115 THEN 'Best-effort (100-115)'
        WHEN priority BETWEEN 116 AND 119 THEN 'Mid-tier (116-119)'
        WHEN priority BETWEEN 120 AND 359 THEN 'Production (120-359)'
        WHEN priority >= 360 THEN 'Monitoring (>=360)'
        ELSE 'NULL'
    END AS priority_tier,
    COUNT(*) AS n
FROM {fqn('instance_events_full')}
GROUP BY type, priority_tier
ORDER BY type, priority_tier
"""
priority_by_type = run_query(sql_priority_by_type)
priority_by_type = priority_by_type.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)
print("Priority tier counts by event type:")
print(priority_by_type)
save_table(priority_by_type, 'google_priority_tier_by_event_type')

# %% [markdown]
# ### 2.3 Null Analysis by Event Type
#
# Overall null rates were computed in notebook 02. Here we investigate whether
# null rates differ by event type — e.g., do SUBMIT events have null machine_id
# (because the instance hasn't been scheduled yet)?

# %%
sql_nulls_by_type = f"""
SELECT
    type,
    COUNT(*) AS total,
    COUNTIF(machine_id IS NULL) AS null_machine_id,
    COUNTIF(cpu_request IS NULL) AS null_cpu_request,
    COUNTIF(memory_request IS NULL) AS null_memory_request,
    COUNTIF(scheduling_class IS NULL) AS null_scheduling_class,
    COUNTIF(priority IS NULL) AS null_priority,
    COUNTIF(collection_id IS NULL) AS null_collection_id,
    COUNTIF(alloc_collection_id IS NULL) AS null_alloc_collection_id
FROM {fqn('instance_events_full')}
GROUP BY type
ORDER BY type
"""
nulls_by_type = run_query(sql_nulls_by_type)

# Compute percentages
null_cols = [c for c in nulls_by_type.columns if c.startswith('null_')]
for col in null_cols:
    pct_col = col.replace('null_', 'pct_null_')
    nulls_by_type = nulls_by_type.with_columns(
        (pl.col(col) / pl.col('total') * 100).round(2).alias(pct_col)
    )

nulls_by_type = nulls_by_type.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)

# Show the percentage columns
pct_cols = ['type', 'type_label', 'total'] + [c for c in nulls_by_type.columns if c.startswith('pct_')]
print("Null rates (%) by event type:")
print(nulls_by_type.select(pct_cols))
save_table(nulls_by_type, 'google_null_rates_by_event_type')

# %% [markdown]
# **Figure 2.3a:** Null rate heatmap by event type.

# %%
# Build matrix for heatmap: rows = event types, columns = fields
pct_col_names = [c for c in nulls_by_type.columns if c.startswith('pct_null_')]
field_labels = [c.replace('pct_null_', '') for c in pct_col_names]
type_labels_list = nulls_by_type['type_label'].to_list()

heatmap_data = nulls_by_type.select(pct_col_names).to_numpy()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(field_labels)))
ax.set_xticklabels(field_labels, rotation=45, ha='right')
ax.set_yticks(range(len(type_labels_list)))
ax.set_yticklabels(type_labels_list)

# Annotate cells
for i in range(len(type_labels_list)):
    for j in range(len(field_labels)):
        val = heatmap_data[i, j]
        color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=8)

plt.colorbar(im, label='Null %')
ax.set_title('Figure 2.3a: Null Rates (%) by Event Type and Column')
fig.tight_layout()
save_figure(fig, 'google_null_heatmap_by_event_type')
plt.show()

# %% [markdown]
# *Figure 2.3a shows how null rates vary across event types. For instance, early
# lifecycle events (SUBMIT, QUEUE) are expected to have null machine_id because
# the instance hasn't been assigned to a machine yet. Understanding these structural
# nulls vs. missing data is critical for imputation strategy.*

# %% [markdown]
# ### 2.4 Temporal Patterns
#
# Event counts binned by hour and day to identify diurnal cycles, weekly patterns,
# and any anomalous periods (e.g., outages, maintenance windows).

# %% [markdown]
# #### 2.4a: Hourly event density over the full trace period.
#
# We bin events into 1-hour buckets and plot total event counts over time. This
# reveals the overall shape of activity during the 31-day trace.

# %%
sql_hourly = f"""
SELECT
    TIMESTAMP_MICROS(CAST(time AS INT64)) AS event_ts,
    -- Bin to hour
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(CAST(time AS INT64)), HOUR) AS hour_bucket,
    type
FROM {fqn('instance_events_full')}
WHERE time > 0 AND time < 9223372036854775807
"""

# Aggregate to hourly counts in BigQuery to avoid pulling billions of rows
sql_hourly_counts = f"""
SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(CAST(time AS INT64)), HOUR) AS hour_bucket,
    COUNT(*) AS event_count
FROM {fqn('instance_events_full')}
WHERE time > 0 AND time < 9223372036854775807
GROUP BY hour_bucket
ORDER BY hour_bucket
"""
hourly_counts = run_query(sql_hourly_counts)
print(f"Hourly buckets: {len(hourly_counts)}")
print(hourly_counts.head(5))

# %%
fig, ax = plt.subplots(figsize=(16, 5))
hours = hourly_counts['hour_bucket'].to_list()
counts = hourly_counts['event_count'].to_list()
ax.plot(hours, counts, linewidth=0.6, color='steelblue')
ax.set_xlabel('Date')
ax.set_ylabel('Events per Hour')
ax.set_title('Figure 2.4a: Hourly Instance Event Volume Over Trace Period')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
fig.tight_layout()
save_figure(fig, 'google_hourly_event_density')
plt.show()

# %% [markdown]
# *Figure 2.4a shows the hourly event volume across the full trace period. Look for:
# (1) diurnal cycles (daily peaks and troughs), (2) anomalous drops that could
# indicate outages or maintenance windows, (3) overall trend (stable vs. growing
# vs. declining). Any anomalous periods should be documented for potential exclusion
# or special handling in train/test splitting.*

# %% [markdown]
# #### 2.4b: Hourly event density broken out by event type.
#
# Same hourly binning but split by event type to see whether failure events follow
# the same temporal pattern as normal operations.

# %%
sql_hourly_by_type = f"""
SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(CAST(time AS INT64)), HOUR) AS hour_bucket,
    type,
    COUNT(*) AS event_count
FROM {fqn('instance_events_full')}
WHERE time > 0 AND time < 9223372036854775807
GROUP BY hour_bucket, type
ORDER BY hour_bucket, type
"""
hourly_by_type = run_query(sql_hourly_by_type)
hourly_by_type = hourly_by_type.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)
print(f"Rows: {len(hourly_by_type):,}")

# %%
# Plot a subset of key types: SCHEDULE (3), EVICT (4), FAIL (5), FINISH (6), KILL (7), LOST (8)
focus_types = [3, 4, 5, 6, 7, 8]
focus_labels = {t: EVENT_TYPE_LABELS[t] for t in focus_types}

fig, axes = plt.subplots(len(focus_types), 1, figsize=(16, 3 * len(focus_types)), sharex=True)

for idx, etype in enumerate(focus_types):
    ax = axes[idx]
    subset = hourly_by_type.filter(pl.col('type') == etype)
    ax.plot(subset['hour_bucket'].to_list(), subset['event_count'].to_list(),
            linewidth=0.6, color=sns.color_palette("tab10")[idx])
    ax.set_ylabel('Count/hr')
    ax.set_title(f'{focus_labels[etype]} (type={etype})', loc='left', fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K' if x >= 1000 else f'{x:.0f}'))

axes[-1].set_xlabel('Date')
fig.suptitle('Figure 2.4b: Hourly Event Volume by Type (Key Types)', fontsize=13, y=1.01)
fig.tight_layout()
save_figure(fig, 'google_hourly_events_by_type')
plt.show()

# %% [markdown]
# *Figure 2.4b breaks out the temporal pattern for key event types. Compare the
# temporal signature of failure types (EVICT, FAIL, LOST) against normal completion
# (FINISH, SCHEDULE). Correlated spikes may indicate cascading failures. If failure
# types show different diurnal patterns, this could inform time-aware features.*

# %% [markdown]
# #### 2.4c: Day-of-week patterns.
#
# Aggregate events by day of week to check for weekly seasonality.

# %%
sql_dow = f"""
SELECT
    EXTRACT(DAYOFWEEK FROM TIMESTAMP_MICROS(CAST(time AS INT64))) AS day_of_week,
    type,
    COUNT(*) AS event_count
FROM {fqn('instance_events_full')}
WHERE time > 0 AND time < 9223372036854775807
GROUP BY day_of_week, type
ORDER BY day_of_week, type
"""
dow_df = run_query(sql_dow)

# Aggregate across all types for overall pattern
dow_total = dow_df.group_by('day_of_week').agg(pl.col('event_count').sum())
dow_total = dow_total.sort('day_of_week')

dow_names = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}

fig, ax = plt.subplots(figsize=(10, 5))
dows = dow_total['day_of_week'].to_list()
dow_labels = [dow_names.get(d, str(d)) for d in dows]
ax.bar(dow_labels, dow_total['event_count'].to_list(), color='steelblue')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Total Events')
ax.set_title('Figure 2.4c: Total Instance Events by Day of Week')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e9:.2f}B'))
fig.tight_layout()
save_figure(fig, 'google_events_by_day_of_week')
plt.show()

# %% [markdown]
# *Figure 2.4c shows total event volume by day of week. Significant variation
# would indicate weekly seasonality that needs to be accounted for in temporal
# feature engineering and train/test splitting (e.g., ensuring each fold covers
# complete weeks).*

# %% [markdown]
# #### 2.4d: Hour-of-day patterns (diurnal cycle).
#
# Aggregate events by hour of day to identify daily usage peaks.

# %%
sql_hod = f"""
SELECT
    EXTRACT(HOUR FROM TIMESTAMP_MICROS(CAST(time AS INT64))) AS hour_of_day,
    type,
    COUNT(*) AS event_count
FROM {fqn('instance_events_full')}
WHERE time > 0 AND time < 9223372036854775807
GROUP BY hour_of_day, type
ORDER BY hour_of_day, type
"""
hod_df = run_query(sql_hod)

# Overall pattern
hod_total = hod_df.group_by('hour_of_day').agg(pl.col('event_count').sum()).sort('hour_of_day')

# Failure types only
failure_types = [4, 5, 8]
hod_failures = hod_df.filter(pl.col('type').is_in(failure_types)).group_by('hour_of_day').agg(
    pl.col('event_count').sum()
).sort('hour_of_day')

fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.bar(hod_total['hour_of_day'].to_list(), hod_total['event_count'].to_list(),
        color='steelblue', alpha=0.7, label='All Events')
ax1.set_xlabel('Hour of Day (PDT)')
ax1.set_ylabel('All Events', color='steelblue')

ax2 = ax1.twinx()
ax2.plot(hod_failures['hour_of_day'].to_list(), hod_failures['event_count'].to_list(),
         color='red', marker='o', linewidth=2, label='Failure Events (4,5,8)')
ax2.set_ylabel('Failure Events (EVICT+FAIL+LOST)', color='red')

fig.suptitle('Figure 2.4d: Events by Hour of Day — All vs. Failures', fontsize=13)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
fig.tight_layout()
save_figure(fig, 'google_events_by_hour_of_day')
plt.show()

# %% [markdown]
# *Figure 2.4d overlays all-event volume (bars) with failure event volume (line)
# by hour of day. If failure rates are higher at certain hours, time-of-day may
# be a useful feature. Note: "failure" types (4, 5, 8) are tentative — this
# classification will be validated in Section 7.*

# %% [markdown]
# ### 2.5 Machine Concentration of Failures
#
# Are failures spread evenly across machines, or concentrated on a small subset?
# This affects whether machine-level features are useful for prediction.

# %%
sql_machine_failures = f"""
SELECT
    machine_id,
    COUNT(*) AS failure_count
FROM {fqn('instance_events_full')}
WHERE type IN (4, 5, 8)  -- EVICT, FAIL, LOST (tentative failure set)
  AND machine_id IS NOT NULL
GROUP BY machine_id
ORDER BY failure_count DESC
"""
machine_failures = run_query(sql_machine_failures)

total_machines_with_failures = len(machine_failures)
total_failure_events = machine_failures['failure_count'].sum()

print(f"Unique machines with failure events: {total_machines_with_failures:,}")
print(f"Total failure events: {total_failure_events:,}")
print(f"\nTop 20 machines by failure count:")
print(machine_failures.head(20))

# %%
# Compute cumulative distribution
machine_failures_sorted = machine_failures.sort('failure_count', descending=True)
cumsum = machine_failures_sorted['failure_count'].cum_sum()
machine_failures_sorted = machine_failures_sorted.with_columns(
    (cumsum / total_failure_events * 100).alias('cumulative_pct')
)

# What % of machines account for 50%, 80%, 90% of failures?
for threshold in [50, 80, 90]:
    n_machines = machine_failures_sorted.filter(
        pl.col('cumulative_pct') <= threshold
    ).height + 1  # +1 for the machine that crosses the threshold
    pct_machines = n_machines / total_machines_with_failures * 100
    print(f"  {threshold}% of failures: top {n_machines:,} machines ({pct_machines:.1f}% of machines with failures)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# CDF of failure concentration
ax = axes[0]
x_pct = np.arange(1, total_machines_with_failures + 1) / total_machines_with_failures * 100
y_pct = machine_failures_sorted['cumulative_pct'].to_list()
ax.plot(x_pct, y_pct, linewidth=1.5, color='steelblue')
ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% of failures')
ax.set_xlabel('% of Machines (ranked by failure count)')
ax.set_ylabel('Cumulative % of Failures')
ax.set_title('Failure Concentration (CDF)')
ax.legend()

# Histogram of per-machine failure counts
ax = axes[1]
counts_arr = machine_failures['failure_count'].to_numpy()
ax.hist(counts_arr, bins=100, color='steelblue', edgecolor='none')
ax.set_xlabel('Failure Events per Machine')
ax.set_ylabel('Number of Machines')
ax.set_title('Distribution of Per-Machine Failure Counts')
ax.set_yscale('log')

fig.suptitle('Figure 2.5a: Machine Failure Concentration', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_machine_failure_concentration')
plt.show()

# %% [markdown]
# *Figure 2.5a (left) shows the cumulative distribution of failures across
# machines — a steep curve means failures are concentrated on few machines.
# (Right) shows the distribution of per-machine failure counts on a log scale.
# If failures are highly concentrated, machine-level features (hardware type,
# capacity, historical failure rate) will be important predictors.*

# %% [markdown]
# ### 2.6 Resource Request Distributions
#
# Histograms of cpu_request and memory_request, broken out by event type.
# Resources are normalized to [0, 1] in the Google traces.

# %%
# Pull a sample for plotting (full table may be too large for in-memory histograms).
# We sample 10M rows, which is sufficient for distribution estimation.
sql_resource_sample = f"""
SELECT
    type,
    cpu_request,
    memory_request
FROM {fqn('instance_events_full')}
WHERE cpu_request IS NOT NULL AND memory_request IS NOT NULL
ORDER BY RAND()
LIMIT 10000000
"""
resource_sample = run_query(sql_resource_sample)
resource_sample = resource_sample.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)
print(f"Sampled rows for resource distributions: {len(resource_sample):,}")

# %% [markdown]
# **Figure 2.6a:** Overall CPU and memory request distributions.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.hist(resource_sample['cpu_request'].to_numpy(), bins=100, color='steelblue',
        edgecolor='none', density=True)
ax.set_xlabel('CPU Request (normalized)')
ax.set_ylabel('Density')
ax.set_title('CPU Request Distribution')

ax = axes[1]
ax.hist(resource_sample['memory_request'].to_numpy(), bins=100, color='coral',
        edgecolor='none', density=True)
ax.set_xlabel('Memory Request (normalized)')
ax.set_ylabel('Density')
ax.set_title('Memory Request Distribution')

fig.suptitle('Figure 2.6a: Resource Request Distributions (10M sample)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_resource_distributions_overall')
plt.show()

# %% [markdown]
# *Figure 2.6a shows the overall distributions of CPU and memory requests. Note
# whether these are uniform, heavy-tailed, bimodal, or concentrated at specific
# values (e.g., many tasks requesting exactly 0.01 CPU). Distribution shape
# informs normalization/scaling strategy for ML models.*

# %% [markdown]
# **Figure 2.6b:** Resource distributions broken out by failure vs. non-failure.

# %%
# Split into failure (EVICT/FAIL/LOST) and non-failure
failure_mask = resource_sample['type'].is_in([4, 5, 8])
rs_failure = resource_sample.filter(failure_mask)
rs_nonfailure = resource_sample.filter(~failure_mask)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# CPU
ax = axes[0]
ax.hist(rs_nonfailure['cpu_request'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='steelblue', label=f'Non-failure (n={len(rs_nonfailure):,})')
ax.hist(rs_failure['cpu_request'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='red', label=f'Failure (n={len(rs_failure):,})')
ax.set_xlabel('CPU Request (normalized)')
ax.set_ylabel('Density')
ax.set_title('CPU Request: Failure vs. Non-failure')
ax.legend()

# Memory
ax = axes[1]
ax.hist(rs_nonfailure['memory_request'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='steelblue', label='Non-failure')
ax.hist(rs_failure['memory_request'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='red', label='Failure')
ax.set_xlabel('Memory Request (normalized)')
ax.set_ylabel('Density')
ax.set_title('Memory Request: Failure vs. Non-failure')
ax.legend()

fig.suptitle('Figure 2.6b: Resource Requests — Failure vs. Non-failure (10M sample)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_resource_distributions_failure_vs_nonfailure')
plt.show()

# %% [markdown]
# *Figure 2.6b compares resource request distributions between failure and
# non-failure events. Differences here suggest resource requests are predictive
# features. If distributions overlap substantially, resource requests alone may
# not discriminate failures.*

# %% [markdown]
# ---
# # Section 3: machine_events Analysis
#
# Machine lifecycle events: ADD (type 1), REMOVE (type 2), UPDATE (type 3).
# Understanding machine churn and capacity helps contextualize instance failures.

# %% [markdown]
# ### 3.1 Machine Event Type Distribution

# %%
MACHINE_EVENT_LABELS = {1: 'ADD', 2: 'REMOVE', 3: 'UPDATE'}

sql_machine_types = f"""
SELECT
    type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT machine_id) AS unique_machines
FROM {fqn('machine_events_full')}
GROUP BY type
ORDER BY type
"""
machine_types = run_query(sql_machine_types)
machine_types = machine_types.with_columns(
    pl.col('type').map_elements(lambda t: MACHINE_EVENT_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)
print(machine_types)
save_table(machine_types, 'google_machine_event_type_distribution')

# %% [markdown]
# ### 3.2 Machine Churn Over Time
#
# How many machines are added and removed over the trace period? High churn
# means the infrastructure is dynamic, which affects feature engineering
# (e.g., can we use historical machine features if machines are short-lived?).

# %%
sql_machine_churn = f"""
SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(CAST(time AS INT64)), DAY) AS day_bucket,
    type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT machine_id) AS unique_machines
FROM {fqn('machine_events_full')}
WHERE time > 0 AND time < 9223372036854775807
GROUP BY day_bucket, type
ORDER BY day_bucket, type
"""
machine_churn = run_query(sql_machine_churn)
machine_churn = machine_churn.with_columns(
    pl.col('type').map_elements(lambda t: MACHINE_EVENT_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)

# %%
fig, ax = plt.subplots(figsize=(16, 5))

for etype, label in MACHINE_EVENT_LABELS.items():
    subset = machine_churn.filter(pl.col('type') == etype)
    ax.plot(subset['day_bucket'].to_list(), subset['unique_machines'].to_list(),
            label=label, linewidth=1.5)

ax.set_xlabel('Date')
ax.set_ylabel('Unique Machines')
ax.set_title('Figure 3.2a: Daily Machine Churn (Unique Machines per Event Type)')
ax.legend()
fig.tight_layout()
save_figure(fig, 'google_machine_churn_daily')
plt.show()

# %% [markdown]
# *Figure 3.2a shows daily machine ADD/REMOVE/UPDATE activity. A stable fleet
# (few adds/removes) simplifies analysis; high churn requires careful handling
# of machine-level features over time.*

# %% [markdown]
# ### 3.3 Capacity Distributions
#
# CPU and memory capacity across all machines (at their most recent ADD or UPDATE event).

# %%
sql_capacity = f"""
SELECT
    machine_id,
    capacity_cpus,
    capacity_memory,
    platform_id
FROM {fqn('machine_events_full')}
WHERE type = 1  -- ADD events give initial capacity
  AND capacity_cpus IS NOT NULL
  AND capacity_memory IS NOT NULL
"""
capacity_df = run_query(sql_capacity)
print(f"Machines with capacity data (ADD events): {len(capacity_df):,}")
print(f"\nCapacity stats:")
print(capacity_df.select(['capacity_cpus', 'capacity_memory']).describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.hist(capacity_df['capacity_cpus'].to_numpy(), bins=50, color='steelblue', edgecolor='none')
ax.set_xlabel('CPU Capacity (normalized)')
ax.set_ylabel('Machine Count')
ax.set_title('CPU Capacity Distribution')

ax = axes[1]
ax.hist(capacity_df['capacity_memory'].to_numpy(), bins=50, color='coral', edgecolor='none')
ax.set_xlabel('Memory Capacity (normalized)')
ax.set_ylabel('Machine Count')
ax.set_title('Memory Capacity Distribution')

fig.suptitle('Figure 3.3a: Machine Capacity Distributions', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_machine_capacity_distributions')
plt.show()

# %% [markdown]
# *Figure 3.3a shows the distribution of CPU and memory capacity across machines.
# Clusters of capacity values indicate hardware tiers (homogeneous racks). Wide
# spread indicates heterogeneous infrastructure.*

# %% [markdown]
# ### 3.4 Platform Diversity
#
# How many distinct platform_ids exist? Platform is a proxy for hardware generation.

# %%
sql_platforms = f"""
SELECT
    platform_id,
    COUNT(DISTINCT machine_id) AS machine_count,
    AVG(capacity_cpus) AS avg_cpu_capacity,
    AVG(capacity_memory) AS avg_memory_capacity
FROM {fqn('machine_events_full')}
WHERE type = 1 AND platform_id IS NOT NULL
GROUP BY platform_id
ORDER BY machine_count DESC
"""
platforms_df = run_query(sql_platforms)
print(f"Distinct platforms: {len(platforms_df)}")
print(platforms_df)
save_table(platforms_df, 'google_platform_diversity')

# %% [markdown]
# ---
# # Section 4: instance_usage Analysis
#
# **WARNING:** This table is ~1.5 TB. All queries in this section use TABLESAMPLE
# or LIMIT to control memory and cost. Results from sampled queries are marked
# with "(SAMPLED)". Full-population queries use BigQuery aggregation (no data
# pulled to Colab).

# %% [markdown]
# ### 4.1 Utilization Distributions
#
# Distributions of average CPU/memory usage and maximum CPU/memory usage.
# These are queried with APPROX_QUANTILES in BigQuery (full population, no
# data pulled to Colab).

# %%
sql_usage_stats = f"""
SELECT
    COUNT(*) AS total_rows,
    AVG(avg_cpu) AS mean_avg_cpu,
    AVG(avg_memory) AS mean_avg_memory,
    AVG(max_cpu) AS mean_max_cpu,
    AVG(max_memory) AS mean_max_memory,
    STDDEV(avg_cpu) AS std_avg_cpu,
    STDDEV(avg_memory) AS std_avg_memory
FROM {fqn('instance_usage_full')}
"""
usage_stats = run_query(sql_usage_stats)
print("Instance Usage Stats (full population):")
print(usage_stats)

# %%
sql_usage_pctiles = f"""
SELECT
    'avg_cpu' AS metric,
    APPROX_QUANTILES(avg_cpu, 100)[OFFSET(25)] AS p25,
    APPROX_QUANTILES(avg_cpu, 100)[OFFSET(50)] AS p50,
    APPROX_QUANTILES(avg_cpu, 100)[OFFSET(75)] AS p75,
    APPROX_QUANTILES(avg_cpu, 100)[OFFSET(95)] AS p95,
    APPROX_QUANTILES(avg_cpu, 100)[OFFSET(99)] AS p99
FROM {fqn('instance_usage_full')}
WHERE avg_cpu IS NOT NULL

UNION ALL

SELECT
    'avg_memory',
    APPROX_QUANTILES(avg_memory, 100)[OFFSET(25)],
    APPROX_QUANTILES(avg_memory, 100)[OFFSET(50)],
    APPROX_QUANTILES(avg_memory, 100)[OFFSET(75)],
    APPROX_QUANTILES(avg_memory, 100)[OFFSET(95)],
    APPROX_QUANTILES(avg_memory, 100)[OFFSET(99)]
FROM {fqn('instance_usage_full')}
WHERE avg_memory IS NOT NULL

UNION ALL

SELECT
    'max_cpu',
    APPROX_QUANTILES(max_cpu, 100)[OFFSET(25)],
    APPROX_QUANTILES(max_cpu, 100)[OFFSET(50)],
    APPROX_QUANTILES(max_cpu, 100)[OFFSET(75)],
    APPROX_QUANTILES(max_cpu, 100)[OFFSET(95)],
    APPROX_QUANTILES(max_cpu, 100)[OFFSET(99)]
FROM {fqn('instance_usage_full')}
WHERE max_cpu IS NOT NULL

UNION ALL

SELECT
    'max_memory',
    APPROX_QUANTILES(max_memory, 100)[OFFSET(25)],
    APPROX_QUANTILES(max_memory, 100)[OFFSET(50)],
    APPROX_QUANTILES(max_memory, 100)[OFFSET(75)],
    APPROX_QUANTILES(max_memory, 100)[OFFSET(95)],
    APPROX_QUANTILES(max_memory, 100)[OFFSET(99)]
FROM {fqn('instance_usage_full')}
WHERE max_memory IS NOT NULL
"""
usage_pctiles = run_query(sql_usage_pctiles)
print("Utilization Percentiles (full population):")
print(usage_pctiles)
save_table(usage_pctiles, 'google_instance_usage_percentiles')

# %% [markdown]
# **Figure 4.1a:** Utilization distributions from a 1% TABLESAMPLE.

# %%
# Pull a 1% sample for histograms
sql_usage_sample = f"""
SELECT
    avg_cpu,
    avg_memory,
    max_cpu,
    max_memory
FROM {fqn('instance_usage_full')}
TABLESAMPLE SYSTEM (1 PERCENT)
"""
usage_sample = run_query(sql_usage_sample)
print(f"Usage sample rows (1% TABLESAMPLE): {len(usage_sample):,}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ('avg_cpu', 'Average CPU Usage', 'steelblue'),
    ('avg_memory', 'Average Memory Usage', 'coral'),
    ('max_cpu', 'Max CPU Usage', 'steelblue'),
    ('max_memory', 'Max Memory Usage', 'coral'),
]

for ax, (col, title, color) in zip(axes.flat, metrics):
    data = usage_sample[col].drop_nulls().to_numpy()
    ax.hist(data, bins=100, color=color, edgecolor='none', density=True)
    ax.set_xlabel(f'{title} (normalized)')
    ax.set_ylabel('Density')
    ax.set_title(title)

fig.suptitle('Figure 4.1a: Instance Utilization Distributions (1% TABLESAMPLE)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_usage_distributions_sampled')
plt.show()

# %% [markdown]
# *Figure 4.1a shows utilization distributions from a 1% sample. These are
# SAMPLED results — exact shape may vary slightly from full population, but
# the general patterns (heavy-tailed, concentrated near zero, etc.) should be
# stable. Compare avg vs. max to understand burstiness of resource usage.*

# %% [markdown]
# ### 4.2 Utilization vs. Request (Over/Under-Provisioning)
#
# Are instances using what they requested? We join instance_usage with
# instance_events (on collection_id + instance_index + machine_id) to compare
# requested vs. actual usage. This query is expensive, so we use LIMIT.

# %% [markdown]
# We compute the ratio in BigQuery to avoid pulling large joins into Colab.
# Using a 1% TABLESAMPLE of instance_usage as the driving table.

# %%
sql_provisioning = f"""
WITH usage_sample AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        avg_cpu,
        avg_memory
    FROM {fqn('instance_usage_full')}
    TABLESAMPLE SYSTEM (1 PERCENT)
    WHERE avg_cpu IS NOT NULL AND avg_memory IS NOT NULL
),
-- Get the most recent SCHEDULE event for each instance to get its resource request
requests AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        cpu_request,
        memory_request,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, machine_id
            ORDER BY time DESC
        ) AS rn
    FROM {fqn('instance_events_full')}
    WHERE type = 3  -- SCHEDULE
      AND cpu_request IS NOT NULL
      AND memory_request IS NOT NULL
)
SELECT
    u.avg_cpu,
    u.avg_memory,
    r.cpu_request,
    r.memory_request,
    SAFE_DIVIDE(u.avg_cpu, r.cpu_request) AS cpu_utilization_ratio,
    SAFE_DIVIDE(u.avg_memory, r.memory_request) AS memory_utilization_ratio
FROM usage_sample u
INNER JOIN requests r
    ON u.collection_id = r.collection_id
    AND u.instance_index = r.instance_index
    AND u.machine_id = r.machine_id
    AND r.rn = 1
LIMIT 5000000
"""
provisioning_df = run_query(sql_provisioning)
print(f"Provisioning comparison rows: {len(provisioning_df):,}")
print(provisioning_df.describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# CPU utilization ratio
ax = axes[0]
cpu_ratio = provisioning_df['cpu_utilization_ratio'].drop_nulls().to_numpy()
cpu_ratio_clipped = np.clip(cpu_ratio, 0, 5)  # Clip for visualization
ax.hist(cpu_ratio_clipped, bins=100, color='steelblue', edgecolor='none', density=True)
ax.axvline(x=1.0, color='red', linestyle='--', label='Ratio = 1.0 (fully utilized)')
ax.set_xlabel('CPU Utilization Ratio (avg_cpu / cpu_request)')
ax.set_ylabel('Density')
ax.set_title('CPU: Usage / Request Ratio')
ax.legend()

# Memory utilization ratio
ax = axes[1]
mem_ratio = provisioning_df['memory_utilization_ratio'].drop_nulls().to_numpy()
mem_ratio_clipped = np.clip(mem_ratio, 0, 5)
ax.hist(mem_ratio_clipped, bins=100, color='coral', edgecolor='none', density=True)
ax.axvline(x=1.0, color='red', linestyle='--', label='Ratio = 1.0 (fully utilized)')
ax.set_xlabel('Memory Utilization Ratio (avg_memory / memory_request)')
ax.set_ylabel('Density')
ax.set_title('Memory: Usage / Request Ratio')
ax.legend()

fig.suptitle('Figure 4.2a: Utilization vs. Request Ratios (SAMPLED, clipped at 5.0)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_utilization_vs_request_ratio')
plt.show()

# %% [markdown]
# *Figure 4.2a shows the ratio of actual average usage to requested resources.
# Ratios < 1 indicate over-provisioning (wasted resources); ratios > 1 indicate
# under-provisioning (usage exceeds request, possible contention). This is SAMPLED
# data — a 1% TABLESAMPLE of instance_usage joined to SCHEDULE events.*

# %% [markdown]
# ### 4.3 Temporal Utilization Patterns
#
# Does average utilization change over the trace period? We compute daily average
# utilization using BigQuery aggregation (full population).

# %%
sql_daily_utilization = f"""
SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(CAST(start_time AS INT64)), DAY) AS day_bucket,
    COUNT(*) AS measurement_count,
    AVG(avg_cpu) AS mean_avg_cpu,
    AVG(avg_memory) AS mean_avg_memory,
    AVG(max_cpu) AS mean_max_cpu,
    AVG(max_memory) AS mean_max_memory
FROM {fqn('instance_usage_full')}
WHERE start_time > 0 AND start_time < 9223372036854775807
GROUP BY day_bucket
ORDER BY day_bucket
"""
daily_utilization = run_query(sql_daily_utilization)
print(f"Daily utilization buckets: {len(daily_utilization)}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

days = daily_utilization['day_bucket'].to_list()

ax = axes[0]
ax.plot(days, daily_utilization['mean_avg_cpu'].to_list(), label='Avg CPU', color='steelblue')
ax.plot(days, daily_utilization['mean_max_cpu'].to_list(), label='Max CPU', color='steelblue', linestyle='--', alpha=0.7)
ax.set_ylabel('CPU Utilization (normalized)')
ax.set_title('Daily CPU Utilization')
ax.legend()

ax = axes[1]
ax.plot(days, daily_utilization['mean_avg_memory'].to_list(), label='Avg Memory', color='coral')
ax.plot(days, daily_utilization['mean_max_memory'].to_list(), label='Max Memory', color='coral', linestyle='--', alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Memory Utilization (normalized)')
ax.set_title('Daily Memory Utilization')
ax.legend()

fig.suptitle('Figure 4.3a: Daily Average Utilization Over Trace Period (full population)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_daily_utilization_trend')
plt.show()

# %% [markdown]
# *Figure 4.3a shows how cluster-wide average utilization evolves over the trace
# period. Stable utilization supports stationarity assumptions. Significant trends
# or shifts would need to be addressed in temporal feature engineering (e.g., using
# relative utilization rather than absolute values).*

# %% [markdown]
# ### 4.4 Per-Machine Utilization Variance
#
# How much does average utilization vary across machines? High variance suggests
# heterogeneous workload placement. Computed via BigQuery aggregation.

# %%
sql_machine_utilization = f"""
SELECT
    machine_id,
    COUNT(*) AS n_measurements,
    AVG(avg_cpu) AS mean_cpu,
    AVG(avg_memory) AS mean_memory,
    STDDEV(avg_cpu) AS std_cpu,
    STDDEV(avg_memory) AS std_memory
FROM {fqn('instance_usage_full')}
WHERE machine_id IS NOT NULL
GROUP BY machine_id
HAVING COUNT(*) >= 100  -- Only machines with enough data
ORDER BY mean_cpu DESC
LIMIT 50000
"""
machine_util = run_query(sql_machine_utilization)
print(f"Machines with >= 100 measurements: {len(machine_util):,}")
print(machine_util.describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.hist(machine_util['mean_cpu'].to_numpy(), bins=100, color='steelblue', edgecolor='none')
ax.set_xlabel('Mean CPU Utilization per Machine')
ax.set_ylabel('Number of Machines')
ax.set_title('Distribution of Mean CPU Utilization Across Machines')

ax = axes[1]
ax.hist(machine_util['mean_memory'].to_numpy(), bins=100, color='coral', edgecolor='none')
ax.set_xlabel('Mean Memory Utilization per Machine')
ax.set_ylabel('Number of Machines')
ax.set_title('Distribution of Mean Memory Utilization Across Machines')

fig.suptitle('Figure 4.4a: Per-Machine Utilization Distributions (SAMPLED, top 50K machines)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_per_machine_utilization')
plt.show()

# %% [markdown]
# *Figure 4.4a shows how average utilization varies across machines. Wide spread
# indicates heterogeneous workload assignment. Bimodal patterns might indicate
# distinct machine roles (e.g., compute-heavy vs. memory-heavy). Results are
# from the top 50K machines by mean CPU — not a random sample.*

# %% [markdown]
# ---
# # Section 5: collection_events Analysis
#
# Collections are groups of instances (jobs). Understanding collection-level
# patterns helps contextualize instance-level failures.

# %% [markdown]
# ### 5.1 Collection Type Distribution

# %%
sql_collection_types = f"""
SELECT
    collection_type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT collection_id) AS unique_collections
FROM {fqn('collection_events_full')}
GROUP BY collection_type
ORDER BY event_count DESC
"""
collection_types = run_query(sql_collection_types)
print(collection_types)
save_table(collection_types, 'google_collection_type_distribution')

# %% [markdown]
# ### 5.2 Scheduling Class Distribution

# %%
sql_sched_class = f"""
SELECT
    scheduling_class,
    COUNT(*) AS event_count,
    COUNT(DISTINCT collection_id) AS unique_collections
FROM {fqn('collection_events_full')}
GROUP BY scheduling_class
ORDER BY scheduling_class
"""
sched_class = run_query(sql_sched_class)
print(sched_class)
save_table(sched_class, 'google_collection_scheduling_class')

# %% [markdown]
# ### 5.3 Priority Distribution

# %%
sql_coll_priority = f"""
SELECT
    CASE
        WHEN priority <= 99 THEN 'Free (<=99)'
        WHEN priority BETWEEN 100 AND 115 THEN 'Best-effort (100-115)'
        WHEN priority BETWEEN 116 AND 119 THEN 'Mid-tier (116-119)'
        WHEN priority BETWEEN 120 AND 359 THEN 'Production (120-359)'
        WHEN priority >= 360 THEN 'Monitoring (>=360)'
        ELSE 'NULL'
    END AS priority_tier,
    COUNT(*) AS event_count,
    COUNT(DISTINCT collection_id) AS unique_collections
FROM {fqn('collection_events_full')}
GROUP BY priority_tier
ORDER BY priority_tier
"""
coll_priority = run_query(sql_coll_priority)
print(coll_priority)
save_table(coll_priority, 'google_collection_priority_distribution')

# %% [markdown]
# **Figure 5.3a:** Collection priority and scheduling class distributions.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Priority tiers
ax = axes[0]
tiers = coll_priority['priority_tier'].to_list()
counts = coll_priority['event_count'].to_list()
ax.barh(tiers, counts, color='steelblue')
ax.set_xlabel('Event Count')
ax.set_title('Collection Events by Priority Tier')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

# Scheduling classes
ax = axes[1]
classes = sched_class['scheduling_class'].to_list()
class_labels = [str(c) for c in classes]
class_counts = sched_class['event_count'].to_list()
ax.barh(class_labels, class_counts, color='coral')
ax.set_xlabel('Event Count')
ax.set_title('Collection Events by Scheduling Class')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

fig.suptitle('Figure 5.3a: Collection Event Distributions', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_collection_distributions')
plt.show()

# %% [markdown]
# *Figure 5.3a shows the distribution of collection events by priority tier (left)
# and scheduling class (right). These distributions reveal the workload mix —
# the proportion of production vs. batch vs. free-tier jobs affects failure
# semantics (e.g., EVICT is expected for free-tier, problematic for production).*

# %% [markdown]
# ### 5.4 Collection Lifecycle Patterns
#
# How are collection events distributed over time? Any correlation with instance
# failure spikes?

# %%
sql_coll_daily = f"""
SELECT
    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(CAST(time AS INT64)), DAY) AS day_bucket,
    type,
    COUNT(*) AS event_count
FROM {fqn('collection_events_full')}
WHERE time > 0 AND time < 9223372036854775807
GROUP BY day_bucket, type
ORDER BY day_bucket, type
"""
coll_daily = run_query(sql_coll_daily)

# Collection event types use the same codes as instance events
coll_daily = coll_daily.with_columns(
    pl.col('type').map_elements(lambda t: EVENT_TYPE_LABELS.get(t, f'UNKNOWN_{t}'), return_dtype=pl.Utf8).alias('type_label')
)

# %%
# Total daily collection events
coll_daily_total = coll_daily.group_by('day_bucket').agg(pl.col('event_count').sum()).sort('day_bucket')

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(coll_daily_total['day_bucket'].to_list(), coll_daily_total['event_count'].to_list(),
        color='steelblue', linewidth=1.5)
ax.set_xlabel('Date')
ax.set_ylabel('Collection Events per Day')
ax.set_title('Figure 5.4a: Daily Collection Event Volume')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
fig.tight_layout()
save_figure(fig, 'google_collection_daily_volume')
plt.show()

# %% [markdown]
# *Figure 5.4a shows daily collection event volume. Compare with instance event
# volume (Figure 2.4a) to see whether collection-level changes drive instance-level
# events.*

# %% [markdown]
# ---
# # Section 6: Cross-Table Analysis
#
# Joining tables to investigate relationships: do failures correlate with machine
# characteristics or resource utilization?

# %% [markdown]
# ### 6.1 Failures vs. Machine Capacity and Platform
#
# Do certain machine capacities or platforms experience more failures? We join
# instance_events (failure events) with machine_events (capacity/platform) on
# machine_id.

# %%
sql_failure_by_capacity = f"""
WITH failure_machines AS (
    SELECT
        machine_id,
        COUNT(*) AS failure_count
    FROM {fqn('instance_events_full')}
    WHERE type IN (4, 5, 8)
      AND machine_id IS NOT NULL
    GROUP BY machine_id
),
machine_info AS (
    SELECT
        machine_id,
        capacity_cpus,
        capacity_memory,
        platform_id,
        ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY time DESC) AS rn
    FROM {fqn('machine_events_full')}
    WHERE type IN (1, 3)  -- ADD or UPDATE
      AND capacity_cpus IS NOT NULL
)
SELECT
    m.machine_id,
    m.capacity_cpus,
    m.capacity_memory,
    m.platform_id,
    COALESCE(f.failure_count, 0) AS failure_count
FROM machine_info m
LEFT JOIN failure_machines f ON m.machine_id = f.machine_id
WHERE m.rn = 1
"""
failure_capacity = run_query(sql_failure_by_capacity)
print(f"Machines with capacity info: {len(failure_capacity):,}")
print(failure_capacity.describe())

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Scatter: CPU capacity vs failure count
ax = axes[0]
ax.scatter(failure_capacity['capacity_cpus'].to_numpy(),
           failure_capacity['failure_count'].to_numpy(),
           alpha=0.1, s=5, color='steelblue')
ax.set_xlabel('CPU Capacity (normalized)')
ax.set_ylabel('Failure Count')
ax.set_title('CPU Capacity vs. Failure Count')
ax.set_yscale('log')

# Scatter: Memory capacity vs failure count
ax = axes[1]
ax.scatter(failure_capacity['capacity_memory'].to_numpy(),
           failure_capacity['failure_count'].to_numpy(),
           alpha=0.1, s=5, color='coral')
ax.set_xlabel('Memory Capacity (normalized)')
ax.set_ylabel('Failure Count')
ax.set_title('Memory Capacity vs. Failure Count')
ax.set_yscale('log')

fig.suptitle('Figure 6.1a: Machine Capacity vs. Failure Count', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_capacity_vs_failures')
plt.show()

# %% [markdown]
# *Figure 6.1a shows the relationship between machine capacity and failure count.
# If certain capacity tiers have more failures, machine capacity is a useful
# feature. However, higher-capacity machines may simply run more instances -
# normalize by instance count in Phase 3 if needed.*

# %% [markdown]
# Failure rate by platform_id.

# %%
failure_by_platform = failure_capacity.group_by('platform_id').agg([
    pl.col('failure_count').sum().alias('total_failures'),
    pl.col('failure_count').mean().alias('avg_failures_per_machine'),
    pl.len().alias('machine_count'),
]).sort('total_failures', descending=True)

print("Failure counts by platform:")
print(failure_by_platform)
save_table(failure_by_platform, 'google_failures_by_platform')

# %% [markdown]
# ### 6.2 Failure Events vs. Utilization
#
# Do instances experience failures when utilization is high? We sample
# instance_usage records for instances that had failure events and compare
# their utilization to a random sample of non-failure instances.
#
# This query is expensive (joins two large tables) so we use TABLESAMPLE.

# %%
sql_failure_utilization = f"""
-- Get instance identifiers that experienced failures
WITH failure_instances AS (
    SELECT DISTINCT
        collection_id,
        instance_index,
        machine_id
    FROM {fqn('instance_events_full')}
    WHERE type IN (4, 5, 8)
      AND machine_id IS NOT NULL
      AND collection_id IS NOT NULL
),
-- Sample utilization records, marking failure vs. non-failure instances
usage_with_label AS (
    SELECT
        u.avg_cpu,
        u.avg_memory,
        u.max_cpu,
        u.max_memory,
        CASE WHEN f.collection_id IS NOT NULL THEN 1 ELSE 0 END AS is_failure_instance
    FROM {fqn('instance_usage_full')} u
    TABLESAMPLE SYSTEM (1 PERCENT)
    LEFT JOIN failure_instances f
        ON u.collection_id = f.collection_id
        AND u.instance_index = f.instance_index
        AND u.machine_id = f.machine_id
    WHERE u.avg_cpu IS NOT NULL AND u.avg_memory IS NOT NULL
)
SELECT *
FROM usage_with_label
LIMIT 5000000
"""
failure_util = run_query(sql_failure_utilization)

n_failure = failure_util.filter(pl.col('is_failure_instance') == 1).height
n_nonfailure = failure_util.filter(pl.col('is_failure_instance') == 0).height
print(f"Failure instance records: {n_failure:,}")
print(f"Non-failure instance records: {n_nonfailure:,}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

fail_data = failure_util.filter(pl.col('is_failure_instance') == 1)
nofail_data = failure_util.filter(pl.col('is_failure_instance') == 0)

# CPU utilization comparison
ax = axes[0]
ax.hist(nofail_data['avg_cpu'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='steelblue', label=f'Non-failure (n={n_nonfailure:,})')
ax.hist(fail_data['avg_cpu'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='red', label=f'Failure (n={n_failure:,})')
ax.set_xlabel('Average CPU Utilization')
ax.set_ylabel('Density')
ax.set_title('Avg CPU: Failure vs. Non-failure Instances')
ax.legend()

# Memory utilization comparison
ax = axes[1]
ax.hist(nofail_data['avg_memory'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='steelblue', label='Non-failure')
ax.hist(fail_data['avg_memory'].to_numpy(), bins=100, density=True,
        alpha=0.6, color='red', label='Failure')
ax.set_xlabel('Average Memory Utilization')
ax.set_ylabel('Density')
ax.set_title('Avg Memory: Failure vs. Non-failure Instances')
ax.legend()

fig.suptitle('Figure 6.2a: Utilization of Failure vs. Non-failure Instances (1% TABLESAMPLE)', fontsize=13, y=1.02)
fig.tight_layout()
save_figure(fig, 'google_utilization_failure_vs_nonfailure')
plt.show()

# %% [markdown]
# *Figure 6.2a compares the utilization distributions of instances that experienced
# failures vs. those that did not. If failure instances show higher utilization,
# this supports resource contention as a failure predictor. SAMPLED — based on
# a 1% TABLESAMPLE of instance_usage joined to failure instance identifiers.*

# %% [markdown]
# ---
# # Section 7: Key Findings Summary
#
# ## Summary of Findings
#

# %% [markdown]
# ### 7.1 Dataset Dimensions and Temporal Coverage
#
# The Google Cluster Traces v3 dataset consists of four primary tables spanning
# approximately 31 days of production Borg cluster activity (May 1–31, 2019). The
# instance_usage table dominates storage at nearly 2 TB, reflecting the granularity of
# per-instance resource telemetry.
#
# | Table | Rows | Columns | Size (GB) | Duration (days) | Key Coverage |
# |-------|------|---------|-----------|-----------------|--------------|
# | instance_events_full | 1,717,317,922 | 12 | 387.45 | 31 days* | 10,005 machines |
# | instance_usage_full | 7,575,500,668 | 19 | 1,991.83 | 31 days | 2.68M timestamps |
# | collection_events_full | 20,807,441 | 14 | 3.12 | 31 days | 5.2M collections |
# | machine_events_full | 46,219 | 7 | 0.01 | 31 days | 10,001 machines |
#
# _*instance_events contains sentinel timestamps (0 = before trace, 2⁶³−1 = after trace)
# requiring filtering._
#
# _Note: machine_attributes_full (1,702,926 rows) omitted from main table — used for
# machine-level feature enrichment only._
#
# **Total dataset size:** ~2,382 GB (2.3 TB).
#
# **Combined rows:** ~9.3 billion across all tables.

# %% [markdown]
# ### 7.2 Event Type Distribution and Failure Definition
#
# The event type distribution reveals the full instance lifecycle. Terminal events
# (types 4–8) represent lifecycle completion outcomes and are the basis for failure
# labeling.
#
# | Type | Label | Count | % Total | Category | Terminal? | Failure? |
# |------|-------|-------|---------|----------|-----------|----------|
# | 0 | SUBMIT | 352,128,320 | 20.50 | Lifecycle | No | ... |
# | 1 | QUEUE | 60,959,018 | 3.55 | Lifecycle | No | ... |
# | 2 | ENABLE | 340,369,139 | 19.82 | Lifecycle | No | ... |
# | 3 | SCHEDULE | 326,295,154 | 19.00 | Lifecycle | No | ... |
# | 4 | **EVICT** | 117,133,729 | 6.82 | Terminal | Yes | Conditional |
# | 5 | **FAIL** | 17,358,057 | 1.01 | Terminal | Yes | Yes |
# | 6 | **FINISH** | 73,611,983 | 4.29 | Terminal | Yes | No (Success) |
# | 7 | KILL | 149,277,877 | 8.69 | Terminal | Yes | No (Excluded) |
# | 8 | **LOST** | 4,351,433 | 0.25 | Terminal | Yes | Yes |
# | 9 | UPDATE_PENDING | 67,373,803 | 3.92 | Update | No | ... |
# | 10 | UPDATE_RUNNING | 208,459,409 | 12.14 | Update | No | ... |
#
# #### 7.2.1 EVICT (Type 4) Analysis: Expected Preemption vs. Failure
#
# Evictions are overwhelmingly concentrated in low-priority tiers. 93.1% of all
# evictions occur in Free (≤99) and Best-effort (100–115) priority tiers, where
# preemption by higher-priority tasks is expected Borg behavior. Only 0.13% of
# evictions affect Production-priority instances.
#
# | Priority Tier | EVICT Count | % of EVICTs | Expected? | Label As |
# |---------------|-------------|-------------|-----------|----------|
# | Free (≤99) | 86,104,965 | 73.5% | Yes | Exclude |
# | Best-effort (100–115) | 22,919,644 | 19.6% | Yes | Exclude |
# | Mid-tier (116–119) | 134,034 | 0.1% | Borderline | Failure* |
# | Production (120–359) | 153,105 | 0.1% | No | Failure |
# | Monitoring (≥360) | 7,821,981 | 6.7% | Partial | TBD† |
#
# _*Mid-tier evictions are rare and may represent scheduling anomalies.
# †Monitoring evictions need further investigation in Phase 3._
#
# #### 7.2.2 Failure Definition Decision
#
# **Primary recommendation (conservative):** Types 5 (FAIL) and 8 (LOST) as failure;
# Type 6 (FINISH) as success. EVICT excluded from primary model; KILL excluded
# (user-initiated cancellation, not system failure).
#
# **Secondary analysis (sensitivity):** Production-priority EVICTs (type 4, priority
# ≥120) added as failures in a sensitivity analysis to test whether production
# evictions exhibit distinct predictive patterns.
#
# **Justification:**
# - FAIL (type 5) represents unambiguous program errors or OOM conditions — clearly a
# system failure.
# - LOST (type 8) indicates missing termination events, likely infrastructure failures
# — the system lost track of the instance.
# - EVICT (type 4) is predominantly expected preemption for low-priority workloads
# (93.1% in Free/Best-effort tiers). Labeling all evictions as failures would conflate
# intentional scheduling decisions with genuine failures.
# - KILL (type 7) represents user-initiated cancellation, not predictable system
# behavior.
#
# **Class imbalance (primary definition):** FINISH: 73,611,983 vs. FAIL+LOST: 21,709,490
# — a 3.4:1 success-to-failure ratio. This is moderate imbalance, manageable with SMOTE,
# cost-sensitive learning, or ensemble balancing without extreme oversampling
# (Li et al., 2021).


# %% [markdown]
# ### 7.3 Null Patterns and Imputation Strategy
#
# Null analysis reveals that missingness in the instance_events table is predominantly
# structural (lifecycle-dependent), not random.
#
# | Event Type | machine_id %null | cpu_req %null | mem_req %null | sched_class | priority | Total Rows |
# |------------|------------------|---------------|---------------|-------------|----------|------------|
# | SUBMIT (0) | 95.48 | 0.01 | 0.01 | 0.0 | 0.0 | 352M |
# | QUEUE (1) | 99.62 | 0.0 | 0.0 | 0.0 | 0.0 | 61M |
# | ENABLE (2) | 96.78 | 0.0 | 0.0 | 0.0 | 0.0 | 340M |
# | SCHEDULE (3) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 326M |
# | EVICT (4) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 117M |
# | FAIL (5) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 17M |
# | FINISH (6) | 4.52 | 0.0 | 0.0 | 0.0 | 0.0 | 74M |
# | KILL (7) | 20.08 | 0.0 | 0.0 | 0.0 | 0.0 | 149M |
# | LOST (8) | 2.59 | 0.0 | 0.0 | 0.0 | 0.0 | 4M |
#
# **Key finding:** machine_id nulls are structural. Pre-scheduling events (SUBMIT,
# QUEUE, ENABLE) have 95–99% null machine_id because no machine has been assigned yet.
# Post-scheduling events (SCHEDULE, EVICT, FAIL) have near-zero null machine_id. This
# is not missing data — it reflects the instance lifecycle.
#
# **Imputation Strategy**
# - **Structural nulls (machine_id):** Do NOT impute. Filter to post-scheduling events
# (types 3–8) when machine_id is needed for analysis. For failure prediction, the
# relevant events (SCHEDULE → terminal) have complete machine_id.
# - **cpu_request / memory_request:** 47,933 nulls out of 1.7B rows (0.003%). These can
# be safely dropped or imputed with median values. Negligible impact on analysis.
# - **instance_usage nulls:** sample_memory is 100% null across all 7.5B rows — this
# column should be dropped entirely. max_memory has 0.57% null. cycles_per_instruction
# and memory_accesses_per_instruction are 20.5% null — investigate whether these are
# structurally missing (certain instance types do not report these metrics).
# - **collection_events nulls:** max_per_machine (99.3% null) and max_per_switch
# (99.7% null) should be dropped. parent_collection_id (36% null) may indicate top-level
# collections without parents.


# %% [markdown]
# ### 7.4 Temporal Patterns and Train/Test Splitting
#
# **Coverage:** All tables span approximately 31 days (May 1–31, 2019). The
# instance_usage table has the cleanest temporal boundaries, with start_time ranging
# from 300,000,000 to 2,678,999,000,000 microseconds.
#
# **Sentinel values in instance_events:** The time column contains values of 0 (event
# occurred before trace start) and 2⁶³−1 (event occurred after trace end). These must
# be filtered or handled as censored observations in temporal analysis.
#
# **Machine events:** 46,219 events across 31 days, covering 10,001 unique machines.
# Machine ADD (27,777), REMOVE (17,941), and UPDATE (501) events capture cluster
# topology changes.
#
# **Implications for Train/Test Splitting**
# - **Temporal split required:** Random splitting would leak future information. The
# recommended approach is chronological partitioning: approximately 70% train
# (days 1–22), 15% validation (days 22–26), 15% test (days 26–31).
# - **Blocked temporal cross-validation:** For hyperparameter tuning, use expanding or
# sliding window CV within the training period to respect temporal ordering
# (Bergmeir & Benítez, 2012).
# - **Open question:** Are there diurnal or weekly patterns in event density? This needs
# to be investigated in Phase 3 temporal analysis to determine whether stratified
# temporal sampling is warranted.

# %% [markdown]
# ### 7.5 Resource and Utilization Patterns
#
# Resource requests and utilization are expressed as normalized values [0, 1] relative
# to the largest machine in the cluster.
#
# | Metric | Mean | Median | P25 | P75 | P95 | P99 |
# |--------|------|--------|-----|-----|-----|-----|
# | cpu_request | 0.0127 | 0.0088 | 0.0041 | 0.0162 | 0.0257 | 0.0599 |
# | memory_request | 0.0073 | 0.0043 | 0.0018 | 0.0093 | 0.0229 | 0.0327 |
# | avg_cpu (usage) | 0.0008 | 0.0008 | 0.0002 | 0.0067 | 0.0288 | 0.0598 |
# | avg_memory (usage) | 0.0018 | 0.0018 | 0.0008 | 0.0045 | 0.0148 | 0.0341 |
# | max_cpu (usage) | 0.0071 | 0.0071 | 0.0012 | 0.0247 | 0.0936 | 0.2031 |
# | max_memory (usage) | 0.0019 | 0.0019 | 0.0008 | 0.0050 | 0.0169 | 0.0344 |
#
# **Key findings:**
# - Resources are heavily right-skewed: median CPU request is 0.88% of maximum machine
# capacity, and median memory request is 0.43%. Most instances request very small
# resource slices.
# - Actual CPU utilization (avg_cpu) is extremely low — median 0.08%, suggesting
# substantial over-provisioning. The gap between requested and utilized resources may
# be a strong failure predictor.
# - max_cpu shows a wider spread (P99 = 20.3%), indicating occasional burst usage. CPU
# burst patterns may differentiate failure-bound instances.
# - FAIL events have the highest average resource requests among terminal events
# (avg CPU 0.0105, avg memory 0.0085 at the event level), suggesting that resource
# pressure correlates with failure.

# %% [markdown]
# ### 7.6 Machine-Level Patterns
#
# Machine events and platform diversity reveal the cluster's physical infrastructure.
#
# | Platform | Machines | Avg CPU Cap | Avg Mem Cap | Total Failures | Avg Fail/Mach | Fail Density |
# |--------|----------|-------------|-------------|----------------|---------------|--------------|
# | Platform A | 3,685 | 0.592 | 0.321 | 42,415,480 | 11,433 | High |
# | Platform B | 2,358 | 0.996 | 0.603 | 55,786,398 | 21,293 | Very High |
# | Platform C | 1,864 | 0.709 | 0.425 | 28,418,787 | 14,981 | High |
# | Platform D | 1,763 | 0.387 | 0.272 | 12,098,751 | 6,820 | Moderate |
#
# _Note: Platform IDs are hashed. Labels (A–D) assigned by machine count for readability._
#
# **Key findings:**
# - Platform B (highest-capacity machines, avg CPU 0.996) has the highest failure density
# at 21,293 failures per machine. This counterintuitive finding suggests that larger
# machines may run more instances or more failure-prone workloads.
# - Four distinct hardware platforms serve approximately 9,670 machines. Platform A has
# the most machines (3,685) but not the highest per-machine failure rate.
# - Machine churn is moderate: 27,777 ADD events and 17,941 REMOVE events over 31 days.
# Machine-level features (time-since-add, failure history) are feasible.
# - Machine capacity varies substantially (CPU capacity ranges 0.387–1.0), suggesting
# heterogeneous hardware that may interact with failure patterns.

# %% [markdown]
# ### 7.7 Open Questions for Phase 3
#
# 1. **Diurnal and weekly temporal patterns:** Do event densities and failure rates vary by
# time of day or day of week? This determines whether temporal features (hour, weekday)
# are informative and whether temporal stratification is needed in the train/test split.
# 2. **Monitoring-tier evictions:** The 7.8M monitoring-priority evictions (6.7% of all
# EVICTs) need investigation. Are these health-check or canary processes that are
# intentionally short-lived, or do they represent genuine infrastructure issues?
# 3. **Instance lifecycle reconstruction:** Can we reconstruct full instance lifecycles
# (SUBMIT → SCHEDULE → terminal) to extract duration and transition features? How many
# instances have clean lifecycle sequences vs. gaps?
# 4. **Usage–failure correlation:** Do instances that eventually fail exhibit different
# utilization profiles (CPU bursts, memory ramps) in the time windows before failure?
# This is the core predictive signal for RQ1.
# 5. **Collection-level failure patterns:** Are failures concentrated in certain
# collections? Collection-type 0 dominates (99.3% of events). Is collection_type
# informative or essentially constant?
# 6. **Cycles_per_instruction and memory_accesses_per_instruction:** These usage metrics
# are 20.5% null. Is missingness correlated with instance type, scheduling class, or
# failure outcome? Could these be strong predictors if properly handled?
# 7. **Sentinel timestamp handling:** How many instances have time=0 or time=MAX_INT64?
# These represent censored observations. Should they be excluded, or can they be handled
# as left/right-censored data?

# %% [markdown]
# ### 7.8 Preliminary Decisions Log
#
# | Decision | Choice | Evidence | Notebook | Literature |
# |----------|--------|----------|----------|------------|
# | Failure event types | Types 5 (FAIL) + 8 (LOST); sensitivity: add production EVICTs | Sec 7.2: 93% EVICTs are Free/Best-effort | 03, Sec 2 | Google (2019); Cheng et al. (2022) |
# | Class imbalance strategy | SMOTE + cost-sensitive learning; 3.4:1 ratio manageable | Sec 7.2: 73.6M success vs 21.7M failure | 03, Sec 2 | Li et al. (2021) |
# | Null handling: machine_id | Do NOT impute; filter to post-scheduling events when needed | Sec 7.3: Structural nulls (95–99%) for pre-scheduling events | 03, Sec 2 | — |
# | Null handling: resources | Drop or median-impute 47,933 rows (0.003% null) | Sec 7.3: Negligible null rate | 03, Sec 2 | — |
# | Null handling: usage cols | Drop sample_memory (100% null); investigate CPI/MAPI (20.5%) | Sec 7.3: sample_memory is empty | 03, Sec 2 | — |
# | Temporal split | Chronological: 70/15/15 train/val/test; blocked temporal CV | Sec 7.4: 31-day coverage | 03, Sec 2 | Bergmeir & Benítez (2012); Raschka (2018) |
# | Machine features | Include platform, capacity, failure history, churn metrics | Sec 7.6: Platform correlates with failure density | 03, Sec 3/6 | Zhang et al. (2023) |
# | KILL handling | Exclude entirely (user-initiated, not predictable) | Sec 7.2: Type 7 is cancellation | 03, Sec 2 | Google (2019) |

