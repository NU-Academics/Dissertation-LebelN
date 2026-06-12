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
# # 01 — BigQuery Caching: Full Population Tables
#
# **Purpose:** Query each of the 5 Google Cluster Traces tables from the public dataset
# and cache the full population into our own `dissertation_lebel` BigQuery dataset.
# After caching, all subsequent queries hit our own tables at zero/minimal cost.
#
# **Run this notebook ONCE.** Re-running will overwrite the cached tables via
# `CREATE OR REPLACE TABLE`.
#
# **Estimated cost:** ~$10-15 in BigQuery on-demand pricing (~2.1 TB scanned total).
# The instance_usage table alone is ~1.5 TB.
#
# **Prerequisites:**
# - Colab Secrets configured: `GCP_PROJECT_ID`, `GITHUB_PAT`
# - BigQuery dataset `dissertation_lebel` created in your project (US region)
# - Run `00_setup_environment.py` first to verify connectivity

# %% [markdown]
# ## 1. Colab Session Setup

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes

# %%
from google.colab import userdata

PROJECT_ID = userdata.get('GCP_PROJECT_ID')
print(f"GCP Project: {PROJECT_ID}")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
OUTPUT_DIR = DRIVE_PATH / 'outputs'
TABLES_DIR = OUTPUT_DIR / 'tables'

for dir_path in [OUTPUT_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# %%
from google.colab import auth
auth.authenticate_user()

from google.cloud import bigquery
bq_client = bigquery.Client(project=PROJECT_ID)

# %% [markdown]
# ## 2. Ensure Dataset Exists

# %%
dataset_id = f"{PROJECT_ID}.dissertation_lebel"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"

try:
    bq_client.get_dataset(dataset_id)
    print(f"Dataset {dataset_id} already exists.")
except Exception:
    dataset = bq_client.create_dataset(dataset, exists_ok=True)
    print(f"Created dataset {dataset_id}.")

# %% [markdown]
# ## 3. Helper Functions

# %%
import time
import polars as pl


def run_cache_query(sql: str, table_name: str) -> dict:
    """Execute a CREATE TABLE cache query and return timing + verification info."""
    print(f"{'=' * 60}")
    print(f"Caching: {table_name}")
    print(f"{'=' * 60}")

    start = time.time()
    job = bq_client.query(sql)
    job.result()  # Block until complete
    elapsed = time.time() - start

    minutes, seconds = divmod(elapsed, 60)
    print(f"  Query completed in {int(minutes)}m {seconds:.1f}s")
    if job.total_bytes_processed:
        tb = job.total_bytes_processed / (1024 ** 4)
        print(f"  Bytes processed: {job.total_bytes_processed / (1024 ** 3):.2f} GB ({tb:.4f} TB)")

    # Verification query against the newly cached table
    verify_sql = f"""
    SELECT
        COUNT(*) AS row_count
    FROM `{PROJECT_ID}.dissertation_lebel.{table_name}`
    """
    verify_result = bq_client.query(verify_sql).to_dataframe()
    row_count = int(verify_result['row_count'][0])

    # Get table metadata for column count and size
    table_ref = bq_client.get_table(f"{PROJECT_ID}.dissertation_lebel.{table_name}")
    num_columns = len(table_ref.schema)
    size_gb = table_ref.num_bytes / (1024 ** 3)

    print(f"  Rows:    {row_count:>15,}")
    print(f"  Columns: {num_columns:>15,}")
    print(f"  Size:    {size_gb:>15.2f} GB")
    print()

    return {
        'table': table_name,
        'rows': row_count,
        'columns': num_columns,
        'size_gb': round(size_gb, 2),
        'query_seconds': round(elapsed, 1),
    }

# %% [markdown]
# ## 4. Cache Table 1: Instance Events (~500 GB)
#
# All instance lifecycle events (SUBMIT, QUEUE, SCHEDULE, EVICT, FAIL, FINISH, etc.).
# No derived columns — raw data only.

# %%
sql_instance_events = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.dissertation_lebel.instance_events_full` AS
SELECT
    time,
    type,
    collection_id,
    scheduling_class,
    collection_type,
    priority,
    instance_index,
    machine_id,
    alloc_collection_id,
    resource_request.cpus as cpu_request,
    resource_request.memory as memory_request,
    constraint
FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_events
"""

result_instance_events = run_cache_query(sql_instance_events, 'instance_events_full')

# %% [markdown]
# ## 5. Cache Table 2: Machine Events (~1 GB)
#
# Machine lifecycle events (ADD, REMOVE, UPDATE). Includes capacity info.

# %%
sql_machine_events = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.dissertation_lebel.machine_events_full` AS
SELECT
    time,
    machine_id,
    type,
    switch_id,
    `capacity`.cpus as capacity_cpus,
    `capacity`.memory as capacity_memory,
    platform_id
FROM `google.com:google-cluster-data`.clusterdata_2019_a.machine_events
"""

result_machine_events = run_cache_query(sql_machine_events, 'machine_events_full')

# %% [markdown]
# ## 6. Cache Table 3: Instance Usage (~1.5 TB)
#
# **WARNING: This is the largest table (~1.5 TB). This single query will cost
# approximately $7-8 in BigQuery on-demand pricing and may take 10-30 minutes
# to complete. Do NOT re-run unless you intend to refresh the cache.**
#
# Resource utilization measurements for every running instance. Contains CPU,
# memory, and performance counter data.

# %%
sql_instance_usage = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.dissertation_lebel.instance_usage_full` AS
SELECT
    start_time,
    end_time,
    collection_id,
    instance_index,
    machine_id,
    alloc_collection_id,
    average_usage.cpus as avg_cpu,
    average_usage.memory as avg_memory,
    maximum_usage.cpus as max_cpu,
    maximum_usage.memory as max_memory,
    random_sample_usage.cpus as sample_cpu,
    random_sample_usage.memory as sample_memory,
    assigned_memory,
    page_cache_memory,
    cycles_per_instruction,
    memory_accesses_per_instruction,
    sample_rate,
    cpu_usage_distribution,
    tail_cpu_usage_distribution
FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
"""

result_instance_usage = run_cache_query(sql_instance_usage, 'instance_usage_full')

# %% [markdown]
# ## 7. Cache Table 4: Collection Events (~50 GB)
#
# Job/collection scheduling events. Includes scheduling metadata, priority,
# and dependency information.

# %%
sql_collection_events = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.dissertation_lebel.collection_events_full` AS
SELECT
    time,
    type,
    collection_id,
    scheduling_class,
    collection_type,
    priority,
    user,
    collection_name,
    parent_collection_id,
    start_after_collection_ids,
    max_per_machine,
    max_per_switch,
    vertical_scaling,
    scheduler
FROM `google.com:google-cluster-data`.clusterdata_2019_a.collection_events
"""

result_collection_events = run_cache_query(sql_collection_events, 'collection_events_full')

# %% [markdown]
# ## 8. Cache Table 5: Machine Attributes (~10 GB)
#
# Machine attribute key-value pairs over time. Useful for hardware heterogeneity analysis.

# %%
sql_machine_attributes = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.dissertation_lebel.machine_attributes_full` AS
SELECT
    time,
    machine_id,
    name,
    value,
    deleted
FROM `google.com:google-cluster-data`.clusterdata_2019_a.machine_attributes
"""

result_machine_attributes = run_cache_query(sql_machine_attributes, 'machine_attributes_full')

# %% [markdown]
# ## 9. Summary

# %%
all_results = [
    result_instance_events,
    result_machine_events,
    result_instance_usage,
    result_collection_events,
    result_machine_attributes,
]

summary_df = pl.DataFrame(all_results)

# Add a totals row
totals = pl.DataFrame([{
    'table': 'TOTAL',
    'rows': summary_df['rows'].sum(),
    'columns': None,
    'size_gb': summary_df['size_gb'].sum(),
    'query_seconds': summary_df['query_seconds'].sum(),
}])
summary_with_totals = pl.concat([summary_df, totals], how='diagonal')

print("=" * 70)
print("CACHING SUMMARY — dissertation_lebel dataset")
print("=" * 70)
print(summary_with_totals)

# %% [markdown]
# ## 10. Save Summary to Drive

# %%
output_path = TABLES_DIR / 'bigquery_cache_summary.csv'
summary_with_totals.write_csv(str(output_path))
print(f"Summary saved to: {output_path}")

# %% [markdown]
# ## Next Steps
#
# All 5 tables are now cached in `dissertation_lebel`. Subsequent queries against
# these tables are free (querying your own dataset, not the public one).
#
