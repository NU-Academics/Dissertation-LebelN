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
# # 08. Google Cluster Traces Preprocessing
#
# **Purpose.** Apply the Phase 2 and Phase 3 front-loaded EDA decisions
# (V01-V13, V25-V28) to produce a preprocessed `instance_events` slice
# suitable for feature engineering in Week 3. The heavy work is pushed to
# BigQuery (transformations and joins against the 1.72B-row
# `instance_events_full` table); Polars LazyFrames handle in-memory
# assertions on materialized result tables.
#
# **Status.** Week 2 deliverable of the 11-week Chapter 4 plan. This
# notebook is exploratory; once each section validates against the EDA
# numbers, the next step extracts the logic into `src/preprocessing/google_traces.py`
# and `src/preprocessing/lifecycle.py`.
#
# **Inputs.**
# - BigQuery cached tables `{PROJECT}.dissertation_lebel.*_full`.
# - `outputs/tables/sentinel_inventory.csv` (F1, V25 reference).
# - `outputs/tables/monitoring_evict_profile.csv` (F3, V27 reference).
# - `outputs/tables/cpi_mapi_within_instance_variance.csv` (F4, V28 reference).
# - `outputs/tables/eda_decisions.csv` (decisions audit trail).
#
# **Outputs.**
# - `{PROJECT}.dissertation_lebel.instance_events_preprocessed` (BigQuery).
# - `{PROJECT}.dissertation_lebel.instance_lifecycle_summary` (BigQuery,
#   Section 5).
# - `gs://{PROJECT}-dissertation-data/google_preprocessed/instance_events_preprocessed/`
#   (partitioned Parquet on GCS; Drive cannot host a 1.7B-row Parquet).
# - `{OUTPUT_DIR}/preprocessed/google/manifest.json` (Drive manifest:
#   GCS prefix, row counts, column count, per-partition SHA256, creation
#   timestamp).
# - `outputs/tables/preprocessing_verification.csv` (Section 6 assertion
#   results, committed to repo).
#
# **Sections.**
# 0. Colab session setup.
# 1. Sentinel timestamp filtering (V25).
# 2. Structural null handling (V03-V05).
# 3. Failure label construction (V01, V08, V27, P04).
# 4. MNAR indicator encoding for CPI and MAPI (V11, V28).
# 5. Lifecycle reconstruction (V09, V10).
# 6. Post-preprocessing assertion suite (regression guard).
# 7. Export to GCS Parquet plus Drive manifest.

# %% [markdown]
# ---
# ## 0. Colab Session Setup

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes google-cloud-storage pyarrow

# %%
import os
import sys
from pathlib import Path

# Clone (or pull) the repo so utils.* is importable. Mirrors the pattern in
# notebook 00. Requires the GITHUB_PAT Colab secret to be set.
from google.colab import userdata

GITHUB_PAT = userdata.get('GITHUB_PAT')
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

# %%
from google.colab import auth

auth.authenticate_user()

# %%
from utils.colab_setup import setup_drive, OUTPUT_DIR
from utils.bq_client import get_client, table_ref, DATASET, PROJECT_ID

setup_drive()
bq_client = get_client()

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_PREPROCESSED_PREFIX = 'google_preprocessed'
GCS_EVENTS_PARQUET_PREFIX = f'{GCS_PREPROCESSED_PREFIX}/instance_events_preprocessed'

print(f"Project:  {PROJECT_ID}")
print(f"Dataset:  {DATASET}")
print(f"Drive:    {OUTPUT_DIR}")
print(f"GCS:      gs://{GCS_BUCKET}/{GCS_PREPROCESSED_PREFIX}/")

# %%
from datetime import datetime, timezone

import polars as pl
import numpy as np

# %% [markdown]
# ### Path constants and helpers

# %%
PREPROCESSED_DIR = OUTPUT_DIR / 'preprocessed' / 'google'
TABLES_DIR = OUTPUT_DIR / 'tables'
for directory in (PREPROCESSED_DIR, TABLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PREPROCESSED_DIR / 'manifest.json'
VERIFICATION_CSV = TABLES_DIR / 'preprocessing_verification.csv'

# Decision-rule constants (load from the artifacts in production, kept inline
# here for readability)
MAX_INT64 = 9_223_372_036_854_775_807
FAIL_LOST_TYPES = (5, 8)
FINISH_TYPE = 6
PROD_EVICT_PRIORITY_LOW = 120
MONITORING_PRIORITY_LOW = 360


def fqn(table: str) -> str:
    """Return fully-qualified BigQuery table name."""
    return table_ref(table)


def run_query(sql: str) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame (small results only)."""
    return pl.from_pandas(bq_client.query(sql).to_dataframe())


def run_ddl(sql: str, label: str) -> None:
    """Execute a DDL/DML statement (CREATE TABLE, EXPORT DATA) without returning rows."""
    print(f"[BQ] Running: {label}")
    job = bq_client.query(sql)
    job.result()
    print(f"  Done. Bytes processed: {job.total_bytes_processed:,}")


def row_count(table: str) -> int:
    """Return total row count for a fully-qualified table reference."""
    sql = f"SELECT COUNT(*) AS n FROM {fqn(table)}"
    return int(run_query(sql)["n"].item())


# %% [markdown]
# ### Verification log
#
# Every section appends one or more `(check, expected, observed, ok)`
# rows here. The full table writes to
# `outputs/tables/preprocessing_verification.csv` at the end of Section 6.

# %%
verification_rows: list[dict] = []


def record_check(check: str, expected: object, observed: object, ok: bool, notes: str = "") -> None:
    """Append a verification row and print a one-line summary."""
    status = "PASS" if ok else "FAIL"
    verification_rows.append({
        "check": check,
        "expected": str(expected),
        "observed": str(observed),
        "ok": ok,
        "notes": notes,
    })
    suffix = f" ({notes})" if notes else ""
    print(f"  [{status}] {check}: expected {expected}, observed {observed}{suffix}")


# %% [markdown]
# ---
# ## 1. Sentinel timestamp filtering (V25)
#
# Drop rows where `time = 0` (left-censoring sentinel) or
# `time = 2^63 - 1` (right-censoring sentinel). Per F1, sentinel rows are
# a ~0.23% slice of `instance_events_full` (~3.88M for `time = 0`,
# ~18K for `time = 2^63 - 1`); dropping them shifts the FAIL_LOST class
# balance from 3.39:1 to ~3.43:1 (negligible). The artifact
# `outputs/tables/sentinel_inventory.csv` enumerates the per-type
# breakdown.

# %%
sentinel_filter_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_events_clean')}
CLUSTER BY collection_id, instance_index AS
SELECT *
FROM {fqn('instance_events_full')}
WHERE time > 0 AND time < {MAX_INT64};
"""

run_ddl(sentinel_filter_sql, "Sentinel filter -> instance_events_clean")

# %%
# Verify row counts before / after match the sentinel inventory totals
counts_sql = f"""
SELECT
    (SELECT COUNT(*) FROM {fqn('instance_events_full')}) AS n_input,
    (SELECT COUNT(*) FROM {fqn('instance_events_clean')}) AS n_output,
    (SELECT COUNTIF(time = 0) FROM {fqn('instance_events_full')}) AS n_zero,
    (SELECT COUNTIF(time = {MAX_INT64}) FROM {fqn('instance_events_full')}) AS n_max
"""
counts_df = run_query(counts_sql)
n_input = int(counts_df["n_input"].item())
n_output = int(counts_df["n_output"].item())
n_zero = int(counts_df["n_zero"].item())
n_max = int(counts_df["n_max"].item())
n_dropped = n_input - n_output

print(f"Input rows:    {n_input:,}")
print(f"Output rows:   {n_output:,}")
print(f"Dropped (zero):    {n_zero:,}")
print(f"Dropped (max_int): {n_max:,}")
print(f"Total dropped:     {n_dropped:,}")

record_check(
    "Section 1: input row count",
    expected=1_717_317_922,
    observed=n_input,
    ok=(n_input == 1_717_317_922),
    notes="Matches the EDA-confirmed instance_events_full row count.",
)
record_check(
    "Section 1: sentinel-drop count = n_zero + n_max",
    expected=n_zero + n_max,
    observed=n_dropped,
    ok=(n_dropped == n_zero + n_max),
    notes="No other rows were silently dropped.",
)

# %% [markdown]
# ---
# ## 2. Structural null handling (V03-V05)
#
# - Drop rows where `cpu_request` or `memory_request` is NULL (V04;
#   ~47,933 rows per Phase 2 EDA, ~0.003% null rate).
# - Drop the column `sample_memory` (100% null in
#   `instance_usage_full`; V05). This drop is applied when
#   `instance_usage_full` is preprocessed in a companion notebook; it is
#   documented here for traceability but not executed on
#   `instance_events_full` because the column is not present here.
# - Drop the columns `max_per_machine` and `max_per_switch` from
#   `collection_events_full` (99%+ null; V05). Same note as above; the
#   drops will be applied in the `collection_events` preprocessing pass.
# - `machine_id` nulls (95-99%) for pre-scheduling event types are
#   structural (V03) and are NOT dropped; the column stays NULL until
#   the SCHEDULE event populates it. Joins downstream must filter to
#   post-scheduling event types as needed.

# %%
null_filter_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_events_requests_clean')}
CLUSTER BY collection_id, instance_index AS
SELECT *
FROM {fqn('instance_events_clean')}
WHERE cpu_request IS NOT NULL
  AND memory_request IS NOT NULL;
"""

run_ddl(null_filter_sql, "Drop null cpu_request / memory_request rows")

# %%
# Verify the dropped count is near the EDA-confirmed 47,933 baseline.
# Note: that baseline was measured before the sentinel filter; expect a
# similar but not identical count here.
n_clean = row_count('instance_events_clean')
n_requests_clean = row_count('instance_events_requests_clean')
n_request_nulls_dropped = n_clean - n_requests_clean

print(f"After sentinel filter:        {n_clean:,}")
print(f"After request null drop:      {n_requests_clean:,}")
print(f"Request-null rows dropped:    {n_request_nulls_dropped:,}")

record_check(
    "Section 2: cpu_request/memory_request null drop count",
    expected="~47,933 (V04 EDA baseline)",
    observed=n_request_nulls_dropped,
    ok=(40_000 <= n_request_nulls_dropped <= 60_000),
    notes="Tolerance band 40K-60K; exact count drifts slightly after sentinel filter.",
)

# %% [markdown]
# ---
# ## 3. Failure label construction (V01, V08, V27, P04)
#
# - `failure_label` = 1 where `type IN (5, 8)`, 0 where `type = 6`,
#   NULL otherwise (V01, primary outcome).
# - `failure_label_sensitivity_prod_evict` adds Production-priority
#   EVICTs (`type = 4 AND priority BETWEEN 120 AND 359`) as positives
#   alongside FAIL_LOST (P04).
# - Monitoring-priority EVICTs (`type = 4 AND priority >= 360`) are
#   excluded from every failure label (V27). The per-instance recurrence
#   signature in `outputs/tables/monitoring_evict_profile.csv` (the
#   `repeats` section) shows the majority of monitoring-EVICT instances
#   cycling through tens to hundreds of evictions, consistent with
#   canary / health-check preemption by Borg rather than failure. The
#   labels below intentionally leave those rows NULL.
# - KILL (type 7) is excluded from every failure label (V08): user-
#   initiated cancellation, not predictable system behavior.

# %%
failure_label_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_events_labeled')}
CLUSTER BY collection_id, instance_index AS
SELECT
    *,
    -- Primary: FAIL + LOST = failure; FINISH = success; everything else NULL.
    CASE
        WHEN type IN ({FAIL_LOST_TYPES[0]}, {FAIL_LOST_TYPES[1]}) THEN 1
        WHEN type = {FINISH_TYPE} THEN 0
        ELSE NULL
    END AS failure_label,

    -- Sensitivity branch: also count Production-priority EVICTs as failures.
    -- Monitoring-priority EVICTs (>= 360) and KILLs stay NULL per V27 and V08.
    CASE
        WHEN type IN ({FAIL_LOST_TYPES[0]}, {FAIL_LOST_TYPES[1]}) THEN 1
        WHEN type = 4 AND priority BETWEEN {PROD_EVICT_PRIORITY_LOW} AND {MONITORING_PRIORITY_LOW - 1} THEN 1
        WHEN type = {FINISH_TYPE} THEN 0
        ELSE NULL
    END AS failure_label_sensitivity_prod_evict
FROM {fqn('instance_events_requests_clean')};
"""

run_ddl(failure_label_sql, "Add failure_label and failure_label_sensitivity_prod_evict")

# %%
# Verify class distributions for both labels
label_dist_sql = f"""
SELECT
    'primary' AS label,
    failure_label AS value,
    COUNT(*) AS n
FROM {fqn('instance_events_labeled')}
GROUP BY failure_label

UNION ALL

SELECT
    'sensitivity_prod_evict' AS label,
    failure_label_sensitivity_prod_evict AS value,
    COUNT(*) AS n
FROM {fqn('instance_events_labeled')}
GROUP BY failure_label_sensitivity_prod_evict

ORDER BY label, value
"""
label_dist_df = run_query(label_dist_sql)
print(label_dist_df)

# %%
# Pull primary counts for the class-balance assertion
primary_pos = int(label_dist_df.filter(
    (pl.col("label") == "primary") & (pl.col("value") == 1)
)["n"].item())
primary_neg = int(label_dist_df.filter(
    (pl.col("label") == "primary") & (pl.col("value") == 0)
)["n"].item())
primary_ratio = primary_neg / primary_pos if primary_pos else float("inf")

print(f"Primary FAIL_LOST positives: {primary_pos:,}")
print(f"Primary FINISH negatives:    {primary_neg:,}")
print(f"FINISH:FAIL_LOST ratio:      {primary_ratio:.2f}:1")

record_check(
    "Section 3: primary class balance",
    expected="~3.4:1 FINISH:FAIL_LOST (V02 baseline)",
    observed=f"{primary_ratio:.2f}:1",
    ok=(3.0 <= primary_ratio <= 4.0),
    notes="Tolerance band 3.0:1 to 4.0:1.",
)

# %%
# Verify monitoring-priority EVICTs are NULL under both labels
monitoring_exclusion_sql = f"""
SELECT
    COUNTIF(type = 4 AND priority >= {MONITORING_PRIORITY_LOW}) AS n_monitoring_evicts,
    COUNTIF(type = 4 AND priority >= {MONITORING_PRIORITY_LOW}
            AND failure_label IS NOT NULL) AS n_monitoring_labeled_primary,
    COUNTIF(type = 4 AND priority >= {MONITORING_PRIORITY_LOW}
            AND failure_label_sensitivity_prod_evict IS NOT NULL) AS n_monitoring_labeled_sens
FROM {fqn('instance_events_labeled')}
"""
mon_df = run_query(monitoring_exclusion_sql)
n_monitoring = int(mon_df["n_monitoring_evicts"].item())
n_mon_primary = int(mon_df["n_monitoring_labeled_primary"].item())
n_mon_sens = int(mon_df["n_monitoring_labeled_sens"].item())

print(f"Monitoring-priority EVICT rows:        {n_monitoring:,}")
print(f"Labeled under primary failure_label:   {n_mon_primary:,}")
print(f"Labeled under sensitivity_prod_evict:  {n_mon_sens:,}")

record_check(
    "Section 3: monitoring EVICT exclusion under primary label (V27)",
    expected=0,
    observed=n_mon_primary,
    ok=(n_mon_primary == 0),
    notes="Monitoring-priority EVICTs must remain NULL under failure_label.",
)
record_check(
    "Section 3: monitoring EVICT exclusion under sensitivity label (V27)",
    expected=0,
    observed=n_mon_sens,
    ok=(n_mon_sens == 0),
    notes="Monitoring-priority EVICTs must remain NULL under failure_label_sensitivity_prod_evict.",
)

# %% [markdown]
# ---
# ## 4. MNAR indicator encoding for CPI and MAPI (V11, V28)
#
# `cycles_per_instruction` (CPI) and `memory_accesses_per_instruction`
# (MAPI) live in `instance_usage_full`. V11 documented that the ~20.5%
# null rate is workload-type driven (MNAR), not platform driven:
# FINISH events are 87.2% null while FAIL_LOST events are 26.8% null.
# V28 then resolved that 39.84% of distinct instances have at least one
# observation with counters present and at least one observation
# without (mixed group). Both encodings are exposed:
#
# - **Per-observation indicator** (`has_cpi_value`, `has_mapi_value`):
#   added to a usage-level table so the original V11 record-level
#   verification still works (87.2% FINISH, 26.8% FAIL_LOST). Useful
#   for assertion / regression and for any feature that operates at the
#   observation level.
# - **Per-instance majority vote**
#   (`has_hardware_counters_majority`): added to a per-instance
#   summary table so feature engineering in Week 3 consumes one stable
#   indicator per (collection_id, instance_index). Per V28, threshold
#   is 50% of observations carrying counters.

# %%
# Per-observation indicators on the full usage table.
# We do NOT drop sample_memory here because it is not on instance_usage_full's
# preprocessed Parquet contract for this notebook; it is dropped when the
# usage table itself is preprocessed in a companion notebook.
usage_observation_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_usage_with_indicators')}
CLUSTER BY collection_id, instance_index AS
SELECT
    *,
    CAST(cycles_per_instruction IS NOT NULL AS INT64) AS has_cpi_value,
    CAST(memory_accesses_per_instruction IS NOT NULL AS INT64) AS has_mapi_value
FROM {fqn('instance_usage_full')};
"""

run_ddl(usage_observation_sql, "Add has_cpi_value / has_mapi_value to instance_usage")

# %%
# Per-instance majority vote (V28). Computed on the full table; result is
# ~74.5M rows and serves as a feature-engineering input.
majority_vote_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_hardware_counters_majority')}
CLUSTER BY collection_id, instance_index AS
WITH per_instance AS (
    SELECT
        collection_id,
        instance_index,
        COUNT(*) AS n_obs,
        COUNTIF(cycles_per_instruction IS NOT NULL) AS n_with_counters
    FROM {fqn('instance_usage_full')}
    GROUP BY collection_id, instance_index
)
SELECT
    collection_id,
    instance_index,
    n_obs,
    n_with_counters,
    SAFE_DIVIDE(n_with_counters, n_obs) AS frac_with_counters,
    CASE
        WHEN n_obs = 0 THEN NULL
        WHEN SAFE_DIVIDE(n_with_counters, n_obs) >= 0.5 THEN 1
        ELSE 0
    END AS has_hardware_counters_majority
FROM per_instance;
"""

run_ddl(majority_vote_sql, "Per-instance majority vote -> instance_hardware_counters_majority")

# %%
# Verify V11 record-level null rates against the EDA target (87.2% FINISH,
# 26.8% FAIL_LOST). Mirrors the V11 query in
# sql/exploration/cpi_mapi_missingness_structure.sql Part C exactly:
# - 5% TABLESAMPLE on instance_usage_full (cost control).
# - Triple-key join on (collection_id, instance_index, machine_id) so that
#   per-machine usage rows are paired with per-machine terminal events.
# - Terminal candidate set is type IN (4, 5, 6, 7, 8); the CASE then
#   labels FAIL_LOST and FINISH only (so EVICT/KILL-terminated instances
#   are not counted in the FAIL_LOST group, which would otherwise dilute
#   the null rate). Sources events from instance_events_full to match
#   V11's pre-preprocessing baseline.
cpi_verification_sql = f"""
WITH usage_sample AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        cycles_per_instruction
    FROM {fqn('instance_usage_full')} TABLESAMPLE SYSTEM (5 PERCENT)
    WHERE machine_id IS NOT NULL
),
terminal_events AS (
    SELECT
        collection_id,
        instance_index,
        machine_id,
        type AS terminal_type,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, machine_id
            ORDER BY time DESC
        ) AS rn
    FROM {fqn('instance_events_full')}
    WHERE type IN (4, 5, 6, 7, 8)
),
labeled AS (
    SELECT
        u.cycles_per_instruction,
        CASE
            WHEN te.terminal_type IN (5, 8) THEN 'FAIL_LOST'
            WHEN te.terminal_type = 6 THEN 'FINISH'
        END AS outcome
    FROM usage_sample u
    INNER JOIN terminal_events te
        ON u.collection_id = te.collection_id
        AND u.instance_index = te.instance_index
        AND u.machine_id = te.machine_id
        AND te.rn = 1
)
SELECT
    outcome,
    COUNT(*) AS n,
    COUNTIF(cycles_per_instruction IS NULL) AS n_null,
    ROUND(COUNTIF(cycles_per_instruction IS NULL) * 100.0 / COUNT(*), 2) AS pct_null
FROM labeled
WHERE outcome IS NOT NULL
GROUP BY outcome
ORDER BY outcome
"""
cpi_df = run_query(cpi_verification_sql)
print(cpi_df)

# %%
finish_null_pct = float(cpi_df.filter(pl.col("outcome") == "FINISH")["pct_null"].item())
fail_null_pct = float(cpi_df.filter(pl.col("outcome") == "FAIL_LOST")["pct_null"].item())

record_check(
    "Section 4: FINISH record-level CPI null rate (V11-exact query, target 87.2%)",
    expected="87.2 +/- 3.0",
    observed=f"{finish_null_pct:.2f}",
    ok=(84.2 <= finish_null_pct <= 90.2),
    notes="Triple-key join, terminals IN (4,5,6,7,8); 5% TABLESAMPLE noise expected.",
)
record_check(
    "Section 4: FAIL_LOST record-level CPI null rate (V11-exact query, target 26.8%)",
    expected="26.8 +/- 3.0",
    observed=f"{fail_null_pct:.2f}",
    ok=(23.8 <= fail_null_pct <= 29.8),
    notes="Triple-key join, terminals IN (4,5,6,7,8); 5% TABLESAMPLE noise expected.",
)

# %%
# Sanity check on the majority-vote table: the 'always_absent' fraction
# should match F4 (~59.97%) and the mixed fraction (~39.84%).
majority_dist_sql = f"""
SELECT
    COUNTIF(n_with_counters = 0) AS always_absent,
    COUNTIF(n_with_counters = n_obs AND n_obs > 0) AS always_present,
    COUNTIF(n_with_counters > 0 AND n_with_counters < n_obs) AS mixed,
    COUNT(*) AS total
FROM {fqn('instance_hardware_counters_majority')}
"""
maj_df = run_query(majority_dist_sql)
total = int(maj_df["total"].item())
mixed = int(maj_df["mixed"].item())
always_absent = int(maj_df["always_absent"].item())
pct_mixed = 100.0 * mixed / total if total else 0.0
pct_always_absent = 100.0 * always_absent / total if total else 0.0

record_check(
    "Section 4: per-instance mixed-pattern fraction (V28 target 39.84%)",
    expected="39.84 +/- 1.0",
    observed=f"{pct_mixed:.2f}",
    ok=(38.84 <= pct_mixed <= 40.84),
)
record_check(
    "Section 4: per-instance always_absent fraction (V28 target 59.97%)",
    expected="59.97 +/- 1.0",
    observed=f"{pct_always_absent:.2f}",
    ok=(58.97 <= pct_always_absent <= 60.97),
)

# %% [markdown]
# ---
# ## 5. Lifecycle reconstruction (V09, V10)
#
# Build a per-instance summary table joining SUBMIT, SCHEDULE, and
# terminal events. Pushed to BigQuery as a single window-function pass
# (never pulls the full 1.7B-row events table into Polars).
# The output schema must expose `submit_time` so
# `src/features/temporal.py` (V26) can derive PDT wall-clock
# components.
#
# Logic anchors on the existing exploration query
# `sql/exploration/instance_lifecycle_reconstruction.sql` (Part B) but
# scales it to the full trace window rather than the 3-day sample used
# there.
#
# **TODO:** Extract the SQL below into
# `src/preprocessing/lifecycle.py::reconstruct_instance_lifecycle()` and
# add a row-count assertion against the EDA-confirmed unique-instance
# count before returning.

# %%
lifecycle_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_lifecycle_summary')}
CLUSTER BY collection_id, instance_index AS
WITH per_instance AS (
    SELECT
        collection_id,
        instance_index,

        -- Counts per event type
        COUNT(*) AS total_events,
        COUNTIF(type = 0) AS submit_count,
        COUNTIF(type = 3) AS schedule_count,
        COUNTIF(type = 4) AS evict_count,
        COUNTIF(type IN ({FAIL_LOST_TYPES[0]}, {FAIL_LOST_TYPES[1]})) AS fail_lost_count,

        -- Lifecycle timestamps (microseconds since trace start; exposed for
        -- downstream temporal feature derivation per V26).
        MIN(IF(type = 0, time, NULL)) AS submit_time,
        MIN(IF(type = 3, time, NULL)) AS first_schedule_time,
        MAX(IF(type = 3, time, NULL)) AS last_schedule_time,
        MAX(time) AS terminal_time,

        -- Terminal event metadata (uses last event by time)
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type,
        ARRAY_AGG(priority ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_priority,
        ARRAY_AGG(scheduling_class ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_scheduling_class,
        ARRAY_AGG(machine_id ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_machine_id,

        -- Resource requests (use the first non-null value over the instance)
        ANY_VALUE(cpu_request) AS cpu_request,
        ANY_VALUE(memory_request) AS memory_request
    FROM {fqn('instance_events_labeled')}
    GROUP BY collection_id, instance_index
),
with_durations AS (
    SELECT
        *,
        -- Resubmission count: 1 SUBMIT means single pass; 2+ means resubmitted
        IF(submit_count > 1, submit_count - 1, 0) AS resubmission_count,

        -- Queue time: SUBMIT -> first SCHEDULE (seconds)
        SAFE_DIVIDE(first_schedule_time - submit_time, 1000000) AS queue_time_sec,

        -- Running duration: last SCHEDULE -> terminal (seconds). Sentinel
        -- filtering in Section 1 ensures the inputs are valid timestamps.
        SAFE_DIVIDE(terminal_time - last_schedule_time, 1000000) AS running_duration_sec,

        -- Total lifecycle (submit -> terminal)
        SAFE_DIVIDE(terminal_time - submit_time, 1000000) AS total_lifecycle_sec,

        -- Outcome category
        CASE
            WHEN terminal_type IN ({FAIL_LOST_TYPES[0]}, {FAIL_LOST_TYPES[1]}) THEN 'FAIL_LOST'
            WHEN terminal_type = {FINISH_TYPE} THEN 'FINISH'
            WHEN terminal_type = 4 AND terminal_priority >= {MONITORING_PRIORITY_LOW} THEN 'EVICT_MONITORING'
            WHEN terminal_type = 4 AND terminal_priority BETWEEN {PROD_EVICT_PRIORITY_LOW}
                                                              AND {MONITORING_PRIORITY_LOW - 1} THEN 'EVICT_PRODUCTION'
            WHEN terminal_type = 4 THEN 'EVICT_LOWER'
            WHEN terminal_type = 7 THEN 'KILL'
            ELSE 'OTHER'
        END AS outcome
    FROM per_instance
)
SELECT *
FROM with_durations
"""

run_ddl(lifecycle_sql, "Reconstruct per-instance lifecycle -> instance_lifecycle_summary")

# %%
# Verify resubmission rate for FAIL_LOST instances against V10's 99.04% target.
# V10 was measured on a 3-day window (days 10-12) per
# sql/exploration/instance_lifecycle_reconstruction.sql Part B, with the
# additional `queue_time_sec >= 0 AND running_duration_sec >= 0` filter that
# excludes instances which failed before being scheduled. This reproduction
# replicates both scoping decisions, then captures the full-trace rate as a
# separate informational row so both numbers are on record.
resub_v10_window_sql = f"""
WITH bounds AS (
    SELECT
        CAST((9 * 86400 + 600) AS INT64) * 1000000 AS sample_start,
        CAST((12 * 86400 + 600) AS INT64) * 1000000 AS sample_end
),
sampled AS (
    SELECT DISTINCT collection_id, instance_index
    FROM {fqn('instance_events_full')}, bounds
    WHERE time BETWEEN sample_start AND sample_end
),
lifecycle AS (
    SELECT
        ie.collection_id,
        ie.instance_index,
        COUNTIF(ie.type = 0) AS submit_count,
        MIN(IF(ie.type = 0, ie.time, NULL)) AS first_submit_time,
        MIN(IF(ie.type = 3, ie.time, NULL)) AS first_schedule_time,
        MAX(IF(ie.type = 3, ie.time, NULL)) AS last_schedule_time,
        MAX(ie.time) AS terminal_time,
        ARRAY_AGG(ie.type ORDER BY ie.time DESC LIMIT 1)[OFFSET(0)] AS terminal_type
    FROM {fqn('instance_events_full')} ie
    INNER JOIN sampled s
        ON ie.collection_id = s.collection_id
        AND ie.instance_index = s.instance_index
    GROUP BY ie.collection_id, ie.instance_index
),
with_durations AS (
    SELECT
        *,
        SAFE_DIVIDE(first_schedule_time - first_submit_time, 1000000) AS queue_time_sec,
        SAFE_DIVIDE(terminal_time - last_schedule_time, 1000000) AS running_duration_sec
    FROM lifecycle
    WHERE terminal_type IN (4, 5, 6, 7, 8)
)
SELECT
    CASE WHEN terminal_type IN (5, 8) THEN 'FAIL_LOST'
         WHEN terminal_type = 6 THEN 'FINISH'
    END AS outcome,
    COUNT(*) AS n_instances,
    COUNTIF(submit_count > 1) AS n_resubmitted,
    ROUND(COUNTIF(submit_count > 1) * 100.0 / COUNT(*), 2) AS pct_resubmitted
FROM with_durations
WHERE terminal_type IN (5, 6, 8)
  AND queue_time_sec >= 0
  AND running_duration_sec >= 0
GROUP BY outcome
ORDER BY outcome
"""
resub_v10_df = run_query(resub_v10_window_sql)
print(resub_v10_df)

# %%
fail_resub_v10 = float(
    resub_v10_df.filter(pl.col("outcome") == "FAIL_LOST")["pct_resubmitted"].item()
)
finish_resub_v10 = float(
    resub_v10_df.filter(pl.col("outcome") == "FINISH")["pct_resubmitted"].item()
)

record_check(
    "Section 5: FAIL_LOST resubmission rate, 3-day window + schedule filter (V10-exact, target 99.04%)",
    expected="99.04 +/- 1.5",
    observed=f"{fail_resub_v10:.2f}",
    ok=(97.54 <= fail_resub_v10 <= 100.5),
    notes="Days 10-12 sample with queue_time_sec >= 0 AND running_duration_sec >= 0 (Part B filter).",
)
record_check(
    "Section 5: FINISH resubmission rate, 3-day window (V10 baseline for ratio)",
    expected="lower than FAIL_LOST",
    observed=f"{finish_resub_v10:.2f}",
    ok=(finish_resub_v10 < fail_resub_v10),
    notes="FAIL_LOST resubmits more than FINISH; the 72x gap drives V10.",
)

# %%
# Informational: full-trace resubmission rate from the lifecycle summary.
# This is consistently lower than V10's 3-day rate because the full trace
# includes single-pass crashes and trace-boundary instances that never had
# the opportunity to resubmit.
resubmission_full_sql = f"""
SELECT
    outcome,
    COUNT(*) AS n_instances,
    COUNTIF(resubmission_count > 0) AS n_resubmitted,
    ROUND(COUNTIF(resubmission_count > 0) * 100.0 / COUNT(*), 2) AS pct_resubmitted
FROM {fqn('instance_lifecycle_summary')}
WHERE outcome IN ('FAIL_LOST', 'FINISH')
GROUP BY outcome
ORDER BY outcome
"""
resub_full_df = run_query(resubmission_full_sql)
print(resub_full_df)
fail_resub_full = float(
    resub_full_df.filter(pl.col("outcome") == "FAIL_LOST")["pct_resubmitted"].item()
)
record_check(
    "Section 5: FAIL_LOST resubmission rate, full trace (informational)",
    expected="<= V10's 3-day rate (selection bias on active instances)",
    observed=f"{fail_resub_full:.2f}",
    ok=True,
    notes="Full-trace rate; lower than V10 because the 3-day window selects active mid-trace instances.",
)

# %%
# Verify the per-instance row count is within the EDA-confirmed range.
# The Google trace's unique-instance count was ~74.5M per F4; lifecycle
# summary should be in that ballpark.
lifecycle_n = row_count('instance_lifecycle_summary')
record_check(
    "Section 5: lifecycle summary row count (F4 baseline ~74.5M)",
    expected="60_000_000 - 100_000_000",
    observed=f"{lifecycle_n:,}",
    ok=(60_000_000 <= lifecycle_n <= 100_000_000),
    notes="Unique (collection_id, instance_index) pairs across the trace.",
)

# %% [markdown]
# ### 5.4 Decisions-log refinement: append V29 (V10 reproduction notes)
#
# Both V10 reproductions from this section are appended to the decisions
# log as V29. The row documents that the original 99.04% rate requires
# the 3-day window plus the Part B `queue_time_sec >= 0 AND
# running_duration_sec >= 0` schedule filter, and captures the
# full-trace rate (~76%) as the long-run population statistic. Mirrors
# the `append_decision` helper in notebook 07b so this cell stays
# self-contained.

# %%
DECISIONS_CSV = OUTPUT_DIR / 'tables' / 'eda_decisions.csv'
DECISIONS_SCHEMA = [
    "id", "category", "dataset", "item", "evidence_or_rationale",
    "source", "applies_to_rq", "status", "next_step",
]


def append_decision(
    decision_id: str,
    category: str,
    dataset: str,
    item: str,
    evidence: str,
    source: str,
    applies_to_rq: str,
    status: str,
    next_step: str,
) -> None:
    """Idempotently append (or overwrite by id) a row to eda_decisions.csv."""
    new_row = pl.DataFrame(
        [(decision_id, category, dataset, item, evidence, source,
          applies_to_rq, status, next_step)],
        schema=DECISIONS_SCHEMA,
        orient="row",
    )
    if DECISIONS_CSV.exists():
        existing = pl.read_csv(str(DECISIONS_CSV))
        existing = existing.filter(pl.col("id") != decision_id)
        combined = pl.concat([existing, new_row], how="vertical_relaxed")
    else:
        combined = new_row
    combined.write_csv(str(DECISIONS_CSV))
    print(f"  Decisions log updated with {decision_id} (total rows: {combined.height})")


append_decision(
    decision_id="V29",
    category="Failure Predictor Refinement",
    dataset="Google",
    item=(
        "Refines V10. Reproducing the 99.04% FAIL_LOST resubmission rate "
        "requires both the 3-day window (days 10-12) and the Part B "
        "queue_time_sec >= 0 AND running_duration_sec >= 0 filter, which "
        "excludes instances that failed before being scheduled (no SCHEDULE "
        "event, so queue_time_sec is NULL). Without that filter the windowed "
        "rate lands near 92%. Across the full 31-day trace the FAIL_LOST "
        "resubmission rate is roughly 76%, because the population then "
        "includes single-pass crashes and trace-boundary instances that "
        "never had the opportunity to resubmit."
    ),
    evidence=(
        f"Notebook 08 Section 5: V10-exact reproduction "
        f"{fail_resub_v10:.2f}% with the schedule filter; full-trace "
        f"lifecycle summary {fail_resub_full:.2f}%. Both rates support "
        "the V10 narrative (resubmission history dominates the failure "
        "population); the gap between them quantifies the selection bias "
        "of the original 3-day sample."
    ),
    source=(
        "notebooks/08_google_preprocessing.py Section 5; "
        "outputs/tables/preprocessing_verification.csv"
    ),
    applies_to_rq="RQ1",
    status="Validated (Phase 3 preprocessing)",
    next_step=(
        "Cite both reproduction modes in working_draft.docx wherever V10 "
        "is referenced: schedule-filtered 3-day rate for the exact V10 "
        "number, full-trace rate for the long-run population statistic."
    ),
)

# %% [markdown]
# ---
# ## 6. Post-preprocessing assertion suite
#
# This section consolidates every assertion already executed in
# Sections 1 through 5, plus a few cross-cutting checks, and writes the
# verification table to disk. Any FAIL row blocks the Section 7 export.
#
# **TODO:** The checks below will move into
# `src/data/validation.py::run_preprocessing_assertions()` once the
# logic stabilizes.

# %%
# Cross-cutting check 1: column count survived end-to-end
schema_df = run_query(f"""
SELECT column_name
FROM `{PROJECT_ID}.dissertation_lebel.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'instance_events_labeled'
ORDER BY ordinal_position
""")
n_cols = schema_df.height
record_check(
    "Section 6: instance_events_labeled column count",
    expected="13 +/- 1 (12 original + 2 labels - 0 dropped)",
    observed=n_cols,
    ok=(13 <= n_cols <= 15),
    notes="Original 12 columns plus failure_label and failure_label_sensitivity_prod_evict.",
)

# %%
# Cross-cutting check 2: no surprise nulls in the failure_label for
# non-terminal event types should produce zero rows of FAIL_LOST or
# FINISH-typed events with a NULL label.
label_consistency_sql = f"""
SELECT
    COUNTIF(type IN ({FAIL_LOST_TYPES[0]}, {FAIL_LOST_TYPES[1]})
            AND failure_label IS NULL) AS n_fail_lost_nulls,
    COUNTIF(type = {FINISH_TYPE} AND failure_label IS NULL) AS n_finish_nulls
FROM {fqn('instance_events_labeled')}
"""
consistency_df = run_query(label_consistency_sql)
n_fail_lost_nulls = int(consistency_df["n_fail_lost_nulls"].item())
n_finish_nulls = int(consistency_df["n_finish_nulls"].item())
record_check(
    "Section 6: FAIL_LOST event rows always carry failure_label = 1",
    expected=0,
    observed=n_fail_lost_nulls,
    ok=(n_fail_lost_nulls == 0),
)
record_check(
    "Section 6: FINISH event rows always carry failure_label = 0",
    expected=0,
    observed=n_finish_nulls,
    ok=(n_finish_nulls == 0),
)

# %%
verification_df = pl.DataFrame(verification_rows)
verification_df.write_csv(str(VERIFICATION_CSV))
print(f"Verification log: {VERIFICATION_CSV}")
print(verification_df.select(["check", "ok"]))

n_failed = verification_df.filter(~pl.col("ok")).height
if n_failed:
    print(f"\n{n_failed} assertion(s) failed. Section 7 export is blocked.")
else:
    print("\nAll assertions passed. Proceeding to export.")

assert n_failed == 0, "Resolve failed assertions before exporting."

# %% [markdown]
# ---
# ## 7. Export to GCS Parquet plus Drive manifest
#
# The full preprocessed events table is ~1.7B rows; Drive cannot host
# the corresponding Parquet at this scale. Export to GCS as a
# date-partitioned Parquet dataset (Polars can `scan_parquet` over the
# prefix lazily) and write a small manifest to Drive describing the
# export.

# %%
export_uri = f"gs://{GCS_BUCKET}/{GCS_EVENTS_PARQUET_PREFIX}/*.parquet"

export_sql = f"""
EXPORT DATA OPTIONS(
    uri='{export_uri}',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT *
FROM {fqn('instance_events_labeled')}
"""

run_ddl(export_sql, f"Export instance_events_labeled -> {export_uri}")

# %%
# Enumerate the produced Parquet files for the manifest. We list via GCS
# rather than via the BQ job result because BQ's EXPORT DATA does not
# return file names directly.
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)
blob_iter = gcs_bucket.list_blobs(prefix=f'{GCS_EVENTS_PARQUET_PREFIX}/')
blobs = [b for b in blob_iter if b.name.endswith('.parquet')]

print(f"Produced {len(blobs)} Parquet files under gs://{GCS_BUCKET}/{GCS_EVENTS_PARQUET_PREFIX}/")
for b in blobs[:5]:
    print(f"  {b.name}  ({b.size:,} bytes)")
if len(blobs) > 5:
    print(f"  ... {len(blobs) - 5} more")

# %%
# Per-file metadata (size, md5_hash from GCS; SHA256 omitted to avoid
# re-downloading each shard). GCS md5_hash is base64-encoded per the
# storage API; we keep it as-is for traceability.
file_records = [
    {
        "path": f"gs://{GCS_BUCKET}/{b.name}",
        "size_bytes": int(b.size),
        "md5_hash": b.md5_hash,
        "updated": b.updated.isoformat() if b.updated else None,
    }
    for b in blobs
]
total_bytes = sum(r["size_bytes"] for r in file_records)
total_rows_exported = row_count('instance_events_labeled')

manifest = {
    "dataset": "google_cluster_traces",
    "stage": "preprocessed",
    "table_name": "instance_events_labeled",
    "bigquery_table": f"{PROJECT_ID}.{DATASET}.instance_events_labeled",
    "gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_EVENTS_PARQUET_PREFIX}/",
    "row_count": total_rows_exported,
    "column_count": n_cols,
    "n_files": len(file_records),
    "total_bytes": total_bytes,
    "files": file_records,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "decisions_applied": [
        "V01 primary failure label (FAIL+LOST positive, FINISH negative)",
        "V03 machine_id structural nulls preserved",
        "V04 cpu_request/memory_request null rows dropped",
        "V08 KILL excluded from labels",
        "V25 sentinel rows (time=0 and time=2^63-1) dropped",
        "V27 monitoring-priority EVICTs excluded from every failure label",
        "P04 prod-priority EVICTs labeled positive under failure_label_sensitivity_prod_evict",
        "V11 + V28 per-observation has_cpi_value indicator + per-instance majority vote",
        "V09 + V10 lifecycle summary table (resubmission count, queue time, running duration)",
    ],
    "downstream_inputs_required": [
        "instance_lifecycle_summary (for Tier 1 features in Week 3)",
        "instance_usage_with_indicators (for Tier 2 features in Week 3)",
        "instance_hardware_counters_majority (for has_hardware_counters in Week 3)",
    ],
}

# %%
import json

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest: {MANIFEST_PATH}")
print(f"Row count exported: {total_rows_exported:,}")
print(f"Total bytes:        {total_bytes:,}")
print(f"File count:         {len(file_records)}")

# %% [markdown]
# ### End-of-notebook smoke test
#
# Lazily scan a small slice of the exported Parquet via Polars to
# confirm the schema and label distribution are as expected.

# %%
sample_uri = f"gs://{GCS_BUCKET}/{GCS_EVENTS_PARQUET_PREFIX}/"
sample_lf = pl.scan_parquet(sample_uri, n_rows=1_000_000)
sample_summary = (
    sample_lf
    .group_by("failure_label")
    .agg(pl.len().alias("n"))
    .sort("failure_label")
    .collect()
)
print("Label distribution in the first 1M scanned rows:")
print(sample_summary)
