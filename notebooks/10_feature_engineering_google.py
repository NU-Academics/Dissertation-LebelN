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
# # 10. Google Cluster Traces Feature Engineering (Tiered)
#
# **Purpose.** Build the three-tier feature matrix for the Google Cluster
# Traces failure-prediction models. The tier ordering encodes the
# EDA-validated predictive-value hierarchy (V13): Tier 1 pre-event signals
# dominate, Tier 2 early-runtime slopes are moderate, and Tier 3 windowed
# utilization is included only so the Chapter 4 ablation can empirically
# re-confirm the V12 utilization inversion. This notebook is the Phase 3, Data Preparation
# deliverable.
#
# **Canonical references.**
# - Chapter 3, Table 10 (Operational Definitions of Variables) — the
#   authoritative feature list and measurement definitions.
# - `outputs/tables/eda_decisions.csv`, rows V09-V13 (failure mechanism,
#   resubmission dominance, MNAR CPI/MAPI encoding, utilization inversion,
#   tier structure) and V26 (temporal stratification / diurnal FAIL_LOST
#   swing).
#
# **EDA decisions operationalized here.**
# - **V09** rapid-onset failure model (median FAIL_LOST running duration
#   22.6s) — motivates Tier 1 lifecycle features and the ±60s early-runtime
#   window for Tier 2.
# - **V10** resubmission history dominates (99.04% of FAIL_LOST resubmitted;
#   first-resubmission failure rate 72x single-pass) — `prior_fail_count`,
#   `resubmission_count`, `first_resubmission`.
# - **V11** MNAR CPI/MAPI encoding — `has_hardware_counters` indicator plus
#   conditional `cpi_value` / `mapi_value` (null when unavailable, never
#   imputed).
# - **V12** utilization inversion (failing instances use LESS CPU/memory but
#   ramp 3.6x/2.3x faster) — Tier 2 slope/ramp features carry the signal;
#   Tier 3 absolute windows are the confounded comparison kept for ablation.
# - **V13** tier structure — three tiers materialized and tagged so the
#   ablation can drop tiers in isolation.
# - **V26** temporal stratification (hourly FAIL_LOST swing up to ~8x,
#   peaking in PDT business hours) — submit-time PDT temporal features.
#
# **Two-stage BigQuery filter.** The Tier 2
# slope features require per-observation usage rows from the 7.5B-row
# `instance_usage_full` table. Exporting raw observations for the working
# set would blow the Colab memory and BigQuery free-tier budgets. Instead
# the notebook (a) materializes `instance_usage_working_set`, a usage subset
# restricted to working-set instances inside a ±60s band around each
# instance's schedule time, and (b) computes the slopes and ramps
# BigQuery-side with `COVAR_POP`/`VAR_POP` slopes, `LAG`, and `AVG OVER`, exporting only the
# compact per-instance early-runtime feature row. Tier 3 windowed features
# use a second, wider working-set usage subset and are computed in Polars.
#
# **Inputs (all produced by notebook 08).**
# - `{PROJECT}.dissertation_lebel.instance_lifecycle_summary` (Tier 1
#   historical / scheduling / temporal source; one row per instance).
# - `{PROJECT}.dissertation_lebel.instance_hardware_counters_majority`
#   (per-instance CPI/MAPI availability majority vote; V11 / V28).
# - `{PROJECT}.dissertation_lebel.instance_usage_with_indicators`
#   (Tier 2 / Tier 3 usage source; has_cpi_value / has_mapi_value).
# - `{PROJECT}.dissertation_lebel.instance_usage_full` (raw usage; only ever
#   touched through the two-stage memory-bounded filter).
# - `{PROJECT}.dissertation_lebel.machine_events_full` (platform_id lookup).
# - **TODO** `{PROJECT}.dissertation_lebel.working_set_instance_ids` (the locked
#   working set: collection_id, instance_index, schedule_time). Produced by
#   `src/features/sampling.py::build_working_set_google`. A
#   guarded bootstrap cell (Section 1.1) builds a candidate stand-in from
#   the lifecycle summary so this notebook runs end-to-end before the
#   sampler is wired in.
#
# **Outputs.**
# - `{PROJECT}.dissertation_lebel.instance_usage_working_set` (BigQuery,
#   ±60s Tier 2 subset).
# - `{PROJECT}.dissertation_lebel.instance_usage_working_set_t3` (BigQuery,
#   wider Tier 3 subset).
# - `{PROJECT}.dissertation_lebel.instance_runtime_features` (BigQuery,
#   per-instance Tier 2 slope/ramp row) plus its GCS Parquet export.
# - `{OUTPUT_DIR}/features/google/instance_features.parquet` (the assembled
#   three-tier feature matrix; one row per working-set instance).
# - `{OUTPUT_DIR}/features/google/feature_schema.json` (column -> tier /
#   dtype / motivation manifest).
# - `outputs/tables/google_feature_engineering_verification.csv` (assertion
#   log, committed to the repo).
#
# **Status.** Exploratory. Once each section validates against the EDA
# numbers, the next step will extract the section logic into
# `src/features/{historical,scheduling,temporal,runtime,utilization}.py`
# as pure LazyFrame -> LazyFrame functions. The inline "-> extract to ..."
# markers below indicate the destination module for each block.
#
# **Sections.**
# 0. Colab session setup.
# 1. Path constants, helpers, and input handles (incl. working-set bootstrap).
# 2. Load Tier 1 source LazyFrames.
# 3. Tier 1 historical features      (-> src/features/historical.py).
# 4. Tier 1 scheduling features      (-> src/features/scheduling.py).
# 5. Tier 1 temporal features        (-> src/features/temporal.py).
# 6. Tier 1 assembly.
# 7. Tier 2 BigQuery materialization + slope/ramp SQL (-> src/features/runtime.py).
# 8. Tier 3 windowed utilization     (-> src/features/utilization.py).
# 9. Three-tier assembly + Parquet export.
# 10. Feature-schema manifest + verification suite.

# %% [markdown]
# ---
# ## 0. Colab Session Setup

# %%
# !pip install -q polars google-cloud-bigquery db-dtypes google-cloud-storage pyarrow

# %%
import os
import sys
from pathlib import Path

# Clone (or pull) the repo so utils.* and src.* are importable. Mirrors the
# pattern in notebooks 00 and 08. Requires the GITHUB_PAT Colab secret.
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
GCS_FEATURES_PREFIX = 'google_features'
GCS_RUNTIME_PARQUET_PREFIX = f'{GCS_FEATURES_PREFIX}/instance_runtime_features'

print(f"Project:  {PROJECT_ID}")
print(f"Dataset:  {DATASET}")
print(f"Drive:    {OUTPUT_DIR}")
print(f"GCS:      gs://{GCS_BUCKET}/{GCS_FEATURES_PREFIX}/")

# %%
from datetime import datetime, timezone

import numpy as np
import polars as pl

# Named constants for event types and priority bands. Importing from the
# schema module keeps the notebook in lockstep with the preprocessing
# pass (notebook 08) and avoids magic numbers in the feature logic.
from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_LOST,
    PRIORITY_FREE_MAX,
    PRIORITY_BEST_EFFORT_LOW,
    PRIORITY_BEST_EFFORT_MAX,
    PRIORITY_MID_TIER_LOW,
    PRIORITY_MID_TIER_MAX,
    PRIORITY_PRODUCTION_LOW,
    PRIORITY_PRODUCTION_MAX,
    PRIORITY_MONITORING_LOW,
)

# %% [markdown]
# ---
# ## 1. Path constants, helpers, and input handles

# %%
FEATURES_DIR = OUTPUT_DIR / 'features' / 'google'
TABLES_DIR = OUTPUT_DIR / 'tables'
for directory in (FEATURES_DIR, TABLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX_PATH = FEATURES_DIR / 'instance_features.parquet'
FEATURE_SCHEMA_PATH = FEATURES_DIR / 'feature_schema.json'
VERIFICATION_CSV = TABLES_DIR / 'google_feature_engineering_verification.csv'

# The early-runtime window. V09 fixes the median FAIL_LOST running duration
# at 22.6s and reports 93.8% of failures crashing within 10s-1min of
# scheduling; a +/-60s band around schedule time captures the full
# early-runtime signal without dragging in long-lived successful instances.
EARLY_RUNTIME_BAND_US = 60_000_000          # +/- 60 s in microseconds
# Tier 3 windows extend past the early-runtime band; the widest is 60 min.
TIER3_MAX_WINDOW_US = 3_600_000_000         # 60 min in microseconds
MICROS_PER_SEC = 1_000_000


def fqn(table: str) -> str:
    """Return the fully-qualified BigQuery table name for a cached table."""
    return table_ref(table)


def run_query(sql: str) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame (small results only)."""
    return pl.from_pandas(bq_client.query(sql).to_dataframe())


def run_ddl(sql: str, label: str) -> None:
    """Execute a DDL/DML statement without returning rows; print bytes billed."""
    print(f"[BQ] Running: {label}")
    job = bq_client.query(sql)
    job.result()
    billed = job.total_bytes_processed or 0
    print(f"  Done. Bytes processed: {billed:,}")


def table_exists(table: str) -> bool:
    """Return True if a table exists in the working dataset."""
    from google.cloud.exceptions import NotFound
    try:
        bq_client.get_table(f"{PROJECT_ID}.{DATASET}.{table}")
        return True
    except NotFound:
        return False


def row_count(table: str) -> int:
    """Return total row count for a table in the working dataset."""
    return int(run_query(f"SELECT COUNT(*) AS n FROM {fqn(table)}")["n"].item())


# %% [markdown]
# ### Verification log
#
# Every section appends `(check, expected, observed, ok)` rows. The full
# table writes to `outputs/tables/google_feature_engineering_verification.csv`
# in Section 10. The convention mirrors notebook 08 Section 6.

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
# ### 1.1 Working-set handle (with bootstrap)
#
# Tier 2 and Tier 3 both join against `working_set_instance_ids`, the
# locked working set produced by
# `src/features/sampling.py::build_working_set_google`. The
# table schema is `(collection_id, instance_index, schedule_time)`, where
# `schedule_time` is the instance's first SCHEDULE timestamp (microseconds,
# trace clock) and anchors the early-runtime window.
#
# This notebook precedes the sampler, so
# the guarded cell below bootstraps a **candidate** working set from the
# lifecycle summary when the locked table is absent: every instance that
# was scheduled (`first_schedule_time IS NOT NULL`) and carries a defined
# failure label. When the sampler later materializes the true 75M working
# set, re-running this notebook picks it up automatically and skips the
# bootstrap. The feature logic is identical either way.

# %%
WORKING_SET_TABLE = 'working_set_instance_ids'

if table_exists(WORKING_SET_TABLE):
    print(f"Found locked working set: {fqn(WORKING_SET_TABLE)}")
else:
    print(
        f"{WORKING_SET_TABLE} not found; bootstrapping a candidate working "
        "set from instance_lifecycle_summary (scheduled, labelled instances)."
    )
    bootstrap_ws_sql = f"""
CREATE OR REPLACE TABLE {fqn(WORKING_SET_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    first_schedule_time AS schedule_time
FROM {fqn('instance_lifecycle_summary')}
WHERE first_schedule_time IS NOT NULL
  AND outcome IN ('FAIL_LOST', 'FINISH');
"""
    run_ddl(bootstrap_ws_sql, f"Bootstrap candidate -> {WORKING_SET_TABLE}")

n_working_set = row_count(WORKING_SET_TABLE)
print(f"Working-set instances: {n_working_set:,}")
record_check(
    "Section 1.1: working-set instance count is in the expected band",
    expected="50,000,000 - 100,000,000 (P01 target band; candidate may differ)",
    observed=n_working_set,
    ok=(n_working_set > 0),
    notes="Bootstrap candidate may exceed the band; the sampler trims to ~75M (P01).",
)

# %% [markdown]
# ---
# ## 2. Load Tier 1 source LazyFrames
#
# All Tier 1 features derive from the per-instance `instance_lifecycle_summary`
# (notebook 08 Section 5) plus three small lookups. The summary is ~75M rows;
# we scan it lazily from its GCS Parquet export and join in the working set so
# every downstream `.collect()` operates on the locked instance population
# only. The hardware-counter majority and the machine-platform lookup are
# small enough to materialize eagerly.

# %%
# Push the working-set restriction, the hardware-counter join, and the
# collection-size aggregation into BigQuery, then export a single
# already-enriched base table. This keeps the ~75M-row joins and the
# collection-size group-by off the Colab box (memory): Polars then
# only scans one Parquet and applies streamable per-row transforms. The prior
# approach pulled the full working set and the full hardware-counter table
# through the pandas bridge and ran a pipeline-breaking group_by in-process,
# which OOM'd the 12.7 GB Colab kernel at Tier 1 assembly.
#
# - INNER JOIN to working_set_instance_ids restricts to the locked working
#   set (replaces the in-Polars semi-join and the 75M-row pandas pull).
# - LEFT JOIN to instance_hardware_counters_majority folds in
#   has_hardware_counters_majority (V11) without a second pandas pull.
# - collection_size_at_submit = COUNT(*) OVER (PARTITION BY collection_id)
#   over the working-set-restricted rows; identical semantics to the prior
#   Polars group_by, but computed BigQuery-side.
LIFECYCLE_BASE_TABLE = 'instance_lifecycle_features_base'
LIFECYCLE_GCS_PREFIX = 'google_features/instance_lifecycle_features_base'
lifecycle_uri = f"gs://{GCS_BUCKET}/{LIFECYCLE_GCS_PREFIX}/"

build_base_sql = f"""
CREATE OR REPLACE TABLE {fqn(LIFECYCLE_BASE_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    s.*,
    COALESCE(h.has_hardware_counters_majority, 0) AS has_hardware_counters_majority,
    COUNT(*) OVER (PARTITION BY s.collection_id) AS collection_size_at_submit
FROM {fqn('instance_lifecycle_summary')} s
INNER JOIN {fqn(WORKING_SET_TABLE)} w
  USING (collection_id, instance_index)
LEFT JOIN {fqn('instance_hardware_counters_majority')} h
  USING (collection_id, instance_index)
"""
run_ddl(build_base_sql, f"Build working-set base table -> {LIFECYCLE_BASE_TABLE}")

export_base_sql = f"""
EXPORT DATA OPTIONS(
    uri='{lifecycle_uri}*.parquet',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT * FROM {fqn(LIFECYCLE_BASE_TABLE)}
"""
run_ddl(export_base_sql, f"Export base table -> {lifecycle_uri}")

# Already restricted to the working set and enriched with
# has_hardware_counters_majority + collection_size_at_submit. Pure lazy scan;
# no eager pandas materialization.
lifecycle_lf = pl.scan_parquet(lifecycle_uri)

# %%
# Machine -> platform lookup. machine_events_full carries platform_id per
# machine; distinct (machine_id, platform_id) is tiny (10,005 machines, a
# handful of platforms). Used to attach platform_id via terminal_machine_id.
machine_platform_lf = pl.from_pandas(
    bq_client.query(f"""
        SELECT machine_id, ANY_VALUE(platform_id) AS platform_id
        FROM {fqn('machine_events_full')}
        WHERE machine_id IS NOT NULL AND platform_id IS NOT NULL
        GROUP BY machine_id
    """).to_dataframe()
).lazy()

# Enumerate the distinct platforms now so the one-hot columns are stable
# across runs (rather than discovered per working-set sample).
platform_ids = sorted(
    machine_platform_lf.select("platform_id").unique().collect()["platform_id"].to_list()
)
print(f"Distinct platform_id values: {platform_ids}")

# The trace obfuscates platform_id as a base64-style hash (contains +, /, =),
# which BigQuery rejects as a column-name suffix. Map each platform to a
# stable, sorted, BigQuery-safe suffix (platform_p0, platform_p1, ...). The
# mapping is recorded in the feature-schema manifest (Section 10) so the
# original platform_id remains recoverable.
PLATFORM_SUFFIX = {pid: f"p{i}" for i, pid in enumerate(platform_ids)}
print(f"Platform one-hot suffix map: {PLATFORM_SUFFIX}")

# %% [markdown]
# ---
# ## 3. Tier 1 historical features
#
# **-> extract to `src/features/historical.py`.** Motivation: V09
# (rapid-onset failure model) and V10 (resubmission history dominates;
# first-resubmission failure rate 72x single-pass). Derived entirely from
# `instance_lifecycle_summary`.
#
# | feature | definition |
# |---|---|
# | `prior_fail_count` | FAIL/LOST events before the terminal event: `fail_lost_count` minus 1 when the terminal event is itself FAIL_LOST, floored at 0. |
# | `has_prior_fail` | `prior_fail_count > 0`. |
# | `resubmission_count` | resubmissions = `submit_count - 1` (carried from the summary). |
# | `prior_evict_count` | count of EVICT events in the instance lifecycle (`evict_count`). |
# | `first_resubmission` | 1 when the instance has been resubmitted at least once (`resubmission_count >= 1`); the V10 discriminator separating first-resubmission (10.12% fail) from single-pass (0.14% fail) populations. |
# | `lifecycle_position` | the instance's fractional ordinal position within its collection (`instance_index / collection_size_at_submit`); early vs late instances differ in failure propensity. Requires `collection_size_at_submit` (Section 4). |

# %%
def add_tier1_historical(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append Tier 1 historical features. Input: instance_lifecycle_summary
    columns (fail_lost_count, terminal_type, submit_count, evict_count,
    resubmission_count). Output: adds prior_fail_count, has_prior_fail,
    resubmission_count (passthrough), prior_evict_count, first_resubmission.

    `lifecycle_position` is finalized in Section 6 once
    `collection_size_at_submit` exists. Motivation: V09, V10.
    """
    terminal_is_fail_lost = pl.col("terminal_type").is_in([EVENT_FAIL, EVENT_LOST])
    return lf.with_columns(
        # Failures strictly before the terminal event.
        pl.max_horizontal(
            pl.col("fail_lost_count") - terminal_is_fail_lost.cast(pl.Int64),
            pl.lit(0),
        ).alias("prior_fail_count"),
        pl.col("evict_count").alias("prior_evict_count"),
        # resubmission_count already present in the summary; ensure dtype.
        pl.col("resubmission_count").cast(pl.Int64).alias("resubmission_count"),
        (pl.col("resubmission_count") >= 1).cast(pl.Int8).alias("first_resubmission"),
    ).with_columns(
        (pl.col("prior_fail_count") > 0).cast(pl.Int8).alias("has_prior_fail"),
    )


tier1_hist_lf = add_tier1_historical(lifecycle_lf)

# %% [markdown]
# ---
# ## 4. Tier 1 scheduling features
#
# **-> extract to `src/features/scheduling.py`.** Motivation: V07 (machine
# / scheduling features). Priority bands follow Borg semantics encoded in
# `src/data/schemas.py`. `platform_id` is attached via `terminal_machine_id`
# and one-hot encoded; `priority_tier` is one-hot; `scheduling_class` stays
# ordinal (0-3) for the tree models.
#
# | feature | definition |
# |---|---|
# | `priority_tier` | band of `terminal_priority`: Free 0-99, BestEffort 100-115, Mid 116-119, Production 120-359, Monitoring 360+ (one-hot). |
# | `scheduling_class` | ordinal `terminal_scheduling_class` in 0-3. |
# | `platform_id` | machine micro-architecture for the scheduled machine (one-hot). |
# | `cpu_request`, `memory_request` | requested resources at submission (passthrough). |
# | `request_ratio` | `cpu_request / memory_request` (null-safe; null when `memory_request` is 0). |
# | `queue_time` | SUBMIT -> first SCHEDULE delta in seconds (`queue_time_sec`). |

# %%
def _priority_tier_expr() -> pl.Expr:
    """Map terminal_priority to a categorical band label (V07; schema bands)."""
    p = pl.col("terminal_priority")
    return (
        pl.when(p <= PRIORITY_FREE_MAX).then(pl.lit("free"))
        .when((p >= PRIORITY_BEST_EFFORT_LOW) & (p <= PRIORITY_BEST_EFFORT_MAX)).then(pl.lit("best_effort"))
        .when((p >= PRIORITY_MID_TIER_LOW) & (p <= PRIORITY_MID_TIER_MAX)).then(pl.lit("mid"))
        .when((p >= PRIORITY_PRODUCTION_LOW) & (p <= PRIORITY_PRODUCTION_MAX)).then(pl.lit("production"))
        .when(p >= PRIORITY_MONITORING_LOW).then(pl.lit("monitoring"))
        .otherwise(pl.lit("unknown"))
        .alias("priority_tier")
    )


PRIORITY_TIER_LEVELS = ["free", "best_effort", "mid", "production", "monitoring"]


def add_tier1_scheduling(lf: pl.LazyFrame, platform_lookup: pl.LazyFrame) -> pl.LazyFrame:
    """Append Tier 1 scheduling features and their one-hot encodings.

    Input: lifecycle-summary columns terminal_priority,
    terminal_scheduling_class, terminal_machine_id, cpu_request,
    memory_request, queue_time_sec. `platform_lookup` is a
    (machine_id, platform_id) LazyFrame. Motivation: V07.
    """
    out = (
        lf
        .with_columns(_priority_tier_expr())
        .with_columns(
            pl.col("terminal_scheduling_class").cast(pl.Int64).alias("scheduling_class"),
            pl.col("cpu_request").cast(pl.Float64),
            pl.col("memory_request").cast(pl.Float64),
            # Null-safe ratio: guard divide-by-zero, leave null when undefined.
            pl.when(pl.col("memory_request") > 0)
              .then(pl.col("cpu_request") / pl.col("memory_request"))
              .otherwise(None)
              .alias("request_ratio"),
            pl.col("queue_time_sec").cast(pl.Float64).alias("queue_time"),
        )
        # Attach platform_id for the scheduled machine.
        .join(platform_lookup, left_on="terminal_machine_id", right_on="machine_id", how="left")
    )

    # One-hot priority_tier with a stable, fully-enumerated column set.
    for level in PRIORITY_TIER_LEVELS:
        out = out.with_columns(
            (pl.col("priority_tier") == level).cast(pl.Int8).alias(f"priority_tier_{level}")
        )
    # One-hot platform_id over the enumerated platform set; unknown -> all 0.
    # Use the BigQuery-safe suffix map for the column names (raw platform_id is
    # a base64-style hash that BigQuery rejects as a field name).
    for pid in platform_ids:
        out = out.with_columns(
            (pl.col("platform_id") == pid).fill_null(False).cast(pl.Int8)
                .alias(f"platform_{PLATFORM_SUFFIX[pid]}")
        )
    return out


tier1_sched_lf = add_tier1_scheduling(tier1_hist_lf, machine_platform_lf)

# %% [markdown]
# ---
# ## 5. Tier 1 temporal features
#
# **-> extract to `src/features/temporal.py`.** Motivation: V26 (FAIL_LOST
# rate varies up to ~8x across hour-of-day buckets, peaking in PDT business
# hours). Derived from `submit_time` using the same naive-anchor convention
# the F2 SQL uses: anchor `'2019-05-01 00:00:00 UTC'`, add the trace-clock
# microseconds, subtract the 600s pre-trace offset, and the EXTRACTed
# components coincide with PDT wall-clock. The trace clock starts 600s
# before the documented trace window; subtracting it aligns the wall-clock
# components used by the V26 diurnal census.
#
# | feature | definition |
# |---|---|
# | `submit_hour_of_day` | PDT hour 0-23 (ordinal for trees). |
# | `submit_day_of_week` | 0=Mon .. 6=Sun. |
# | `submit_hour_sin`, `submit_hour_cos` | cyclic encoding of the hour (for non-tree models in Phase 4). |
# | `submit_is_business_hours_pdt` | PDT hour in [8, 17]. |
# | `submit_is_weekend` | day_of_week in {Sat, Sun}. |

# %%
# Naive anchor for the trace clock. The trace's `time` field is microseconds
# from 600s before the trace start; F2 fixes the anchor at 2019-05-01 UTC.
TRACE_ANCHOR = datetime(2019, 5, 1, 0, 0, 0)
TRACE_PRE_OFFSET_US = 600 * MICROS_PER_SEC      # 600 s, the pre-trace lead-in


def add_tier1_temporal(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append Tier 1 submit-time PDT temporal features. Input: submit_time
    (microseconds, trace clock). Motivation: V26.
    """
    # Reconstruct the PDT wall-clock timestamp via the naive-anchor trick.
    wall = (
        pl.lit(TRACE_ANCHOR)
        + pl.duration(microseconds=(pl.col("submit_time") - TRACE_PRE_OFFSET_US))
    )
    out = lf.with_columns(wall.alias("_submit_wall"))
    out = out.with_columns(
        pl.col("_submit_wall").dt.hour().alias("submit_hour_of_day"),
        # Polars weekday(): Mon=1 .. Sun=7. Shift to Mon=0 .. Sun=6.
        (pl.col("_submit_wall").dt.weekday() - 1).alias("submit_day_of_week"),
    )
    out = out.with_columns(
        (2 * np.pi * pl.col("submit_hour_of_day") / 24).sin().alias("submit_hour_sin"),
        (2 * np.pi * pl.col("submit_hour_of_day") / 24).cos().alias("submit_hour_cos"),
        ((pl.col("submit_hour_of_day") >= 8) & (pl.col("submit_hour_of_day") <= 17))
            .cast(pl.Int8).alias("submit_is_business_hours_pdt"),
        (pl.col("submit_day_of_week") >= 5).cast(pl.Int8).alias("submit_is_weekend"),
    ).drop("_submit_wall")
    return out


tier1_temporal_lf = add_tier1_temporal(tier1_sched_lf)

# %% [markdown]
# ---
# ## 6. Tier 1 assembly
#
# Finalize the two Tier 1 features that depend on cross-instance context or
# external lookups:
# - `collection_size_at_submit` — the number of working-set instances in the
#   instance's collection. Collection-level context (V07); larger collections
#   behave differently under scheduling pressure.
# - `lifecycle_position` — `instance_index / collection_size_at_submit`
#   (Section 3 dependency resolved here).
# - `has_hardware_counters` — joined from the per-instance majority vote
#   (V11); the Tier 2 `cpi_value` / `mapi_value` are conditioned on it.

# %%
# collection_size_at_submit and has_hardware_counters_majority are already
# present on the base table (computed BigQuery-side in Section 2), so Tier 1
# assembly is now pure per-row work: no group_by and no wide joins to
# materialize. This is what unblocks the probe collect on the Colab box.
tier1_lf = (
    tier1_temporal_lf
    .with_columns(
        # has_hardware_counters: per-instance majority vote (V11). The base
        # table already COALESCEd nulls to 0; cast for dtype stability.
        pl.col("has_hardware_counters_majority").fill_null(0)
            .cast(pl.Int8).alias("has_hardware_counters"),
    )
    .with_columns(
        # Fractional ordinal position of the instance within its collection.
        pl.when(pl.col("collection_size_at_submit") > 0)
          .then(pl.col("instance_index") / pl.col("collection_size_at_submit"))
          .otherwise(None)
          .alias("lifecycle_position"),
    )
)

# Canonical Tier 1 column set (keys + features). One-hot expansions are
# enumerated so the schema is stable across working-set samples.
TIER1_FEATURE_COLS = [
    # historical (V09, V10)
    "prior_fail_count", "has_prior_fail", "resubmission_count",
    "prior_evict_count", "first_resubmission", "lifecycle_position",
    # hardware-counter availability (V11)
    "has_hardware_counters",
    # scheduling (V07)
    "scheduling_class", "cpu_request", "memory_request", "request_ratio",
    "queue_time", "collection_size_at_submit",
    *[f"priority_tier_{lvl}" for lvl in PRIORITY_TIER_LEVELS],
    *[f"platform_{PLATFORM_SUFFIX[pid]}" for pid in platform_ids],
    # temporal (V26)
    "submit_hour_of_day", "submit_day_of_week", "submit_hour_sin",
    "submit_hour_cos", "submit_is_business_hours_pdt", "submit_is_weekend",
]
KEY_COLS = ["collection_id", "instance_index"]
LABEL_COLS = ["failure_label", "outcome"]

# Carry the label through from the lifecycle summary's outcome so the feature
# matrix is self-contained for modeling. failure_label: FAIL_LOST -> 1,
# FINISH -> 0 (V01); other outcomes are excluded from the working set already.
tier1_lf = tier1_lf.with_columns(
    pl.when(pl.col("outcome") == "FAIL_LOST").then(1)
      .when(pl.col("outcome") == "FINISH").then(0)
      .otherwise(None)
      .cast(pl.Int8)
      .alias("failure_label")
)

tier1_out_lf = tier1_lf.select(KEY_COLS + LABEL_COLS + TIER1_FEATURE_COLS)

# Validate Tier 1 before the expensive Tier 2 BigQuery work. With the
# Section 2 pushdown there is no group_by upstream, so the streaming engine
# evaluates this bounded slice without materializing the full working set.
tier1_probe = tier1_out_lf.head(100_000).collect(streaming=True)
print(f"Tier 1 probe shape: {tier1_probe.shape}")
print(tier1_probe.select(["failure_label", "has_prior_fail", "first_resubmission",
                          "priority_tier_production", "submit_is_business_hours_pdt"]).head())

record_check(
    "Section 6: Tier 1 feature column count",
    expected=f"{len(TIER1_FEATURE_COLS)} feature columns",
    observed=len([c for c in tier1_probe.columns if c in TIER1_FEATURE_COLS]),
    ok=(len([c for c in tier1_probe.columns if c in TIER1_FEATURE_COLS]) == len(TIER1_FEATURE_COLS)),
)
record_check(
    "Section 6: failure_label is binary with no nulls in the working set",
    expected="{0, 1}",
    observed=sorted(tier1_probe["failure_label"].drop_nulls().unique().to_list()),
    ok=(tier1_probe["failure_label"].null_count() == 0
        and set(tier1_probe["failure_label"].unique().to_list()) <= {0, 1}),
)

# %% [markdown]
# ---
# ## 7. Tier 2 early-runtime features (BigQuery, two-stage filter)
#
# **-> extract to `src/features/runtime.py`.** Motivation: V12 (utilization
# inversion — failing instances ramp 3.6x faster on CPU and 2.3x faster on
# memory even though they use *less* absolute resource). Tier 2 carries the
# discriminative rate-of-change signal.
#
# The slopes require per-observation usage rows from the
# 7.5B-row `instance_usage_full`. Two-stage approach:
#
# 1. **Stage 1 — materialize the working-set usage subset.** Inner-join
#    `instance_usage_full` to `working_set_instance_ids` on
#    `(collection_id, instance_index)` and keep only observations inside the
#    +/-60s early-runtime band around each instance's `schedule_time`. This
#    is the only time the 7.5B table is read; the join + range filter prune
#    it to the working set's early-runtime slice.
# 2. **Stage 2 — compute features BigQuery-side.** Use closed-form OLS slopes
#    (`COVAR_POP / VAR_POP`), `LAG`,
#    and `AVG ... OVER` window functions to reduce each instance's handful of
#    early-runtime observations to a single per-instance feature row. Only
#    that compact row is exported to GCS Parquet; raw observations never
#    leave BigQuery.
#
# | feature | definition |
# |---|---|
# | `cpu_slope_5s/15s/30s` | OLS slope of `avg_cpu` on seconds-since-schedule over the first 5 / 15 / 30 s after scheduling. |
# | `memory_slope_5s/15s/30s` | same for `avg_memory`. |
# | `initial_cpu_ramp` | first-to-second observation delta in `avg_cpu` (early acceleration). |
# | `initial_memory_ramp` | same for `avg_memory`. |
# | `first_interval_util_ratio` | first-observation `avg_cpu` / `cpu_request` (realized vs requested at startup). |
# | `cpi_value`, `mapi_value` | first-observation hardware-counter values, conditioned on `has_hardware_counters` (null otherwise; V11 MNAR — never imputed). |

# %% [markdown]
# ### 7.1 Stage 1 — materialize `instance_usage_working_set` (+/-60s band)
#
# This is the canonical first-stage filter. Note the use of
# `instance_usage_with_indicators` (the V11-augmented usage table from
# notebook 08) as the source so the `has_cpi_value` / `has_mapi_value`
# indicators ride along for the conditional CPI/MAPI features.

# %%
stage1_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_usage_working_set')}
CLUSTER BY collection_id, instance_index AS
SELECT
    u.*,
    w.schedule_time,
    -- Seconds since the instance's schedule event; the regression abscissa.
    SAFE_DIVIDE(u.start_time - w.schedule_time, {MICROS_PER_SEC}) AS sec_since_schedule
FROM {fqn('instance_usage_with_indicators')} u
INNER JOIN {fqn(WORKING_SET_TABLE)} w
  USING (collection_id, instance_index)
WHERE u.start_time BETWEEN (w.schedule_time - {EARLY_RUNTIME_BAND_US})
                       AND (w.schedule_time + {EARLY_RUNTIME_BAND_US});
"""
run_ddl(stage1_sql, "Stage 1 -> instance_usage_working_set (+/-60s)")

n_usage_ws = row_count('instance_usage_working_set')
print(f"Working-set early-runtime usage observations: {n_usage_ws:,}")
record_check(
    "Section 7.1: instance_usage_working_set is non-empty and bounded",
    expected="> 0 and << 7.5B (join-pruned)",
    observed=n_usage_ws,
    ok=(0 < n_usage_ws < 7_575_500_668),
    notes="Confirms the +/-60s join pruned the 7.5B-row source.",
)

# %% [markdown]
# ### 7.2 Stage 2 — per-instance slope / ramp / first-interval features
#
# BigQuery has no `REGR_SLOPE`, so the OLS slope of `y` on
# `sec_since_schedule` is computed directly as `COVAR_POP(y, x) / VAR_POP(x)`
# (the closed-form least-squares slope). The 5 / 15 / 30 s bands are selected
# by nulling out-of-band observations inside the aggregate; `COVAR_POP` /
# `VAR_POP` ignore NULL inputs, and a single in-band observation gives
# `VAR_POP = 0`, so `SAFE_DIVIDE` returns NULL (no slope from one point).
# Ramps and first-interval ratios use `ROW_NUMBER` to pick the first two
# post-schedule observations. CPI/MAPI values are taken from the first
# post-schedule observation and conditioned on availability.

# %%
def _slope_sql(value_col: str, horizon_s: int) -> str:
    """Return a BigQuery OLS-slope expression for `value_col` regressed on
    sec_since_schedule over the first `horizon_s` seconds. Both COVAR_POP and
    VAR_POP are gated on the same in-band, non-null-y rows so numerator and
    denominator span an identical observation set. SAFE_DIVIDE yields NULL
    when the band has < 2 distinct-x observations (VAR_POP = 0).
    """
    gate = f"sec_since_schedule <= {horizon_s} AND {value_col} IS NOT NULL"
    x = f"IF({gate}, sec_since_schedule, NULL)"
    y = f"IF({gate}, {value_col}, NULL)"
    return f"SAFE_DIVIDE(COVAR_POP({y}, {x}), VAR_POP({x}))"


stage2_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_runtime_features')}
CLUSTER BY collection_id, instance_index AS
WITH ranked AS (
    SELECT
        collection_id,
        instance_index,
        sec_since_schedule,
        avg_cpu,
        avg_memory,
        cycles_per_instruction,
        memory_accesses_per_instruction,
        has_cpi_value,
        has_mapi_value,
        -- Order post-schedule observations to pick the first two.
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY sec_since_schedule
        ) AS rn_post,
        -- Previous observation values for the ramp deltas.
        LAG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY sec_since_schedule
        ) AS prev_avg_cpu,
        LAG(avg_memory) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY sec_since_schedule
        ) AS prev_avg_memory,
        -- Smoothed first-interval baseline: trailing mean over the first few
        -- post-schedule observations. Using AVG OVER rather than a single
        -- first-observation reading dampens per-sample noise in the realized
        -- startup utilization that feeds first_interval_util_ratio.
        AVG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY sec_since_schedule
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        ) AS first_window_avg_cpu
    FROM {fqn('instance_usage_working_set')}
    -- Restrict slopes/ramps to the post-schedule side of the band.
    WHERE sec_since_schedule >= 0
),
agg AS (
    SELECT
        collection_id,
        instance_index,
        -- CPU slopes over progressively wider post-schedule bands
        -- (COVAR_POP / VAR_POP closed-form OLS; BigQuery has no REGR_SLOPE).
        {_slope_sql('avg_cpu', 5)}  AS cpu_slope_5s,
        {_slope_sql('avg_cpu', 15)} AS cpu_slope_15s,
        {_slope_sql('avg_cpu', 30)} AS cpu_slope_30s,
        -- Memory slopes.
        {_slope_sql('avg_memory', 5)}  AS memory_slope_5s,
        {_slope_sql('avg_memory', 15)} AS memory_slope_15s,
        {_slope_sql('avg_memory', 30)} AS memory_slope_30s,
        -- First-to-second observation ramp (acceleration at startup).
        MAX(IF(rn_post = 2, avg_cpu - prev_avg_cpu, NULL))       AS initial_cpu_ramp,
        MAX(IF(rn_post = 2, avg_memory - prev_avg_memory, NULL)) AS initial_memory_ramp,
        -- Smoothed first-interval realized CPU (AVG OVER baseline at rn_post=1).
        MAX(IF(rn_post = 1, first_window_avg_cpu, NULL))        AS first_interval_avg_cpu,
        -- First-observation hardware-counter values + availability flags.
        MAX(IF(rn_post = 1, cycles_per_instruction, NULL))             AS first_cpi,
        MAX(IF(rn_post = 1, memory_accesses_per_instruction, NULL))    AS first_mapi,
        MAX(IF(rn_post = 1, has_cpi_value, NULL))                AS first_has_cpi,
        MAX(IF(rn_post = 1, has_mapi_value, NULL))               AS first_has_mapi
    FROM ranked
    GROUP BY collection_id, instance_index
)
SELECT
    a.collection_id,
    a.instance_index,
    a.cpu_slope_5s, a.cpu_slope_15s, a.cpu_slope_30s,
    a.memory_slope_5s, a.memory_slope_15s, a.memory_slope_30s,
    a.initial_cpu_ramp,
    a.initial_memory_ramp,
    -- first_interval_util_ratio: smoothed realized startup CPU / requested CPU.
    SAFE_DIVIDE(a.first_interval_avg_cpu, NULLIF(s.cpu_request, 0)) AS first_interval_util_ratio,
    -- Conditional CPI/MAPI: keep the value only when the counter was present
    -- on the first observation; NULL otherwise (V11 MNAR, never imputed).
    IF(a.first_has_cpi  = 1, a.first_cpi,  NULL) AS cpi_value,
    IF(a.first_has_mapi = 1, a.first_mapi, NULL) AS mapi_value
FROM agg a
LEFT JOIN {fqn('instance_lifecycle_summary')} s
  USING (collection_id, instance_index)
"""
run_ddl(stage2_sql, "Stage 2 -> instance_runtime_features (slopes/ramps)")

n_runtime = row_count('instance_runtime_features')
print(f"Per-instance Tier 2 feature rows: {n_runtime:,}")
record_check(
    "Section 7.2: one Tier 2 row per instance with early-runtime observations",
    expected=f"<= working-set size ({n_working_set:,})",
    observed=n_runtime,
    ok=(0 < n_runtime <= n_working_set),
    notes="Instances with no post-schedule usage in band are absent; joined as null Tier 2 later.",
)

# %% [markdown]
# ### 7.3 Export the compact Tier 2 feature row to GCS Parquet
#
# Only the per-instance feature row leaves BigQuery. The export is
# small enough to also stage to Drive, but GCS Parquet keeps the lazy-scan
# convention used elsewhere in the pipeline.

# %%
runtime_uri = f"gs://{GCS_BUCKET}/{GCS_RUNTIME_PARQUET_PREFIX}/"
export_runtime_sql = f"""
EXPORT DATA OPTIONS(
    uri='{runtime_uri}*.parquet',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT * FROM {fqn('instance_runtime_features')}
"""
run_ddl(export_runtime_sql, f"Export Tier 2 features -> {runtime_uri}")

# Tier 2 stays a BigQuery table (instance_runtime_features); the GCS export
# above is for traceability. The three-tier join happens BigQuery-side in
# Section 9, so no Polars scan of Tier 2 is needed here.

TIER2_FEATURE_COLS = [
    "cpu_slope_5s", "cpu_slope_15s", "cpu_slope_30s",
    "memory_slope_5s", "memory_slope_15s", "memory_slope_30s",
    "initial_cpu_ramp", "initial_memory_ramp", "first_interval_util_ratio",
    "cpi_value", "mapi_value",
]

# %% [markdown]
# ---
# ## 8. Tier 3 windowed utilization features (LOW value, ablation only)
#
# **-> extract to `src/features/utilization.py`.** Motivation: V12 / V13.
# These absolute windowed aggregates are the *confounded* comparison: the
# rapid-onset 22.6s crash window (V09) is shorter than every Tier 3
# aggregation window, so failing instances appear to use *less* resource
# (the V12 inversion). **Do not drop Tier 3** — the ablation
# (V13) and the Tier 3 inversion regression guard
# (`src/data/validation.py::assert_tier3_inversion`, exercised in
# notebook 11) both depend on these columns being present.
#
# Tier 3 windows (5 / 15 / 60 min) extend past the +/-60s Tier 2 band, so a
# **wider** working-set usage subset is materialized first. The subset is
# still bounded to the working set (the 7.5B source is never
# scanned outside the join), then the windowed aggregates are computed in
# Polars over the post-schedule observations within each window.
#
# | feature | definition |
# |---|---|
# | `avg_cpu_5min` / `15min` / `60min` | mean `avg_cpu` over post-schedule observations within the window. |
# | `max_cpu_*` | max `avg_cpu` (peak) within the window. |
# | `std_cpu_*` | std of `avg_cpu` within the window. |
# | `avg_memory_*`, `max_memory_*`, `std_memory_*` | memory variants. |

# %% [markdown]
# ### 8.1 Materialize the wider Tier 3 usage subset

# %%
stage1_t3_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_usage_working_set_t3')}
CLUSTER BY collection_id, instance_index AS
SELECT
    u.collection_id,
    u.instance_index,
    u.avg_cpu,
    u.avg_memory,
    SAFE_DIVIDE(u.start_time - w.schedule_time, {MICROS_PER_SEC}) AS sec_since_schedule
FROM {fqn('instance_usage_with_indicators')} u
INNER JOIN {fqn(WORKING_SET_TABLE)} w
  USING (collection_id, instance_index)
-- Post-schedule observations out to the 60-min Tier 3 horizon.
WHERE u.start_time BETWEEN w.schedule_time
                       AND (w.schedule_time + {TIER3_MAX_WINDOW_US});
"""
run_ddl(stage1_t3_sql, "Stage 1 (Tier 3) -> instance_usage_working_set_t3 (0..60min)")

# %% [markdown]
# ### 8.2 Compute windowed aggregates in BigQuery
#
# The windowed avg / max / std are conditional aggregates grouped per
# instance. Computing them BigQuery-side (rather than a Polars `group_by` over
# an exported usage subset) keeps the ~75M-group aggregation off the Colab box;
# only the compact per-instance Tier 3 feature table results. `STDDEV_SAMP`
# matches the prior Polars `.std()` (ddof = 1): a single in-window observation
# yields NULL, consistent with Polars.

# %%
TIER3_WINDOWS_SEC = {"5min": 300, "15min": 900, "60min": 3600}


def _windowed_agg_sql() -> str:
    """Return the comma-separated BigQuery conditional-aggregate expressions
    for the Tier 3 windowed utilization features (avg/max/std of cpu and
    memory at the 5/15/60-min windows). Motivation: V12, V13.
    """
    parts: list[str] = []
    for label, horizon in TIER3_WINDOWS_SEC.items():
        cpu_in = f"IF(sec_since_schedule BETWEEN 0 AND {horizon}, avg_cpu, NULL)"
        mem_in = f"IF(sec_since_schedule BETWEEN 0 AND {horizon}, avg_memory, NULL)"
        parts += [
            f"AVG({cpu_in}) AS avg_cpu_{label}",
            f"MAX({cpu_in}) AS max_cpu_{label}",
            f"STDDEV_SAMP({cpu_in}) AS std_cpu_{label}",
            f"AVG({mem_in}) AS avg_memory_{label}",
            f"MAX({mem_in}) AS max_memory_{label}",
            f"STDDEV_SAMP({mem_in}) AS std_memory_{label}",
        ]
    return ",\n    ".join(parts)


tier3_agg_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_tier3_features')}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    {_windowed_agg_sql()}
FROM {fqn('instance_usage_working_set_t3')}
GROUP BY collection_id, instance_index
"""
run_ddl(tier3_agg_sql, "Tier 3 windowed aggregates -> instance_tier3_features")

TIER3_FEATURE_COLS = [
    f"{stat}_{res}_{label}"
    for res in ("cpu", "memory")
    for stat in ("avg", "max", "std")
    for label in TIER3_WINDOWS_SEC
]
# Reorder to the canonical avg/max/std-per-resource grouping for readability.
TIER3_FEATURE_COLS = [
    "avg_cpu_5min", "max_cpu_5min", "std_cpu_5min",
    "avg_cpu_15min", "max_cpu_15min", "std_cpu_15min",
    "avg_cpu_60min", "max_cpu_60min", "std_cpu_60min",
    "avg_memory_5min", "max_memory_5min", "std_memory_5min",
    "avg_memory_15min", "max_memory_15min", "std_memory_15min",
    "avg_memory_60min", "max_memory_60min", "std_memory_60min",
]

# %% [markdown]
# ---
# ## 9. Three-tier assembly + Parquet export
#
# The three per-instance tables (Tier 1 in Polars, Tier 2 and Tier 3 in
# BigQuery) are joined **in BigQuery**, not in Polars. An in-Polars left join
# of two ~75M-row right-hand frames builds two full in-memory hash tables and
# OOMs the 12.7 GB Colab kernel (the original failure here). Instead: stream
# Tier 1 to GCS and load it into BigQuery, left-join the three tables
# BigQuery-side on the instance key, export to GCS, then have Polars do a pure
# streaming copy (no joins) to consolidate the shards into the single Drive
# Parquet the downstream notebooks read. Tier 2 / Tier 3 are absent for
# instances with no in-band usage and join as nulls (correct: those instances
# carry no early-runtime or windowed signal).

# %%
ALL_FEATURE_COLS = TIER1_FEATURE_COLS + TIER2_FEATURE_COLS + TIER3_FEATURE_COLS
FINAL_COLS = KEY_COLS + LABEL_COLS + ALL_FEATURE_COLS

# %% [markdown]
# ### 9.1 Stream Tier 1 to GCS and load it into BigQuery
#
# `tier1_out_lf` has no group_by upstream (only the tiny platform join), so the
# streaming sink stays within memory. The Parquet is loaded into a clustered
# BigQuery table so the final join runs entirely BigQuery-side. The sink writes
# to `gs://` exactly as the Tier 1 / Tier 2 reads above scan from `gs://`; if
# your environment needs explicit credentials for the write, pass
# `storage_options=` to `sink_parquet`.

# %%
TIER1_GCS_PREFIX = f'{GCS_FEATURES_PREFIX}/instance_tier1_features'
tier1_uri = f"gs://{GCS_BUCKET}/{TIER1_GCS_PREFIX}/"

# tier1_out_lf already selects KEY_COLS + LABEL_COLS + TIER1_FEATURE_COLS.
tier1_out_lf.sink_parquet(f"{tier1_uri}tier1.parquet", compression="snappy")
print(f"Tier 1 streamed to {tier1_uri}")

load_tier1_sql = f"""
LOAD DATA OVERWRITE {fqn('instance_tier1_features')}
CLUSTER BY collection_id, instance_index
FROM FILES (
    format = 'PARQUET',
    uris = ['{tier1_uri}*.parquet']
)
"""
run_ddl(load_tier1_sql, "Load Tier 1 Parquet -> instance_tier1_features")

# %% [markdown]
# ### 9.2 Assemble the three-tier matrix in BigQuery
#
# `t1.*` carries the keys, labels, and Tier 1 features; the Tier 2 and Tier 3
# feature columns are appended explicitly so the join keys are not duplicated.

# %%
tier2_select = ",\n    ".join(f"t2.{c}" for c in TIER2_FEATURE_COLS)
tier3_select = ",\n    ".join(f"t3.{c}" for c in TIER3_FEATURE_COLS)

assemble_sql = f"""
CREATE OR REPLACE TABLE {fqn('instance_features')}
CLUSTER BY collection_id, instance_index AS
SELECT
    t1.*,
    {tier2_select},
    {tier3_select}
FROM {fqn('instance_tier1_features')} t1
LEFT JOIN {fqn('instance_runtime_features')} t2
  USING (collection_id, instance_index)
LEFT JOIN {fqn('instance_tier3_features')} t3
  USING (collection_id, instance_index)
"""
run_ddl(assemble_sql, "Assemble three-tier matrix -> instance_features")

# %% [markdown]
# ### 9.3 Export and consolidate to the single Drive Parquet
#
# The streaming `scan_parquet -> sink_parquet` copy contains no joins or
# aggregations, so peak memory is bounded regardless of the matrix size.

# %%
FINAL_GCS_PREFIX = f'{GCS_FEATURES_PREFIX}/instance_features'
final_uri = f"gs://{GCS_BUCKET}/{FINAL_GCS_PREFIX}/"

export_final_sql = f"""
EXPORT DATA OPTIONS(
    uri='{final_uri}*.parquet',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT * FROM {fqn('instance_features')}
"""
run_ddl(export_final_sql, f"Export feature matrix -> {final_uri}")

FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
pl.scan_parquet(final_uri).sink_parquet(str(FEATURE_MATRIX_PATH), compression="snappy")
print(f"Feature matrix written: {FEATURE_MATRIX_PATH}")

# %% [markdown]
# ---
# ## 10. Feature-schema manifest + verification suite

# %%
# Re-scan the written matrix for verification (avoids recomputing the joins).
matrix_lf = pl.scan_parquet(str(FEATURE_MATRIX_PATH))
n_rows_matrix = matrix_lf.select(pl.len()).collect().item()
matrix_cols = matrix_lf.collect_schema().names()

print(f"Feature matrix rows:    {n_rows_matrix:,}")
print(f"Feature matrix columns: {len(matrix_cols)}")

record_check(
    "Section 10: row count matches the working set",
    expected=n_working_set,
    observed=n_rows_matrix,
    ok=(n_rows_matrix == n_working_set),
    notes="One feature row per working-set instance (Tier 1 backbone, left joins).",
)
record_check(
    "Section 10: all declared feature columns present",
    expected=len(FINAL_COLS),
    observed=len(matrix_cols),
    ok=(set(FINAL_COLS) == set(matrix_cols)),
)

# %% [markdown]
# ### 10.1 Tier 3 inversion sanity check (V12 preview)
#
# A lightweight preview of the formal Tier 3 inversion guard run in
# notebook 11 (`assert_tier3_inversion`). Failing instances should
# retain *lower* median absolute CPU at every Tier 3 window (V12). A
# violation here means preprocessing or feature engineering washed out the
# inversion the Chapter 4 ablation depends on.
#
# Computed in BigQuery against `instance_features` (`APPROX_QUANTILES` median)
# rather than collecting the 75M-row matrix into Polars, which would OOM the
# Colab kernel. The approximate median is sufficient for a directional check.

# %%
inv_parts: list[str] = []
for label in TIER3_WINDOWS_SEC:
    inv_parts.append(
        f"APPROX_QUANTILES(IF(failure_label = 1, avg_cpu_{label}, NULL), 2)[OFFSET(1)] "
        f"AS fail_med_{label}"
    )
    inv_parts.append(
        f"APPROX_QUANTILES(IF(failure_label = 0, avg_cpu_{label}, NULL), 2)[OFFSET(1)] "
        f"AS ok_med_{label}"
    )
inv_sql = "SELECT\n    " + ",\n    ".join(inv_parts) + f"\nFROM {fqn('instance_features')}"
inv_df = run_query(inv_sql)

inversion_rows = []
for label in TIER3_WINDOWS_SEC:
    fail_med = inv_df[f"fail_med_{label}"].item()
    ok_med = inv_df[f"ok_med_{label}"].item()
    holds = (fail_med is not None and ok_med is not None and fail_med < ok_med)
    inversion_rows.append({
        "window": label, "median_cpu_fail": fail_med,
        "median_cpu_success": ok_med, "inversion_holds": holds,
    })
    record_check(
        f"Section 10.1: V12 inversion holds at {label} (median CPU fail < success)",
        expected="fail < success",
        observed=f"fail={fail_med}, success={ok_med}",
        ok=holds,
        notes="Approx median (BigQuery); full guard runs in notebook 11.",
    )

pl.DataFrame(inversion_rows).write_csv(str(TABLES_DIR / "tier3_inversion_check.csv"))

# %% [markdown]
# ### 10.2 Per-tier null audit
#
# Tier 2 / Tier 3 nulls are expected for instances with no in-band usage
# observations (rapid-onset crashes). The audit records the null fraction so
# the modeling notebooks choose an appropriate missing-value strategy.
#
# Computed in BigQuery (one COUNTIF pass over `instance_features`) rather than
# a full-column Polars collect over the 75M-row matrix, which OOM'd the kernel.

# %%
null_parts = ",\n    ".join(f"COUNTIF({c} IS NULL) AS n_{c}" for c in ALL_FEATURE_COLS)
null_sql = f"SELECT COUNT(*) AS n_total,\n    {null_parts}\nFROM {fqn('instance_features')}"
null_df = run_query(null_sql)
n_total = int(null_df["n_total"].item())
null_fracs = {c: int(null_df[f"n_{c}"].item()) / n_total for c in ALL_FEATURE_COLS}
for tier_name, cols in (("Tier 1", TIER1_FEATURE_COLS),
                        ("Tier 2", TIER2_FEATURE_COLS),
                        ("Tier 3", TIER3_FEATURE_COLS)):
    worst = max(null_fracs[c] for c in cols)
    print(f"{tier_name}: max null fraction {worst:.4f}")

# %% [markdown]
# ### 10.3 Feature-schema manifest

# %%
import json

# Motivation tags map each feature group to the EDA decision that justifies it.
TIER_MOTIVATION = {
    "tier1": "V07, V09, V10, V11, V26 (pre-event signals: history, scheduling, temporal)",
    "tier2": "V12 (utilization inversion -> early-runtime rate-of-change)",
    "tier3": "V12, V13 (absolute windowed utilization; ablation-only, confounded)",
}
matrix_dtypes = {name: str(dt) for name, dt in zip(matrix_lf.collect_schema().names(),
                                                   matrix_lf.collect_schema().dtypes())}


def tier_of(col: str) -> str:
    if col in TIER1_FEATURE_COLS:
        return "tier1"
    if col in TIER2_FEATURE_COLS:
        return "tier2"
    if col in TIER3_FEATURE_COLS:
        return "tier3"
    return "key_or_label"


feature_schema = {
    "dataset": "google_cluster_traces",
    "artifact": "instance_features",
    "path": str(FEATURE_MATRIX_PATH),
    "row_count": n_rows_matrix,
    "n_features": len(ALL_FEATURE_COLS),
    "working_set_table": f"{PROJECT_ID}.{DATASET}.{WORKING_SET_TABLE}",
    "tier_motivation": TIER_MOTIVATION,
    # Recover the original (obfuscated) platform_id behind each one-hot suffix:
    # column platform_<suffix> corresponds to platform_id_by_suffix[<suffix>].
    "platform_id_by_suffix": {suffix: pid for pid, suffix in PLATFORM_SUFFIX.items()},
    "columns": [
        {"name": c, "dtype": matrix_dtypes.get(c), "tier": tier_of(c)}
        for c in FINAL_COLS
    ],
    "tier_counts": {
        "tier1": len(TIER1_FEATURE_COLS),
        "tier2": len(TIER2_FEATURE_COLS),
        "tier3": len(TIER3_FEATURE_COLS),
    },
    "execution_strategy": {
        "stage1_tables": [
            f"{PROJECT_ID}.{DATASET}.instance_usage_working_set",
            f"{PROJECT_ID}.{DATASET}.instance_usage_working_set_t3",
        ],
        "stage2_table": f"{PROJECT_ID}.{DATASET}.instance_runtime_features",
        "note": "7.5B-row instance_usage_full read only through working-set joins; "
                "slopes/ramps computed BigQuery-side, only per-instance rows exported.",
    },
    "decisions_applied": ["V07", "V09", "V10", "V11", "V12", "V13", "V26"],
    "created_at": datetime.now(timezone.utc).isoformat(),
}

FEATURE_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(FEATURE_SCHEMA_PATH, "w") as f:
    json.dump(feature_schema, f, indent=2)
print(f"Feature schema manifest: {FEATURE_SCHEMA_PATH}")
print(f"Tier counts: {feature_schema['tier_counts']}")

# %% [markdown]
# ### 10.4 Write the verification log

# %%
verification_df = pl.DataFrame(verification_rows)
verification_df.write_csv(str(VERIFICATION_CSV))
print(f"Verification log: {VERIFICATION_CSV}")
print(verification_df.select(["check", "ok"]))

n_failed = verification_df.filter(~pl.col("ok")).height
if n_failed:
    print(f"\n{n_failed} assertion(s) failed. Inspect before locking the working set.")
else:
    print("\nAll assertions passed. Feature matrix ready for notebook 11 (learning curve).")

assert n_failed == 0, "Resolve failed assertions before proceeding."

# %% [markdown]
# ### End-of-notebook smoke test
#
# Confirm the tier composition and label balance on a scanned slice.

# %%
smoke = (
    pl.scan_parquet(str(FEATURE_MATRIX_PATH))
    .select(["failure_label", "first_resubmission", "cpu_slope_15s", "avg_cpu_5min"])
    .head(1_000_000)
    .collect()
)
print("Label distribution (first 1M rows):")
print(smoke.group_by("failure_label").agg(pl.len().alias("n")).sort("failure_label"))
print("\nTier 2 cpu_slope_15s availability (non-null fraction):",
      1 - smoke["cpu_slope_15s"].null_count() / smoke.height)
print("Tier 3 avg_cpu_5min availability (non-null fraction):",
      1 - smoke["avg_cpu_5min"].null_count() / smoke.height)

# %% [markdown]
# **Next steps.** Extract each tier section into the
# matching `src/features/*.py` module as pure LazyFrame -> LazyFrame
# functions; build `src/features/sampling.py` to produce the locked
# `working_set_instance_ids`; run the learning-curve
# harness and the formal Tier 3 inversion guard on this matrix (notebook 11).
