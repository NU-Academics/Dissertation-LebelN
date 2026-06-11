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
# ### 1.1 Working-set handle (sampler-locked)
#
# Tier 1/2/3 all join against `working_set_instance_ids`, the locked working set
# produced by the collection-level sampler
# (`src/features/sampling.py::build_working_set_sql`, the BigQuery production path
# of the unit-tested `build_working_set_google`). The table schema is
# `(collection_id, instance_index, schedule_time)`, where `schedule_time` is the
# instance's first SCHEDULE timestamp (microseconds, trace clock) and anchors the
# early-runtime window.
#
# The sampler retains **every** collection with at least one FAIL/LOST instance
# (full failure retention, P01/V02) and stratified-samples the successful
# collections by their modal `(priority_tier, scheduling_class)` so the instance
# marginals are preserved (V07). It reads the per-instance
# `instance_lifecycle_summary`, so the 1.72B-row events table is never scanned
# here.
#
# **Population reality (P01).** The eligible instance population (scheduled
# instances with a FINISH or FAIL/LOST terminal) is ~35M, which is below the
# `WORKING_SET_TARGET_M` million target and below the 50M P01 floor. With the
# target exceeding the population, the sampler returns the *full* eligible
# population (no subsampling): failure retention is then total and the success
# marginals are preserved exactly. Section 1.1a confirms that full-population
# retention.
#
# This working set scopes the **instance-grain** Tier 1/2/3 matrices only. The
# RQ1 modeling matrix lives at the **episode grain** and is a full census built
# from `instance_events_labeled` over all instances (Section 11), not from this
# working set, so it is unaffected by the lock. The 50-100M P01 figure is a
# modeling-row commitment and is verified at the episode grain in Section 11.3,
# where the census lands at ~90M episodes, inside the band. (For reference, the
# ~35M working-set instances themselves carry ~41M episodes, a smaller and
# distinct quantity reported in the manifest.)

# %%
from dataclasses import asdict
import json

from src.features.sampling import SamplingManifest, build_working_set_sql

WORKING_SET_TABLE = 'working_set_instance_ids'
WORKING_SET_TARGET_M = 75                        # P01 midpoint of the 50-100M band.
WORKING_SET_TARGET_INSTANCES = int(WORKING_SET_TARGET_M * 1_000_000)
REBUILD_WORKING_SET = False                      # set True to re-lock from scratch.
WORKING_SET_MANIFEST_PATH = TABLES_DIR / 'google_working_set_manifest.json'

if table_exists(WORKING_SET_TABLE) and not REBUILD_WORKING_SET:
    print(f"Using existing locked working set: {fqn(WORKING_SET_TABLE)}")
else:
    print(
        f"Locking the working set via the collection-level sampler "
        f"(target {WORKING_SET_TARGET_INSTANCES:,} instances; full failure retention, "
        "stratified success sampling)."
    )
    run_ddl(
        build_working_set_sql(
            fqn('instance_lifecycle_summary'),
            fqn(WORKING_SET_TABLE),
            target_instances=WORKING_SET_TARGET_INSTANCES,
        ),
        f"Build working set (sampler) -> {WORKING_SET_TABLE}",
    )

n_working_set = row_count(WORKING_SET_TABLE)
print(f"Working-set instances: {n_working_set:,}")
# The 50-100M P01 band is a modeling-row commitment verified at the episode grain
# in Section 11.3 (the RQ1 episode census is built independently of this working
# set). At the instance grain the eligible population is ~35M (below the band), so
# the sampler returns the full population rather than subsampling; Section 1.1a
# asserts that full-population retention.

# %% [markdown]
# #### 1.1a Working-set composition manifest and verification
#
# Build the `SamplingManifest` from BigQuery aggregates and assert the two sampler
# contract properties before the working set is used downstream: (1) full failure
# retention (every FAIL/LOST instance is kept), and (2) the successful instances'
# `(priority_tier, scheduling_class)` marginals are preserved within 2%. The
# per-stratum table is written into the manifest for traceability. The aggregates
# scan only the per-instance lifecycle summary, never the raw events.

# %%
# Priority band CASE on the lifecycle summary (mirrors the sampler / V07 bands).
_WS_TIER_CASE = f"""CASE
        WHEN terminal_priority <= {PRIORITY_FREE_MAX} THEN 'free'
        WHEN terminal_priority BETWEEN {PRIORITY_BEST_EFFORT_LOW} AND {PRIORITY_BEST_EFFORT_MAX} THEN 'best_effort'
        WHEN terminal_priority BETWEEN {PRIORITY_MID_TIER_LOW} AND {PRIORITY_MID_TIER_MAX} THEN 'mid'
        WHEN terminal_priority BETWEEN {PRIORITY_PRODUCTION_LOW} AND {PRIORITY_PRODUCTION_MAX} THEN 'production'
        WHEN terminal_priority >= {PRIORITY_MONITORING_LOW} THEN 'monitoring'
        ELSE 'unknown' END"""

_WS_INST_CTE = f"""
WITH inst AS (
    SELECT
        s.collection_id, s.instance_index,
        {_WS_TIER_CASE} AS priority_tier,
        s.terminal_scheduling_class AS scheduling_class,
        IF(s.outcome = 'FAIL_LOST', 1, 0) AS is_failure,
        s.schedule_count AS n_episodes,
        (w.collection_id IS NOT NULL) AS in_ws
    FROM {fqn('instance_lifecycle_summary')} s
    LEFT JOIN {fqn(WORKING_SET_TABLE)} w USING (collection_id, instance_index)
    WHERE s.first_schedule_time IS NOT NULL AND s.outcome IN ('FAIL_LOST', 'FINISH')
),
coll AS (
    SELECT
        collection_id,
        MAX(is_failure) AS has_failure,
        MAX(IF(in_ws, 1, 0)) AS in_ws_coll,
        COUNT(*) AS n_instances,
        SUM(IF(in_ws, 1, 0)) AS n_in_ws,
        SUM(IF(in_ws, n_episodes, 0)) AS episodes_in_ws
    FROM inst
    GROUP BY collection_id
)
"""

ws_agg = run_query(_WS_INST_CTE + """
SELECT
    COUNT(*) AS total_collections,
    COUNTIF(has_failure = 1) AS retained_failure_collections,
    COUNTIF(has_failure = 0 AND in_ws_coll = 1) AS sampled_success_collections,
    SUM(IF(has_failure = 1, n_instances, 0)) AS failure_instances,
    SUM(IF(has_failure = 1, n_in_ws, 0)) AS failure_instances_in_ws,
    SUM(n_instances) AS eligible_instances,
    SUM(n_in_ws) AS total_instances,
    SUM(episodes_in_ws) AS total_episodes
FROM coll
""")
ws_row = ws_agg.row(0, named=True)

ws_strata = run_query(_WS_INST_CTE + """
SELECT
    i.priority_tier,
    i.scheduling_class,
    COUNT(*) AS population_instances,
    COUNTIF(i.in_ws) AS sampled_instances
FROM inst i
JOIN coll c USING (collection_id)
WHERE c.has_failure = 0
GROUP BY i.priority_tier, i.scheduling_class
ORDER BY i.priority_tier, i.scheduling_class
""")

_pop_total = max(int(ws_strata["population_instances"].sum()), 1)
_samp_total = max(int(ws_strata["sampled_instances"].sum()), 1)
strata_rows = [
    {
        "priority_tier": r["priority_tier"],
        "scheduling_class": int(r["scheduling_class"]),
        "population_instances": int(r["population_instances"]),
        "sampled_instances": int(r["sampled_instances"]),
        "population_frac": int(r["population_instances"]) / _pop_total,
        "sampled_frac": int(r["sampled_instances"]) / _samp_total,
    }
    for r in ws_strata.iter_rows(named=True)
]

working_set_manifest = SamplingManifest(
    total_collections=int(ws_row["total_collections"]),
    retained_failure_collections=int(ws_row["retained_failure_collections"]),
    sampled_success_collections=int(ws_row["sampled_success_collections"]),
    total_instances=int(ws_row["total_instances"]),
    total_episodes=int(ws_row["total_episodes"]),
    target_instances=WORKING_SET_TARGET_INSTANCES,
    stratification=strata_rows,
)
with open(WORKING_SET_MANIFEST_PATH, "w") as f:
    json.dump(asdict(working_set_manifest), f, indent=2)
print(f"Working-set manifest: {WORKING_SET_MANIFEST_PATH}")
print(f"  collections: {working_set_manifest.total_collections:,} "
      f"(failure retained {working_set_manifest.retained_failure_collections:,}, "
      f"success sampled {working_set_manifest.sampled_success_collections:,})")
print(f"  instances: {working_set_manifest.total_instances:,} | "
      f"their episodes: {working_set_manifest.total_episodes:,} "
      "(carrier-population episode count, not the RQ1 episode-census matrix; "
      "that is sized in Section 11.3)")

# (1) Full failure retention: every FAIL/LOST instance is in the working set.
record_check(
    "Section 1.1a: full failure retention (all FAIL/LOST instances kept)",
    expected=int(ws_row["failure_instances"]),
    observed=int(ws_row["failure_instances_in_ws"]),
    ok=(int(ws_row["failure_instances_in_ws"]) == int(ws_row["failure_instances"])),
    notes="Sampler keeps every failure-containing collection in full (P01/V02).",
)

# (1b) Full-population retention: the eligible instance population (FINISH or
# FAIL/LOST, scheduled) is below the P01 floor, so the sampler returns all of it.
# Confirm no eligible instance was dropped (working set == eligible population).
record_check(
    "Section 1.1a: full eligible population retained (target exceeds population)",
    expected=int(ws_row["eligible_instances"]),
    observed=int(ws_row["total_instances"]),
    ok=(int(ws_row["total_instances"]) == int(ws_row["eligible_instances"])),
    notes="Eligible instance population (~35M) is below the 50M P01 floor; full "
          "population is used rather than subsampled.",
)

# Note: the P01 50-100M band is a modeling-row commitment verified at the episode
# grain in Section 11.3, against the episode-census row count. It is not asserted
# here because the RQ1 episode matrix is a full census built from
# `instance_events_labeled` over all instances, not from this working set. The
# manifest's `total_episodes` (below) is only the episode count belonging to the
# working-set instances (the instance-grain carrier), which is a different and
# smaller quantity than the episode-census modeling matrix.


# (2) Success marginals preserved within 2% (priority_tier and scheduling_class).
def _ws_marginal(rows: list[dict], key: str, frac: str) -> dict:
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0.0) + r[frac]
    return out


_pop_prio = _ws_marginal(strata_rows, "priority_tier", "population_frac")
_samp_prio = _ws_marginal(strata_rows, "priority_tier", "sampled_frac")
_pop_cls = _ws_marginal(strata_rows, "scheduling_class", "population_frac")
_samp_cls = _ws_marginal(strata_rows, "scheduling_class", "sampled_frac")
_ws_max_drift = max(
    [abs(_samp_prio.get(k, 0.0) - v) for k, v in _pop_prio.items()]
    + [abs(_samp_cls.get(k, 0.0) - v) for k, v in _pop_cls.items()]
    + [0.0]
)
record_check(
    "Section 1.1a: success (priority_tier, scheduling_class) marginals preserved within 2%",
    expected="max |sampled_frac - population_frac| <= 0.02",
    observed=f"max drift {_ws_max_drift:.4f}",
    ok=(_ws_max_drift <= 0.02),
    notes="Stratified success sampling (V07); trivially satisfied (drift ~0) while "
          "the full population is retained, and the active guard if a larger "
          "eligible population later triggers real subsampling.",
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
tier1_probe = tier1_out_lf.head(100_000).collect(engine="streaming")
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
# ---
# ## 11. Episode-grain reconstruction (per-attempt redesign)
#
# **Why this part exists.** Sections 2-10 build the matrix at the *instance*
# grain (one row per instance, terminal outcome as the label). Notebook 11's
# prediction-point ablation showed that grain leaks: an at-submission +
# lifecycle-history model scored MCC ~0.93, far above what pre-event signals
# can legitimately deliver, because the instance-grain history features
# (`prior_fail_count`, `resubmission_count`) are computed over the *whole*
# lifecycle, including the very resubmissions that lead to the terminal label.
# The history therefore peeks at the outcome.
#
# The fix is to model at the *scheduled-episode* grain. An episode is one
# `sched_seq` group: the events from a SCHEDULE up to (but not into) the next
# SCHEDULE. Each scheduled run gets its own row, its own terminal, and history
# computed **strictly from prior episodes only**. This removes the peek and, as
# notebook 11b confirmed, also rebalances the classes (episode-grain neg:pos is
# ~4.6:1 versus ~78:1 at the instance grain).
#
# **What 11b established (locked rules).**
# - Episode = `sched_seq` group with at least one SCHEDULE. `sched_seq` is the
#   running count of SCHEDULE events within an instance (the construction from
#   notebook 11b Section 3).
# - Terminal = first terminal-type event in the group. FAIL/LOST -> positive,
#   FINISH -> negative, EVICT/KILL excluded (V01, V08, V27). 99.1% of failures
#   are post-schedule, so the episode grain captures them cleanly.
# - Open episodes (scheduled, no terminal before the trace ends; ~1.2%) are
#   dropped as right-censored trace-boundary truncation.
# - Multi-terminal episodes (~5.4%) take the first terminal.
# - FINISH "doubling" is not redundancy: 99.2% of finishing instances finish
#   exactly once, and the recurring tail (0.8%, up to 1,894 finishes) finishes
#   once *per distinct `sched_seq`*, i.e. genuine separate scheduled runs. They
#   become legitimate separate negatives. The only safeguards they require are
#   modeling-stage, not here: a per-instance negative cap and a group-aware
#   train/test split by instance key (Section 11.5, applied in notebook 11).
#
# **Grain change.** The episode key is
# `(collection_id, instance_index, sched_seq)`. History becomes strictly-prior
# cumulative counts over an instance's earlier episodes. Static submission
# attributes (cpu_request, memory_request, priority, scheduling_class) are
# constant within an instance and broadcast to each episode; the per-episode
# scheduled machine and schedule time come from the episode's own SCHEDULE
# event, and the per-episode queue time uses the SUBMIT that initiated that
# attempt.
#
# **Scope of this part (Phase A).** Build the leakage-free episode Tier 1
# matrix: segmentation, label, strictly-prior history, per-episode scheduling /
# temporal features, exported for notebook 11 to re-run the prediction-point
# ablation on. The Tier 2 / Tier 3 usage rewire to episode grain (each episode
# carries its own +/-60s and 0..60min usage windows, assigned by schedule
# interval) is Phase B and is documented as the next step in Section 11.6.
#
# **Outputs.**
# - `{PROJECT}.dissertation_lebel.episode_segments_history` (BigQuery, every
#   `sched_seq >= 1` episode with terminal + strictly-prior history; all
#   terminal types retained so history counts prior evicts/kills too).
# - `{PROJECT}.dissertation_lebel.episode_lifecycle_features_base` (BigQuery,
#   modeling episodes only: label in {0, 1}, Tier 1 episode features).
# - `{OUTPUT_DIR}/features/google/episode_features_tier1.parquet` (Drive).

# %%
from src.data.schemas import EVENT_SCHEDULE, EVENT_SUBMIT, EVENT_FINISH, EVENT_KILL

EPISODE_EVENTS_TABLE = 'instance_events_labeled'
EPISODE_HISTORY_TABLE = 'episode_segments_history'
EPISODE_BASE_TABLE = 'episode_lifecycle_features_base'
EPISODE_GCS_PREFIX = f'{GCS_FEATURES_PREFIX}/episode_features_tier1'
EPISODE_MATRIX_PATH = FEATURES_DIR / 'episode_features_tier1.parquet'

TERMINAL_TYPES_SQL = f"{EVENT_EVICT}, {EVENT_FAIL}, {EVENT_FINISH}, {EVENT_KILL}, {EVENT_LOST}"

# Episode-grain checks live in their own list so the instance-grain
# verification log (Section 10) is left untouched.
episode_verification_rows: list[dict] = []


def record_episode_check(check: str, expected: object, observed: object, ok: bool, notes: str = "") -> None:
    """Append an episode-grain verification row and print a one-line summary."""
    status = "PASS" if ok else "FAIL"
    episode_verification_rows.append({
        "check": check, "expected": str(expected), "observed": str(observed),
        "ok": ok, "notes": notes,
    })
    suffix = f" ({notes})" if notes else ""
    print(f"  [{status}] {check}: expected {expected}, observed {observed}{suffix}")


# %% [markdown]
# ### 11.1 Segment episodes and attach strictly-prior history
#
# One BigQuery pass over `instance_events_labeled`:
# - `ev` tags every event with `sched_seq` (running SCHEDULE count) and a
#   running last-SUBMIT time, so each SCHEDULE row knows the submit that opened
#   its attempt.
# - `seg` collapses to one row per `(instance, sched_seq >= 1)` episode: the
#   episode's schedule time and scheduled machine (from its SCHEDULE event), the
#   attempt's submit time, and the first terminal event by time.
# - `hist` adds strictly-prior cumulative history with a window framed
#   `UNBOUNDED PRECEDING AND 1 PRECEDING` (the current episode is excluded, which
#   is exactly what removes the lifecycle peek). History is computed over *all*
#   episodes, including EVICT/KILL/open, so prior counts are complete; the
#   modeling filter to FAIL_LOST/FINISH happens later in Section 11.2.

# %%
build_history_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_HISTORY_TABLE)}
CLUSTER BY collection_id, instance_index AS
WITH ev AS (
    SELECT
        collection_id,
        instance_index,
        time,
        type,
        machine_id,
        COUNTIF(type = {EVENT_SCHEDULE}) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS sched_seq,
        -- Most recent SUBMIT at or before this event; at the SCHEDULE row this
        -- is the submit that initiated the attempt (per-episode queue anchor).
        MAX(IF(type = {EVENT_SUBMIT}, time, NULL)) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_last_submit
    FROM {fqn(EPISODE_EVENTS_TABLE)}
),
seg AS (
    SELECT
        collection_id,
        instance_index,
        sched_seq,
        -- The episode's SCHEDULE event anchors time, machine, and the attempt
        -- submit. MIN over the (single) SCHEDULE row in the group selects it.
        MIN(IF(type = {EVENT_SCHEDULE}, time, NULL))               AS schedule_time,
        MIN(IF(type = {EVENT_SCHEDULE}, machine_id, NULL))         AS scheduled_machine_id,
        MIN(IF(type = {EVENT_SCHEDULE}, running_last_submit, NULL)) AS attempt_submit_time,
        -- First terminal event by time within the episode (locked rule).
        ARRAY_AGG(
            IF(type IN ({TERMINAL_TYPES_SQL}), type, NULL)
            IGNORE NULLS ORDER BY time LIMIT 1
        )[SAFE_OFFSET(0)] AS terminal_type,
        COUNTIF(type IN ({TERMINAL_TYPES_SQL})) AS n_terminal_events
    FROM ev
    WHERE sched_seq >= 1
    GROUP BY collection_id, instance_index, sched_seq
)
SELECT
    collection_id,
    instance_index,
    sched_seq,
    schedule_time,
    scheduled_machine_id,
    attempt_submit_time,
    terminal_type,
    n_terminal_events,
    -- Strictly-prior history: window excludes the current episode.
    COUNT(*) OVER w                                              AS prior_episode_count,
    COUNTIF(terminal_type IN ({EVENT_FAIL}, {EVENT_LOST})) OVER w AS prior_fail_count,
    COUNTIF(terminal_type = {EVENT_FINISH}) OVER w               AS prior_finish_count,
    COUNTIF(terminal_type = {EVENT_EVICT}) OVER w                AS prior_evict_count
FROM seg
WINDOW w AS (
    PARTITION BY collection_id, instance_index
    ORDER BY sched_seq
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
"""
run_ddl(build_history_sql, f"Segment episodes + strictly-prior history -> {EPISODE_HISTORY_TABLE}")

n_episodes_all = row_count(EPISODE_HISTORY_TABLE)
print(f"Total scheduled episodes (sched_seq >= 1): {n_episodes_all:,}")

# Segmentation sanity: terminal composition and the open-episode share.
seg_audit = run_query(f"""
SELECT
    COUNT(*) AS n_episodes,
    COUNTIF(terminal_type IN ({EVENT_FAIL}, {EVENT_LOST})) AS n_fail_lost,
    COUNTIF(terminal_type = {EVENT_FINISH})                AS n_finish,
    COUNTIF(terminal_type = {EVENT_EVICT})                 AS n_evict,
    COUNTIF(terminal_type = {EVENT_KILL})                  AS n_kill,
    COUNTIF(terminal_type IS NULL)                         AS n_open,
    COUNTIF(n_terminal_events > 1)                         AS n_multi_terminal
FROM {fqn(EPISODE_HISTORY_TABLE)}
""")
print(seg_audit)
_open_frac = seg_audit["n_open"].item() / seg_audit["n_episodes"].item()
_multi_frac = seg_audit["n_multi_terminal"].item() / seg_audit["n_episodes"].item()
record_episode_check(
    "Section 11.1: open-episode share is small (right-censored, dropped)",
    expected="< 0.05 (11b observed ~0.012)",
    observed=round(_open_frac, 4),
    ok=(_open_frac < 0.05),
)
record_episode_check(
    "Section 11.1: multi-terminal share is small (first-terminal rule)",
    expected="< 0.10 (11b observed ~0.054)",
    observed=round(_multi_frac, 4),
    ok=(_multi_frac < 0.10),
)

# %% [markdown]
# ### 11.2 Build the modeling base (label + Tier 1 episode features)
#
# Filter to terminal in {FAIL, LOST, FINISH} (drops EVICT/KILL via V08/V27 and
# open episodes via the right-censoring rule), then attach the Tier 1 feature
# block:
# - **historical (strictly-prior):** `prior_fail_count`, `prior_evict_count`,
#   `resubmission_count` (= prior episode count), `has_prior_fail`,
#   `first_resubmission`. These are now leakage-free by construction.
# - **scheduling (V07):** static submission attributes from
#   `instance_lifecycle_summary`, per-episode `queue_time`
#   (schedule_time - attempt_submit_time), `priority_tier` and `platform`
#   one-hots. `platform` uses the episode's own scheduled machine.
# - **temporal (V26):** computed from the episode's schedule time via the same
#   naive-anchor convention as Section 5 (anchor 2019-05-01, minus the 600s
#   pre-trace offset). Columns keep the `submit_*` names so notebook 11's
#   feature lists carry over; they are per-attempt schedule-time features.
#
# One-hot column sets reuse the Section 4/5 enumerations
# (`PRIORITY_TIER_LEVELS`, `platform_ids`, `PLATFORM_SUFFIX`) so the episode
# matrix is column-compatible with the instance matrix.

# %%
# Build one-hot and temporal SQL fragments from the enumerations defined in
# Sections 2/4 so the episode columns line up with the instance columns.
# Explicit, readable priority-tier band CASE -> one-hot (mirrors _priority_tier_expr).
# Uses submit_priority (the value at the instance's FIRST event), NOT
# terminal_priority. terminal_priority is the priority at the death event and
# leaks the outcome at an at-submission prediction point (NB12 Sec 3.2-3.4: the
# terminal-derived tiers near-deterministically encode failure, e.g. monitoring
# tier 99.4% FAIL). submit_priority is the leak-free at-submission attribute.
_priority_tier_case = f"""
    CASE
        WHEN s.submit_priority <= {PRIORITY_FREE_MAX} THEN 'free'
        WHEN s.submit_priority BETWEEN {PRIORITY_BEST_EFFORT_LOW} AND {PRIORITY_BEST_EFFORT_MAX} THEN 'best_effort'
        WHEN s.submit_priority BETWEEN {PRIORITY_MID_TIER_LOW} AND {PRIORITY_MID_TIER_MAX} THEN 'mid'
        WHEN s.submit_priority BETWEEN {PRIORITY_PRODUCTION_LOW} AND {PRIORITY_PRODUCTION_MAX} THEN 'production'
        WHEN s.submit_priority >= {PRIORITY_MONITORING_LOW} THEN 'monitoring'
        ELSE 'unknown'
    END
"""
_priority_onehot_sql = ",\n    ".join(
    f"IF(({_priority_tier_case.strip()}) = '{lvl}', 1, 0) AS priority_tier_{lvl}"
    for lvl in PRIORITY_TIER_LEVELS
)
_platform_onehot_sql = ",\n    ".join(
    f"IF(m.platform_id = '{pid}', 1, 0) AS platform_{PLATFORM_SUFFIX[pid]}"
    for pid in platform_ids
)

# Per-episode wall clock from the schedule time (naive-anchor; same convention
# as Section 5). EXTRACT components match the PDT census used by V26.
_ep_wall_sql = (
    f"TIMESTAMP_ADD(TIMESTAMP '2019-05-01 00:00:00', "
    f"INTERVAL CAST(h.schedule_time - {TRACE_PRE_OFFSET_US} AS INT64) MICROSECOND)"
)

build_episode_base_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_BASE_TABLE)}
CLUSTER BY collection_id, instance_index AS
WITH plat AS (
    SELECT machine_id, ANY_VALUE(platform_id) AS platform_id
    FROM {fqn('machine_events_full')}
    WHERE machine_id IS NOT NULL AND platform_id IS NOT NULL
    GROUP BY machine_id
),
base AS (
    SELECT
        h.*,
        {_ep_wall_sql} AS _sched_wall
    FROM {fqn(EPISODE_HISTORY_TABLE)} h
    WHERE h.terminal_type IN ({EVENT_FAIL}, {EVENT_LOST}, {EVENT_FINISH})
)
SELECT
    h.collection_id,
    h.instance_index,
    h.sched_seq,
    -- Label (V01): FAIL/LOST -> 1, FINISH -> 0.
    IF(h.terminal_type IN ({EVENT_FAIL}, {EVENT_LOST}), 1, 0) AS failure_label,
    IF(h.terminal_type IN ({EVENT_FAIL}, {EVENT_LOST}), 'FAIL_LOST', 'FINISH') AS outcome,
    -- Tier 1 historical (strictly-prior; leakage-free).
    h.prior_fail_count,
    h.prior_evict_count,
    h.prior_episode_count AS resubmission_count,
    IF(h.prior_fail_count > 0, 1, 0) AS has_prior_fail,
    IF(h.prior_episode_count >= 1, 1, 0) AS first_resubmission,
    -- Tier 1 scheduling (static submission attributes + per-episode queue time).
    s.cpu_request,
    s.memory_request,
    SAFE_DIVIDE(s.cpu_request, NULLIF(s.memory_request, 0)) AS request_ratio,
    CAST(s.submit_scheduling_class AS INT64) AS scheduling_class,  -- submit-time, not terminal (leak-free)
    SAFE_DIVIDE(h.schedule_time - h.attempt_submit_time, {MICROS_PER_SEC}) AS queue_time,
    COALESCE(hc.has_hardware_counters_majority, 0) AS has_hardware_counters,
    {_priority_onehot_sql},
    {_platform_onehot_sql},
    -- Tier 1 temporal (V26), from the episode's schedule wall clock.
    EXTRACT(HOUR FROM h._sched_wall) AS submit_hour_of_day,
    MOD(EXTRACT(DAYOFWEEK FROM h._sched_wall) + 5, 7) AS submit_day_of_week,
    SIN(2 * ACOS(-1) * EXTRACT(HOUR FROM h._sched_wall) / 24) AS submit_hour_sin,
    COS(2 * ACOS(-1) * EXTRACT(HOUR FROM h._sched_wall) / 24) AS submit_hour_cos,
    IF(EXTRACT(HOUR FROM h._sched_wall) BETWEEN 8 AND 17, 1, 0) AS submit_is_business_hours_pdt,
    IF(MOD(EXTRACT(DAYOFWEEK FROM h._sched_wall) + 5, 7) >= 5, 1, 0) AS submit_is_weekend
FROM base h
LEFT JOIN {fqn('instance_lifecycle_summary')} s
  USING (collection_id, instance_index)
LEFT JOIN {fqn('instance_hardware_counters_majority')} hc
  USING (collection_id, instance_index)
LEFT JOIN plat m
  ON m.machine_id = h.scheduled_machine_id
"""
run_ddl(build_episode_base_sql, f"Build episode Tier 1 base -> {EPISODE_BASE_TABLE}")

# %% [markdown]
# ### 11.3 Label balance and leakage guardrails
#
# Two checks. First, the episode-grain class balance should match 11b
# (neg:pos ~4.6:1, ~18% positive); a large drift means the segmentation or the
# terminal rule changed. Second, a direct leakage guard: at the *first* episode
# of every instance (`resubmission_count = 0`) the strictly-prior history must
# be all zeros. If any first episode shows prior history, the window frame is
# wrong and the peek is back.

# %%
balance = run_query(f"""
SELECT
    COUNT(*) AS n,
    COUNTIF(failure_label = 1) AS n_pos,
    COUNTIF(failure_label = 0) AS n_neg
FROM {fqn(EPISODE_BASE_TABLE)}
""")
n_ep = int(balance["n"].item())
n_pos = int(balance["n_pos"].item())
n_neg = int(balance["n_neg"].item())
pos_frac = n_pos / n_ep if n_ep else 0.0
neg_pos_ratio = (n_neg / n_pos) if n_pos else float("inf")
print(f"Episodes: {n_ep:,} | positive {n_pos:,} ({pos_frac:.3f}) | neg:pos {neg_pos_ratio:.2f}:1")
record_episode_check(
    "Section 11.3: episode-grain positive fraction matches 11b",
    expected="~0.18 (accept 0.10 - 0.30)",
    observed=round(pos_frac, 3),
    ok=(0.10 <= pos_frac <= 0.30),
)
record_episode_check(
    "Section 11.3: episode-grain neg:pos ratio matches 11b",
    expected="~4.6:1 (accept 2:1 - 8:1)",
    observed=round(neg_pos_ratio, 2),
    ok=(2.0 <= neg_pos_ratio <= 8.0),
)
# P01 working-set-size commitment (50-100M modeling rows), verified at the grain
# the RQ1 model actually fits on. The episode census is a full-population matrix
# (all instances, episodes with a FAIL/LOST/FINISH terminal), so no subsampling
# is applied; adequacy beyond the band is established by the P05 learning curve.
record_episode_check(
    "Section 11.3: episode census is in the P01 band (50-100M modeling rows)",
    expected="50,000,000 - 100,000,000 episodes (P01)",
    observed=n_ep,
    ok=(50_000_000 <= n_ep <= 100_000_000),
    notes="Full episode census (V30 per-attempt grain); P01 band re-cast from "
          "instance events to modeling rows. P05 learning curve confirms adequacy.",
)

leak_guard = run_query(f"""
SELECT
    COUNTIF(resubmission_count = 0
            AND (prior_fail_count > 0 OR prior_evict_count > 0
                 OR has_prior_fail = 1 OR first_resubmission = 1)) AS n_first_with_history
FROM {fqn(EPISODE_BASE_TABLE)}
""")
n_first_leak = int(leak_guard["n_first_with_history"].item())
record_episode_check(
    "Section 11.3: first episodes carry zero prior history (strictly-prior guard)",
    expected=0,
    observed=n_first_leak,
    ok=(n_first_leak == 0),
    notes="Any nonzero means the strictly-prior window frame leaked the current episode.",
)

# %% [markdown]
# ### 11.4 Export the episode Tier 1 matrix
#
# Export to GCS, then a pure streaming copy to the single Drive Parquet that
# notebook 11 will scan. No joins or aggregations in the copy, so memory is
# bounded regardless of episode count.

# %%
episode_uri = f"gs://{GCS_BUCKET}/{EPISODE_GCS_PREFIX}/"
export_episode_sql = f"""
EXPORT DATA OPTIONS(
    uri='{episode_uri}*.parquet',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT * FROM {fqn(EPISODE_BASE_TABLE)}
"""
run_ddl(export_episode_sql, f"Export episode Tier 1 matrix -> {episode_uri}")

EPISODE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
pl.scan_parquet(episode_uri).sink_parquet(str(EPISODE_MATRIX_PATH), compression="snappy")
print(f"Episode Tier 1 matrix written: {EPISODE_MATRIX_PATH}")

episode_verification_df = pl.DataFrame(episode_verification_rows)
EPISODE_VERIFICATION_CSV = TABLES_DIR / "google_episode_reconstruction_verification.csv"
episode_verification_df.write_csv(str(EPISODE_VERIFICATION_CSV))
print(f"Episode verification log: {EPISODE_VERIFICATION_CSV}")
print(episode_verification_df.select(["check", "ok"]))

n_episode_failed = episode_verification_df.filter(~pl.col("ok")).height
assert n_episode_failed == 0, "Resolve failed episode-grain assertions before modeling."

# %% [markdown]
# ### 11.5 Modeling-stage helpers: per-instance negative cap + group split
#
# These do NOT change the base table (per the decision to keep the full episode
# record for auditing). They are applied in notebook 11 when assembling the
# training frame:
#
# - **Per-instance negative cap.** The recurring tail (~1,419 instances, up to
#   1,894 finishes each) would otherwise flood the negative class with highly
#   correlated episodes from a handful of instances. Cap the number of negative
#   (FINISH) episodes kept per instance at `CAP_NEG_PER_INSTANCE` (default 5),
#   sampled deterministically. Positives are never capped.
# - **Group-aware split.** Split train/test by instance key, never by episode,
#   so no instance's episodes straddle the split (which would let a recurring
#   instance's near-identical episodes leak across the boundary).
#
# Both are pure-Polars and deterministic (seed P14 = 42).

# %%
CAP_NEG_PER_INSTANCE = 5
SEED_P14 = 42


def cap_negative_episodes(lf: pl.LazyFrame, cap: int = CAP_NEG_PER_INSTANCE,
                          seed: int = SEED_P14) -> pl.LazyFrame:
    """Keep all positive episodes; keep at most `cap` negative episodes per
    instance, chosen by a deterministic per-instance hash ordering. Controls the
    recurring-instance negative flood without touching the base table.
    """
    ranked = lf.with_columns(
        # Deterministic per-episode order within an instance via a hashed key.
        (pl.col("collection_id").cast(pl.Utf8) + "_"
         + pl.col("instance_index").cast(pl.Utf8) + "_"
         + pl.col("sched_seq").cast(pl.Utf8) + f"_{seed}").hash().alias("_ep_hash"),
    ).with_columns(
        pl.col("_ep_hash").rank("ordinal").over(["collection_id", "instance_index"]).alias("_neg_rank"),
    )
    keep = (pl.col("failure_label") == 1) | (pl.col("_neg_rank") <= cap)
    return ranked.filter(keep).drop(["_ep_hash", "_neg_rank"])


def group_train_test_split(lf: pl.LazyFrame, test_frac: float = 0.2,
                           seed: int = SEED_P14) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split episodes into train/test by INSTANCE key (group-aware), so an
    instance's episodes never straddle the split. Deterministic via a hash of
    the instance key into [0, 1).
    """
    keyed = lf.with_columns(
        ((pl.col("collection_id").cast(pl.Utf8) + "_"
          + pl.col("instance_index").cast(pl.Utf8) + f"_{seed}").hash() % 1_000_000 / 1_000_000)
        .alias("_grp_u")
    )
    train = keyed.filter(pl.col("_grp_u") >= test_frac).drop("_grp_u")
    test = keyed.filter(pl.col("_grp_u") < test_frac).drop("_grp_u")
    return train, test


# Quick demonstration on the written matrix (bounded slice for the probe).
_episode_lf = pl.scan_parquet(str(EPISODE_MATRIX_PATH))
_capped_pos = _episode_lf.pipe(cap_negative_episodes).filter(pl.col("failure_label") == 1).select(pl.len()).collect().item()
_capped_neg = _episode_lf.pipe(cap_negative_episodes).filter(pl.col("failure_label") == 0).select(pl.len()).collect().item()
print(f"After per-instance negative cap ({CAP_NEG_PER_INSTANCE}): "
      f"pos={_capped_pos:,} neg={_capped_neg:,} neg:pos={_capped_neg / max(_capped_pos, 1):.2f}:1")

# %% [markdown]
# ### 11.6 Next step (Phase B): Tier 2 / Tier 3 at episode grain
#
# Phase A delivers the leakage-free Tier 1 episode matrix. To bring the runtime
# and windowed-utilization tiers to the episode grain, rewire Sections 7-8 so
# the usage joins key on the episode:
# - Replace the single per-instance `schedule_time` with the per-episode
#   `schedule_time` from `episode_segments_history`, and assign each usage
#   observation to the episode whose `[schedule_time, next_schedule_time)`
#   interval contains it (a join on instance key plus an interval predicate, or
#   an `ASOF`-style match), before applying the +/-60s (Tier 2) and 0..60min
#   (Tier 3) windows.
# - Aggregate slopes/ramps/windows per `(collection_id, instance_index,
#   sched_seq)` instead of per instance, then left-join onto
#   `episode_lifecycle_features_base`.
# Then re-run notebook 11's prediction-point ablation on the full episode
# matrix; the at-submission + strictly-prior-history MCC should fall from the
# leaked ~0.93 to a defensible level, confirming the redesign.

# %% [markdown]
# ---
# ## 12. Episode-grain Tier 2 / Tier 3 (Phase B) + full episode matrix
#
# Phase A (Section 11) delivered the leakage-free episode Tier 1 matrix.
# Notebook 11 Section 3.8 confirmed the fix: at the episode grain the
# submission+history MCC drops by ~0.27 from the leaked instance-grain value,
# and strictly-prior history adds almost nothing once it cannot see the label.
# Phase B brings the runtime (Tier 2) and windowed-utilization (Tier 3) tiers to
# the episode grain so the at-scheduling and early-runtime prediction points
# (where the RQ1 >0.90 target is actually tested) can be evaluated.
#
# **Usage-to-episode assignment.** The instance-grain Sections 7-8 used a single
# per-instance `schedule_time` and a symmetric +/-60s band. At the episode grain
# a recurring instance has many schedules, and a naive band would let one usage
# observation fall into two episodes' windows (cross-episode contamination). The
# fix is interval assignment: each usage observation belongs to exactly the
# episode whose half-open schedule interval `[schedule_time, next_schedule_time)`
# contains it. Tier 2 keeps the first 60s of that interval; Tier 3 keeps up to
# 60min, clipped at the next schedule. For rapidly-resubmitting tasks the Tier 3
# window is therefore legitimately short (the next run's usage is never
# attributed to this episode), which is the honest behavior.
#
# Everything reuses the validated slope (`_slope_sql`) and windowed-aggregate
# (`_windowed_agg_sql`) builders from Sections 7-8; only the grain changes
# (partition / group / join keys gain `sched_seq`).
#
# **Outputs.**
# - `{PROJECT}.dissertation_lebel.episode_schedule_intervals`
# - `{PROJECT}.dissertation_lebel.episode_usage_working_set` (Tier 2 subset)
# - `{PROJECT}.dissertation_lebel.episode_usage_working_set_t3` (Tier 3 subset)
# - `{PROJECT}.dissertation_lebel.episode_runtime_features` (Tier 2 per episode)
# - `{PROJECT}.dissertation_lebel.episode_tier3_features` (Tier 3 per episode)
# - `{PROJECT}.dissertation_lebel.episode_features` (full episode matrix)
# - `{OUTPUT_DIR}/features/google/episode_features.parquet` (Drive)

# %%
EPISODE_INTERVALS_TABLE = 'episode_schedule_intervals'
EPISODE_USAGE_T2_TABLE = 'episode_usage_working_set'
EPISODE_USAGE_T3_TABLE = 'episode_usage_working_set_t3'
EPISODE_RUNTIME_TABLE = 'episode_runtime_features'
EPISODE_TIER3_TABLE = 'episode_tier3_features'
EPISODE_FEATURES_TABLE = 'episode_features'
EPISODE_FULL_GCS_PREFIX = f'{GCS_FEATURES_PREFIX}/episode_features'

# %% [markdown]
# ### 12.1 Per-episode schedule intervals
#
# `next_schedule_time` is the same instance's next episode schedule (NULL on the
# last episode -> open-ended). Built from `episode_segments_history` so every
# scheduled run (including EVICT/KILL/open) bounds the interval, which is correct
# for assigning usage to the run that was actually executing.

# %%
build_intervals_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_INTERVALS_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    sched_seq,
    schedule_time,
    LEAD(schedule_time) OVER (
        PARTITION BY collection_id, instance_index ORDER BY sched_seq
    ) AS next_schedule_time
FROM {fqn(EPISODE_HISTORY_TABLE)}
"""
run_ddl(build_intervals_sql, f"Build episode schedule intervals -> {EPISODE_INTERVALS_TABLE}")

# %% [markdown]
# ### 12.2 Tier 2 episode usage subset + slope/ramp features
#
# Stage 1 assigns each usage observation to its containing episode interval and
# keeps the first 60s post-schedule. Stage 2 reuses `_slope_sql` with the grain
# extended to `(collection_id, instance_index, sched_seq)`.

# %%
ep_stage1_t2_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_USAGE_T2_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    u.*,
    e.sched_seq,
    e.schedule_time,
    SAFE_DIVIDE(u.start_time - e.schedule_time, {MICROS_PER_SEC}) AS sec_since_schedule
FROM {fqn('instance_usage_with_indicators')} u
INNER JOIN {fqn(EPISODE_INTERVALS_TABLE)} e
  USING (collection_id, instance_index)
WHERE u.start_time >= e.schedule_time
  AND (e.next_schedule_time IS NULL OR u.start_time < e.next_schedule_time)
  AND u.start_time <= e.schedule_time + {EARLY_RUNTIME_BAND_US}
"""
run_ddl(ep_stage1_t2_sql, f"Stage 1 (episode Tier 2) -> {EPISODE_USAGE_T2_TABLE}")

ep_stage2_t2_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_RUNTIME_TABLE)}
CLUSTER BY collection_id, instance_index AS
WITH ranked AS (
    SELECT
        collection_id, instance_index, sched_seq,
        sec_since_schedule, avg_cpu, avg_memory,
        cycles_per_instruction, memory_accesses_per_instruction,
        has_cpi_value, has_mapi_value,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
        ) AS rn_post,
        LAG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
        ) AS prev_avg_cpu,
        LAG(avg_memory) OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
        ) AS prev_avg_memory,
        AVG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        ) AS first_window_avg_cpu
    FROM {fqn(EPISODE_USAGE_T2_TABLE)}
    WHERE sec_since_schedule >= 0
),
agg AS (
    SELECT
        collection_id, instance_index, sched_seq,
        {_slope_sql('avg_cpu', 5)}  AS cpu_slope_5s,
        {_slope_sql('avg_cpu', 15)} AS cpu_slope_15s,
        {_slope_sql('avg_cpu', 30)} AS cpu_slope_30s,
        {_slope_sql('avg_memory', 5)}  AS memory_slope_5s,
        {_slope_sql('avg_memory', 15)} AS memory_slope_15s,
        {_slope_sql('avg_memory', 30)} AS memory_slope_30s,
        MAX(IF(rn_post = 2, avg_cpu - prev_avg_cpu, NULL))       AS initial_cpu_ramp,
        MAX(IF(rn_post = 2, avg_memory - prev_avg_memory, NULL)) AS initial_memory_ramp,
        MAX(IF(rn_post = 1, first_window_avg_cpu, NULL))         AS first_interval_avg_cpu,
        MAX(IF(rn_post = 1, cycles_per_instruction, NULL))          AS first_cpi,
        MAX(IF(rn_post = 1, memory_accesses_per_instruction, NULL)) AS first_mapi,
        MAX(IF(rn_post = 1, has_cpi_value, NULL))                AS first_has_cpi,
        MAX(IF(rn_post = 1, has_mapi_value, NULL))               AS first_has_mapi
    FROM ranked
    GROUP BY collection_id, instance_index, sched_seq
)
SELECT
    a.collection_id, a.instance_index, a.sched_seq,
    a.cpu_slope_5s, a.cpu_slope_15s, a.cpu_slope_30s,
    a.memory_slope_5s, a.memory_slope_15s, a.memory_slope_30s,
    a.initial_cpu_ramp, a.initial_memory_ramp,
    SAFE_DIVIDE(a.first_interval_avg_cpu, NULLIF(s.cpu_request, 0)) AS first_interval_util_ratio,
    IF(a.first_has_cpi  = 1, a.first_cpi,  NULL) AS cpi_value,
    IF(a.first_has_mapi = 1, a.first_mapi, NULL) AS mapi_value
FROM agg a
LEFT JOIN {fqn('instance_lifecycle_summary')} s
  USING (collection_id, instance_index)
"""
run_ddl(ep_stage2_t2_sql, f"Stage 2 (episode Tier 2) -> {EPISODE_RUNTIME_TABLE}")

# %% [markdown]
# ### 12.3 Tier 3 episode usage subset + windowed aggregates
#
# Same interval assignment, horizon 60min, clipped at the next schedule. Reuses
# `_windowed_agg_sql` with the grain extended to include `sched_seq`.

# %%
ep_stage1_t3_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_USAGE_T3_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    u.collection_id,
    u.instance_index,
    e.sched_seq,
    u.avg_cpu,
    u.avg_memory,
    SAFE_DIVIDE(u.start_time - e.schedule_time, {MICROS_PER_SEC}) AS sec_since_schedule
FROM {fqn('instance_usage_with_indicators')} u
INNER JOIN {fqn(EPISODE_INTERVALS_TABLE)} e
  USING (collection_id, instance_index)
WHERE u.start_time >= e.schedule_time
  AND (e.next_schedule_time IS NULL OR u.start_time < e.next_schedule_time)
  AND u.start_time <= e.schedule_time + {TIER3_MAX_WINDOW_US}
"""
run_ddl(ep_stage1_t3_sql, f"Stage 1 (episode Tier 3) -> {EPISODE_USAGE_T3_TABLE}")

ep_tier3_agg_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_TIER3_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    sched_seq,
    {_windowed_agg_sql()}
FROM {fqn(EPISODE_USAGE_T3_TABLE)}
GROUP BY collection_id, instance_index, sched_seq
"""
run_ddl(ep_tier3_agg_sql, f"Episode Tier 3 windowed aggregates -> {EPISODE_TIER3_TABLE}")

# %% [markdown]
# ### 12.4 Assemble the full episode matrix and export
#
# Left-join Tier 2 / Tier 3 onto the episode Tier 1 base on the episode key.
# Episodes with no in-band usage (rapid-onset crashes) carry null Tier 2/3,
# exactly as at the instance grain.

# %%
ep_tier2_select = ",\n    ".join(f"t2.{c}" for c in TIER2_FEATURE_COLS)
ep_tier3_select = ",\n    ".join(f"t3.{c}" for c in TIER3_FEATURE_COLS)

assemble_episode_sql = f"""
CREATE OR REPLACE TABLE {fqn(EPISODE_FEATURES_TABLE)}
CLUSTER BY collection_id, instance_index AS
SELECT
    t1.*,
    {ep_tier2_select},
    {ep_tier3_select}
FROM {fqn(EPISODE_BASE_TABLE)} t1
LEFT JOIN {fqn(EPISODE_RUNTIME_TABLE)} t2
  USING (collection_id, instance_index, sched_seq)
LEFT JOIN {fqn(EPISODE_TIER3_TABLE)} t3
  USING (collection_id, instance_index, sched_seq)
"""
run_ddl(assemble_episode_sql, f"Assemble full episode matrix -> {EPISODE_FEATURES_TABLE}")

n_episode_full = row_count(EPISODE_FEATURES_TABLE)
n_episode_base = row_count(EPISODE_BASE_TABLE)
print(f"Full episode matrix rows: {n_episode_full:,}")
record_episode_check(
    "Section 12.4: full episode matrix row count matches the episode base",
    expected=n_episode_base,
    observed=n_episode_full,
    ok=(n_episode_full == n_episode_base),
    notes="Tier 1 backbone with Tier 2/3 left joins; one row per modeling episode.",
)

# Tier 2 / Tier 3 availability (non-null fraction of a representative column).
ep_avail = run_query(f"""
SELECT
    COUNT(*) AS n,
    COUNTIF(cpu_slope_15s IS NOT NULL) AS n_t2,
    COUNTIF(avg_cpu_5min IS NOT NULL)  AS n_t3
FROM {fqn(EPISODE_FEATURES_TABLE)}
""")
_t2_avail = ep_avail["n_t2"].item() / ep_avail["n"].item()
_t3_avail = ep_avail["n_t3"].item() / ep_avail["n"].item()
print(f"Episode Tier 2 availability (cpu_slope_15s non-null): {_t2_avail:.3f}")
print(f"Episode Tier 3 availability (avg_cpu_5min non-null):  {_t3_avail:.3f}")
record_episode_check(
    "Section 12.4: Tier 2/3 are populated for a non-trivial share of episodes",
    expected="> 0 (rapid-onset crashes legitimately lack usage)",
    observed=f"t2={_t2_avail:.3f}, t3={_t3_avail:.3f}",
    ok=(_t2_avail > 0 and _t3_avail > 0),
)

ep_full_uri = f"gs://{GCS_BUCKET}/{EPISODE_FULL_GCS_PREFIX}/"
export_episode_full_sql = f"""
EXPORT DATA OPTIONS(
    uri='{ep_full_uri}*.parquet',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT * FROM {fqn(EPISODE_FEATURES_TABLE)}
"""
run_ddl(export_episode_full_sql, f"Export full episode matrix -> {ep_full_uri}")

# The full episode matrix is ~90M rows. Its durable copies are the BigQuery
# table and the GCS export above; notebook 11 Section 3.8 reads the BigQuery
# table directly, so no multi-GB Drive sink is written here (large Drive-FUSE
# writes routinely fail to flush before a Colab runtime recycles). Scan the GCS
# export with ``pl.scan_parquet(ep_full_uri)`` if a local copy is ever needed.
print(f"Full episode matrix is durable in BigQuery ({fqn(EPISODE_FEATURES_TABLE)}) "
      f"and GCS ({ep_full_uri}).")

# Re-write the episode verification log with the Phase B checks appended.
episode_verification_df = pl.DataFrame(episode_verification_rows)
episode_verification_df.write_csv(str(EPISODE_VERIFICATION_CSV))
print(f"Episode verification log (Phase A + B): {EPISODE_VERIFICATION_CSV}")
print(episode_verification_df.select(["check", "ok"]))
assert episode_verification_df.filter(~pl.col("ok")).height == 0, \
    "Resolve failed episode-grain assertions before modeling."

# %% [markdown]
# **Next steps.** Notebook 11 Section 3.8 reads the durable BigQuery
# `episode_features` table for the full prediction-point ablation (all five
# points, now including at-scheduling and early-runtime) and the honest RQ1
# curve; the at-submission floor is ~0.67
# MCC and the runtime tiers are where the >0.90 target is tested. Extract each
# tier section into the matching `src/features/*.py` module as pure
# LazyFrame -> LazyFrame functions; build `src/features/sampling.py` to produce
# the locked `working_set_instance_ids`; run the learning-curve harness and the
# formal Tier 3 inversion guard on this matrix (notebook 11).
