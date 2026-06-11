"""Polars schemas, sentinel constants, and priority-tier bands.

Schemas describe the cached BigQuery tables in
``{project}.dissertation_lebel.*_full`` as inspected in notebook 02 and
referenced throughout notebooks 03 and 08. Each schema lists the columns
expected after Polars receives them via either
``pl.from_pandas(bq_client.query(...).to_dataframe())`` or
``pl.scan_parquet`` over a BigQuery EXPORT DATA dump.

Sentinel constants come from F1 / V25 (notebook 07b Section 2; notebook
08 Section 1). Priority-tier bands and event-type codes come from
notebook 03 Section 7.2; the Production/Monitoring split is load-bearing
for the V27 monitoring exclusion and the P04 production-EVICT
sensitivity branch.
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Sentinel timestamps (V25).
#
# Rows where ``time`` equals ``SENTINEL_TIME_BEFORE`` (left-censoring marker,
# value 0) or ``SENTINEL_TIME_AFTER`` (right-censoring marker, 2**63 - 1)
# are dropped before any lifecycle reconstruction. See notebook 07b
# Section 2 for the F1 evidence: dropping shifts the FAIL_LOST class
# balance from 3.39:1 to ~3.43:1, which is negligible.
# ---------------------------------------------------------------------------
SENTINEL_TIME_BEFORE: int = 0
SENTINEL_TIME_AFTER: int = 2**63 - 1


# ---------------------------------------------------------------------------
# Priority-tier bands.
#
# These follow Google Borg's documented priority semantics. The
# Production / Monitoring split at 360 anchors the V27 monitoring
# exclusion and the P04 production-EVICT sensitivity branch.
# ---------------------------------------------------------------------------
PRIORITY_FREE_MAX: int = 99
PRIORITY_BEST_EFFORT_LOW: int = 100
PRIORITY_BEST_EFFORT_MAX: int = 115
PRIORITY_MID_TIER_LOW: int = 116
PRIORITY_MID_TIER_MAX: int = 119
PRIORITY_PRODUCTION_LOW: int = 120
PRIORITY_PRODUCTION_MAX: int = 359
PRIORITY_MONITORING_LOW: int = 360


# ---------------------------------------------------------------------------
# Event-type codes (instance_events.type).
#
# Names mirror the official Google Cluster Traces v3 release notes.
# Using named constants in preprocessing avoids magic numbers in the
# failure-label CASE expressions and makes regressions easier to spot.
# ---------------------------------------------------------------------------
EVENT_SUBMIT: int = 0
EVENT_QUEUE: int = 1
EVENT_ENABLE: int = 2
EVENT_SCHEDULE: int = 3
EVENT_EVICT: int = 4
EVENT_FAIL: int = 5
EVENT_FINISH: int = 6
EVENT_KILL: int = 7
EVENT_LOST: int = 8
EVENT_UPDATE_PENDING: int = 9
EVENT_UPDATE_RUNNING: int = 10


# ---------------------------------------------------------------------------
# Failure-label primitives (V01, V08).
#
# Consumed by both ``src/preprocessing/google_traces.py`` (label
# construction) and ``src/data/validation.py`` (regression assertions).
# ---------------------------------------------------------------------------
FAIL_LOST_TYPES: tuple[int, int] = (EVENT_FAIL, EVENT_LOST)
FINISH_TYPE: int = EVENT_FINISH
TERMINAL_TYPES: tuple[int, ...] = (
    EVENT_EVICT, EVENT_FAIL, EVENT_FINISH, EVENT_KILL, EVENT_LOST,
)


# ---------------------------------------------------------------------------
# Polars schemas for the cached BigQuery tables.
#
# INT64 columns round-trip as ``pl.Int64`` after the pandas conversion.
# The trace release uses STRUCT columns for resource_request,
# average_usage, maximum_usage, etc.; these are pre-flattened in the
# cached tables (see notebook 01). Repeated string columns (e.g.
# ``constraint``) appear as ``pl.List(pl.Utf8)``. Callers that do not
# need a column can omit it when constructing a LazyFrame.
# ---------------------------------------------------------------------------
INSTANCE_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Int64,
    "type": pl.Int64,
    "collection_id": pl.Int64,
    "instance_index": pl.Int64,
    "machine_id": pl.Int64,
    "alloc_collection_id": pl.Int64,
    "alloc_instance_index": pl.Int64,
    "scheduling_class": pl.Int64,
    "priority": pl.Int64,
    "cpu_request": pl.Float64,
    "memory_request": pl.Float64,
    "constraint": pl.List(pl.Utf8),
}

MACHINE_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Int64,
    "type": pl.Int64,
    "machine_id": pl.Int64,
    "platform_id": pl.Utf8,
    "capacity_cpus": pl.Float64,
    "capacity_memory": pl.Float64,
    "missing_data_reason": pl.Int64,
}

INSTANCE_USAGE_SCHEMA: dict[str, pl.DataType] = {
    "collection_id": pl.Int64,
    "instance_index": pl.Int64,
    "machine_id": pl.Int64,
    "alloc_collection_id": pl.Int64,
    "alloc_instance_index": pl.Int64,
    "start_time": pl.Int64,
    "end_time": pl.Int64,
    "avg_cpu": pl.Float64,
    "avg_memory": pl.Float64,
    "max_cpu": pl.Float64,
    "max_memory": pl.Float64,
    "sample_cpu": pl.Float64,
    "sample_memory": pl.Float64,
    "assigned_memory": pl.Float64,
    "page_cache_memory": pl.Float64,
    "cycles_per_instruction": pl.Float64,
    "memory_accesses_per_instruction": pl.Float64,
    "sample_rate": pl.Float64,
}

COLLECTION_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Int64,
    "type": pl.Int64,
    "collection_id": pl.Int64,
    "collection_type": pl.Int64,
    "priority": pl.Int64,
    "scheduling_class": pl.Int64,
    "parent_collection_id": pl.Int64,
    "alloc_collection_id": pl.Int64,
    "max_per_machine": pl.Int64,
    "max_per_switch": pl.Int64,
    "vertical_scaling": pl.Int64,
    "scheduler": pl.Int64,
    "missing_type": pl.Int64,
    "constraint": pl.List(pl.Utf8),
}

MACHINE_ATTRIBUTES_SCHEMA: dict[str, pl.DataType] = {
    "time": pl.Int64,
    "machine_id": pl.Int64,
    "name": pl.Utf8,
    "value": pl.Utf8,
    "deleted": pl.Int64,
}


# ---------------------------------------------------------------------------
# Post-preprocessing schemas.
#
# After notebook 08 Section 3, instance_events_labeled adds two label
# columns to the cached schema. After Section 4, instance_usage gains
# two per-observation indicator columns and the per-instance
# majority-vote summary is materialized as a separate table.
# ---------------------------------------------------------------------------
PREPROCESSED_INSTANCE_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    **INSTANCE_EVENTS_SCHEMA,
    "failure_label": pl.Int64,
    "failure_label_sensitivity_prod_evict": pl.Int64,
}

INSTANCE_USAGE_WITH_INDICATORS_SCHEMA: dict[str, pl.DataType] = {
    **INSTANCE_USAGE_SCHEMA,
    "has_cpi_value": pl.Int64,
    "has_mapi_value": pl.Int64,
}

INSTANCE_HARDWARE_COUNTERS_MAJORITY_SCHEMA: dict[str, pl.DataType] = {
    "collection_id": pl.Int64,
    "instance_index": pl.Int64,
    "n_obs": pl.Int64,
    "n_with_counters": pl.Int64,
    "frac_with_counters": pl.Float64,
    "has_hardware_counters_majority": pl.Int64,
}

INSTANCE_LIFECYCLE_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "collection_id": pl.Int64,
    "instance_index": pl.Int64,
    "total_events": pl.Int64,
    "submit_count": pl.Int64,
    "schedule_count": pl.Int64,
    "evict_count": pl.Int64,
    "fail_lost_count": pl.Int64,
    "submit_time": pl.Int64,
    "first_schedule_time": pl.Int64,
    "last_schedule_time": pl.Int64,
    "terminal_time": pl.Int64,
    "terminal_type": pl.Int64,
    "terminal_priority": pl.Int64,
    "terminal_scheduling_class": pl.Int64,
    "terminal_machine_id": pl.Int64,
    "submit_priority": pl.Int64,
    "submit_scheduling_class": pl.Int64,
    "cpu_request": pl.Float64,
    "memory_request": pl.Float64,
    "resubmission_count": pl.Int64,
    "queue_time_sec": pl.Float64,
    "running_duration_sec": pl.Float64,
    "total_lifecycle_sec": pl.Float64,
    "outcome": pl.Utf8,
}


# Convenience registry for table-name to schema lookups.
CACHED_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "instance_events_full": INSTANCE_EVENTS_SCHEMA,
    "machine_events_full": MACHINE_EVENTS_SCHEMA,
    "instance_usage_full": INSTANCE_USAGE_SCHEMA,
    "collection_events_full": COLLECTION_EVENTS_SCHEMA,
    "machine_attributes_full": MACHINE_ATTRIBUTES_SCHEMA,
}

PREPROCESSED_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "instance_events_labeled": PREPROCESSED_INSTANCE_EVENTS_SCHEMA,
    "instance_usage_with_indicators": INSTANCE_USAGE_WITH_INDICATORS_SCHEMA,
    "instance_hardware_counters_majority": INSTANCE_HARDWARE_COUNTERS_MAJORITY_SCHEMA,
    "instance_lifecycle_summary": INSTANCE_LIFECYCLE_SUMMARY_SCHEMA,
}
