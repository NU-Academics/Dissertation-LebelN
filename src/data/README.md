# `src/data/` - Schemas, Constants, and Assertions

Two modules:

- `schemas.py`: Polars schemas for the five cached BigQuery tables plus the four preprocessed outputs, sentinel constants, priority-tier bands, event-type codes, and convenience registries keyed by table name.
- `validation.py`: assertion helpers used by the post-preprocessing assertion suite in `notebooks/08_google_preprocessing.py` Section 6 and by tests under `tests/test_preprocessing.py`.

## `schemas.py`

### Constants

The module exports the primitives every other Phase 3 module imports rather than reaching for magic numbers. Highlights:

- `SENTINEL_TIME_BEFORE = 0` and `SENTINEL_TIME_AFTER = 2**63 - 1` from the F1 sentinel inventory (`V25`).
- Priority-tier bands `PRIORITY_FREE_MAX` (99), `PRIORITY_BEST_EFFORT_LOW` / `MAX`, `PRIORITY_MID_TIER_LOW` / `MAX`, `PRIORITY_PRODUCTION_LOW` (120) / `MAX` (359), and `PRIORITY_MONITORING_LOW` (360). The Production / Monitoring split at 360 anchors the V27 monitoring-EVICT exclusion and the P04 production-EVICT sensitivity branch.
- Event-type codes `EVENT_SUBMIT` through `EVENT_UPDATE_RUNNING` corresponding to the official Google Cluster Traces v3 release.
- Failure-label primitives `FAIL_LOST_TYPES = (5, 8)`, `FINISH_TYPE = 6`, `TERMINAL_TYPES = (4, 5, 6, 7, 8)`.

### Schemas

Polars schemas for the five cached tables:

| Constant | Source table | Notes |
|----------|--------------|-------|
| `INSTANCE_EVENTS_SCHEMA` | `instance_events_full` | 12 columns. `constraint` is a `pl.List(pl.Utf8)`. |
| `MACHINE_EVENTS_SCHEMA` | `machine_events_full` | Includes `platform_id`, `capacity_cpus`, `capacity_memory`. |
| `INSTANCE_USAGE_SCHEMA` | `instance_usage_full` | 18 numeric columns; CPI and MAPI live here. |
| `COLLECTION_EVENTS_SCHEMA` | `collection_events_full` | Includes `collection_type`, `parent_collection_id`. |
| `MACHINE_ATTRIBUTES_SCHEMA` | `machine_attributes_full` | Long-format key/value pairs per machine. |

Plus four post-preprocessing schemas: `PREPROCESSED_INSTANCE_EVENTS_SCHEMA` (cached schema + failure labels), `INSTANCE_USAGE_WITH_INDICATORS_SCHEMA` (usage + `has_cpi_value`, `has_mapi_value`), `INSTANCE_HARDWARE_COUNTERS_MAJORITY_SCHEMA`, and `INSTANCE_LIFECYCLE_SUMMARY_SCHEMA`.

Two registries `CACHED_SCHEMAS` and `PREPROCESSED_SCHEMAS` let callers look up a schema by table name.

## `validation.py`

Assertions raise `AssertionFailedError` on tolerance violation and return the observed value on success so callers can log it. Each function is a single-purpose check against an EDA-confirmed invariant.

- `assert_class_balance(lf, expected_ratio, tolerance=0.05, label_column="failure_label")`. Confirms the negative:positive class ratio is within the tolerance band. The V02 baseline for the primary Google failure label is roughly 3.4:1.
- `assert_null_rate(lf, column, expected_rate, tolerance=0.01)`. Confirms the null rate of a column is within `+/- tolerance` of the expected fraction. Used for V04 and V11 cross-checks.
- `assert_tier3_inversion(lf, cpu_column="avg_cpu", label_column="failure_label")`. The regression guard for V12: failing instances must continue to exhibit lower median absolute CPU utilization than successful ones (FAIL_LOST median 0.012 vs FINISH 0.081 in Phase 2). If preprocessing or feature engineering accidentally washes out the inversion, the Chapter 4 Tier 3 ablation loses its empirical anchor.
- `assert_monitoring_evict_excluded(lf, label_columns=..., ...)`. The regression guard for V27: monitoring-priority EVICT rows (type 4, priority >= 360) must remain NULL under every failure label. The F3.2 repeats distribution is the load-bearing evidence for the exclusion.
- `assert_row_count(lf, expected_min, expected_max)`. Used by the lifecycle reconstruction guard and by section-level row-count checks in notebook 08.

## When to extend

Add a new schema or constant here when a notebook validates a new table layout or a new structural assumption. Add a new assertion here when a Validated decision in `eda_decisions.csv` would silently regress if preprocessing changed without notice. Keep helpers single-purpose; do not stack unrelated checks into one function.
