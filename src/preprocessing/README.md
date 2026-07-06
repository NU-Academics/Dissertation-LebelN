# `src/preprocessing/` - Per-Event Transforms and Lifecycle Reconstruction

Three modules:

- `google_traces.py`: pure Polars LazyFrame transforms for the per-event Google preprocessing steps (sentinel filter, failure labeling, MNAR hardware-counter encoding, machine-attribute join).
- `lifecycle.py`: BigQuery-backed reconstruction of the per-instance lifecycle summary. Splits out from `google_traces.py` because the SUBMIT / SCHEDULE / terminal join runs across the full 1.72B-row events table and does not fit in Colab memory.
- `backblaze.py`: pure Polars LazyFrame transforms for the Backblaze row-level preprocessing (SSD exclusion, era assignment, SMART schema reconciliation, availability indicators, drive-model canonicalization, censoring marker).

## `google_traces.py`

Four functions. Each is pure: LazyFrame in, LazyFrame out, no I/O. Each maps to one or more rows in `outputs/tables/eda_decisions.csv`.

### `filter_sentinel_timestamps(lf, time_column="time")`

Drops rows where `time` equals 0 (left-censoring marker) or `2**63 - 1` (right-censoring marker). Implements `V25`. F1 (notebook 07b Section 2) confirmed sentinel rows are about 0.23% of `instance_events_full` and dropping them shifts the FAIL_LOST class balance from 3.39:1 to 3.43:1, which is negligible. Accepts `start_time` as the alternate column when applied to `instance_usage_full`.

### `apply_failure_label(lf, sensitivity_branch=None)`

Adds `failure_label` (primary, `V01`) and optionally `failure_label_sensitivity_prod_evict` (`P04`):

- Primary label: 1 where `type IN (5, 8)` (FAIL or LOST), 0 where `type = 6` (FINISH), NULL otherwise.
- Sensitivity branch `"prod_evict"`: additionally labels Production-priority EVICTs (`type = 4 AND 120 <= priority < 360`) as 1.

Monitoring-priority EVICTs (`priority >= 360`) are excluded from every failure label by construction (`V27`). The F3.2 repeats distribution showed those rows are canary / health-check preemptions rather than failures. KILL events (`type = 7`) are excluded by construction (`V08`): user-initiated cancellation is not predictable system behavior.

Raises `ValueError` when `sensitivity_branch` is not `None` or `"prod_evict"`.

### `encode_hardware_counters_mnar(lf, per_instance_majority=False, cpi_column=..., mapi_column=...)`

Adds the MNAR null-pattern indicators for CPI and MAPI:

- Per-observation (`V11`, always added): `has_cpi_value` and `has_mapi_value` are 1 where the underlying counter is non-null and 0 otherwise. The V11 record-level FINISH / FAIL_LOST null asymmetry (87.2% vs 26.8%) is preserved at this level and is what notebook 08 Section 4 verifies against the V11-exact triple-key join.
- Per-instance majority vote (`V28`, opt-in): `has_hardware_counters_majority` is 1 when at least half of the instance's observations have a non-null CPI value, 0 otherwise. F4 found that 39.84% of instances flip the indicator within their lifetime, so the per-instance vote is the recommended encoding for downstream feature engineering.

### `attach_machine_attributes(events_lf, attrs_lf, attribute_columns=None)`

For every `machine_id` in `attrs_lf`, takes the row with the largest `time` and joins it back onto `events_lf` on `machine_id`. Implements `V07`: include platform_id and capacity as machine-level features. Callers typically pass `machine_events_full` directly; the function also accepts a pre-pivoted `machine_attributes_full` LazyFrame so long as it is keyed by `machine_id` and carries a `time` column.

## `lifecycle.py`

### `reconstruct_instance_lifecycle(bq_client, project_id, dataset, source_table, output_table, gcs_bucket, gcs_prefix, skip_materialize_if_exists, expected_row_count_range)`

Three-step BigQuery-side reconstruction:

1. **Materialize**: `CREATE OR REPLACE TABLE {project}.{dataset}.{output_table}` clustered by `(collection_id, instance_index)`. The DDL is a single GROUP BY pass over the source events table with the sentinel filter applied inside the CTE (`V25`). Produces the per-instance summary documented by `INSTANCE_LIFECYCLE_SUMMARY_SCHEMA`.
2. **Verify**: counts the materialized rows and raises `ValueError` if outside `expected_row_count_range`. The default `(60M, 100M)` brackets the F4-confirmed ~74.5M unique-instance count with slack.
3. **Export and return**: `EXPORT DATA` writes partitioned Parquet to `gs://{bucket}/{prefix}/*.parquet`, then returns `pl.scan_parquet(prefix)` so callers receive a lazy scan rather than a 75M-row eager pull.

The output schema exposes `submit_time` (microseconds since trace start), satisfying the V26 contract for the temporal-feature derivation in `src/features/temporal.py`. It also exposes `submit_priority` and `submit_scheduling_class` (the priority and scheduling class at the instance's first event, `ARRAY_AGG(... ORDER BY time ASC LIMIT 1)`): these are the leak-free at-submission sources for the episode-grain `priority_tier` / `scheduling_class`, whereas the `terminal_*` counterparts (last event by time) are retained for outcome labeling only because the terminal value encodes the outcome (`V35`).

Implements `V09` (rapid-onset failure model), `V10` (resubmission history dominates), and `V29` (V10 reproduction caveats and full-trace contrast). The window-function pattern follows `sql/exploration/instance_lifecycle_reconstruction.sql` Part B; the validated full-trace adaptation lives in `notebooks/08_google_preprocessing.py` Section 5.

## `backblaze.py`

Six pure functions for the Backblaze row-level pass. Each is LazyFrame in, LazyFrame out, no I/O; the calling notebook (`09_backblaze_preprocessing.py`) composes them and owns reads and writes. The SMART schema is not stable across 2013-2025, so reconciliation guarantees a fixed column set and availability indicators mark where a non-universal column was actually collected rather than imputing it.

### `filter_hdds_only(lf, ssd_models=None, model_column="model")`

Drops SSD rows so the modeling axis is HDD-only (the 13-year temporal consistency required for the drift analysis). The daily schema has no drive-type flag, so SSDs are identified by an explicit model set sourced from notebook 05 and passed in by the caller (notebook 09 flags 18 model families by an `SSD` / `DELLBOSS` / `MTFDDAK` / `WDC WDS` token, keeping early Samsung SpinPoint HDDs that a bare-manufacturer match would drop). A `None` or empty set is a no-op.

### `assign_era(lf, era_constants=None, date_column="date", era_column="era")`

Annotates each row with the SMART schema era of its observation date from `BACKBLAZE_ERAS` (`src/data/schemas.py`, `V44`). Dates outside every era range receive `"unknown"` so the caller can triage them.

### `reconcile_smart_schema(lf, smart_ids, keep_normalized=True)`

Guarantees a `smart_{id}_raw` column (and optionally the `_normalized` sibling) for every requested SMART id, adding absent columns as all-null so every cleaned file shares one schema across the evolving years. Absent columns are added as null, never imputed.

### `encode_smart_availability_indicators(lf, smart_ids)`

Adds a `has_smart_{id}` indicator (1 where `smart_{id}_raw` is non-null) for each requested id (`V16`). Whether an era-gated attribute (notably 187 and 188) was collected is itself signal, so the model can use era-gated columns without imputing the periods that did not collect them.

### `canonicalize_drive_model(lf, aliases=None, model_column="model")`

Adds `model_canonical` (optional alias folding, validated in notebook 05) and `manufacturer` (derived from the canonical model-name prefix using the Backblaze naming conventions) (`V18`).

### `mark_censoring(lf, serial_column="serial_number", date_column="date", failure_column="failure")`

Classifies each drive's final observation for the survival framing: `is_last_obs`, `failure_observed` (terminal row with `failure = 1`), and `censored` (drive left the fleet without a recorded failure). Provided for per-drive-scale use; at full population scale notebook 09 computes the terminal table map-reduce style to bound memory.

The paired post-preprocessing assertions live in `src/data/validation.py` (`assert_failure_event_count`, `assert_one_row_per_drive_day`, `assert_era_assignment_complete`, `assert_fleet_expansion`).

## Tests

`tests/test_preprocessing.py` covers the four pure functions in `google_traces.py` and the six in `backblaze.py` (a three-era synthetic fixture exercises SSD filtering, era binning, schema reconciliation, availability indicators, the canonicalizer aliases, and the censoring marker, plus the four Backblaze validation asserts). `tests/test_lifecycle.py` covers the lifecycle reconstruction semantics via a Polars-native reference implementation; the BigQuery-backed path is validated end-to-end against the EDA-confirmed statistics in notebook 08.
