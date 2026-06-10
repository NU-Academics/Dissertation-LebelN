"""BigQuery-backed per-instance lifecycle reconstruction.

Splits out from ``google_traces.py`` because the SUBMIT / SCHEDULE /
terminal join runs across the full 1.72B-row ``instance_events_full``
table and is best executed in BigQuery rather than in-process Polars.
The module materializes the per-instance summary as a clustered
BigQuery table, optionally exports it to GCS as Parquet, and returns
a Polars LazyFrame that scans the export.

Cross-references:
- V09 (rapid-onset failure model), V10 (resubmission history dominates),
  V29 (V10 reproduction caveats and full-trace contrast) all rely on
  the summary produced here.
- The window-function pattern follows
  ``sql/exploration/instance_lifecycle_reconstruction.sql`` Part B; the
  validated full-trace adaptation lives in notebook 08 Section 5.
- V26 requires ``submit_time`` to be exposed in the output schema so
  ``src/features/temporal.py`` can derive submit-PDT temporal features.
"""

from __future__ import annotations

import polars as pl
from google.cloud import bigquery

from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_KILL,
    EVENT_LOST,
    EVENT_SCHEDULE,
    EVENT_SUBMIT,
    PRIORITY_MONITORING_LOW,
    PRIORITY_PRODUCTION_LOW,
    SENTINEL_TIME_AFTER,
    SENTINEL_TIME_BEFORE,
)

# Default row-count window. F4 reported 74,490,982 unique
# (collection_id, instance_index) pairs; the band below leaves slack
# for the sentinel-filter drop and for re-runs against alternate
# source tables.
_DEFAULT_ROW_COUNT_RANGE: tuple[int, int] = (60_000_000, 100_000_000)


def reconstruct_instance_lifecycle(
    bq_client: bigquery.Client,
    project_id: str,
    dataset: str = "dissertation_lebel",
    source_table: str = "instance_events_labeled",
    output_table: str = "instance_lifecycle_summary",
    gcs_bucket: str | None = None,
    gcs_prefix: str = "google_preprocessed/instance_lifecycle_summary",
    skip_materialize_if_exists: bool = False,
    expected_row_count_range: tuple[int, int] = _DEFAULT_ROW_COUNT_RANGE,
) -> pl.LazyFrame:
    """Compute per-instance lifecycle features via BigQuery window functions.

    The function performs three steps:

    1. ``CREATE OR REPLACE TABLE`` materializes the per-instance summary
       as ``{project_id}.{dataset}.{output_table}`` clustered by
       ``(collection_id, instance_index)``. Sentinel-bearing rows are
       filtered inside the CTE (V25). Skipped when
       ``skip_materialize_if_exists=True``.
    2. Verifies the row count falls inside
       ``expected_row_count_range``. The default ``(60M, 100M)`` brackets
       the EDA-confirmed ~74.5M unique-instance count (F4) with slack on
       both sides. ``ValueError`` is raised on mismatch.
    3. ``EXPORT DATA`` writes the materialized table to GCS as
       partitioned Parquet at
       ``gs://{gcs_bucket}/{gcs_prefix}/*.parquet`` and a
       ``pl.scan_parquet`` LazyFrame over the prefix is returned. Lazy
       scanning keeps the ~75M-row summary out of Colab memory; the
       caller can ``.collect()`` a working subset after filtering.

    The output schema exposes ``submit_time`` (microseconds since trace
    start), satisfying the contract that
    ``src/features/temporal.py`` depends on (V26).

    Args:
        bq_client: authenticated BigQuery client.
        project_id: GCP project that owns ``dataset``.
        dataset: BigQuery dataset name. Defaults to ``dissertation_lebel``.
        source_table: cached preprocessed events table to read from.
            Default ``instance_events_labeled`` (produced by notebook 08
            Section 3, sentinel-filtered + label-augmented). Pass
            ``instance_events_full`` to reconstruct against the raw
            trace.
        output_table: name of the materialized summary table.
        gcs_bucket: GCS bucket for the Parquet export. When ``None``,
            defaults to ``{project_id}-dissertation-data`` to match the
            convention used by notebooks 04, 05, 06, and 08.
        gcs_prefix: prefix inside the bucket for the export.
        skip_materialize_if_exists: when True, skip the
            ``CREATE OR REPLACE`` step. Useful for re-running the
            verification and re-export without re-materializing.
        expected_row_count_range: tuple ``(low, high)``. The function
            verifies the materialized row count is inside this band and
            raises ``ValueError`` if not.

    Returns:
        Polars LazyFrame backed by ``pl.scan_parquet`` over the GCS
        prefix that holds the export.

    Raises:
        ValueError: when the materialized row count is outside
            ``expected_row_count_range``.
    """
    if gcs_bucket is None:
        gcs_bucket = f"{project_id}-dissertation-data"

    fq_source = f"`{project_id}.{dataset}.{source_table}`"
    fq_output = f"`{project_id}.{dataset}.{output_table}`"

    # Step 1. Materialize.
    if not skip_materialize_if_exists:
        bq_client.query(_build_lifecycle_ddl(fq_source, fq_output)).result()

    # Step 2. Verify row count against the F4 baseline.
    n_rows = int(
        bq_client.query(f"SELECT COUNT(*) AS n FROM {fq_output}")
                 .to_dataframe()["n"].iloc[0]
    )
    lo, hi = expected_row_count_range
    if not (lo <= n_rows <= hi):
        raise ValueError(
            f"reconstruct_instance_lifecycle: {output_table} row count "
            f"{n_rows:,} is outside the expected window [{lo:,}, {hi:,}]. "
            "Investigate the source table and the sentinel filter before "
            "trusting downstream features."
        )

    # Step 3. Export to GCS as Parquet and return a lazy scan.
    export_uri = f"gs://{gcs_bucket}/{gcs_prefix}/*.parquet"
    export_sql = f"""
EXPORT DATA OPTIONS(
    uri='{export_uri}',
    format='PARQUET',
    compression='SNAPPY',
    overwrite=true
) AS
SELECT * FROM {fq_output}
"""
    bq_client.query(export_sql).result()

    scan_prefix = f"gs://{gcs_bucket}/{gcs_prefix}/"
    return pl.scan_parquet(scan_prefix)


def _build_lifecycle_ddl(fq_source: str, fq_output: str) -> str:
    """Return the ``CREATE OR REPLACE TABLE`` DDL for the lifecycle summary.

    The DDL mirrors notebook 08 Section 5's inline SQL: a single
    GROUP BY per ``(collection_id, instance_index)`` that exposes
    submit / first-schedule / last-schedule / terminal timestamps,
    terminal-event metadata, derived durations (``queue_time_sec``,
    ``running_duration_sec``, ``total_lifecycle_sec``), and a
    categorical outcome derived from the terminal event's type and
    priority. The sentinel filter (V25) is applied inside the CTE so
    that downstream lifecycle features are computed only on valid
    timestamps.
    """
    return f"""
CREATE OR REPLACE TABLE {fq_output}
CLUSTER BY collection_id, instance_index AS
WITH per_instance AS (
    SELECT
        collection_id,
        instance_index,
        COUNT(*) AS total_events,
        COUNTIF(type = {EVENT_SUBMIT}) AS submit_count,
        COUNTIF(type = {EVENT_SCHEDULE}) AS schedule_count,
        COUNTIF(type = {EVENT_EVICT}) AS evict_count,
        COUNTIF(type IN ({EVENT_FAIL}, {EVENT_LOST})) AS fail_lost_count,
        MIN(IF(type = {EVENT_SUBMIT}, time, NULL)) AS submit_time,
        MIN(IF(type = {EVENT_SCHEDULE}, time, NULL)) AS first_schedule_time,
        MAX(IF(type = {EVENT_SCHEDULE}, time, NULL)) AS last_schedule_time,
        MAX(time) AS terminal_time,
        ARRAY_AGG(type ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_type,
        ARRAY_AGG(priority ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_priority,
        ARRAY_AGG(scheduling_class ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_scheduling_class,
        ARRAY_AGG(machine_id ORDER BY time DESC LIMIT 1)[OFFSET(0)] AS terminal_machine_id,
        -- Submission-time (FIRST event by time) priority / scheduling class:
        -- the leak-free at-submission values. terminal_* above are end-of-life
        -- attributes that encode the outcome and must not be used as submission
        -- features (see NB12 Sec 3.2-3.4 leak diagnosis; eda_decisions V33-leak).
        ARRAY_AGG(priority ORDER BY time ASC LIMIT 1)[OFFSET(0)] AS submit_priority,
        ARRAY_AGG(scheduling_class ORDER BY time ASC LIMIT 1)[OFFSET(0)] AS submit_scheduling_class,
        ANY_VALUE(cpu_request) AS cpu_request,
        ANY_VALUE(memory_request) AS memory_request
    FROM {fq_source}
    WHERE time > {SENTINEL_TIME_BEFORE}
      AND time < {SENTINEL_TIME_AFTER}
    GROUP BY collection_id, instance_index
),
with_durations AS (
    SELECT
        *,
        IF(submit_count > 1, submit_count - 1, 0) AS resubmission_count,
        SAFE_DIVIDE(first_schedule_time - submit_time, 1000000) AS queue_time_sec,
        SAFE_DIVIDE(terminal_time - last_schedule_time, 1000000) AS running_duration_sec,
        SAFE_DIVIDE(terminal_time - submit_time, 1000000) AS total_lifecycle_sec,
        CASE
            WHEN terminal_type IN ({EVENT_FAIL}, {EVENT_LOST}) THEN 'FAIL_LOST'
            WHEN terminal_type = {EVENT_FINISH} THEN 'FINISH'
            WHEN terminal_type = {EVENT_EVICT}
                 AND terminal_priority >= {PRIORITY_MONITORING_LOW} THEN 'EVICT_MONITORING'
            WHEN terminal_type = {EVENT_EVICT}
                 AND terminal_priority BETWEEN {PRIORITY_PRODUCTION_LOW}
                                          AND {PRIORITY_MONITORING_LOW - 1} THEN 'EVICT_PRODUCTION'
            WHEN terminal_type = {EVENT_EVICT} THEN 'EVICT_LOWER'
            WHEN terminal_type = {EVENT_KILL} THEN 'KILL'
            ELSE 'OTHER'
        END AS outcome
    FROM per_instance
)
SELECT * FROM with_durations
"""
