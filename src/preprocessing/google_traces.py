"""Per-event preprocessing transforms for Google Cluster Traces v3.

Pure Polars LazyFrame functions. No I/O. Each function takes one or more
LazyFrames in and returns a LazyFrame with the preprocessing transform
applied. The functions encode the Phase 2 and Phase 3 front-loaded
decisions captured in ``outputs/tables/eda_decisions.csv``.

Cross-references to the validated logic:
- V25: sentinel filtering (notebook 07b Section 2; notebook 08 Section 1).
- V01, V08, V27, P04: failure label construction (notebook 08 Section 3).
- V11, V28: MNAR encoding of CPI / MAPI counters (notebook 08 Section 4).
- V07: machine-attribute join (notebook 03 Section 7.6; Zhang et al., 2023).

The cross-event lifecycle reconstruction lives in
``src/preprocessing/lifecycle.py`` because that pass is best executed in
BigQuery against the full 1.72B-row events table.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FINISH,
    FAIL_LOST_TYPES,
    PRIORITY_MONITORING_LOW,
    PRIORITY_PRODUCTION_LOW,
    SENTINEL_TIME_AFTER,
    SENTINEL_TIME_BEFORE,
)

_VALID_SENSITIVITY_BRANCHES: tuple[str | None, ...] = (None, "prod_evict")


def filter_sentinel_timestamps(
    lf: pl.LazyFrame, time_column: str = "time",
) -> pl.LazyFrame:
    """Drop rows where ``time_column`` carries a sentinel value (V25).

    The Google trace uses ``0`` as a left-censoring marker and
    ``2**63 - 1`` as a right-censoring marker. F1 (notebook 07b
    Section 2) confirmed that sentinel-bearing rows are about 0.23% of
    instance_events_full, and dropping them shifts the FAIL_LOST class
    balance from 3.39:1 to 3.43:1, which is negligible.

    Args:
        lf: LazyFrame with a timestamp column.
        time_column: name of the timestamp column. Default ``time`` for
            instance_events; set to ``start_time`` when applying to
            instance_usage.

    Returns:
        LazyFrame with sentinel rows removed.
    """
    return lf.filter(
        (pl.col(time_column) > SENTINEL_TIME_BEFORE)
        & (pl.col(time_column) < SENTINEL_TIME_AFTER)
    )


def apply_failure_label(
    lf: pl.LazyFrame,
    sensitivity_branch: str | None = None,
) -> pl.LazyFrame:
    """Add failure label columns to an instance_events LazyFrame.

    **Primary label (V01).** Adds ``failure_label``:
        - 1 where ``type`` is FAIL (5) or LOST (8).
        - 0 where ``type`` is FINISH (6).
        - NULL otherwise.

    **Sensitivity branch ``"prod_evict"`` (P04).** When
    ``sensitivity_branch="prod_evict"``, also adds
    ``failure_label_sensitivity_prod_evict`` which additionally labels
    Production-priority EVICTs (``type = 4 AND 120 <= priority < 360``)
    as 1.

    Monitoring-priority EVICTs (``priority >= 360``) are excluded from
    every failure label by construction (V27). The F3.2 repeats
    distribution showed those rows are canary / health-check
    preemptions rather than failures.

    KILL events (``type = 7``) are excluded by construction (V08):
    user-initiated cancellation, not predictable system behavior.

    Args:
        lf: instance_events LazyFrame with ``type`` and ``priority``
            columns.
        sensitivity_branch: ``None`` (primary label only) or
            ``"prod_evict"`` (primary plus sensitivity column).

    Returns:
        LazyFrame with one or two new label columns.

    Raises:
        ValueError: when ``sensitivity_branch`` is not in the allowed
            set.
    """
    if sensitivity_branch not in _VALID_SENSITIVITY_BRANCHES:
        raise ValueError(
            f"sensitivity_branch must be one of {_VALID_SENSITIVITY_BRANCHES}; "
            f"got {sensitivity_branch!r}."
        )

    primary = (
        pl.when(pl.col("type").is_in(list(FAIL_LOST_TYPES)))
          .then(pl.lit(1).cast(pl.Int64))
          .when(pl.col("type") == EVENT_FINISH)
          .then(pl.lit(0).cast(pl.Int64))
          .otherwise(pl.lit(None).cast(pl.Int64))
          .alias("failure_label")
    )

    out = lf.with_columns(primary)

    if sensitivity_branch == "prod_evict":
        sensitivity = (
            pl.when(pl.col("type").is_in(list(FAIL_LOST_TYPES)))
              .then(pl.lit(1).cast(pl.Int64))
              .when(
                  (pl.col("type") == EVENT_EVICT)
                  & (pl.col("priority") >= PRIORITY_PRODUCTION_LOW)
                  & (pl.col("priority") < PRIORITY_MONITORING_LOW)
              )
              .then(pl.lit(1).cast(pl.Int64))
              .when(pl.col("type") == EVENT_FINISH)
              .then(pl.lit(0).cast(pl.Int64))
              .otherwise(pl.lit(None).cast(pl.Int64))
              .alias("failure_label_sensitivity_prod_evict")
        )
        out = out.with_columns(sensitivity)

    return out


def encode_hardware_counters_mnar(
    lf: pl.LazyFrame,
    per_instance_majority: bool = False,
    cpi_column: str = "cycles_per_instruction",
    mapi_column: str = "memory_accesses_per_instruction",
) -> pl.LazyFrame:
    """Encode the CPI / MAPI MNAR null pattern as feature indicators.

    **Per-observation indicators (V11; always added).**
    ``has_cpi_value`` and ``has_mapi_value`` are 1 where the underlying
    counter column is non-null and 0 otherwise. The 87.2% / 26.8%
    record-level FINISH / FAIL_LOST asymmetry is preserved at this
    level, which is what notebook 08 Section 4 verifies.

    **Per-instance majority vote (V28; opt-in).** When
    ``per_instance_majority=True``, ``has_hardware_counters_majority``
    is added: 1 when at least half of the instance's observations
    carry a non-null CPI value, 0 otherwise. The grouping key is
    ``(collection_id, instance_index)``. F4 found that 39.84% of
    instances have at least one mixed observation, so this opt-in is
    the recommended encoding for downstream feature engineering.

    Args:
        lf: instance_usage-shaped LazyFrame with the CPI / MAPI columns
            present and, when ``per_instance_majority=True``, the
            ``collection_id`` / ``instance_index`` keys.
        per_instance_majority: when True, append the majority-vote
            indicator.
        cpi_column: column name for cycles_per_instruction.
        mapi_column: column name for memory_accesses_per_instruction.

    Returns:
        LazyFrame with the indicator columns appended.
    """
    out = lf.with_columns([
        pl.col(cpi_column).is_not_null().cast(pl.Int64).alias("has_cpi_value"),
        pl.col(mapi_column).is_not_null().cast(pl.Int64).alias("has_mapi_value"),
    ])
    if per_instance_majority:
        out = out.with_columns(
            pl.col("has_cpi_value")
              .mean()
              .over(["collection_id", "instance_index"])
              .ge(0.5)
              .cast(pl.Int64)
              .alias("has_hardware_counters_majority")
        )
    return out


def attach_machine_attributes(
    events_lf: pl.LazyFrame,
    attrs_lf: pl.LazyFrame,
    attribute_columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Join the latest machine-level attributes onto an events LazyFrame (V07).

    For every ``machine_id`` present in ``attrs_lf``, the row with the
    largest ``time`` is kept and joined back into ``events_lf`` on
    ``machine_id``. Events whose machine_id is missing or has no entry
    in ``attrs_lf`` retain NULL attributes.

    V07 motivates including platform_id and capacity as machine-level
    features (notebook 03 Section 7.6; Zhang et al., 2023). The
    canonical attribute source is ``machine_events_full`` whose schema
    exposes ``platform_id``, ``capacity_cpus``, and ``capacity_memory``.
    Callers may also pass a pre-pivoted ``machine_attributes_full``
    LazyFrame as long as the table is keyed by ``machine_id`` and
    carries a ``time`` column.

    Args:
        events_lf: LazyFrame with a ``machine_id`` column.
        attrs_lf: machine-level LazyFrame with ``machine_id`` and
            ``time`` columns plus one or more attribute columns.
        attribute_columns: optional explicit list of attribute columns
            to attach. Defaults to every column in ``attrs_lf`` other
            than ``machine_id`` and ``time``.

    Returns:
        LazyFrame with the joined attribute columns. Row count matches
        ``events_lf``; no event rows are dropped.
    """
    if attribute_columns is None:
        attrs_columns = attrs_lf.collect_schema().names()
        attribute_columns = [
            c for c in attrs_columns if c not in {"machine_id", "time"}
        ]

    latest = (
        attrs_lf
        .filter(pl.col("machine_id").is_not_null())
        .sort(["machine_id", "time"], descending=[False, True])
        .group_by("machine_id")
        .agg([pl.col(c).first().alias(c) for c in attribute_columns])
    )
    return events_lf.join(latest, on="machine_id", how="left")
