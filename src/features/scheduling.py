"""Tier 1 scheduling, priority, and platform features for Google Cluster Traces.

Tier 1 (pre-event) scheduling signals are available at or before scheduling
time and rank among the highest-value predictors. This module encodes the
Borg priority bands, the ordinal scheduling class, the one-hot machine
platform, the requested resources, and the queue time, all derived from the
lifecycle summary (``src/preprocessing/lifecycle.py``).

Validated in ``notebooks/10_feature_engineering_google.py`` Section 4 against
the working set before extraction into this module.

Cross-references (``outputs/tables/eda_decisions.csv``):
- V07 - machine / scheduling features.
- V11 - CPI/MAPI availability indicator (the Tier 1 ``has_hardware_counters``
  majority flag that gates the Tier 2 conditional counter values).

Priority bands are imported from ``src/data/schemas.py`` so this module stays
in lockstep with the preprocessing pass. Each function is a pure
``LazyFrame -> LazyFrame`` transform that performs no I/O.

``platform_id`` is obfuscated in the trace as a base64-style hash (containing
``+``, ``/``, ``=``), which is illegal as a BigQuery / Parquet field name, so
the one-hot columns use stable index suffixes (``platform_p0`` ...) produced
by :func:`platform_suffix_map`.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    PRIORITY_BEST_EFFORT_LOW,
    PRIORITY_BEST_EFFORT_MAX,
    PRIORITY_FREE_MAX,
    PRIORITY_MID_TIER_LOW,
    PRIORITY_MID_TIER_MAX,
    PRIORITY_MONITORING_LOW,
    PRIORITY_PRODUCTION_LOW,
    PRIORITY_PRODUCTION_MAX,
)

# Stable, fully-enumerated priority-tier level set (one-hot column suffixes).
PRIORITY_TIER_LEVELS: list[str] = ["free", "best_effort", "mid", "production", "monitoring"]


def priority_tier_expr() -> pl.Expr:
    """Return the expression mapping ``submit_priority`` to a band label.

    Bands follow Borg semantics (``src/data/schemas.py``): Free 0-99,
    Best-effort 100-115, Mid 116-119, Production 120-359, Monitoring 360+.
    Sourced from the submit-time priority (the value at the instance's first
    event), not terminal_priority, which is the death-time value and leaks the
    outcome at the prediction point (V35). Motivation: V07.
    """
    p = pl.col("submit_priority")
    return (
        pl.when(p <= PRIORITY_FREE_MAX).then(pl.lit("free"))
        .when((p >= PRIORITY_BEST_EFFORT_LOW) & (p <= PRIORITY_BEST_EFFORT_MAX)).then(pl.lit("best_effort"))
        .when((p >= PRIORITY_MID_TIER_LOW) & (p <= PRIORITY_MID_TIER_MAX)).then(pl.lit("mid"))
        .when((p >= PRIORITY_PRODUCTION_LOW) & (p <= PRIORITY_PRODUCTION_MAX)).then(pl.lit("production"))
        .when(p >= PRIORITY_MONITORING_LOW).then(pl.lit("monitoring"))
        .otherwise(pl.lit("unknown"))
        .alias("priority_tier")
    )


def platform_suffix_map(platform_ids: list[str]) -> dict[str, str]:
    """Map each (obfuscated) ``platform_id`` to a stable, field-name-safe
    suffix ``p0, p1, ...`` in sorted order.

    The mapping is deterministic for a fixed platform set, so the one-hot
    column names are stable across working-set samples. Record the inverse
    mapping alongside any artifact so the original ``platform_id`` remains
    recoverable.
    """
    return {pid: f"p{i}" for i, pid in enumerate(sorted(platform_ids))}


def platform_onehot_columns(platform_ids: list[str]) -> list[str]:
    """Return the enumerated one-hot platform column names for a platform set."""
    suffix = platform_suffix_map(platform_ids)
    return [f"platform_{suffix[pid]}" for pid in sorted(platform_ids)]


def add_scheduling_features(
    lf: pl.LazyFrame,
    platform_lookup: pl.LazyFrame,
    platform_ids: list[str],
) -> pl.LazyFrame:
    """Append the Tier 1 scheduling/priority/platform features.

    Expected input columns (from ``instance_lifecycle_summary``):
        - ``submit_priority`` (Int64), ``submit_scheduling_class`` (Int64),
          ``terminal_machine_id`` (Int64), ``cpu_request`` (Float64),
          ``memory_request`` (Float64), ``queue_time_sec`` (Float64).
          Priority and scheduling class use the submit-time values to avoid the
          terminal-sourcing leak (V35); terminal_machine_id is the scheduled
          machine, used only for the platform join.

    Args:
        lf: per-instance lifecycle LazyFrame.
        platform_lookup: ``(machine_id, platform_id)`` LazyFrame used to
            attach the platform of the scheduled machine.
        platform_ids: the full set of distinct ``platform_id`` values; drives
            the stable one-hot column enumeration via
            :func:`platform_suffix_map`.

    Appended output columns:
        - ``priority_tier`` (Utf8) plus one-hot ``priority_tier_{level}`` (Int8)
          for every level in :data:`PRIORITY_TIER_LEVELS`.
        - ``scheduling_class`` (Int64): ordinal 0-3.
        - ``platform_id`` (Utf8, raw) plus one-hot ``platform_{suffix}`` (Int8)
          for every platform in ``platform_ids``.
        - ``cpu_request``, ``memory_request`` (Float64): passthrough.
        - ``request_ratio`` (Float64): ``cpu_request / memory_request``,
          null when ``memory_request <= 0``.
        - ``queue_time`` (Float64): ``queue_time_sec`` (SUBMIT -> first
          SCHEDULE delta, seconds).

    Motivation: V07.
    """
    suffix = platform_suffix_map(platform_ids)

    out = (
        lf
        .with_columns(priority_tier_expr())
        .with_columns(
            pl.col("submit_scheduling_class").cast(pl.Int64).alias("scheduling_class"),
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
    for pid in sorted(platform_ids):
        out = out.with_columns(
            (pl.col("platform_id") == pid).fill_null(False).cast(pl.Int8)
                .alias(f"platform_{suffix[pid]}")
        )
    return out


def add_hardware_counter_flag(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append ``has_hardware_counters`` from the per-instance majority vote.

    Expected input column:
        - ``has_hardware_counters_majority`` (Int64): 1 when the instance's
          usage observations carry CPI or MAPI by majority vote (V11 / V28),
          0 or null otherwise.

    Appended output column:
        - ``has_hardware_counters`` (Int8): the majority flag, nulls filled
          to 0. This Tier 1 availability indicator gates the Tier 2 conditional
          ``cpi_value`` / ``mapi_value`` (see ``runtime.py``); the values are
          MNAR and must never be imputed.

    Motivation: V11 - workload-type-driven CPI/MAPI missingness (MNAR).
    """
    return lf.with_columns(
        pl.col("has_hardware_counters_majority").fill_null(0)
          .cast(pl.Int8).alias("has_hardware_counters"),
    )


def scheduling_feature_cols(platform_ids: list[str]) -> list[str]:
    """Return the full Tier 1 scheduling feature-column list for a platform set
    (excludes the raw ``priority_tier`` / ``platform_id`` label columns)."""
    return [
        "scheduling_class",
        "cpu_request",
        "memory_request",
        "request_ratio",
        "queue_time",
        *[f"priority_tier_{lvl}" for lvl in PRIORITY_TIER_LEVELS],
        *platform_onehot_columns(platform_ids),
    ]
