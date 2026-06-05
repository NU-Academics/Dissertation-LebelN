"""Tier 2 early-runtime features for Google Cluster Traces failure prediction.

Tier 2 (early-runtime) signals are of moderate expected value: failing
instances use *less* absolute CPU/memory than successful ones, but ramp 3.6x
faster on CPU and 2.3x faster on memory (the V12 utilization inversion). The
discriminative signal therefore lives in the rate-of-change, not the level.
This module derives the per-instance slopes, startup ramps, the
realized-vs-requested ratio, and the conditional hardware-counter values from
the early-runtime usage observations in a +/-60s band around schedule time.

Validated in ``notebooks/10_feature_engineering_google.py`` Section 7 before
extraction into this module.

Cross-references (``outputs/tables/eda_decisions.csv``):
- V12 - utilization inversion; rate-of-change carries the signal.
- V11 - CPI/MAPI are MNAR (workload-type-driven missingness); the values are
  kept only when present (conditioned on ``has_hardware_counters``) and are
  never imputed.

Two equivalent code paths are provided:

1. :func:`add_runtime_features` - a pure ``LazyFrame -> LazyFrame`` transform
   that computes the features in Polars from a per-observation usage
   LazyFrame. This is the canonical, unit-testable API; it is correct at any
   scale but materializes the group-by in process.

2. :func:`build_usage_working_set_sql` / :func:`build_runtime_features_sql` -
   BigQuery SQL builders capturing the **validated production path**. At the
   full working-set scale (~35M instances over a 7.5B-row usage source), the
   notebook runs the two-stage memory-bounded filter and computes the slopes
   BigQuery-side (``COVAR_POP / VAR_POP``, ``LAG``, ``AVG OVER``), exporting
   only the compact per-instance feature row. The Polars and SQL paths encode
   identical feature definitions (the OLS slope ``COVAR_POP/VAR_POP`` equals
   the closed-form ``(n*Sxy - Sx*Sy)/(n*Sxx - Sx^2)`` used here).
"""

from __future__ import annotations

import polars as pl

# Early-runtime window and slope horizons (seconds). V09 fixes the median
# FAIL_LOST running duration at 22.6s; a +/-60s band captures the full
# early-runtime signal, and the 5/15/30s slope horizons resolve the startup ramp.
EARLY_RUNTIME_BAND_US: int = 60_000_000
SLOPE_HORIZONS_S: tuple[int, int, int] = (5, 15, 30)
MICROS_PER_SEC: int = 1_000_000

KEY_COLS: list[str] = ["collection_id", "instance_index"]

RUNTIME_FEATURE_COLS: list[str] = [
    "cpu_slope_5s", "cpu_slope_15s", "cpu_slope_30s",
    "memory_slope_5s", "memory_slope_15s", "memory_slope_30s",
    "initial_cpu_ramp", "initial_memory_ramp", "first_interval_util_ratio",
    "cpi_value", "mapi_value",
]


# ---------------------------------------------------------------------------
# Pure Polars path.
# ---------------------------------------------------------------------------
def _slope_expr(value_col: str, horizon_s: int) -> pl.Expr:
    """Closed-form OLS slope of ``value_col`` on ``sec_since_schedule`` over the
    first ``horizon_s`` seconds, computed inside a ``group_by(...).agg(...)``.

    slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2); null when the denominator is 0
    (fewer than two distinct-x in-band observations). Equivalent to
    ``COVAR_POP(y, x) / VAR_POP(x)``. Motivation: V12.
    """
    sec = pl.col("sec_since_schedule")
    mask = (sec >= 0) & (sec <= horizon_s) & pl.col(value_col).is_not_null()
    x = sec.filter(mask)
    y = pl.col(value_col).filter(mask)
    n = mask.sum()
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    return pl.when(denom != 0).then((n * sxy - sx * sy) / denom).otherwise(None)


def add_runtime_features(
    usage_lf: pl.LazyFrame,
    requests_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Reduce per-observation early-runtime usage to one Tier 2 feature row
    per instance.

    Expected ``usage_lf`` columns (the +/-60s working-set usage subset; see
    :func:`build_usage_working_set_sql`):
        - ``collection_id`` (Int64), ``instance_index`` (Int64).
        - ``sec_since_schedule`` (Float64): seconds since the SCHEDULE event.
        - ``avg_cpu`` (Float64), ``avg_memory`` (Float64).
        - ``cycles_per_instruction`` (Float64),
          ``memory_accesses_per_instruction`` (Float64).
        - ``has_cpi_value`` (Int64), ``has_mapi_value`` (Int64): per-observation
          availability indicators (V11).

    Args:
        usage_lf: early-runtime usage observations.
        requests_lf: optional ``(collection_id, instance_index, cpu_request)``
            LazyFrame used to form ``first_interval_util_ratio``. When omitted,
            that ratio is null.

    Appended output columns (one row per instance):
        - ``cpu_slope_{5s,15s,30s}`` / ``memory_slope_{5s,15s,30s}`` (Float64):
          OLS slope of avg_cpu / avg_memory on seconds-since-schedule over the
          first 5 / 15 / 30 s.
        - ``initial_cpu_ramp`` / ``initial_memory_ramp`` (Float64): first-to-
          second observation delta (startup acceleration).
        - ``first_interval_util_ratio`` (Float64): smoothed first-interval
          avg_cpu (mean of the first up to three post-schedule observations)
          divided by ``cpu_request``.
        - ``cpi_value`` / ``mapi_value`` (Float64): first post-schedule
          observation hardware-counter value, kept only when present on that
          observation (else null; V11 MNAR - never imputed).

    Motivation: V12 (rate-of-change), V11 (conditional counters).
    """
    sec = "sec_since_schedule"
    # Restrict to the post-schedule side of the band, mirroring the BigQuery
    # ``ranked`` CTE (WHERE sec_since_schedule >= 0).
    post = usage_lf.filter(pl.col(sec) >= 0)

    cpu_ord = pl.col("avg_cpu").sort_by(sec)
    mem_ord = pl.col("avg_memory").sort_by(sec)

    agg = post.group_by(KEY_COLS).agg(
        _slope_expr("avg_cpu", 5).alias("cpu_slope_5s"),
        _slope_expr("avg_cpu", 15).alias("cpu_slope_15s"),
        _slope_expr("avg_cpu", 30).alias("cpu_slope_30s"),
        _slope_expr("avg_memory", 5).alias("memory_slope_5s"),
        _slope_expr("avg_memory", 15).alias("memory_slope_15s"),
        _slope_expr("avg_memory", 30).alias("memory_slope_30s"),
        # First-to-second observation ramp (null when < 2 observations).
        pl.when(pl.len() >= 2)
          .then(cpu_ord.slice(1, 1).first() - cpu_ord.first())
          .otherwise(None).alias("initial_cpu_ramp"),
        pl.when(pl.len() >= 2)
          .then(mem_ord.slice(1, 1).first() - mem_ord.first())
          .otherwise(None).alias("initial_memory_ramp"),
        # Smoothed first-interval baseline: mean of the first up to 3 obs
        # (the AVG OVER (CURRENT ROW AND 2 FOLLOWING) at rn_post = 1).
        cpu_ord.head(3).mean().alias("_first_interval_avg_cpu"),
        # First post-schedule hardware-counter values + availability flags.
        pl.col("cycles_per_instruction").sort_by(sec).first().alias("_first_cpi"),
        pl.col("memory_accesses_per_instruction").sort_by(sec).first().alias("_first_mapi"),
        pl.col("has_cpi_value").sort_by(sec).first().alias("_first_has_cpi"),
        pl.col("has_mapi_value").sort_by(sec).first().alias("_first_has_mapi"),
    )

    if requests_lf is not None:
        agg = agg.join(requests_lf.select([*KEY_COLS, "cpu_request"]), on=KEY_COLS, how="left")
    else:
        agg = agg.with_columns(pl.lit(None, dtype=pl.Float64).alias("cpu_request"))

    return agg.with_columns(
        # Realized startup CPU / requested CPU.
        pl.when(pl.col("cpu_request") > 0)
          .then(pl.col("_first_interval_avg_cpu") / pl.col("cpu_request"))
          .otherwise(None)
          .alias("first_interval_util_ratio"),
        # Conditional counter values (V11 MNAR): keep only when present.
        pl.when(pl.col("_first_has_cpi") == 1).then(pl.col("_first_cpi"))
          .otherwise(None).alias("cpi_value"),
        pl.when(pl.col("_first_has_mapi") == 1).then(pl.col("_first_mapi"))
          .otherwise(None).alias("mapi_value"),
    ).drop(
        "_first_interval_avg_cpu", "_first_cpi", "_first_mapi",
        "_first_has_cpi", "_first_has_mapi", "cpu_request",
    )


# ---------------------------------------------------------------------------
# Validated BigQuery production path (two-stage memory-bounded filter).
# ---------------------------------------------------------------------------
def build_usage_working_set_sql(
    usage_table: str,
    working_set_table: str,
    out_table: str,
    band_us: int = EARLY_RUNTIME_BAND_US,
) -> str:
    """Return the Stage-1 DDL materializing the +/-``band_us`` early-runtime
    usage subset (first stage).

    Inner-joins the (7.5B-row) ``usage_table`` to ``working_set_table`` on the
    instance key and keeps only observations inside the band around each
    instance's ``schedule_time``. ``usage_table`` should be the V11-augmented
    ``instance_usage_with_indicators`` so the counter availability flags ride
    along. Arguments are fully-qualified, back-ticked BigQuery table names.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
SELECT
    u.*,
    w.schedule_time,
    SAFE_DIVIDE(u.start_time - w.schedule_time, {MICROS_PER_SEC}) AS sec_since_schedule
FROM {usage_table} u
INNER JOIN {working_set_table} w
  USING (collection_id, instance_index)
WHERE u.start_time BETWEEN (w.schedule_time - {band_us})
                       AND (w.schedule_time + {band_us});
"""


def slope_sql(value_col: str, horizon_s: int) -> str:
    """Return a BigQuery OLS-slope expression for ``value_col`` on
    ``sec_since_schedule`` over the first ``horizon_s`` seconds.

    BigQuery has no ``REGR_SLOPE``; the slope is ``COVAR_POP(y, x) /
    VAR_POP(x)`` with both gated on the same in-band, non-null-y rows.
    ``SAFE_DIVIDE`` yields NULL when the band has < 2 distinct-x observations.
    """
    gate = f"sec_since_schedule <= {horizon_s} AND {value_col} IS NOT NULL"
    x = f"IF({gate}, sec_since_schedule, NULL)"
    y = f"IF({gate}, {value_col}, NULL)"
    return f"SAFE_DIVIDE(COVAR_POP({y}, {x}), VAR_POP({x}))"


def build_runtime_features_sql(
    usage_working_set_table: str,
    lifecycle_summary_table: str,
    out_table: str,
) -> str:
    """Return the Stage-2 DDL reducing the early-runtime usage subset to one
    Tier 2 feature row per instance (second stage).

    Uses ``ROW_NUMBER`` / ``LAG`` / ``AVG OVER`` for the ramps and the smoothed
    first-interval baseline, and :func:`slope_sql` for the OLS slopes.
    ``lifecycle_summary_table`` supplies ``cpu_request`` for the
    ``first_interval_util_ratio``. Arguments are fully-qualified, back-ticked
    BigQuery table names. Mirrors :func:`add_runtime_features`.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
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
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index ORDER BY sec_since_schedule
        ) AS rn_post,
        LAG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index ORDER BY sec_since_schedule
        ) AS prev_avg_cpu,
        LAG(avg_memory) OVER (
            PARTITION BY collection_id, instance_index ORDER BY sec_since_schedule
        ) AS prev_avg_memory,
        AVG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index ORDER BY sec_since_schedule
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        ) AS first_window_avg_cpu
    FROM {usage_working_set_table}
    WHERE sec_since_schedule >= 0
),
agg AS (
    SELECT
        collection_id,
        instance_index,
        {slope_sql('avg_cpu', 5)}  AS cpu_slope_5s,
        {slope_sql('avg_cpu', 15)} AS cpu_slope_15s,
        {slope_sql('avg_cpu', 30)} AS cpu_slope_30s,
        {slope_sql('avg_memory', 5)}  AS memory_slope_5s,
        {slope_sql('avg_memory', 15)} AS memory_slope_15s,
        {slope_sql('avg_memory', 30)} AS memory_slope_30s,
        MAX(IF(rn_post = 2, avg_cpu - prev_avg_cpu, NULL))       AS initial_cpu_ramp,
        MAX(IF(rn_post = 2, avg_memory - prev_avg_memory, NULL)) AS initial_memory_ramp,
        MAX(IF(rn_post = 1, first_window_avg_cpu, NULL))        AS first_interval_avg_cpu,
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
    SAFE_DIVIDE(a.first_interval_avg_cpu, NULLIF(s.cpu_request, 0)) AS first_interval_util_ratio,
    IF(a.first_has_cpi  = 1, a.first_cpi,  NULL) AS cpi_value,
    IF(a.first_has_mapi = 1, a.first_mapi, NULL) AS mapi_value
FROM agg a
LEFT JOIN {lifecycle_summary_table} s
  USING (collection_id, instance_index)
"""
