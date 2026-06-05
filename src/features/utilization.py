"""Tier 3 windowed utilization features for Google Cluster Traces.

Tier 3 (windowed absolute utilization) has LOW expected predictive value and
is included **only for the Chapter 4 ablation**. These absolute aggregates are
the confounded comparison behind the V12 utilization inversion: because the
rapid-onset crash window (median 22.6s, V09) is shorter than every Tier 3
window, failing instances appear to use *less* resource. **Do not drop these
columns** - the feature ablation (V13) and the Tier 3 inversion
regression guard (``src/data/validation.py::assert_tier3_inversion``) both
depend on them, and the inversion was confirmed on the working set (median
avg_cpu fail 0.00125 < success 0.00178 at all windows).

Validated in ``notebooks/10_feature_engineering_google.py`` Section 8 before
extraction into this module.

Cross-references (``outputs/tables/eda_decisions.csv``):
- V12 - utilization inversion (failing instances use less absolute resource).
- V13 - tier structure; Tier 3 retained for ablation despite low value.

Two equivalent code paths are provided:

1. :func:`add_windowed_utilization` - a pure ``LazyFrame -> LazyFrame``
   transform computing the windowed aggregates in Polars.
2. :func:`build_windowed_utilization_sql` - the BigQuery SQL builder capturing
   the **validated production path** (used by the notebook at full working-set
   scale to keep the ~35M-group aggregation off the Colab box). Polars
   ``.std()`` (ddof = 1) matches BigQuery ``STDDEV_SAMP``: a single in-window
   observation yields NULL in both.
"""

from __future__ import annotations

import polars as pl

# Tier 3 aggregation windows (post-schedule seconds). The 60-min window extends
# well past the early-runtime band, hence the separate (wider) usage subset.
TIER3_WINDOWS_SEC: dict[str, int] = {"5min": 300, "15min": 900, "60min": 3600}
TIER3_MAX_WINDOW_US: int = 3_600_000_000  # 60 min in microseconds.
MICROS_PER_SEC: int = 1_000_000

KEY_COLS: list[str] = ["collection_id", "instance_index"]

# Canonical Tier 3 column order (avg/max/std per resource per window).
UTILIZATION_FEATURE_COLS: list[str] = [
    "avg_cpu_5min", "max_cpu_5min", "std_cpu_5min",
    "avg_cpu_15min", "max_cpu_15min", "std_cpu_15min",
    "avg_cpu_60min", "max_cpu_60min", "std_cpu_60min",
    "avg_memory_5min", "max_memory_5min", "std_memory_5min",
    "avg_memory_15min", "max_memory_15min", "std_memory_15min",
    "avg_memory_60min", "max_memory_60min", "std_memory_60min",
]


def add_windowed_utilization(usage_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce per-observation usage to one Tier 3 feature row per instance.

    Expected ``usage_lf`` columns (the 0..60min working-set usage subset; see
    :func:`build_windowed_utilization_sql`):
        - ``collection_id`` (Int64), ``instance_index`` (Int64).
        - ``avg_cpu`` (Float64), ``avg_memory`` (Float64).
        - ``sec_since_schedule`` (Float64): seconds since the SCHEDULE event.

    Appended output columns (one row per instance): for each window in
    :data:`TIER3_WINDOWS_SEC` and each resource in (cpu, memory), the mean,
    max, and (sample, ddof = 1) std of the resource over post-schedule
    observations within the window: ``{avg,max,std}_{cpu,memory}_{5min,15min,60min}``.

    Motivation: V12, V13 (ablation-only confounded comparison).
    """
    aggs: list[pl.Expr] = []
    for label, horizon in TIER3_WINDOWS_SEC.items():
        in_win = (pl.col("sec_since_schedule") >= 0) & (pl.col("sec_since_schedule") <= horizon)
        cpu_w = pl.when(in_win).then(pl.col("avg_cpu")).otherwise(None)
        mem_w = pl.when(in_win).then(pl.col("avg_memory")).otherwise(None)
        aggs += [
            cpu_w.mean().alias(f"avg_cpu_{label}"),
            cpu_w.max().alias(f"max_cpu_{label}"),
            cpu_w.std().alias(f"std_cpu_{label}"),
            mem_w.mean().alias(f"avg_memory_{label}"),
            mem_w.max().alias(f"max_memory_{label}"),
            mem_w.std().alias(f"std_memory_{label}"),
        ]
    return usage_lf.group_by(KEY_COLS).agg(aggs)


def build_tier3_usage_subset_sql(
    usage_table: str,
    working_set_table: str,
    out_table: str,
    horizon_us: int = TIER3_MAX_WINDOW_US,
) -> str:
    """Return the DDL materializing the 0..``horizon_us`` working-set usage
    subset for the Tier 3 windows (the source is read only through the
    working-set join, keeping it out of in-process memory). Arguments are
    fully-qualified, back-ticked
    BigQuery table names.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
SELECT
    u.collection_id,
    u.instance_index,
    u.avg_cpu,
    u.avg_memory,
    SAFE_DIVIDE(u.start_time - w.schedule_time, {MICROS_PER_SEC}) AS sec_since_schedule
FROM {usage_table} u
INNER JOIN {working_set_table} w
  USING (collection_id, instance_index)
WHERE u.start_time BETWEEN w.schedule_time
                       AND (w.schedule_time + {horizon_us});
"""


def _windowed_agg_sql() -> str:
    """Return the comma-separated BigQuery conditional-aggregate expressions for
    the Tier 3 windowed utilization features (mirrors
    :func:`add_windowed_utilization`; ``STDDEV_SAMP`` == Polars ``.std()``)."""
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


def build_windowed_utilization_sql(usage_subset_table: str, out_table: str) -> str:
    """Return the DDL computing the per-instance Tier 3 windowed features from
    the usage subset (the validated production path). Arguments are
    fully-qualified, back-ticked BigQuery table names.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    {_windowed_agg_sql()}
FROM {usage_subset_table}
GROUP BY collection_id, instance_index
"""
