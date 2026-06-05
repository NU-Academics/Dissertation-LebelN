"""Tier 1 historical features for Google Cluster Traces failure prediction.

Tier 1 (pre-event) signals carry the highest expected predictive value in the
EDA-validated feature hierarchy. This module derives the per-instance
historical signals from the lifecycle summary produced by
``src/preprocessing/lifecycle.py`` (and notebook 08 Section 5).

Validated in ``notebooks/10_feature_engineering_google.py`` Section 3 against
the working set (35,133,137 instances) before extraction into this module.

Cross-references (``outputs/tables/eda_decisions.csv``):
- V09 - rapid-onset failure model (median FAIL_LOST running duration 22.6s).
- V10 - resubmission history dominates; first-resubmission failure rate is
  ~72x the single-pass rate.

Each function is a pure ``LazyFrame -> LazyFrame`` transform that appends
columns and performs no I/O.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import EVENT_FAIL, EVENT_LOST


def add_historical_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append the Tier 1 historical features.

    Expected input columns (from ``instance_lifecycle_summary``):
        - ``fail_lost_count`` (Int64): FAIL/LOST events in the lifecycle.
        - ``terminal_type`` (Int64): event type of the terminal event.
        - ``evict_count`` (Int64): EVICT events in the lifecycle.
        - ``resubmission_count`` (Int64): submits beyond the first.

    Appended output columns:
        - ``prior_fail_count`` (Int64): FAIL/LOST events strictly before the
          terminal event (``fail_lost_count`` minus 1 when the terminal event
          is itself FAIL_LOST, floored at 0).
        - ``has_prior_fail`` (Int8): 1 when ``prior_fail_count > 0``.
        - ``resubmission_count`` (Int64): passthrough, dtype-normalized.
        - ``prior_evict_count`` (Int64): ``evict_count``.
        - ``first_resubmission`` (Int8): 1 when the instance has been
          resubmitted at least once (``resubmission_count >= 1``). This is the
          V10 discriminator separating the first-resubmission population
          (10.12% FAIL_LOST) from the single-pass population (0.14%).

    Motivation: V09 (rapid-onset failure), V10 (resubmission dominance).
    """
    # Motivation: V09/V10 - failures concentrate in resubmitted instances, and
    # a failing terminal event must be excluded so prior_fail_count counts
    # only *previous* attempts.
    terminal_is_fail_lost = pl.col("terminal_type").is_in([EVENT_FAIL, EVENT_LOST])
    return lf.with_columns(
        pl.max_horizontal(
            pl.col("fail_lost_count") - terminal_is_fail_lost.cast(pl.Int64),
            pl.lit(0),
        ).alias("prior_fail_count"),
        pl.col("evict_count").alias("prior_evict_count"),
        pl.col("resubmission_count").cast(pl.Int64).alias("resubmission_count"),
        # Motivation: V10 - first-resubmission vs single-pass failure rate (72x).
        (pl.col("resubmission_count") >= 1).cast(pl.Int8).alias("first_resubmission"),
    ).with_columns(
        (pl.col("prior_fail_count") > 0).cast(pl.Int8).alias("has_prior_fail"),
    )


def add_lifecycle_position(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append ``lifecycle_position``, the instance's fractional ordinal
    position within its collection.

    Expected input columns:
        - ``instance_index`` (Int64): the instance's index in its collection.
        - ``collection_size_at_submit`` (Int64): number of working-set
          instances in the collection (computed collection-side; see
          notebook 10 Section 2 / 6).

    Appended output column:
        - ``lifecycle_position`` (Float64): ``instance_index /
          collection_size_at_submit``; null when the size is non-positive.
          Early vs late instances within a collection differ in failure
          propensity.

    Separated from :func:`add_historical_features` because it depends on the
    collection-level ``collection_size_at_submit`` aggregate rather than on
    per-instance lifecycle columns alone. Motivation: V07, V09.
    """
    return lf.with_columns(
        pl.when(pl.col("collection_size_at_submit") > 0)
          .then(pl.col("instance_index") / pl.col("collection_size_at_submit"))
          .otherwise(None)
          .alias("lifecycle_position"),
    )


# Canonical Tier 1 historical feature names (excludes the keys they are
# appended to). Useful for column-set assertions in tests and notebook 11.
HISTORICAL_FEATURE_COLS: list[str] = [
    "prior_fail_count",
    "has_prior_fail",
    "resubmission_count",
    "prior_evict_count",
    "first_resubmission",
    "lifecycle_position",
]
