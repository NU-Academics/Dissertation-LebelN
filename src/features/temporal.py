"""Tier 1 submit-time temporal features for Google Cluster Traces.

The FAIL_LOST rate varies up to ~8x across hour-of-day buckets, peaking during
PDT business hours, so the SUBMIT wall-clock time is a Tier 1 (pre-event)
predictor. This module derives the temporal features from ``submit_time``.

Validated in ``notebooks/10_feature_engineering_google.py`` Section 5 against
the working set before extraction into this module.

Cross-reference (``outputs/tables/eda_decisions.csv``):
- V26 - temporal stratification; hourly FAIL_LOST swing up to ~8x, peaking in
  PDT business hours; weekly swing ~1.29x.

PDT wall-clock convention (matches the F2 SQL in
``notebooks/07b_phase3_front_loaded_eda.py``): the trace ``time`` field is
microseconds measured from 600 seconds before the documented trace start.
Anchoring at the naive timestamp ``'2019-05-01 00:00:00 UTC'``, adding the
trace-clock microseconds, and subtracting the 600-second pre-trace offset
yields a naive timestamp whose EXTRACTed hour / day-of-week coincide with PDT
wall-clock by arithmetic coincidence (the trace is PDT-local and the anchor is
chosen so no timezone conversion is needed). This module reproduces that exact
arithmetic so the engineered features align with the V26 diurnal census.

The single public function is a pure ``LazyFrame -> LazyFrame`` transform that
performs no I/O.
"""

from __future__ import annotations

import math
from datetime import datetime

import polars as pl

# Naive anchor and pre-trace offset for the trace clock (V26 / F2 convention).
TRACE_ANCHOR: datetime = datetime(2019, 5, 1, 0, 0, 0)
MICROS_PER_SEC: int = 1_000_000
TRACE_PRE_OFFSET_US: int = 600 * MICROS_PER_SEC  # 600 s pre-trace lead-in.

# PDT business-hours window (inclusive), and the weekend day-of-week codes
# (Mon=0 ... Sun=6).
BUSINESS_HOURS_PDT: tuple[int, int] = (8, 17)
WEEKEND_DOW_MIN: int = 5  # Saturday and Sunday are 5 and 6.


def add_temporal_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append the Tier 1 submit-time PDT temporal features.

    Expected input column:
        - ``submit_time`` (Int64): microseconds on the trace clock (600 s
          before the trace start; V25/V26 sentinel rows already excluded
          upstream in notebook 08).

    Appended output columns:
        - ``submit_hour_of_day`` (Int): PDT hour 0-23 (ordinal for trees).
        - ``submit_day_of_week`` (Int): 0=Mon ... 6=Sun.
        - ``submit_hour_sin`` / ``submit_hour_cos`` (Float64): cyclic encoding
          of the hour, ``sin/cos(2*pi*hour/24)`` (for non-tree models).
        - ``submit_is_business_hours_pdt`` (Int8): 1 when the PDT hour is in
          [8, 17].
        - ``submit_is_weekend`` (Int8): 1 when day-of-week is Sat or Sun.

    Motivation: V26 - diurnal FAIL_LOST swing peaking in PDT business hours.
    """
    # Reconstruct the PDT wall-clock timestamp via the naive-anchor trick (V26).
    wall = (
        pl.lit(TRACE_ANCHOR)
        + pl.duration(microseconds=(pl.col("submit_time") - TRACE_PRE_OFFSET_US))
    )
    lo, hi = BUSINESS_HOURS_PDT
    return (
        lf.with_columns(wall.alias("_submit_wall"))
        .with_columns(
            pl.col("_submit_wall").dt.hour().alias("submit_hour_of_day"),
            # Polars weekday(): Mon=1 ... Sun=7. Shift to Mon=0 ... Sun=6.
            (pl.col("_submit_wall").dt.weekday() - 1).alias("submit_day_of_week"),
        )
        .with_columns(
            (2 * math.pi * pl.col("submit_hour_of_day") / 24).sin().alias("submit_hour_sin"),
            (2 * math.pi * pl.col("submit_hour_of_day") / 24).cos().alias("submit_hour_cos"),
            ((pl.col("submit_hour_of_day") >= lo) & (pl.col("submit_hour_of_day") <= hi))
                .cast(pl.Int8).alias("submit_is_business_hours_pdt"),
            (pl.col("submit_day_of_week") >= WEEKEND_DOW_MIN)
                .cast(pl.Int8).alias("submit_is_weekend"),
        )
        .drop("_submit_wall")
    )


# Canonical Tier 1 temporal feature names (excludes the keys they are appended
# to). Useful for column-set assertions in tests and notebook 11.
TEMPORAL_FEATURE_COLS: list[str] = [
    "submit_hour_of_day",
    "submit_day_of_week",
    "submit_hour_sin",
    "submit_hour_cos",
    "submit_is_business_hours_pdt",
    "submit_is_weekend",
]
