"""SMART feature engineering for the Backblaze Hard Drive dataset.

Pure Polars LazyFrame transforms. No I/O. Each function adds a family of feature
columns and returns a LazyFrame, so the calling notebook composes them and owns
all reads and writes.

**Ordering contract.** Every rolling, lag, and per-drive feature here is computed
within ``over("serial_number")`` and assumes the input is sorted by
``(serial_number, date)``. The caller establishes that ordering once per
processing partition (a drive's full history must sit in one partition, so
partitioning is by drive, never by row). Because the ordering is per drive and
strictly by date, no future row enters a past window, which is the leakage
contract for the temporal split (test period 2023-2025).

**Feature tiers** follow the EDA-validated predictive-value structure recorded in
``outputs/tables/eda_decisions.csv``:

- Tier 1 (``add_tier1_smart_features``): pre-event signals on the primary SMART
  attributes (V14), zero-inflation indicators (V19, O05), degradation-onset
  timing, manufacturer identity (V18), and capacity.
- Tier 2 (``add_tier2_rolling_features``): engineered SMART dynamics, rolling
  location and upper-quantile statistics and rate-of-change on the primary
  attributes, and a reduced set on the secondary attributes (V15).
- Tier 3 (``add_tier3_drift_features``): drift-aware cohort features for the
  online-learning analysis (RQ5) and the era-gated SMART 187 / 188 handling
  (V16, the era census).
- Targets (``add_multi_horizon_targets``): the 7 / 14 / 30-day failure horizons
  (P07). Targets look forward by construction and are labels, not features.

**Windows** are expressed in rows. The preprocessed data is one row per
drive-day, so an ``N``-row window is an ``N``-day window wherever a drive is
observed on consecutive days; short gaps make a row window slightly shorter in
calendar time than ``N`` days, which is acceptable for these degradation
statistics.

**Note on drive-model target encoding (V18, O07).** Target encoding of
``drive_model`` against the historical per-model failure rate is a fit-on-prior-
years operation that depends on the temporal split, so it is applied at modeling
time (prior years only) rather than baked in here. This module carries
``drive_model`` through as a categorical and one-hot encodes ``manufacturer``.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import BACKBLAZE_ERAS

PRIMARY_SMART_IDS: tuple[int, ...] = (5, 197, 198)
SECONDARY_SMART_IDS: tuple[int, ...] = (4, 12, 193, 240, 1, 7, 9)
ERA_GATED_SMART_IDS: tuple[int, ...] = (187, 188)
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30)
ONE_HOT_MANUFACTURERS: tuple[str, ...] = (
    "Seagate", "HGST", "WDC", "Toshiba", "Hitachi",
)
SERIAL: str = "serial_number"
CAPACITY_TB: float = 1_000_000_000_000.0


def _first_nonzero_days_since(raw_col: str, date_col: str = "date") -> pl.Expr:
    """Days since a drive's first non-zero reading of ``raw_col``.

    Leakage-safe: the first-non-zero date is a fixed per-drive event, and the
    feature is only populated for rows on or after it (rows before it, where the
    first non-zero still lies in the future, are null).
    """
    first_nz = (
        pl.when(pl.col(raw_col) > 0).then(pl.col(date_col)).min().over(SERIAL)
    )
    return (
        pl.when(pl.col(date_col) >= first_nz)
        .then((pl.col(date_col) - first_nz).dt.total_days())
        .otherwise(None)
        .alias(f"days_since_first_nonzero_{raw_col.replace('_raw', '')}")
    )


def add_tier1_smart_features(
    lf: pl.LazyFrame,
    primary_smart_ids: tuple[int, ...] = PRIMARY_SMART_IDS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    manufacturers: tuple[str, ...] = ONE_HOT_MANUFACTURERS,
) -> pl.LazyFrame:
    """Add Tier 1 pre-event SMART features (V14, V18, V19).

    For each primary SMART id: rolling "any non-zero in the last ``w`` days"
    indicators (zero-inflation, V19 / O05) and the days-since-first-non-zero
    degradation-onset timer. Also one-hot ``manufacturer`` (V18) and
    ``capacity_tb``. Requires input sorted by ``(serial_number, date)``.

    Args:
        lf: preprocessed Backblaze LazyFrame with ``smart_{id}_raw`` columns,
            ``manufacturer``, ``capacity_bytes``, ``serial_number``, ``date``.
        primary_smart_ids: primary SMART ids (default 5, 197, 198).
        windows: row windows for the non-zero indicators.
        manufacturers: manufacturers to one-hot encode.

    Returns:
        LazyFrame with the Tier 1 columns appended.
    """
    exprs: list[pl.Expr] = []
    for sid in primary_smart_ids:
        raw = f"smart_{sid}_raw"
        nonzero = (pl.col(raw) > 0).cast(pl.Int8)
        for w in windows:
            exprs.append(
                nonzero.rolling_max(window_size=w, min_samples=1)
                .over(SERIAL)
                .cast(pl.Int8)
                .alias(f"has_nonzero_smart_{sid}_{w}d")
            )
        exprs.append(_first_nonzero_days_since(raw))

    for mfr in manufacturers:
        exprs.append(
            (pl.col("manufacturer") == mfr).cast(pl.Int8).alias(f"is_mfr_{mfr.lower()}")
        )
    exprs.append((pl.col("capacity_bytes") / CAPACITY_TB).alias("capacity_tb"))
    return lf.with_columns(exprs)


def _rolling_block(raw: str, sid: int, windows: tuple[int, ...]) -> list[pl.Expr]:
    """Rolling mean, p95, p99, and std of one SMART raw column over ``windows``."""
    block: list[pl.Expr] = []
    for w in windows:
        base = pl.col(raw)
        block.extend([
            base.rolling_mean(window_size=w, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollmean_{w}d"),
            base.rolling_quantile(quantile=0.95, window_size=w, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollp95_{w}d"),
            base.rolling_quantile(quantile=0.99, window_size=w, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollp99_{w}d"),
            base.rolling_std(window_size=w, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollstd_{w}d"),
        ])
    return block


def add_tier2_rolling_features(
    lf: pl.LazyFrame,
    primary_smart_ids: tuple[int, ...] = PRIMARY_SMART_IDS,
    secondary_smart_ids: tuple[int, ...] = SECONDARY_SMART_IDS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pl.LazyFrame:
    """Add Tier 2 engineered SMART dynamics (V15, V19).

    Primary SMART ids get rolling mean / p95 / p99 / std over each window plus
    1-day and 7-day rate-of-change. Secondary SMART ids get a reduced set
    (30-day rolling mean, 7-day delta, 30-day non-zero indicator) to keep
    dimensionality bounded (V15). Requires input sorted by
    ``(serial_number, date)``.

    Args:
        lf: preprocessed Backblaze LazyFrame sorted by ``(serial_number, date)``.
        primary_smart_ids: primary SMART ids (default 5, 197, 198).
        secondary_smart_ids: secondary SMART ids (default 4, 12, 193, 240, 1,
            7, 9).
        windows: row windows for the primary rolling statistics.

    Returns:
        LazyFrame with the Tier 2 columns appended.
    """
    exprs: list[pl.Expr] = []
    for sid in primary_smart_ids:
        raw = f"smart_{sid}_raw"
        exprs.extend(_rolling_block(raw, sid, windows))
        exprs.append(
            (pl.col(raw) - pl.col(raw).shift(1).over(SERIAL)).alias(f"smart_{sid}_delta_1d")
        )
        exprs.append(
            (pl.col(raw) - pl.col(raw).shift(7).over(SERIAL)).alias(f"smart_{sid}_delta_7d")
        )

    for sid in secondary_smart_ids:
        raw = f"smart_{sid}_raw"
        exprs.append(
            pl.col(raw).rolling_mean(window_size=30, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollmean_30d")
        )
        exprs.append(
            (pl.col(raw) - pl.col(raw).shift(7).over(SERIAL)).alias(f"smart_{sid}_delta_7d")
        )
        exprs.append(
            (pl.col(raw) > 0).cast(pl.Int8).rolling_max(window_size=30, min_samples=1)
            .over(SERIAL).cast(pl.Int8).alias(f"has_nonzero_smart_{sid}_30d")
        )
    return lf.with_columns(exprs)


def _era_availability_expr(smart_id: int, era_constants: list) -> pl.Expr:
    """1 where the row's era lists ``smart_id`` in its available set, else 0."""
    expr = pl.when(pl.lit(False)).then(pl.lit(0, dtype=pl.Int8))
    for _start, _end, name, ids in era_constants:
        if smart_id in ids:
            expr = expr.when(pl.col("era") == name).then(pl.lit(1, dtype=pl.Int8))
    return expr.otherwise(pl.lit(0, dtype=pl.Int8)).alias(f"era_smart_{smart_id}_available")


def add_tier3_drift_features(
    lf: pl.LazyFrame,
    dataset_start_date,
    era_constants: list | None = None,
    era_gated_smart_ids: tuple[int, ...] = ERA_GATED_SMART_IDS,
) -> pl.LazyFrame:
    """Add Tier 3 drift-aware and era-gated features (V16, RQ5).

    Cohort features for the drift analysis: fleet age (days since the dataset
    start), per-drive age, and calendar ``year`` / ``month`` / ``quarter``. For
    each era-gated SMART id (187, 188): an era-availability flag from the census
    plus a 30-day rolling mean, 30-day rolling p95, and 7-day delta, which are
    null in eras and rows where the attribute was not collected. Requires input
    sorted by ``(serial_number, date)``.

    Args:
        lf: preprocessed Backblaze LazyFrame with ``era`` and the era-gated
            ``smart_{id}_raw`` columns.
        dataset_start_date: the earliest observation date across the whole
            dataset (a ``datetime.date``), passed in so ``fleet_age_days`` is
            consistent across processing partitions.
        era_constants: ``BACKBLAZE_ERAS`` (default) for the era-availability
            flags.
        era_gated_smart_ids: SMART ids gated by era (default 187, 188).

    Returns:
        LazyFrame with the Tier 3 columns appended.
    """
    if era_constants is None:
        era_constants = BACKBLAZE_ERAS

    exprs: list[pl.Expr] = [
        (pl.col("date") - pl.lit(dataset_start_date)).dt.total_days().alias("fleet_age_days"),
        (pl.col("date") - pl.col("date").min().over(SERIAL)).dt.total_days().alias("drive_age_days"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
    ]
    for sid in era_gated_smart_ids:
        raw = f"smart_{sid}_raw"
        exprs.append(_era_availability_expr(sid, era_constants))
        exprs.append(
            pl.col(raw).rolling_mean(window_size=30, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollmean_30d")
        )
        exprs.append(
            pl.col(raw).rolling_quantile(quantile=0.95, window_size=30, min_samples=1)
            .over(SERIAL).alias(f"smart_{sid}_rollp95_30d")
        )
        exprs.append(
            (pl.col(raw) - pl.col(raw).shift(7).over(SERIAL)).alias(f"smart_{sid}_delta_7d")
        )
    return lf.with_columns(exprs)


def add_multi_horizon_targets(
    lf: pl.LazyFrame,
    horizons: tuple[int, ...] = DEFAULT_WINDOWS,
    failure_column: str = "failure",
    date_column: str = "date",
) -> pl.LazyFrame:
    """Add multi-horizon failure targets (P07).

    ``failure_within_{h}d`` is 1 when the drive fails within ``h`` days on or
    after this observation, else 0. Backblaze marks ``failure = 1`` on a drive's
    removal day, so the per-drive maximum failure date is the failure date; a row
    is positive for horizon ``h`` when that date is between the row date and
    ``h`` days later. These are labels and legitimately look forward.

    Args:
        lf: Backblaze LazyFrame with ``serial_number``, ``date``, and the binary
            failure column.
        horizons: day horizons (default 7, 14, 30).
        failure_column: binary failure column.
        date_column: observation-date column.

    Returns:
        LazyFrame with one ``failure_within_{h}d`` column per horizon.
    """
    failure_date = (
        pl.when(pl.col(failure_column) == 1).then(pl.col(date_column)).max().over(SERIAL)
    )
    days_to_failure = (failure_date - pl.col(date_column)).dt.total_days()
    exprs = [
        ((days_to_failure >= 0) & (days_to_failure <= h))
        .fill_null(False)
        .cast(pl.Int8)
        .alias(f"failure_within_{h}d")
        for h in horizons
    ]
    return lf.with_columns(exprs)
