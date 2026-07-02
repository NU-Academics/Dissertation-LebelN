"""Unit tests for ``src/features/backblaze_smart.py``.

Each test builds a small synthetic, date-sorted Polars LazyFrame and checks one
feature family, including the leakage contract (no future row enters a past
window). No GCS, Drive, or subprocess.

Run with::

    pytest tests/test_backblaze_smart.py -v
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.data.schemas import BACKBLAZE_ERAS
from src.features.backblaze_smart import (
    add_multi_horizon_targets,
    add_tier1_smart_features,
    add_tier2_rolling_features,
    add_tier3_drift_features,
)


def _drive_frame() -> pl.LazyFrame:
    """One drive (Seagate) observed daily 2022-06-01..06, failing on the last day.

    smart_5_raw is zero-inflated: zeros until 2022-06-03, then rising. Includes
    the columns the three tiers read.
    """
    dates = [date(2022, 6, d) for d in range(1, 7)]
    n = len(dates)
    return pl.LazyFrame(
        {
            "serial_number": ["Z1"] * n,
            "date": dates,
            "failure": [0, 0, 0, 0, 0, 1],
            "capacity_bytes": [4_000_000_000_000] * n,
            "manufacturer": ["Seagate"] * n,
            "model_canonical": ["ST4000DM000"] * n,
            "era": ["recent_2021_2025"] * n,
            "smart_5_raw": [0, 0, 4, 8, 16, 32],
            "smart_197_raw": [0, 0, 0, 2, 4, 8],
            "smart_198_raw": [0, 0, 0, 0, 2, 4],
            "smart_4_raw": [10, 10, 11, 11, 12, 12],
            "smart_12_raw": [1, 1, 1, 1, 1, 1],
            "smart_193_raw": [5, 6, 7, 8, 9, 10],
            "smart_240_raw": [100, 110, 120, 130, 140, 150],
            "smart_1_raw": [0, 0, 0, 0, 0, 0],
            "smart_7_raw": [3, 3, 3, 3, 3, 3],
            "smart_9_raw": [24, 48, 72, 96, 120, 144],
            "smart_187_raw": [0, 1, 2, 3, 4, 5],
            "smart_188_raw": [0, 0, 1, 1, 2, 2],
        }
    ).sort(["serial_number", "date"])


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


def test_tier1_nonzero_and_onset() -> None:
    """Non-zero indicators and days-since-first-non-zero are leakage-safe."""
    result = add_tier1_smart_features(_drive_frame(), windows=(3,)).collect()

    # smart_5 first non-zero on 2022-06-03; the 3-day rolling indicator is 1
    # from that day on and 0 before.
    assert result["has_nonzero_smart_5_3d"].to_list() == [0, 0, 1, 1, 1, 1]

    # days_since_first_nonzero is null before the first non-zero, 0 on the day,
    # then increments; no future information leaks into earlier rows.
    dsf = result["days_since_first_nonzero_smart_5"].to_list()
    assert dsf == [None, None, 0, 1, 2, 3]

    # One-hot manufacturer and capacity.
    assert result["is_mfr_seagate"].to_list() == [1] * 6
    assert result["is_mfr_hgst"].to_list() == [0] * 6
    assert result["capacity_tb"].to_list() == [4.0] * 6


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------


def test_tier2_rolling_and_delta() -> None:
    """Rolling statistics and rate-of-change respect per-drive ordering."""
    result = add_tier2_rolling_features(_drive_frame(), windows=(3,)).collect()

    # 3-day rolling mean of smart_5 (0,0,4,8,16,32): last window (8,16,32)->18.67.
    rm = result["smart_5_rollmean_3d"].to_list()
    assert abs(rm[-1] - (8 + 16 + 32) / 3) < 1e-9
    assert rm[0] == 0.0  # first row, window of one zero

    # 1-day delta of smart_5: null, 0, 4, 4, 8, 16.
    assert result["smart_5_delta_1d"].to_list() == [None, 0, 4, 4, 8, 16]

    # 7-day delta has no 7-back row in a 6-row drive, so all null.
    assert result["smart_5_delta_7d"].to_list() == [None] * 6

    # p95 column exists for the primary set.
    assert "smart_197_rollp95_3d" in result.columns

    # Secondary reduced set: only the three documented columns per id.
    assert "smart_193_rollmean_30d" in result.columns
    assert "smart_193_delta_7d" in result.columns
    assert "has_nonzero_smart_193_30d" in result.columns
    # Secondary ids do not get the full rolling block.
    assert "smart_193_rollp95_3d" not in result.columns


# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------


def test_tier3_drift_and_era_gating() -> None:
    """Cohort ages, calendar parts, and era-availability flags are correct."""
    start = date(2013, 4, 1)
    result = add_tier3_drift_features(_drive_frame(), dataset_start_date=start).collect()

    # fleet age counts from the dataset start; drive age from the drive's first day.
    assert result["fleet_age_days"][0] == (date(2022, 6, 1) - start).days
    assert result["drive_age_days"].to_list() == [0, 1, 2, 3, 4, 5]
    assert result["year"].to_list() == [2022] * 6
    assert result["quarter"].to_list() == [2] * 6

    # SMART 187/188 are NOT in the recent era's available set (V44), so the
    # era-availability flag is 0 for these recent-era rows even though the raw
    # column carries values.
    assert set(result["era_smart_187_available"].to_list()) == {0}
    assert "smart_187_rollmean_30d" in result.columns


def test_tier3_era_flag_true_in_standard_era() -> None:
    """In the standard era the SMART 187 availability flag is 1."""
    lf = _drive_frame().with_columns(pl.lit("standard_2014_2021").alias("era"))
    result = add_tier3_drift_features(
        lf, dataset_start_date=date(2013, 4, 1), era_constants=BACKBLAZE_ERAS
    ).collect()
    assert set(result["era_smart_187_available"].to_list()) == {1}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_multi_horizon_targets() -> None:
    """Failure horizons flag the correct pre-failure window; censored drives are 0."""
    result = add_multi_horizon_targets(_drive_frame(), horizons=(3, 30)).collect()

    # The drive fails on 2022-06-06. Within 3 days: rows on 06-03..06-06 -> 1.
    assert result["failure_within_3d"].to_list() == [0, 0, 1, 1, 1, 1]
    # Within 30 days: every row (all within 30 days of the failure) -> 1.
    assert result["failure_within_30d"].to_list() == [1, 1, 1, 1, 1, 1]

    # A never-failing drive yields all zeros at every horizon.
    healthy = _drive_frame().with_columns(pl.lit(0).alias("failure"))
    hres = add_multi_horizon_targets(healthy, horizons=(7,)).collect()
    assert hres["failure_within_7d"].to_list() == [0] * 6
