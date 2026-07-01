"""Unit tests for ``src/preprocessing/google_traces.py``.

Each test builds a small synthetic Polars LazyFrame hand-crafted to hit
the branches of one transform. No BigQuery, no Drive, no GCS.

Run with::

    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.data.schemas import (
    BACKBLAZE_ERAS,
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_KILL,
    EVENT_LOST,
    EVENT_SUBMIT,
    PRIORITY_MONITORING_LOW,
    PRIORITY_PRODUCTION_LOW,
    SENTINEL_TIME_AFTER,
    SENTINEL_TIME_BEFORE,
)
from src.data.validation import (
    AssertionFailedError,
    assert_era_assignment_complete,
    assert_failure_event_count,
    assert_fleet_expansion,
    assert_one_row_per_drive_day,
)
from src.preprocessing.backblaze import (
    assign_era,
    canonicalize_drive_model,
    encode_smart_availability_indicators,
    filter_hdds_only,
    mark_censoring,
    reconcile_smart_schema,
)
from src.preprocessing.google_traces import (
    apply_failure_label,
    encode_hardware_counters_mnar,
    filter_sentinel_timestamps,
)


# ---------------------------------------------------------------------------
# 1. Sentinel filtering (V25)
# ---------------------------------------------------------------------------


def test_filter_sentinel_timestamps_drops_zero_and_max() -> None:
    """Sentinel rows are removed, every other row survives."""
    lf = pl.LazyFrame(
        {
            "time": [
                100,
                SENTINEL_TIME_BEFORE,
                200,
                SENTINEL_TIME_AFTER,
                300,
                SENTINEL_TIME_BEFORE,
                400,
                SENTINEL_TIME_AFTER,
                500,
            ],
            "type": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    result = filter_sentinel_timestamps(lf).collect()

    assert result.height == 5
    assert result["time"].to_list() == [100, 200, 300, 400, 500]
    assert result["type"].to_list() == [0, 2, 4, 6, 8]


def test_filter_sentinel_timestamps_accepts_alternate_time_column() -> None:
    """The ``time_column`` argument lets the caller filter usage rows."""
    lf = pl.LazyFrame(
        {
            "start_time": [
                SENTINEL_TIME_BEFORE,
                1_000_000,
                SENTINEL_TIME_AFTER,
                2_000_000,
            ],
            "collection_id": [10, 20, 30, 40],
        }
    )

    result = filter_sentinel_timestamps(lf, time_column="start_time").collect()

    assert result.height == 2
    assert result["collection_id"].to_list() == [20, 40]


# ---------------------------------------------------------------------------
# 2-4. Failure label construction (V01, V08, V27, P04)
# ---------------------------------------------------------------------------


def _every_event_type_lf() -> pl.LazyFrame:
    """Fixture covering every event type with neutral priority."""
    return pl.LazyFrame(
        {
            "time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "type": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "priority": [100] * 11,
        }
    )


def test_failure_label_primary() -> None:
    """FAIL and LOST map to 1, FINISH maps to 0, every other type is NULL."""
    result = apply_failure_label(_every_event_type_lf()).collect()
    labels = dict(zip(result["type"].to_list(), result["failure_label"].to_list()))

    assert labels[EVENT_FAIL] == 1
    assert labels[EVENT_LOST] == 1
    assert labels[EVENT_FINISH] == 0
    for t in [0, 1, 2, 3, EVENT_EVICT, EVENT_KILL, 9, 10]:
        assert labels[t] is None, f"type {t} should be NULL, got {labels[t]!r}"

    # The sensitivity column is not added when sensitivity_branch is None.
    assert "failure_label_sensitivity_prod_evict" not in result.columns


def test_failure_label_sensitivity_prod_evict() -> None:
    """Production-priority EVICTs get 1 in the sensitivity column; primary stays NULL."""
    lf = pl.LazyFrame(
        {
            "time": list(range(1, 11)),
            "type": [
                EVENT_EVICT,  # 0: free EVICT, below production
                EVENT_EVICT,  # 1: production low
                EVENT_EVICT,  # 2: production mid
                EVENT_EVICT,  # 3: production max
                EVENT_EVICT,  # 4: monitoring low
                EVENT_FAIL,   # 5
                EVENT_FINISH, # 6
                EVENT_LOST,   # 7
                EVENT_KILL,   # 8
                EVENT_EVICT,  # 9: monitoring high
            ],
            "priority": [
                100,
                PRIORITY_PRODUCTION_LOW,
                200,
                PRIORITY_MONITORING_LOW - 1,
                PRIORITY_MONITORING_LOW,
                0,
                0,
                0,
                200,
                500,
            ],
        }
    )

    result = apply_failure_label(lf, sensitivity_branch="prod_evict").collect()
    primary = result["failure_label"].to_list()
    sens = result["failure_label_sensitivity_prod_evict"].to_list()

    expected_primary = [None, None, None, None, None, 1, 0, 1, None, None]
    expected_sens = [None, 1, 1, 1, None, 1, 0, 1, None, None]

    assert primary == expected_primary
    assert sens == expected_sens


def test_failure_label_excludes_monitoring_evict() -> None:
    """Regression guard for V27.

    Monitoring-priority EVICT rows (type=4, priority>=360) must remain
    NULL under both the primary failure label and the prod-EVICT
    sensitivity branch. A production-priority EVICT at the boundary
    confirms the sensitivity branch is still active for non-monitoring
    rows.
    """
    lf = pl.LazyFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "type": [EVENT_EVICT] * 5,
            "priority": [
                PRIORITY_MONITORING_LOW,
                PRIORITY_MONITORING_LOW + 40,
                PRIORITY_MONITORING_LOW + 140,
                PRIORITY_MONITORING_LOW + 640,
                PRIORITY_MONITORING_LOW - 1,  # last is production
            ],
        }
    )

    primary_result = apply_failure_label(lf).collect()
    sens_result = apply_failure_label(lf, sensitivity_branch="prod_evict").collect()

    primary = primary_result["failure_label"].to_list()
    sens = sens_result["failure_label_sensitivity_prod_evict"].to_list()

    # Every row is NULL under the primary label (EVICT excluded by V01 + V27).
    assert primary == [None, None, None, None, None]
    # Under the sensitivity branch the four monitoring rows stay NULL,
    # only the production-boundary row gets labeled 1.
    assert sens == [None, None, None, None, 1]


def test_failure_label_invalid_sensitivity_branch_raises() -> None:
    """``sensitivity_branch`` must be one of the documented options."""
    lf = pl.LazyFrame({"time": [1], "type": [EVENT_FAIL], "priority": [0]})

    with pytest.raises(ValueError, match="sensitivity_branch"):
        apply_failure_label(lf, sensitivity_branch="bogus")


# ---------------------------------------------------------------------------
# 5-6. MNAR hardware-counter indicators (V11, V28)
# ---------------------------------------------------------------------------


def _build_mnar_observation_fixture() -> pl.LazyFrame:
    """Hand-crafted usage rows matching V11's 87.2% / 26.8% null targets.

    1000 rows total: 500 FINISH-associated rows at 87.2% null, 500
    FAIL_LOST-associated rows at 26.8% null. CPI and MAPI are nulled
    together to match V11's perfect-correlation finding (notebook 03
    follow-up query Part E).
    """
    n_finish_null = 436          # 500 * 0.872
    n_finish_present = 64
    n_fail_null = 134            # 500 * 0.268
    n_fail_present = 366

    failure_label = (
        [0] * (n_finish_null + n_finish_present)
        + [1] * (n_fail_null + n_fail_present)
    )
    cpi: list[float | None] = (
        [None] * n_finish_null
        + [1.9] * n_finish_present
        + [None] * n_fail_null
        + [2.6] * n_fail_present
    )
    mapi = list(cpi)
    return pl.LazyFrame(
        {
            "collection_id": list(range(len(failure_label))),
            "instance_index": [0] * len(failure_label),
            "cycles_per_instruction": cpi,
            "memory_accesses_per_instruction": mapi,
            "failure_label": failure_label,
        }
    )


def test_mnar_indicators_per_observation() -> None:
    """``has_cpi_value`` and ``has_mapi_value`` mirror the per-row null pattern.

    The aggregate null rate per outcome group reproduces V11's targets
    within tolerance: roughly 87.2% null for FINISH and 26.8% for
    FAIL_LOST.
    """
    lf = _build_mnar_observation_fixture()
    result = encode_hardware_counters_mnar(lf).collect()

    # Per-row indicators flip exactly with the underlying null pattern.
    cpi = result["cycles_per_instruction"].to_list()
    mapi = result["memory_accesses_per_instruction"].to_list()
    has_cpi = result["has_cpi_value"].to_list()
    has_mapi = result["has_mapi_value"].to_list()

    for c, h in zip(cpi, has_cpi):
        assert (c is None and h == 0) or (c is not None and h == 1)
    for m, h in zip(mapi, has_mapi):
        assert (m is None and h == 0) or (m is not None and h == 1)

    # Aggregate null rate per outcome group matches V11 within +/- 1pp.
    rates = (
        result.group_by("failure_label")
        .agg(pl.col("cycles_per_instruction").is_null().mean().alias("null_rate"))
        .sort("failure_label")
    )
    by_label = dict(
        zip(rates["failure_label"].to_list(), rates["null_rate"].to_list())
    )
    assert abs(by_label[0] - 0.872) < 0.01, f"FINISH null rate {by_label[0]:.4f} != 0.872"
    assert abs(by_label[1] - 0.268) < 0.01, f"FAIL_LOST null rate {by_label[1]:.4f} != 0.268"


def test_mnar_indicators_per_instance_majority() -> None:
    """Per-instance majority vote (V28). Five hand-crafted instances cover the four branches.

    F4 confirmed 39.84% of instances flip the indicator within their
    lifetime, so the per-instance majority vote is the recommended
    encoding. The fixture covers all-present, all-absent, majority-
    present, majority-absent, and exactly-50% cases.
    """
    cpi: list[float | None] = (
        [1.0, 1.0, 1.0, 1.0, 1.0]                          # A: all present
        + [None, None, None, None, None]                   # B: all absent
        + [1.0, 1.0, 1.0, None, None]                      # C: 3 of 5 present (60%)
        + [1.0, 1.0, None, None, None]                     # D: 2 of 5 present (40%)
        + [1.0, 1.0, None, None]                           # E: 2 of 4 present (50%)
    )
    collection_ids = [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 4
    instance_indices = [0] * len(collection_ids)
    mapi: list[float | None] = [1.0] * len(collection_ids)

    lf = pl.LazyFrame(
        {
            "collection_id": collection_ids,
            "instance_index": instance_indices,
            "cycles_per_instruction": cpi,
            "memory_accesses_per_instruction": mapi,
        }
    )

    result = encode_hardware_counters_mnar(lf, per_instance_majority=True).collect()
    assert "has_hardware_counters_majority" in result.columns

    by_instance = (
        result.group_by("collection_id")
        .agg(pl.col("has_hardware_counters_majority").first().alias("majority"))
        .sort("collection_id")
    )
    majority = dict(
        zip(by_instance["collection_id"].to_list(), by_instance["majority"].to_list())
    )

    assert majority[1] == 1, "all-present instance must vote 1"
    assert majority[2] == 0, "all-absent instance must vote 0"
    assert majority[3] == 1, "60% present majority must vote 1"
    assert majority[4] == 0, "40% present majority must vote 0"
    assert majority[5] == 1, "exactly 50% present must vote 1 (>= 0.5 threshold)"

    # The per-observation indicator is still added when majority is requested.
    assert "has_cpi_value" in result.columns
    assert "has_mapi_value" in result.columns


def test_mnar_indicators_opt_in_skips_majority_column() -> None:
    """When ``per_instance_majority=False`` the majority column is not added."""
    lf = pl.LazyFrame(
        {
            "collection_id": [1, 1],
            "instance_index": [0, 0],
            "cycles_per_instruction": [None, 1.0],
            "memory_accesses_per_instruction": [None, 1.0],
        }
    )

    result = encode_hardware_counters_mnar(lf, per_instance_majority=False).collect()
    assert "has_cpi_value" in result.columns
    assert "has_mapi_value" in result.columns
    assert "has_hardware_counters_majority" not in result.columns


# ---------------------------------------------------------------------------
# Backblaze preprocessing (V14-V19, the schema-era census, V44)
#
# A single synthetic fixture spans all three SMART schema eras and both
# HDD and SSD models. SMART 187 is populated only on the standard-era rows so
# the availability indicator and the era gating can be exercised.
# ---------------------------------------------------------------------------


_SSD_MODELS = frozenset({"Samsung SSD 850 EVO"})


def _backblaze_fixture() -> pl.LazyFrame:
    """Drive-day rows across the three eras plus an SSD drive to exclude.

    - ZHDD3 (Toshiba), 2 days in the early era (2013), healthy.
    - ZHDD1 (Seagate), 3 days in the standard era (2014), fails on its last day.
    - ZHDD2 (HGST), 2 days in the recent era (2022), healthy (censored).
    - ZSSD1 (Samsung SSD), 2 days in the recent era, to be filtered out.

    smart_187_raw is non-null only on the standard-era rows; smart_5_raw,
    smart_5_normalized, and smart_197_raw are populated throughout.
    """
    dates = [
        date(2013, 5, 1), date(2013, 5, 2),                       # ZHDD3 early
        date(2014, 5, 1), date(2014, 5, 2), date(2014, 5, 3),     # ZHDD1 standard
        date(2022, 6, 1), date(2022, 6, 2),                       # ZHDD2 recent
        date(2022, 6, 1), date(2022, 6, 2),                       # ZSSD1 recent SSD
    ]
    serials = [
        "ZHDD3", "ZHDD3",
        "ZHDD1", "ZHDD1", "ZHDD1",
        "ZHDD2", "ZHDD2",
        "ZSSD1", "ZSSD1",
    ]
    models = [
        "TOSHIBA MG07ACA14TA", "TOSHIBA MG07ACA14TA",
        "ST4000DM000", "ST4000DM000", "ST4000DM000",
        "HGST HMS5C4040BLE640", "HGST HMS5C4040BLE640",
        "Samsung SSD 850 EVO", "Samsung SSD 850 EVO",
    ]
    failure = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    smart_187_raw: list[float | None] = [
        None, None,
        100.0, 100.0, 100.0,
        None, None,
        None, None,
    ]
    n = len(dates)
    return pl.LazyFrame(
        {
            "date": dates,
            "serial_number": serials,
            "model": models,
            "failure": failure,
            "smart_5_raw": [0.0] * n,
            "smart_5_normalized": [100.0] * n,
            "smart_197_raw": [0.0] * n,
            "smart_187_raw": smart_187_raw,
        }
    )


def test_filter_hdds_only_removes_ssd_models() -> None:
    """SSD model rows are dropped; HDD rows survive intact."""
    result = filter_hdds_only(_backblaze_fixture(), ssd_models=_SSD_MODELS).collect()
    assert result.height == 7
    assert "ZSSD1" not in result["serial_number"].to_list()
    assert set(result["serial_number"].to_list()) == {"ZHDD1", "ZHDD2", "ZHDD3"}


def test_filter_hdds_only_noop_without_models() -> None:
    """No SSD list means no filtering (the caller asserts counts downstream)."""
    result = filter_hdds_only(_backblaze_fixture()).collect()
    assert result.height == 9


def test_assign_era_bins_by_date() -> None:
    """Each row gets the era whose inclusive date range contains its date."""
    result = assign_era(_backblaze_fixture()).collect()
    by_serial = (
        result.group_by("serial_number")
        .agg(pl.col("era").unique().alias("eras"))
    )
    eras = {
        row["serial_number"]: row["eras"]
        for row in by_serial.iter_rows(named=True)
    }
    assert eras["ZHDD3"] == ["early_2013_2014"]
    assert eras["ZHDD1"] == ["standard_2014_2021"]
    assert eras["ZHDD2"] == ["recent_2021_2025"]


def test_assign_era_marks_out_of_range_unknown() -> None:
    """A date before the census window is labeled ``unknown``."""
    lf = pl.LazyFrame({"date": [date(2010, 1, 1)], "serial_number": ["X"]})
    result = assign_era(lf).collect()
    assert result["era"].to_list() == ["unknown"]


def test_reconcile_smart_schema_adds_missing_columns() -> None:
    """Missing SMART columns are added as all-null; present ones are kept."""
    result = reconcile_smart_schema(
        _backblaze_fixture(), smart_ids=(5, 197, 187, 188, 999)
    ).collect()
    # 188 and 999 were absent and must now exist (raw + normalized).
    for col in ["smart_188_raw", "smart_188_normalized", "smart_999_raw",
                "smart_999_normalized"]:
        assert col in result.columns, f"{col} should be added"
        assert result[col].null_count() == result.height, f"{col} should be all-null"
    # A pre-existing column is untouched.
    assert result["smart_5_raw"].null_count() == 0


def test_encode_smart_availability_indicators_track_nullness() -> None:
    """``has_smart_{id}`` is 1 where the raw column is non-null, else 0."""
    lf = reconcile_smart_schema(_backblaze_fixture(), smart_ids=(187, 188))
    result = encode_smart_availability_indicators(lf, smart_ids=(187, 188)).collect()
    # 187 is populated only on the standard-era ZHDD1 rows (3 of 9).
    assert result["has_smart_187"].sum() == 3
    # 188 was reconciled to all-null, so its indicator is always 0.
    assert result["has_smart_188"].sum() == 0


def test_canonicalize_drive_model_derives_manufacturer() -> None:
    """Manufacturer is read from the canonical model-name prefix."""
    result = canonicalize_drive_model(_backblaze_fixture()).collect()
    by_serial = {
        row["serial_number"]: row["manufacturer"]
        for row in result.select(["serial_number", "manufacturer"])
        .unique()
        .iter_rows(named=True)
    }
    assert by_serial["ZHDD1"] == "Seagate"
    assert by_serial["ZHDD2"] == "HGST"
    assert by_serial["ZHDD3"] == "Toshiba"
    assert by_serial["ZSSD1"] == "Samsung"


def test_canonicalize_drive_model_applies_aliases() -> None:
    """An alias folds onto a canonical identity before manufacturer derivation."""
    aliases = {"ST4000DM000": "ST4000DM000A"}
    result = canonicalize_drive_model(_backblaze_fixture(), aliases=aliases).collect()
    seagate = result.filter(pl.col("serial_number") == "ZHDD1")
    assert seagate["model_canonical"].unique().to_list() == ["ST4000DM000A"]
    assert seagate["manufacturer"].unique().to_list() == ["Seagate"]


def test_mark_censoring_flags_terminal_observations() -> None:
    """The last row per drive is an observed failure or a censoring event."""
    hdd = filter_hdds_only(_backblaze_fixture(), ssd_models=_SSD_MODELS)
    result = mark_censoring(hdd).collect()

    # ZHDD1 fails on its last day: failure_observed on the 2014-05-03 row only.
    z1 = result.filter(pl.col("serial_number") == "ZHDD1").sort("date")
    assert z1["is_last_obs"].to_list() == [0, 0, 1]
    assert z1["failure_observed"].to_list() == [0, 0, 1]
    assert z1["censored"].to_list() == [0, 0, 0]

    # ZHDD2 simply leaves the fleet: censored on its last day, never a failure.
    z2 = result.filter(pl.col("serial_number") == "ZHDD2").sort("date")
    assert z2["is_last_obs"].to_list() == [0, 1]
    assert z2["failure_observed"].to_list() == [0, 0]
    assert z2["censored"].to_list() == [0, 1]


# ---------------------------------------------------------------------------
# Backblaze post-preprocessing assertions (validation.py)
# ---------------------------------------------------------------------------


def test_assert_failure_event_count_pass_and_fail() -> None:
    """The fixture carries exactly one failure event."""
    lf = _backblaze_fixture()
    assert assert_failure_event_count(lf, expected=1) == 1
    with pytest.raises(AssertionFailedError, match="failures"):
        assert_failure_event_count(lf, expected=5)


def test_assert_one_row_per_drive_day_pass_and_fail() -> None:
    """A unique grain passes; a duplicated pair raises."""
    assert assert_one_row_per_drive_day(_backblaze_fixture()) == 0
    dup = pl.LazyFrame(
        {
            "serial_number": ["A", "A"],
            "date": [date(2022, 1, 1), date(2022, 1, 1)],
        }
    )
    with pytest.raises(AssertionFailedError, match="duplicated"):
        assert_one_row_per_drive_day(dup)


def test_assert_era_assignment_complete_pass_and_fail() -> None:
    """All known eras pass; an out-of-range date raises."""
    ok = assign_era(filter_hdds_only(_backblaze_fixture(), ssd_models=_SSD_MODELS))
    assert assert_era_assignment_complete(ok) == 0
    bad = assign_era(pl.LazyFrame({"date": [date(2010, 1, 1)], "serial_number": ["X"]}))
    with pytest.raises(AssertionFailedError, match="unknown"):
        assert_era_assignment_complete(bad)


def test_assert_fleet_expansion_pass_and_fail() -> None:
    """Distinct drives after the cutoff must exceed those before it."""
    growing = pl.LazyFrame(
        {
            "serial_number": ["A", "B", "C", "D", "E"],
            "date": [
                date(2016, 1, 1), date(2016, 1, 2),
                date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3),
            ],
        }
    )
    before, after = assert_fleet_expansion(growing, cutoff_date="2020-01-01")
    assert (before, after) == (2, 3)

    shrinking = pl.LazyFrame(
        {
            "serial_number": ["A", "B", "C", "D"],
            "date": [
                date(2016, 1, 1), date(2016, 1, 2), date(2016, 1, 3),
                date(2022, 1, 1),
            ],
        }
    )
    with pytest.raises(AssertionFailedError, match="exceed"):
        assert_fleet_expansion(shrinking, cutoff_date="2020-01-01")


def test_backblaze_eras_constant_is_well_formed() -> None:
    """BACKBLAZE_ERAS carries three eras with the expected 187/188 placement."""
    assert len(BACKBLAZE_ERAS) == 3
    names = [name for _s, _e, name, _ids in BACKBLAZE_ERAS]
    assert names == ["early_2013_2014", "standard_2014_2021", "recent_2021_2025"]
    # SMART 187/188 are confined to the standard era (V16, V44).
    by_name = {name: set(ids) for _s, _e, name, ids in BACKBLAZE_ERAS}
    assert {187, 188}.issubset(by_name["standard_2014_2021"])
    assert 187 not in by_name["early_2013_2014"]
    assert 187 not in by_name["recent_2021_2025"]
