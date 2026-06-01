"""Unit tests for ``src/preprocessing/google_traces.py``.

Each test builds a small synthetic Polars LazyFrame hand-crafted to hit
the branches of one transform. No BigQuery, no Drive, no GCS.

Run with::

    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data.schemas import (
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
