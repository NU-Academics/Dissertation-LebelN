"""Unit tests for the lifecycle reconstruction semantics.

The production lifecycle reconstruction lives in
``src/preprocessing/lifecycle.py`` and runs as BigQuery DDL. Unit tests
exercise the data shape the BigQuery query is expected to return rather
than the query itself; the BigQuery path is validated end-to-end
against the EDA-confirmed statistics during notebook 08's section 5
smoke run.

The helper ``_reconstruct_lifecycle_polars`` is a Polars-native mirror
of the BigQuery DDL. Tests build small synthetic event tables, run the
helper, and assert on the resulting per-instance summary.

Run with::

    pytest tests/test_lifecycle.py -v
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_LOST,
    EVENT_SCHEDULE,
    EVENT_SUBMIT,
    FAIL_LOST_TYPES,
)
from src.preprocessing.google_traces import filter_sentinel_timestamps


# ---------------------------------------------------------------------------
# Polars reference reconstruction
#
# This helper mirrors the semantics of
# ``src/preprocessing/lifecycle.py::_build_lifecycle_ddl`` on a Polars
# LazyFrame. It exists for testability: the BigQuery query is too heavy
# for unit tests, but the algorithm itself can be exercised on small
# synthetic event tables.
# ---------------------------------------------------------------------------


def _reconstruct_lifecycle_polars(events_lf: pl.LazyFrame) -> pl.DataFrame:
    """Polars equivalent of the BigQuery lifecycle reconstruction.

    Returns a per-instance summary with the same columns the BigQuery
    DDL emits, plus a ``has_prior_fail`` boolean. ``has_prior_fail`` is
    True when at least one FAIL or LOST event precedes the instance's
    final (largest-time) event in the trace.

    Args:
        events_lf: instance_events-shaped LazyFrame.

    Returns:
        Polars DataFrame keyed by (collection_id, instance_index).
    """
    # Compute max time per instance up front so the has_prior_fail logic
    # can compare each FAIL/LOST event's time to the instance's terminal.
    with_max_time = events_lf.with_columns(
        pl.col("time")
        .max()
        .over(["collection_id", "instance_index"])
        .alias("_max_time")
    )

    prior_fail = (
        with_max_time
        .filter(
            pl.col("type").is_in(list(FAIL_LOST_TYPES))
            & (pl.col("time") < pl.col("_max_time"))
        )
        .group_by(["collection_id", "instance_index"])
        .agg(pl.lit(True).alias("has_prior_fail"))
    )

    summary = events_lf.group_by(["collection_id", "instance_index"]).agg(
        [
            pl.len().alias("total_events"),
            (pl.col("type") == EVENT_SUBMIT).sum().alias("submit_count"),
            (pl.col("type") == EVENT_SCHEDULE).sum().alias("schedule_count"),
            (pl.col("type") == EVENT_EVICT).sum().alias("evict_count"),
            pl.col("type")
            .is_in(list(FAIL_LOST_TYPES))
            .sum()
            .alias("fail_lost_count"),
            pl.when(pl.col("type") == EVENT_SUBMIT)
            .then(pl.col("time"))
            .min()
            .alias("submit_time"),
            pl.when(pl.col("type") == EVENT_SCHEDULE)
            .then(pl.col("time"))
            .min()
            .alias("first_schedule_time"),
            pl.when(pl.col("type") == EVENT_SCHEDULE)
            .then(pl.col("time"))
            .max()
            .alias("last_schedule_time"),
            pl.col("time").max().alias("terminal_time"),
            pl.col("type")
            .sort_by("time", descending=True)
            .first()
            .alias("terminal_type"),
        ]
    )

    summary = summary.with_columns(
        [
            pl.when(pl.col("submit_count") > 1)
            .then(pl.col("submit_count") - 1)
            .otherwise(0)
            .alias("resubmission_count"),
            (
                (pl.col("first_schedule_time") - pl.col("submit_time")) / 1_000_000.0
            ).alias("queue_time_sec"),
            (
                (pl.col("terminal_time") - pl.col("last_schedule_time")) / 1_000_000.0
            ).alias("running_duration_sec"),
            (
                (pl.col("terminal_time") - pl.col("submit_time")) / 1_000_000.0
            ).alias("total_lifecycle_sec"),
        ]
    )

    summary = summary.join(
        prior_fail, on=["collection_id", "instance_index"], how="left"
    ).with_columns(pl.col("has_prior_fail").fill_null(False))

    return summary.sort(["collection_id", "instance_index"]).collect()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _events(rows: list[dict]) -> pl.LazyFrame:
    """Build a small instance_events-shaped LazyFrame from a row list."""
    # Fill in defaults for columns the lifecycle helper does not require
    # in every row but Polars needs in the schema.
    defaults = {
        "priority": 0,
        "scheduling_class": 0,
        "machine_id": None,
        "cpu_request": None,
        "memory_request": None,
    }
    completed = []
    for row in rows:
        merged = dict(defaults)
        merged.update(row)
        completed.append(merged)
    return pl.LazyFrame(completed)


# ---------------------------------------------------------------------------
# 1. Resubmission count
# ---------------------------------------------------------------------------


def test_lifecycle_resubmission_count() -> None:
    """An instance with two SUBMIT events produces ``resubmission_count = 1``.

    Lifecycle: SUBMIT, SCHEDULE, FAIL, SUBMIT, SCHEDULE, FINISH. The
    second SUBMIT counts as a resubmission.
    """
    events = _events(
        [
            {"collection_id": 1, "instance_index": 0, "time": 1_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 2_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 3_000_000, "type": EVENT_FAIL},
            {"collection_id": 1, "instance_index": 0, "time": 4_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 5_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 6_000_000, "type": EVENT_FINISH},
        ]
    )

    summary = _reconstruct_lifecycle_polars(events)

    row = summary.to_dicts()[0]
    assert row["submit_count"] == 2
    assert row["resubmission_count"] == 1
    assert row["fail_lost_count"] == 1
    assert row["terminal_type"] == EVENT_FINISH


# ---------------------------------------------------------------------------
# 2. Prior fail flag
# ---------------------------------------------------------------------------


def test_lifecycle_prior_fail_flag() -> None:
    """``has_prior_fail`` is True when the first run terminated with FAIL.

    Instance 1: SUBMIT, SCHEDULE, FAIL, SUBMIT, SCHEDULE, FINISH.
        First run failed before the second succeeded. has_prior_fail True.
    Instance 2: SUBMIT, SCHEDULE, FINISH.
        Single-pass success. has_prior_fail False.
    Instance 3: SUBMIT, SCHEDULE, FAIL.
        Single-pass failure. The FAIL is the terminal, so there is no
        run prior to it. has_prior_fail False.
    """
    events = _events(
        [
            # Instance 1: fail then resubmit and succeed
            {"collection_id": 1, "instance_index": 0, "time": 1_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 2_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 3_000_000, "type": EVENT_FAIL},
            {"collection_id": 1, "instance_index": 0, "time": 4_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 5_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 6_000_000, "type": EVENT_FINISH},
            # Instance 2: single-pass success
            {"collection_id": 2, "instance_index": 0, "time": 1_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 2, "instance_index": 0, "time": 2_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 2, "instance_index": 0, "time": 3_000_000, "type": EVENT_FINISH},
            # Instance 3: single-pass failure (FAIL is the terminal)
            {"collection_id": 3, "instance_index": 0, "time": 1_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 3, "instance_index": 0, "time": 2_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 3, "instance_index": 0, "time": 3_000_000, "type": EVENT_FAIL},
        ]
    )

    summary = _reconstruct_lifecycle_polars(events)
    flags = dict(
        zip(summary["collection_id"].to_list(), summary["has_prior_fail"].to_list())
    )

    assert flags[1] is True, "instance with prior FAIL must set has_prior_fail True"
    assert flags[2] is False, "single-pass success must set has_prior_fail False"
    assert flags[3] is False, "single-pass FAIL has no prior run; flag must be False"


# ---------------------------------------------------------------------------
# 3. Queue time
# ---------------------------------------------------------------------------


def test_lifecycle_queue_time() -> None:
    """``queue_time_sec = (first_schedule_time - submit_time) / 1e6``."""
    events = _events(
        [
            {"collection_id": 1, "instance_index": 0, "time": 10_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 12_500_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 15_000_000, "type": EVENT_FINISH},
        ]
    )

    summary = _reconstruct_lifecycle_polars(events)
    row = summary.to_dicts()[0]

    expected_queue = (12_500_000 - 10_000_000) / 1_000_000.0
    assert row["queue_time_sec"] == expected_queue
    assert row["queue_time_sec"] == 2.5


def test_lifecycle_queue_time_with_resubmission_uses_first_schedule() -> None:
    """The queue time anchors on the FIRST schedule after the FIRST submit."""
    events = _events(
        [
            {"collection_id": 1, "instance_index": 0, "time": 10_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 12_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 14_000_000, "type": EVENT_FAIL},
            {"collection_id": 1, "instance_index": 0, "time": 15_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 17_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 20_000_000, "type": EVENT_FINISH},
        ]
    )

    summary = _reconstruct_lifecycle_polars(events)
    row = summary.to_dicts()[0]

    # First SCHEDULE is at 12s; first SUBMIT is at 10s; queue_time = 2s.
    assert row["queue_time_sec"] == 2.0
    # Last SCHEDULE is at 17s; terminal at 20s; running_duration = 3s.
    assert row["running_duration_sec"] == 3.0


# ---------------------------------------------------------------------------
# 4. Sentinel-aware running duration
# ---------------------------------------------------------------------------


def test_lifecycle_running_duration_sentinel_aware() -> None:
    """Sentinel-bearing terminals do not inflate ``running_duration_sec``.

    Without the sentinel filter, the terminal time becomes 2**63 - 1
    and ``running_duration_sec`` blows up into the 9e15-second range,
    which is the source of the EDA-discovered 2.3B-second mean before
    V25 was applied. With ``filter_sentinel_timestamps`` applied first
    the running duration stays finite and physically plausible.
    """
    raw_events = _events(
        [
            {"collection_id": 1, "instance_index": 0, "time": 1_000_000, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 2_000_000, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 5_000_000, "type": EVENT_FINISH},
            # Sentinel terminal that the pre-V25 pipeline would have kept.
            {
                "collection_id": 1,
                "instance_index": 0,
                "time": 2**63 - 1,
                "type": EVENT_FAIL,
            },
        ]
    )

    # Without the sentinel filter the FAIL terminal dominates running_duration.
    bad_summary = _reconstruct_lifecycle_polars(raw_events)
    bad_running = bad_summary.to_dicts()[0]["running_duration_sec"]
    assert bad_running > 1e9, (
        "Sanity check: a sentinel terminal should produce a >1B-second running "
        "duration without the V25 filter."
    )

    # With the sentinel filter applied first, the FINISH at t=5s becomes
    # the terminal and the running duration is the realistic 3 seconds.
    clean_summary = _reconstruct_lifecycle_polars(
        filter_sentinel_timestamps(raw_events)
    )
    clean_row = clean_summary.to_dicts()[0]
    assert clean_row["terminal_type"] == EVENT_FINISH
    assert clean_row["running_duration_sec"] == 3.0
    assert clean_row["total_lifecycle_sec"] == 4.0


# ---------------------------------------------------------------------------
# Schema parity smoke test
# ---------------------------------------------------------------------------


def test_lifecycle_schema_matches_bigquery_ddl() -> None:
    """The Polars reference exposes the same per-instance columns as the BQ DDL.

    Keeps the helper aligned with ``_build_lifecycle_ddl`` so the BQ and
    Polars paths can be cross-checked at any time.
    """
    events = _events(
        [
            {"collection_id": 1, "instance_index": 0, "time": 1, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 2, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 3, "type": EVENT_FINISH},
        ]
    )

    summary = _reconstruct_lifecycle_polars(events)

    required = {
        "collection_id",
        "instance_index",
        "total_events",
        "submit_count",
        "schedule_count",
        "evict_count",
        "fail_lost_count",
        "submit_time",
        "first_schedule_time",
        "last_schedule_time",
        "terminal_time",
        "terminal_type",
        "resubmission_count",
        "queue_time_sec",
        "running_duration_sec",
        "total_lifecycle_sec",
        "has_prior_fail",
    }
    missing = required - set(summary.columns)
    assert not missing, f"missing columns vs BQ DDL parity: {sorted(missing)}"


# Lost events also count toward fail_lost_count (V01 includes type=8).
def test_lifecycle_counts_lost_with_fail() -> None:
    """``fail_lost_count`` includes both FAIL and LOST events."""
    events = _events(
        [
            {"collection_id": 1, "instance_index": 0, "time": 1, "type": EVENT_SUBMIT},
            {"collection_id": 1, "instance_index": 0, "time": 2, "type": EVENT_SCHEDULE},
            {"collection_id": 1, "instance_index": 0, "time": 3, "type": EVENT_LOST},
        ]
    )

    summary = _reconstruct_lifecycle_polars(events)
    row = summary.to_dicts()[0]
    assert row["fail_lost_count"] == 1
    assert row["terminal_type"] == EVENT_LOST
