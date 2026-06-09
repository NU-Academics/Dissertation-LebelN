"""Unit tests for the scheduled-episode (per-attempt) feature logic.

The production episode reconstruction lives in ``src/features/episodes.py`` and
runs as BigQuery DDL at the ~325M-episode scale. These tests exercise the
Polars reference :func:`segment_episodes_polars` (a faithful mirror of
:func:`build_episode_history_sql`) on small synthetic event tables, the two
modeling-stage helpers, and the SQL builders' rendered shape.

The key property under test is the leakage fix: strictly-prior history must
exclude the current episode, so a first episode always carries zero prior
counts (the same guard notebook 10 Section 11.3 asserts at scale).

Run with::

    pytest tests/test_episodes.py -v
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_SCHEDULE,
    EVENT_SUBMIT,
)
from src.features.episodes import (
    build_episode_history_sql,
    build_episode_intervals_sql,
    build_episode_runtime_features_sql,
    build_episode_tier3_features_sql,
    build_episode_usage_subset_sql,
    cap_negative_episodes,
    group_train_test_split,
    segment_episodes_polars,
)


def _events(rows: list[tuple[int, int, int, int, int | None]]) -> pl.LazyFrame:
    """Build an instance_events-shaped LazyFrame from
    (collection_id, instance_index, time, type, machine_id) tuples."""
    return pl.LazyFrame(
        rows,
        schema=["collection_id", "instance_index", "time", "type", "machine_id"],
        orient="row",
    )


# ---------------------------------------------------------------------------
# Segmentation + strictly-prior history
# ---------------------------------------------------------------------------
def test_segmentation_two_episodes_strictly_prior_history():
    # Instance (1,1): submit -> schedule -> FAIL ; resubmit -> schedule -> FINISH.
    events = _events([
        (1, 1, 0, EVENT_SUBMIT, None),
        (1, 1, 10, EVENT_SCHEDULE, 7),
        (1, 1, 12, EVENT_FAIL, 7),
        (1, 1, 13, EVENT_SUBMIT, None),
        (1, 1, 20, EVENT_SCHEDULE, 9),
        (1, 1, 30, EVENT_FINISH, 9),
    ])
    seg = segment_episodes_polars(events).sort("sched_seq")

    assert seg.height == 2
    ep1 = seg.row(0, named=True)
    ep2 = seg.row(1, named=True)

    # Episode 1: FAIL, scheduled at t=10 off the t=0 submit, no prior history.
    assert ep1["sched_seq"] == 1
    assert ep1["schedule_time"] == 10
    assert ep1["attempt_submit_time"] == 0
    assert ep1["scheduled_machine_id"] == 7
    assert ep1["terminal_type"] == EVENT_FAIL
    assert ep1["prior_episode_count"] == 0
    assert ep1["prior_fail_count"] == 0
    assert ep1["prior_finish_count"] == 0

    # Episode 2: FINISH, scheduled at t=20 off the t=13 resubmit; prior history
    # sees exactly the FAIL from episode 1 (strictly prior, current excluded).
    assert ep2["sched_seq"] == 2
    assert ep2["schedule_time"] == 20
    assert ep2["attempt_submit_time"] == 13
    assert ep2["terminal_type"] == EVENT_FINISH
    assert ep2["prior_episode_count"] == 1
    assert ep2["prior_fail_count"] == 1
    assert ep2["prior_finish_count"] == 0


def test_first_episodes_carry_zero_prior_history():
    # Two instances, each with multiple episodes incl. an EVICT prior.
    events = _events([
        # (2,1): single FINISH episode, never submitted before its schedule.
        (2, 1, 5, EVENT_SCHEDULE, 1),
        (2, 1, 8, EVENT_FINISH, 1),
        # (3,1): EVICT episode then FAIL episode.
        (3, 1, 1, EVENT_SCHEDULE, 2),
        (3, 1, 2, EVENT_EVICT, 2),
        (3, 1, 3, EVENT_SCHEDULE, 2),
        (3, 1, 4, EVENT_FAIL, 2),
    ])
    seg = segment_episodes_polars(events)

    # Strictly-prior guard: every first episode has zero prior counts.
    firsts = seg.filter(pl.col("prior_episode_count") == 0)
    assert firsts["prior_fail_count"].sum() == 0
    assert firsts["prior_finish_count"].sum() == 0
    assert firsts["prior_evict_count"].sum() == 0
    # (2,1) never submitted: attempt_submit_time is null.
    inst21 = seg.filter((pl.col("collection_id") == 2) & (pl.col("instance_index") == 1)).row(0, named=True)
    assert inst21["attempt_submit_time"] is None

    # (3,1) second episode sees the prior EVICT but no prior FAIL.
    ep2 = seg.filter((pl.col("collection_id") == 3) & (pl.col("sched_seq") == 2)).row(0, named=True)
    assert ep2["prior_evict_count"] == 1
    assert ep2["prior_fail_count"] == 0
    assert ep2["terminal_type"] == EVENT_FAIL


def test_multi_terminal_takes_first_by_time():
    # An episode with two terminals (LOST then FINISH within the same sched_seq):
    # the first by time wins, and n_terminal_events records the multiplicity.
    from src.data.schemas import EVENT_LOST
    events = _events([
        (4, 1, 1, EVENT_SCHEDULE, 3),
        (4, 1, 5, EVENT_LOST, 3),
        (4, 1, 9, EVENT_FINISH, 3),
    ])
    seg = segment_episodes_polars(events)
    row = seg.row(0, named=True)
    assert row["terminal_type"] == EVENT_LOST
    assert row["n_terminal_events"] == 2


# ---------------------------------------------------------------------------
# Modeling-stage helpers
# ---------------------------------------------------------------------------
def _modeling_frame() -> pl.LazyFrame:
    rows = []
    # Instance A: 1 positive + 6 negative episodes (recurring tail).
    for s in range(7):
        rows.append({"collection_id": 1, "instance_index": 1, "sched_seq": s + 1,
                     "failure_label": 1 if s == 0 else 0})
    # Instance B: 1 positive + 1 negative.
    rows.append({"collection_id": 2, "instance_index": 1, "sched_seq": 1, "failure_label": 1})
    rows.append({"collection_id": 2, "instance_index": 1, "sched_seq": 2, "failure_label": 0})
    return pl.LazyFrame(rows)


def test_cap_negative_episodes_keeps_positives_caps_negatives():
    capped = _modeling_frame().pipe(cap_negative_episodes, cap=2).collect()
    # All positives retained.
    assert capped.filter(pl.col("failure_label") == 1).height == 2
    # Instance A negatives capped to 2 (from 6); instance B's single negative kept.
    a_neg = capped.filter((pl.col("collection_id") == 1) & (pl.col("failure_label") == 0)).height
    b_neg = capped.filter((pl.col("collection_id") == 2) & (pl.col("failure_label") == 0)).height
    assert a_neg == 2
    assert b_neg == 1


def test_group_split_keeps_instances_whole():
    train, test = group_train_test_split(_modeling_frame(), test_frac=0.5)
    tr, te = train.collect(), test.collect()

    def insts(df: pl.DataFrame) -> set[tuple[int, int]]:
        return set(map(tuple, df.select(["collection_id", "instance_index"]).unique().rows()))

    # No instance straddles the split, and every row is assigned exactly once.
    assert insts(tr).isdisjoint(insts(te))
    assert tr.height + te.height == _modeling_frame().collect().height


# ---------------------------------------------------------------------------
# SQL builders render at the episode grain
# ---------------------------------------------------------------------------
def test_sql_builders_render_balanced_and_episode_grained():
    ev, hist, ints = "`p.d.events`", "`p.d.hist`", "`p.d.ints`"
    usage, summ, out = "`p.d.usage`", "`p.d.summary`", "`p.d.out`"

    sqls = {
        "history": build_episode_history_sql(ev, hist),
        "intervals": build_episode_intervals_sql(hist, ints),
        "usage_t2": build_episode_usage_subset_sql(usage, ints, out, 60_000_000),
        "usage_t3": build_episode_usage_subset_sql(usage, ints, out, 3_600_000_000, tier3=True),
        "runtime": build_episode_runtime_features_sql(usage, summ, out),
        "tier3": build_episode_tier3_features_sql(usage, out),
    }
    for name, sql in sqls.items():
        assert sql.count("(") == sql.count(")"), f"{name}: unbalanced parens"
        assert "sched_seq" in sql, f"{name}: missing episode grain"

    # The strictly-prior window frame is what removes the leak.
    assert "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING" in sqls["history"]
    # Interval assignment (not a symmetric band) keys Tier 2/3 usage.
    assert "next_schedule_time IS NULL OR" in sqls["usage_t2"]
    # Episode Tier 2/3 group at the 3-part grain.
    assert "GROUP BY collection_id, instance_index, sched_seq" in sqls["runtime"]
    assert "GROUP BY collection_id, instance_index, sched_seq" in sqls["tier3"]
