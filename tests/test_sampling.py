"""Unit tests for the Google collection-level working-set sampler.

`src/features/sampling.py::build_working_set_google` retains every
failure-containing collection in full and stratified-samples the successful
collections so the `(priority_tier, scheduling_class)` instance marginals are
preserved. These tests build a small synthetic instance_events table with a
known stratum structure and assert the two contract properties: full failure
retention, and marginal preservation within 2%.

Run with::

    pytest tests/test_sampling.py -v
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import EVENT_FAIL, EVENT_FINISH, EVENT_SCHEDULE, EVENT_SUBMIT
from src.features.sampling import build_working_set_google, build_working_set_sql

# Priorities that fall in distinct bands (schemas.py): free <= 99, production 120-359.
PRIO = {"free": 50, "production": 200}


def _instance_rows(cid: int, idx: int, prio: int, sclass: int, terminal: int) -> list[tuple]:
    """Three events for one instance: SUBMIT -> SCHEDULE -> terminal, all sharing
    the same priority and scheduling_class. Columns match INSTANCE_EVENTS_SCHEMA
    order used by the test frame: collection_id, instance_index, time, type,
    priority, scheduling_class."""
    return [
        (cid, idx, 0, EVENT_SUBMIT, prio, sclass),
        (cid, idx, 10, EVENT_SCHEDULE, prio, sclass),
        (cid, idx, 20, terminal, prio, sclass),
    ]


def _build_events() -> pl.LazyFrame:
    rows: list[tuple] = []
    cid = 0
    # Success population: 4 homogeneous strata x 20 collections x 5 instances.
    # 80 success collections, 400 success instances, 100 per stratum.
    for tier, prio in PRIO.items():
        for sclass in (0, 1):
            for _c in range(20):
                cid += 1
                for k in range(5):
                    rows += _instance_rows(cid, k, prio, sclass, EVENT_FINISH)
    # Failure collections: one per stratum, each 1 FAIL + 1 FINISH instance.
    failure_ids: list[int] = []
    for tier, prio in PRIO.items():
        for sclass in (0, 1):
            cid += 1
            failure_ids.append(cid)
            rows += _instance_rows(cid, 0, prio, sclass, EVENT_FAIL)
            rows += _instance_rows(cid, 1, prio, sclass, EVENT_FINISH)
    lf = pl.LazyFrame(
        rows,
        schema=["collection_id", "instance_index", "time", "type", "priority", "scheduling_class"],
        orient="row",
    )
    return lf, failure_ids


def _marginals(strata: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Collapse the per-(tier, class) stratification rows into priority_tier and
    scheduling_class marginals for both the population and the sampled subset."""
    pop_prio, samp_prio, pop_cls, samp_cls = {}, {}, {}, {}
    for r in strata:
        pop_prio[r["priority_tier"]] = pop_prio.get(r["priority_tier"], 0.0) + r["population_frac"]
        samp_prio[r["priority_tier"]] = samp_prio.get(r["priority_tier"], 0.0) + r["sampled_frac"]
        pop_cls[r["scheduling_class"]] = pop_cls.get(r["scheduling_class"], 0.0) + r["population_frac"]
        samp_cls[r["scheduling_class"]] = samp_cls.get(r["scheduling_class"], 0.0) + r["sampled_frac"]
    return pop_prio, samp_prio, pop_cls, samp_cls


def test_all_failure_collections_retained_and_subsampled():
    events, failure_ids = _build_events()
    # Force subsampling: failures retain 8 instances; aim for ~0.5 of 400 success.
    ws_lf, manifest = build_working_set_google(events, target_instances=208)
    ws = ws_lf.collect()
    ws_collections = set(ws["collection_id"].unique().to_list())

    # 1. Every failure-containing collection is retained in full.
    assert set(failure_ids).issubset(ws_collections)
    assert manifest.retained_failure_collections == len(failure_ids)
    # Every FAIL instance survives.
    fail_rows = ws.filter(pl.col("type") == EVENT_FAIL)
    assert fail_rows.height == len(failure_ids)

    # 2. Subsampling actually happened (not all 84 collections kept).
    assert manifest.sampled_success_collections < 80
    assert manifest.total_collections == 84
    # Size lands at the target (exact for this clean construction).
    assert manifest.total_instances == 208
    # Episode count is reported (one SCHEDULE per instance here).
    assert manifest.total_episodes == manifest.total_instances


def test_priority_and_scheduling_marginals_preserved_within_2pct():
    events, _ = _build_events()
    _ws_lf, manifest = build_working_set_google(events, target_instances=208)
    pop_prio, samp_prio, pop_cls, samp_cls = _marginals(manifest.stratification)

    for tier in pop_prio:
        assert abs(samp_prio[tier] - pop_prio[tier]) <= 0.02, f"priority_tier {tier} marginal drifted"
    for sclass in pop_cls:
        assert abs(samp_cls[sclass] - pop_cls[sclass]) <= 0.02, f"scheduling_class {sclass} marginal drifted"
    # Sanity: the marginals are the expected balanced 0.5 / 0.5.
    assert abs(pop_prio["free"] - 0.5) < 1e-9
    assert abs(pop_cls[0] - 0.5) < 1e-9


def test_build_working_set_sql_renders():
    sql = build_working_set_sql("`p.d.instance_lifecycle_summary`", "`p.d.working_set_instance_ids`",
                                target_instances=75_000_000, seed=42)
    assert sql.count("(") == sql.count(")"), "unbalanced parens"
    # Full failure retention is unconditional (no fraction on the failure branch).
    assert "SELECT collection_id FROM coll WHERE has_failure = 1" in sql
    # Stratified prefix selection on the success branch.
    assert "PARTITION BY stratum_key" in sql
    assert "cum_instances <= frac.f * sr.stratum_instances" in sql
    # Reads the per-instance summary, never the raw events table.
    assert "instance_lifecycle_summary" in sql
    assert "schedule_count AS n_episodes" in sql


def test_target_larger_than_population_keeps_everything():
    events, failure_ids = _build_events()
    # Default 75M target dwarfs the synthetic data: no subsampling.
    ws_lf, manifest = build_working_set_google(events, target_size_M=75)
    ws = ws_lf.collect()
    assert manifest.sampled_success_collections == 80  # all success collections kept
    assert set(failure_ids).issubset(set(ws["collection_id"].to_list()))
    # Marginals trivially preserved when nothing is dropped.
    pop_prio, samp_prio, _pc, _sc = _marginals(manifest.stratification)
    for tier in pop_prio:
        assert abs(samp_prio[tier] - pop_prio[tier]) <= 0.02
