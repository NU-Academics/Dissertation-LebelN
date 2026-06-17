"""Collection-level working-set construction for Google Cluster Traces.

The full Google population is too large to model directly, so RQ1-RQ4 train on
a **working set**: a representative subsample built with two non-negotiable
properties.

1. **Full failure retention.** Every collection that contains at least one FAIL
   (type 5) or LOST (type 8) instance is kept in its entirety. Failures are rare
   (P01/V02), so none are discarded; sampling only ever thins the abundant
   successful collections.
2. **Stratified success sampling.** The successful (non-failure) collections are
   subsampled so the joint ``(priority_tier, scheduling_class)`` distribution of
   the retained successful **instances** matches the successful population within
   a small tolerance (the marginals are preserved). This keeps the priority and
   scheduling-class mix representative rather than letting random sampling drift
   it.

Sampling happens at the **collection** grain (whole collections are kept or
dropped) so within-collection structure is never broken. That keeps the working
set valid for the scheduled-episode regrain (``src/features/episodes.py``, V30):
every episode of a retained instance is present, so strictly-prior history is
intact. The size target is expressed in **unique instances** (the P01 ``50-100
million instance events`` commitment, default 75M as the midpoint); the manifest
also reports the scheduled-episode count for the regrained modeling stage.

Validated against the Phase 3 plan and ``eda_decisions.csv`` rows P01 (working-set
size), V02 (imbalance handling), V07 (priority/scheduling features), and V30-V32
(the episode regrain that consumes this working set).

Scale note: like the other ``src/features`` modules this provides two paths.
:func:`build_working_set_google` is the pure-Polars reference, correct at any
scale and unit-tested on small synthetic frames, but it materializes the
per-instance summary in process. :func:`build_working_set_sql` is the validated
production path: it runs the identical selection BigQuery-side off the per-instance
``instance_lifecycle_summary`` (so the 1.72B-row events table is never scanned)
and is what notebook 10 Section 1.1 calls to lock ``working_set_instance_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from src.data.schemas import (
    EVENT_FAIL,
    EVENT_LOST,
    EVENT_SCHEDULE,
    PRIORITY_BEST_EFFORT_LOW,
    PRIORITY_BEST_EFFORT_MAX,
    PRIORITY_FREE_MAX,
    PRIORITY_MID_TIER_LOW,
    PRIORITY_MID_TIER_MAX,
    PRIORITY_MONITORING_LOW,
    PRIORITY_PRODUCTION_LOW,
    PRIORITY_PRODUCTION_MAX,
)

INSTANCE_KEY: list[str] = ["collection_id", "instance_index"]
DEFAULT_SEED: int = 42
TARGET_BAND_M: tuple[int, int] = (50, 100)  # P01 acceptable working-set band.


@dataclass
class SamplingManifest:
    """Provenance record for a constructed Google working set.

    Attributes:
        total_collections: distinct collections in the source events.
        retained_failure_collections: collections kept in full because they
            contain >= 1 FAIL/LOST instance.
        sampled_success_collections: successful collections selected by the
            stratified sampler.
        total_instances: unique instances in the working set (the size metric).
        total_episodes: scheduled episodes (SCHEDULE events) in the working set,
            for the episode-grain modeling stage (informational).
        target_instances: the instance-count target the sampler aimed for.
        size_metric: the unit ``total_instances`` is counted in.
        seed: deterministic sampling seed (P14).
        stratification: one row per ``(priority_tier, scheduling_class)`` over the
            **successful** instances, with the population and sampled instance
            counts and fractions used to verify marginal preservation.
    """

    total_collections: int
    retained_failure_collections: int
    sampled_success_collections: int
    total_instances: int
    total_episodes: int
    target_instances: int
    size_metric: str = "instances"
    seed: int = DEFAULT_SEED
    stratification: list[dict] = field(default_factory=list)


def _priority_tier_expr(priority: pl.Expr) -> pl.Expr:
    """Map a ``priority`` expression to its Borg band label. Mirrors
    ``scheduling.priority_tier_expr`` but parameterized by column so it can run
    on the per-instance representative priority (bands from ``schemas.py``)."""
    return (
        pl.when(priority <= PRIORITY_FREE_MAX).then(pl.lit("free"))
        .when((priority >= PRIORITY_BEST_EFFORT_LOW) & (priority <= PRIORITY_BEST_EFFORT_MAX)).then(pl.lit("best_effort"))
        .when((priority >= PRIORITY_MID_TIER_LOW) & (priority <= PRIORITY_MID_TIER_MAX)).then(pl.lit("mid"))
        .when((priority >= PRIORITY_PRODUCTION_LOW) & (priority <= PRIORITY_PRODUCTION_MAX)).then(pl.lit("production"))
        .when(priority >= PRIORITY_MONITORING_LOW).then(pl.lit("monitoring"))
        .otherwise(pl.lit("unknown"))
    )


def _per_instance_summary(events_lf: pl.LazyFrame) -> pl.DataFrame:
    """Reduce instance_events to one row per instance: terminal type, the
    submit-time (first-event) priority_tier / scheduling_class, the event and
    scheduled-episode counts, and the failure flag (terminal FAIL/LOST, V01).
    Priority and scheduling class use the first event, not the terminal one, so
    the stratification keys do not encode the outcome (V35)."""
    return (
        events_lf.group_by(INSTANCE_KEY)
        .agg(
            pl.col("type").sort_by("time").last().alias("terminal_type"),
            pl.col("priority").sort_by("time").first().alias("priority"),
            pl.col("scheduling_class").sort_by("time").first().alias("scheduling_class"),
            pl.len().alias("n_events"),
            (pl.col("type") == EVENT_SCHEDULE).sum().alias("n_episodes"),
        )
        .with_columns(
            _priority_tier_expr(pl.col("priority")).alias("priority_tier"),
            pl.col("terminal_type").is_in([EVENT_FAIL, EVENT_LOST]).alias("is_failure"),
        )
        .with_columns(
            (pl.col("priority_tier") + "|" + pl.col("scheduling_class").cast(pl.Utf8)).alias("stratum_key")
        )
        .collect()
    )


def _select_success_collections(
    success_coll: pl.DataFrame, budget_instances: int, seed: int
) -> set[int]:
    """Pick successful collections so that, within each modal
    ``(priority_tier, scheduling_class)`` stratum, the selected instance count is
    as close as possible to ``fraction * stratum_population`` where
    ``fraction = budget / total_success_instances``. Whole collections are
    selected in a deterministic hash order, which preserves the joint stratum
    distribution (hence both marginals)."""
    total_success = int(success_coll["n_instances"].sum())
    if total_success == 0:
        return set()
    if budget_instances >= total_success:
        return set(success_coll["collection_id"].to_list())

    frac = budget_instances / total_success
    chosen: set[int] = set()
    for (_stratum,), grp in success_coll.group_by("stratum_key", maintain_order=True):
        pop = int(grp["n_instances"].sum())
        target = frac * pop
        ordered = grp.with_columns(
            (pl.col("collection_id").cast(pl.Utf8) + f"_{seed}").hash().alias("_h")
        ).sort("_h")
        cum = 0
        for row in ordered.iter_rows(named=True):
            nxt = cum + row["n_instances"]
            if nxt <= target:
                chosen.add(row["collection_id"])
                cum = nxt
            else:
                # Include the crossing collection only if it lands closer to target.
                if abs(target - nxt) < abs(target - cum):
                    chosen.add(row["collection_id"])
                break
    return chosen


def build_working_set_google(
    events_lf: pl.LazyFrame,
    target_size_M: float = 75,
    seed: int = DEFAULT_SEED,
    target_instances: int | None = None,
) -> tuple[pl.LazyFrame, SamplingManifest]:
    """Construct a Google working set: retain all failure-containing collections,
    stratified-sample the successful collections, return the working-set
    LazyFrame and its :class:`SamplingManifest`.

    Args:
        events_lf: instance_events-shaped LazyFrame (columns ``collection_id``,
            ``instance_index``, ``time``, ``type``, ``priority``,
            ``scheduling_class``).
        target_size_M: target working-set size in **millions of instances**
            (default 75, the midpoint of the P01 50-100M band).
        seed: deterministic sampling seed (P14).
        target_instances: optional explicit instance target overriding
            ``target_size_M`` (used in tests; production uses ``target_size_M``).

    Returns:
        ``(working_set_lf, manifest)`` where ``working_set_lf`` is ``events_lf``
        filtered to the selected collections and ``manifest`` records the counts
        and the per-stratum success-instance marginals.
    """
    if target_instances is None:
        target_instances = int(round(target_size_M * 1_000_000))

    inst = _per_instance_summary(events_lf)

    coll = inst.group_by("collection_id").agg(
        pl.col("is_failure").any().alias("has_failure"),
        pl.len().alias("n_instances"),
        pl.col("n_episodes").sum().alias("n_episodes"),
        # Modal stratum for the collection (deterministic tie-break).
        pl.col("stratum_key").mode().sort().first().alias("stratum_key"),
    )

    failure_coll = coll.filter(pl.col("has_failure"))
    success_coll = coll.filter(~pl.col("has_failure"))

    retained_failure_instances = int(failure_coll["n_instances"].sum())
    budget = max(0, target_instances - retained_failure_instances)
    selected_success_ids = _select_success_collections(success_coll, budget, seed)

    failure_ids = set(failure_coll["collection_id"].to_list())
    selected_ids = failure_ids | selected_success_ids

    # Instance-level success marginals: population vs the sampled subset. The
    # success population is the instances in success COLLECTIONS (the sampling
    # universe), not all non-failure instances; a FINISH instance inside a
    # retained failure collection is excluded since it is never sampled here.
    inst_lab = inst.join(coll.select(["collection_id", "has_failure"]), on="collection_id")
    success_pop = inst_lab.filter(~pl.col("has_failure"))
    sampled_success = success_pop.filter(pl.col("collection_id").is_in(list(selected_success_ids)))
    pop_total = max(success_pop.height, 1)
    samp_total = max(sampled_success.height, 1)
    pop_strata = (
        success_pop.group_by(["priority_tier", "scheduling_class"]).len().rename({"len": "population_instances"})
    )
    samp_strata = (
        sampled_success.group_by(["priority_tier", "scheduling_class"]).len().rename({"len": "sampled_instances"})
    )
    strata = (
        pop_strata.join(samp_strata, on=["priority_tier", "scheduling_class"], how="left")
        .with_columns(pl.col("sampled_instances").fill_null(0))
        .with_columns(
            (pl.col("population_instances") / pop_total).alias("population_frac"),
            (pl.col("sampled_instances") / samp_total).alias("sampled_frac"),
        )
        .sort(["priority_tier", "scheduling_class"])
    )

    selected_inst = inst.filter(pl.col("collection_id").is_in(list(selected_ids)))
    manifest = SamplingManifest(
        total_collections=coll.height,
        retained_failure_collections=failure_coll.height,
        sampled_success_collections=len(selected_success_ids),
        total_instances=selected_inst.height,
        total_episodes=int(selected_inst["n_episodes"].sum()),
        target_instances=target_instances,
        seed=seed,
        stratification=strata.to_dicts(),
    )

    working_set_lf = events_lf.filter(pl.col("collection_id").is_in(list(selected_ids)))
    return working_set_lf, manifest


# ---------------------------------------------------------------------------
# Validated BigQuery production path (population scale).
# ---------------------------------------------------------------------------
def build_working_set_sql(
    lifecycle_summary_table: str,
    out_table: str,
    target_instances: int,
    seed: int = DEFAULT_SEED,
) -> str:
    """Return the DDL that builds the locked working set at population scale.

    Mirrors :func:`build_working_set_google` but reads the per-instance
    ``instance_lifecycle_summary`` (so the 1.72B-row events table is never
    scanned) and runs the selection BigQuery-side: retain every collection with a
    FAIL/LOST instance, then keep successful collections within each modal
    ``(priority_tier, scheduling_class)`` stratum up to ``f * stratum_instances``
    where ``f = (target - failure_instances) / success_instances``. Collections
    are ordered by a deterministic ``FARM_FINGERPRINT`` hash so the prefix kept
    per stratum preserves the joint distribution (hence both marginals). Output
    columns: ``collection_id``, ``instance_index``, ``schedule_time`` (the locked
    working-set table the feature notebook joins against). Arguments are
    fully-qualified, back-ticked BigQuery table names.
    """
    tier_case = f"""CASE
            WHEN submit_priority <= {PRIORITY_FREE_MAX} THEN 'free'
            WHEN submit_priority BETWEEN {PRIORITY_BEST_EFFORT_LOW} AND {PRIORITY_BEST_EFFORT_MAX} THEN 'best_effort'
            WHEN submit_priority BETWEEN {PRIORITY_MID_TIER_LOW} AND {PRIORITY_MID_TIER_MAX} THEN 'mid'
            WHEN submit_priority BETWEEN {PRIORITY_PRODUCTION_LOW} AND {PRIORITY_PRODUCTION_MAX} THEN 'production'
            WHEN submit_priority >= {PRIORITY_MONITORING_LOW} THEN 'monitoring'
            ELSE 'unknown'
        END"""
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
WITH inst AS (
    SELECT
        collection_id,
        instance_index,
        first_schedule_time AS schedule_time,
        {tier_case} AS priority_tier,
        submit_scheduling_class AS scheduling_class,
        IF(outcome = 'FAIL_LOST', 1, 0) AS is_failure,
        schedule_count AS n_episodes
    FROM {lifecycle_summary_table}
    WHERE first_schedule_time IS NOT NULL
      AND outcome IN ('FAIL_LOST', 'FINISH')
),
coll AS (
    SELECT
        collection_id,
        MAX(is_failure) AS has_failure,
        COUNT(*) AS n_instances,
        APPROX_TOP_COUNT(
            CONCAT(priority_tier, '|', CAST(scheduling_class AS STRING)), 1
        )[OFFSET(0)].value AS stratum_key
    FROM inst
    GROUP BY collection_id
),
totals AS (
    SELECT
        SUM(IF(has_failure = 0, n_instances, 0)) AS success_instances,
        SUM(IF(has_failure = 1, n_instances, 0)) AS failure_instances
    FROM coll
),
frac AS (
    SELECT
        CASE
            WHEN success_instances = 0 THEN 0.0
            WHEN GREATEST({target_instances} - failure_instances, 0) >= success_instances THEN 1.0
            ELSE GREATEST({target_instances} - failure_instances, 0) / success_instances
        END AS f
    FROM totals
),
success_ranked AS (
    SELECT
        collection_id,
        SUM(n_instances) OVER (
            PARTITION BY stratum_key
            ORDER BY FARM_FINGERPRINT(CONCAT(CAST(collection_id AS STRING), '_{seed}'))
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cum_instances,
        SUM(n_instances) OVER (PARTITION BY stratum_key) AS stratum_instances
    FROM coll
    WHERE has_failure = 0
),
selected AS (
    SELECT collection_id FROM coll WHERE has_failure = 1
    UNION ALL
    SELECT sr.collection_id
    FROM success_ranked sr
    CROSS JOIN frac
    WHERE sr.cum_instances <= frac.f * sr.stratum_instances
)
SELECT
    i.collection_id,
    i.instance_index,
    i.schedule_time
FROM inst i
JOIN selected s USING (collection_id)
"""
