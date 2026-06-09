"""Scheduled-episode (per-attempt) features for Google Cluster Traces.

The instance-grain feature matrix (``notebooks/10`` Sections 2-10) labels one
row per instance with its terminal outcome and derives history from the *whole*
lifecycle. Notebook 11's prediction-point ablation showed that grain leaks: a
submission + lifecycle-history model scored an inflated MCC because the history
counts (``prior_fail_count`` / ``resubmission_count``) span the very
resubmissions that produce the terminal label, so they peek at the outcome
(instance-grain submission+history MCC ~0.95 vs episode-grain ~0.68).

The fix is to model at the *scheduled-episode* grain. An episode is one
``sched_seq`` group: the events from a SCHEDULE up to (but not into) the next
SCHEDULE. ``sched_seq`` is the running count of SCHEDULE events within an
instance. Each scheduled run becomes one row, its terminal is the first
terminal-type event in the group, and history is computed **strictly from prior
episodes only** (window framed ``UNBOUNDED PRECEDING AND 1 PRECEDING``). This
removes the peek and rebalances the classes (episode-grain neg:pos ~3.7:1 vs
~78:1 at the instance grain).

Validated in ``notebooks/11b_attempt_structure_google.py`` (segmentation),
``notebooks/10_feature_engineering_google.py`` Sections 11-12 (rebuild +
episode Tier 2/3), and ``notebooks/11_preprocessed_eda_google.py`` Section 3.8
(leakage re-check) before extraction into this module.

Cross-references (``outputs/tables/eda_decisions.csv``):
- V30 - scheduled-episode modeling grain (removes the lifecycle-history leak).
- V31 - V10 reframing: strictly-prior history adds negligible incremental
  predictive value (~+0.015 MCC) once it cannot see the label.
- V32 - episode-grain imbalance and recurring-tail handling (per-instance
  negative cap + instance-keyed group split).
- V33 - episode-grain RQ1 prediction-point curve (early-runtime meets >0.90).
- V01 / V08 / V27 - FAIL/LOST positive, FINISH negative, EVICT/KILL excluded.
- V09 / V12 / V13 - rapid-onset mechanism, utilization inversion, tier structure.

Two equivalent code paths are provided, mirroring the rest of ``src/features``:

1. Pure-Polars references and helpers (unit-testable, used in tests and at small
   scale): :func:`segment_episodes_polars`, :func:`cap_negative_episodes`,
   :func:`group_train_test_split`.
2. BigQuery SQL builders capturing the **validated production path** at the
   ~325M-episode scale: :func:`build_episode_history_sql`,
   :func:`build_episode_intervals_sql`, :func:`build_episode_usage_subset_sql`,
   :func:`build_episode_runtime_features_sql`,
   :func:`build_episode_tier3_features_sql`. The Tier 2 slope expression reuses
   :func:`src.features.runtime.slope_sql` and the Tier 3 windowed aggregates
   reuse :func:`src.features.utilization._windowed_agg_sql`, so the episode and
   instance paths encode identical feature definitions; only the grain changes
   (an extra ``sched_seq`` key) and usage is assigned to episodes by schedule
   interval rather than a symmetric band.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_KILL,
    EVENT_LOST,
    EVENT_SCHEDULE,
    EVENT_SUBMIT,
)
from src.features.runtime import slope_sql
from src.features.utilization import _windowed_agg_sql

# Grains and constants.
INSTANCE_KEY: list[str] = ["collection_id", "instance_index"]
EPISODE_KEY: list[str] = ["collection_id", "instance_index", "sched_seq"]
MICROS_PER_SEC: int = 1_000_000
EARLY_RUNTIME_BAND_US: int = 60_000_000        # Tier 2 post-schedule horizon.
TIER3_MAX_WINDOW_US: int = 3_600_000_000       # Tier 3 post-schedule horizon (60 min).

# All terminal event types (used to bound episodes and pick the terminal); the
# modeling subset keeps only FAIL/LOST (positive) and FINISH (negative).
TERMINAL_TYPES: tuple[int, ...] = (EVENT_EVICT, EVENT_FAIL, EVENT_FINISH, EVENT_KILL, EVENT_LOST)
POSITIVE_TYPES: tuple[int, int] = (EVENT_FAIL, EVENT_LOST)
MODELING_TERMINAL_TYPES: tuple[int, int, int] = (EVENT_FAIL, EVENT_LOST, EVENT_FINISH)

# Modeling-stage defaults (V32, P14).
DEFAULT_NEG_CAP: int = 5
DEFAULT_SEED: int = 42


def _types_sql(types: tuple[int, ...]) -> str:
    """Comma-separated event-type codes for a BigQuery ``IN (...)`` list."""
    return ", ".join(str(t) for t in types)


# ---------------------------------------------------------------------------
# Validated BigQuery production path.
# ---------------------------------------------------------------------------
def build_episode_history_sql(events_table: str, out_table: str) -> str:
    """Return the DDL segmenting episodes and attaching strictly-prior history.

    One pass over ``events_table`` (the sentinel-filtered, label-enriched
    ``instance_events_labeled``): tag every event with ``sched_seq`` and a
    running last-SUBMIT time, collapse to one row per ``sched_seq >= 1`` episode
    (schedule time, scheduled machine, attempt submit time, first terminal by
    time), then add strictly-prior cumulative history with a window framed
    ``UNBOUNDED PRECEDING AND 1 PRECEDING`` (the current episode is excluded,
    which is what removes the lifecycle peek). All terminal types are retained
    so the prior counts are complete; the modeling filter happens downstream.
    Mirrors :func:`segment_episodes_polars`. Arguments are fully-qualified,
    back-ticked BigQuery table names.
    """
    term = _types_sql(TERMINAL_TYPES)
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
WITH ev AS (
    SELECT
        collection_id,
        instance_index,
        time,
        type,
        machine_id,
        COUNTIF(type = {EVENT_SCHEDULE}) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS sched_seq,
        MAX(IF(type = {EVENT_SUBMIT}, time, NULL)) OVER (
            PARTITION BY collection_id, instance_index
            ORDER BY time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_last_submit
    FROM {events_table}
),
seg AS (
    SELECT
        collection_id,
        instance_index,
        sched_seq,
        MIN(IF(type = {EVENT_SCHEDULE}, time, NULL))               AS schedule_time,
        MIN(IF(type = {EVENT_SCHEDULE}, machine_id, NULL))         AS scheduled_machine_id,
        MIN(IF(type = {EVENT_SCHEDULE}, running_last_submit, NULL)) AS attempt_submit_time,
        ARRAY_AGG(
            IF(type IN ({term}), type, NULL)
            IGNORE NULLS ORDER BY time LIMIT 1
        )[SAFE_OFFSET(0)] AS terminal_type,
        COUNTIF(type IN ({term})) AS n_terminal_events
    FROM ev
    WHERE sched_seq >= 1
    GROUP BY collection_id, instance_index, sched_seq
)
SELECT
    collection_id,
    instance_index,
    sched_seq,
    schedule_time,
    scheduled_machine_id,
    attempt_submit_time,
    terminal_type,
    n_terminal_events,
    COUNT(*) OVER w                                              AS prior_episode_count,
    COUNTIF(terminal_type IN ({_types_sql(POSITIVE_TYPES)})) OVER w AS prior_fail_count,
    COUNTIF(terminal_type = {EVENT_FINISH}) OVER w              AS prior_finish_count,
    COUNTIF(terminal_type = {EVENT_EVICT}) OVER w               AS prior_evict_count
FROM seg
WINDOW w AS (
    PARTITION BY collection_id, instance_index
    ORDER BY sched_seq
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
"""


def build_episode_intervals_sql(history_table: str, out_table: str) -> str:
    """Return the DDL deriving per-episode schedule intervals.

    ``next_schedule_time`` is the same instance's next episode schedule (NULL on
    the last episode -> open-ended). Built from the episode-history table so
    every scheduled run (including EVICT/KILL/open) bounds the interval, which is
    correct for assigning usage to the run that was actually executing.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    sched_seq,
    schedule_time,
    LEAD(schedule_time) OVER (
        PARTITION BY collection_id, instance_index ORDER BY sched_seq
    ) AS next_schedule_time
FROM {history_table}
"""


def build_episode_usage_subset_sql(
    usage_table: str,
    intervals_table: str,
    out_table: str,
    horizon_us: int,
    *,
    tier3: bool = False,
) -> str:
    """Return the DDL assigning each usage observation to its containing episode
    interval and keeping the first ``horizon_us`` post-schedule.

    Interval assignment (``[schedule_time, next_schedule_time)``) makes each
    observation belong to exactly one episode, so a recurring instance's runs
    never contaminate each other (a symmetric band would double-count). Set
    ``tier3=True`` to keep only the columns the Tier 3 windows need (avg_cpu,
    avg_memory); otherwise the full V11-augmented usage row is carried for the
    Tier 2 slope/ramp/counter features. ``usage_table`` should be
    ``instance_usage_with_indicators``. Arguments are fully-qualified,
    back-ticked BigQuery table names.
    """
    if tier3:
        select_cols = """    u.collection_id,
    u.instance_index,
    e.sched_seq,
    u.avg_cpu,
    u.avg_memory,
    SAFE_DIVIDE(u.start_time - e.schedule_time, {micros}) AS sec_since_schedule""".format(
            micros=MICROS_PER_SEC
        )
    else:
        select_cols = """    u.*,
    e.sched_seq,
    e.schedule_time,
    SAFE_DIVIDE(u.start_time - e.schedule_time, {micros}) AS sec_since_schedule""".format(
            micros=MICROS_PER_SEC
        )
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
SELECT
{select_cols}
FROM {usage_table} u
INNER JOIN {intervals_table} e
  USING (collection_id, instance_index)
WHERE u.start_time >= e.schedule_time
  AND (e.next_schedule_time IS NULL OR u.start_time < e.next_schedule_time)
  AND u.start_time <= e.schedule_time + {horizon_us}
"""


def build_episode_runtime_features_sql(
    usage_subset_table: str,
    lifecycle_summary_table: str,
    out_table: str,
) -> str:
    """Return the DDL reducing the episode Tier 2 usage subset to one
    slope/ramp/counter row per episode.

    Identical to the instance-grain Stage 2 (``runtime.build_runtime_features_sql``)
    except the grain gains ``sched_seq`` in every partition / group / select.
    Reuses :func:`src.features.runtime.slope_sql` for the OLS slopes.
    ``lifecycle_summary_table`` supplies ``cpu_request`` (constant within an
    instance) for ``first_interval_util_ratio``.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
WITH ranked AS (
    SELECT
        collection_id, instance_index, sched_seq,
        sec_since_schedule, avg_cpu, avg_memory,
        cycles_per_instruction, memory_accesses_per_instruction,
        has_cpi_value, has_mapi_value,
        ROW_NUMBER() OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
        ) AS rn_post,
        LAG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
        ) AS prev_avg_cpu,
        LAG(avg_memory) OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
        ) AS prev_avg_memory,
        AVG(avg_cpu) OVER (
            PARTITION BY collection_id, instance_index, sched_seq
            ORDER BY sec_since_schedule
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        ) AS first_window_avg_cpu
    FROM {usage_subset_table}
    WHERE sec_since_schedule >= 0
),
agg AS (
    SELECT
        collection_id, instance_index, sched_seq,
        {slope_sql('avg_cpu', 5)}  AS cpu_slope_5s,
        {slope_sql('avg_cpu', 15)} AS cpu_slope_15s,
        {slope_sql('avg_cpu', 30)} AS cpu_slope_30s,
        {slope_sql('avg_memory', 5)}  AS memory_slope_5s,
        {slope_sql('avg_memory', 15)} AS memory_slope_15s,
        {slope_sql('avg_memory', 30)} AS memory_slope_30s,
        MAX(IF(rn_post = 2, avg_cpu - prev_avg_cpu, NULL))       AS initial_cpu_ramp,
        MAX(IF(rn_post = 2, avg_memory - prev_avg_memory, NULL)) AS initial_memory_ramp,
        MAX(IF(rn_post = 1, first_window_avg_cpu, NULL))         AS first_interval_avg_cpu,
        MAX(IF(rn_post = 1, cycles_per_instruction, NULL))          AS first_cpi,
        MAX(IF(rn_post = 1, memory_accesses_per_instruction, NULL)) AS first_mapi,
        MAX(IF(rn_post = 1, has_cpi_value, NULL))                AS first_has_cpi,
        MAX(IF(rn_post = 1, has_mapi_value, NULL))               AS first_has_mapi
    FROM ranked
    GROUP BY collection_id, instance_index, sched_seq
)
SELECT
    a.collection_id, a.instance_index, a.sched_seq,
    a.cpu_slope_5s, a.cpu_slope_15s, a.cpu_slope_30s,
    a.memory_slope_5s, a.memory_slope_15s, a.memory_slope_30s,
    a.initial_cpu_ramp, a.initial_memory_ramp,
    SAFE_DIVIDE(a.first_interval_avg_cpu, NULLIF(s.cpu_request, 0)) AS first_interval_util_ratio,
    IF(a.first_has_cpi  = 1, a.first_cpi,  NULL) AS cpi_value,
    IF(a.first_has_mapi = 1, a.first_mapi, NULL) AS mapi_value
FROM agg a
LEFT JOIN {lifecycle_summary_table} s
  USING (collection_id, instance_index)
"""


def build_episode_tier3_features_sql(usage_subset_table: str, out_table: str) -> str:
    """Return the DDL computing the per-episode Tier 3 windowed features.

    Identical to ``utilization.build_windowed_utilization_sql`` except grouped at
    the episode grain. Reuses :func:`src.features.utilization._windowed_agg_sql`.
    """
    return f"""
CREATE OR REPLACE TABLE {out_table}
CLUSTER BY collection_id, instance_index AS
SELECT
    collection_id,
    instance_index,
    sched_seq,
    {_windowed_agg_sql()}
FROM {usage_subset_table}
GROUP BY collection_id, instance_index, sched_seq
"""


# ---------------------------------------------------------------------------
# Pure-Polars reference + modeling-stage helpers.
# ---------------------------------------------------------------------------
def segment_episodes_polars(events_lf: pl.LazyFrame) -> pl.DataFrame:
    """Polars mirror of :func:`build_episode_history_sql` for testing.

    Builds the per-episode summary with strictly-prior history from an
    ``instance_events``-shaped LazyFrame (columns ``collection_id``,
    ``instance_index``, ``time``, ``type``, and optionally ``machine_id``).

    Returns one row per ``sched_seq >= 1`` episode keyed by
    :data:`EPISODE_KEY`, with ``schedule_time``, ``attempt_submit_time``,
    ``terminal_type``, ``n_terminal_events``, and the strictly-prior counts
    ``prior_episode_count`` / ``prior_fail_count`` / ``prior_finish_count`` /
    ``prior_evict_count``. The prior counts exclude the current episode, so a
    first episode (``prior_episode_count == 0``) always has zero prior history.
    """
    has_machine = "machine_id" in events_lf.collect_schema().names()
    base = events_lf
    if not has_machine:
        base = base.with_columns(pl.lit(None).alias("machine_id"))

    # Order within instance by time; tag sched_seq (running SCHEDULE count) and
    # the running last-SUBMIT time.
    ev = (
        base.sort([*INSTANCE_KEY, "time"])
        .with_columns(
            (pl.col("type") == EVENT_SCHEDULE).cast(pl.Int64)
            .cum_sum().over(INSTANCE_KEY).alias("sched_seq"),
            pl.when(pl.col("type") == EVENT_SUBMIT).then(pl.col("time")).otherwise(None)
            .forward_fill().over(INSTANCE_KEY).alias("running_last_submit"),
        )
        .filter(pl.col("sched_seq") >= 1)
    )

    is_term = pl.col("type").is_in(list(TERMINAL_TYPES))
    is_sched = pl.col("type") == EVENT_SCHEDULE
    # Rows are already time-ordered within instance, hence within each episode.
    seg = (
        ev.group_by(EPISODE_KEY)
        .agg(
            pl.col("time").filter(is_sched).min().alias("schedule_time"),
            pl.col("machine_id").filter(is_sched).first().alias("scheduled_machine_id"),
            pl.col("running_last_submit").filter(is_sched).first().alias("attempt_submit_time"),
            # First terminal by time (robust to within-group row order).
            pl.col("type").filter(is_term)
            .sort_by(pl.col("time").filter(is_term)).first().alias("terminal_type"),
            is_term.sum().alias("n_terminal_events"),
        )
        .sort(EPISODE_KEY)
    )

    is_fail = pl.col("terminal_type").is_in(list(POSITIVE_TYPES))
    is_finish = pl.col("terminal_type") == EVENT_FINISH
    is_evict = pl.col("terminal_type") == EVENT_EVICT
    # Strictly-prior cumulative counts: inclusive cum_sum minus the current row.
    seg = seg.with_columns(
        pl.int_range(pl.len()).over(INSTANCE_KEY).alias("prior_episode_count"),
        (is_fail.cast(pl.Int64).cum_sum().over(INSTANCE_KEY)
         - is_fail.cast(pl.Int64)).alias("prior_fail_count"),
        (is_finish.cast(pl.Int64).cum_sum().over(INSTANCE_KEY)
         - is_finish.cast(pl.Int64)).alias("prior_finish_count"),
        (is_evict.cast(pl.Int64).cum_sum().over(INSTANCE_KEY)
         - is_evict.cast(pl.Int64)).alias("prior_evict_count"),
    )
    return seg.collect()


def cap_negative_episodes(
    lf: pl.LazyFrame,
    cap: int = DEFAULT_NEG_CAP,
    seed: int = DEFAULT_SEED,
    label_col: str = "failure_label",
) -> pl.LazyFrame:
    """Keep all positive episodes; keep at most ``cap`` negative episodes per
    instance, chosen by a deterministic per-episode hash ordering (V32).

    Controls the recurring-instance negative flood (a few instances reschedule
    hundreds to ~1,894 times) without touching the source matrix. Positives are
    never capped. Expects ``collection_id``, ``instance_index``, ``sched_seq``,
    and ``label_col`` columns.
    """
    ranked = lf.with_columns(
        (pl.col("collection_id").cast(pl.Utf8) + "_"
         + pl.col("instance_index").cast(pl.Utf8) + "_"
         + pl.col("sched_seq").cast(pl.Utf8) + f"_{seed}").hash().alias("_ep_hash"),
    ).with_columns(
        pl.col("_ep_hash").rank("ordinal").over(INSTANCE_KEY).alias("_neg_rank"),
    )
    keep = (pl.col(label_col) == 1) | (pl.col("_neg_rank") <= cap)
    return ranked.filter(keep).drop(["_ep_hash", "_neg_rank"])


def group_train_test_split(
    lf: pl.LazyFrame,
    test_frac: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split episodes into train/test by INSTANCE key (group-aware) so an
    instance's episodes never straddle the split (V32).

    Deterministic via a hash of the instance key into ``[0, 1)``; instances with
    hash ``< test_frac`` form the test side. Prevents a recurring instance's
    near-identical episodes from leaking across the boundary.
    """
    keyed = lf.with_columns(
        ((pl.col("collection_id").cast(pl.Utf8) + "_"
          + pl.col("instance_index").cast(pl.Utf8) + f"_{seed}").hash() % 1_000_000 / 1_000_000)
        .alias("_grp_u")
    )
    train = keyed.filter(pl.col("_grp_u") >= test_frac).drop("_grp_u")
    test = keyed.filter(pl.col("_grp_u") < test_frac).drop("_grp_u")
    return train, test
