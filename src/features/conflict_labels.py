"""RQ2 conflict labeling for Google Cluster Traces.

RQ2 asks whether supervised classifiers can predict, at the moment a scheduling
conflict is detected, whether that conflict will *resolve cleanly* (all affected
work finishes) or *escalate* (contention turns into failure or persistent
eviction churn). This module derives the labeled conflict episodes the RQ2
notebook trains on.

Three conflict types are labeled, each as a pure ``LazyFrame -> LazyFrame``
transform consistent with the rest of :mod:`src.features` (no I/O, append-only):

1. :func:`label_resource_contention` - machine-time windows where the summed
   resource *requests* of the instances resident on a machine exceed that
   machine's capacity (over-subscription).
2. :func:`label_priority_inversion` - an EVICT (type 4) whose evicted instance
   held a *higher* priority than the instance scheduled into its place shortly
   after on the same machine.
3. :func:`label_scheduling_violations` - collections whose SCHEDULE-to-EVICT or
   SCHEDULE-to-FAIL ratios spike inside a short window.

Every labeler returns the same envelope::

    conflict_id | conflict_type | machine_id | start_time | end_time
    <detection-time feature columns ...> | resolution_outcome

:func:`build_conflict_dataset` diagonally concatenates the three into one frame
for pooled modeling (the per-type feature columns differ, so the union carries
nulls that the notebook imputes with a missing-indicator, exactly as RQ1 does).

Leakage discipline (the lesson carried from the RQ1 feature-source leak, V35).
Every feature column is computable from information available at the conflict's
``start_time``: counts, requests, priorities, capacities, and ratios observed up
to detection. The ``resolution_outcome`` label is derived from how each conflict
*ends* (terminal event types of the affected instances, or the next window's
escalation), and nothing derived from the ending may appear as a feature. Split
the resulting episodes by ``conflict_id`` (or ``machine_id``) so no episode
straddles train and test.

First-pass thresholds. The detection thresholds below (over-subscription ratio,
the replacement-matching window, the violation ratios) are starting values. The
RQ2 notebook calibrates them against the observed conflict rate and the
collection-type concentration before the modeling run, and the chosen values are
recorded in the decisions audit trail. Treat the module constants as defaults,
not as fixed findings.

Cross-references (``outputs/tables/eda_decisions.csv``): the RQ2 conflict
taxonomy and the >0.80 resolution-success target (Chapter 3 RQ2 Study
Procedures); V01 / V08 / V27 (FAIL/LOST = failure, FINISH = success, monitoring
EVICTs excluded); the collection-type concentration open question resolved in
the RQ2 notebook.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    EVENT_EVICT,
    EVENT_FAIL,
    EVENT_FINISH,
    EVENT_LOST,
    EVENT_SCHEDULE,
    PRIORITY_MONITORING_LOW,
)

# ---------------------------------------------------------------------------
# Conflict taxonomy and the shared output envelope
# ---------------------------------------------------------------------------
CONFLICT_TYPES: tuple[str, ...] = (
    "resource_contention",
    "priority_inversion",
    "scheduling_violation",
)

#: Label column every labeler emits (1 = clean resolution, 0 = escalation).
LABEL_COLUMN: str = "resolution_outcome"

#: Envelope columns that identify an episode but are NOT model features. The
#: notebook drops these before training (``conflict_type`` is kept and one-hot
#: encoded, so it is deliberately absent here).
META_COLUMNS: tuple[str, ...] = (
    "conflict_id",
    "machine_id",
    "start_time",
    "end_time",
)

# ---------------------------------------------------------------------------
# Tunable detection constants (first-pass defaults; calibrated in the notebook)
# ---------------------------------------------------------------------------
#: Microseconds per time bucket used for windowed detection (5 min).
WINDOW_US: int = 5 * 60 * 1_000_000

#: A contention window is flagged when summed requests exceed this multiple of
#: machine capacity on either resource axis.
OVERSUBSCRIPTION_RATIO: float = 1.0

#: Minimum number of concurrently resident instances for a contention window to
#: count (a single over-large request is not contention).
MIN_CONCURRENT: int = 2

#: Cap on the number of windows a single residency interval may span, so a
#: pathologically long-lived instance cannot explode the binning. Residencies
#: longer than this are truncated at detection (documented limitation).
MAX_RESIDENCY_WINDOWS: int = 288  # 24h at 5-min windows

#: Window after an EVICT in which a SCHEDULE on the same machine is treated as
#: the replacing instance (1 min).
INVERSION_MATCH_US: int = 60 * 1_000_000

#: Minimum SCHEDULE events in a collection window for a scheduling-violation
#: ratio to be meaningful.
MIN_SCHEDULE_EVENTS: int = 3

#: SCHEDULE-to-EVICT and SCHEDULE-to-FAIL ratios above which a collection window
#: is flagged as a scheduling violation.
VIOLATION_EVICT_RATIO: float = 0.5
VIOLATION_FAIL_RATIO: float = 0.25

_TERMINAL_FAIL = (EVENT_FAIL, EVENT_LOST)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _instance_summary(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per-instance request, submit-time priority / scheduling class, and
    terminal outcome flags, derived from the raw event stream.

    The outcome flags (``finished`` / ``failed`` / ``evict_count`` /
    ``has_terminal``) feed the label only; the request and submit-time priority /
    scheduling class are detection-time attributes safe to use as features.
    ``has_terminal`` is the censoring guard: an instance whose terminal event
    (FINISH or FAIL/LOST) is not observed inside the working-set window is
    right-censored, so its resolution outcome is unknown and the caller drops it
    rather than mislabeling unobserved completion as escalation.
    """
    return events_lf.group_by("collection_id", "instance_index").agg(
        cpu_request=pl.col("cpu_request").max(),
        memory_request=pl.col("memory_request").max(),
        submit_priority=pl.col("priority").sort_by("time").first(),
        scheduling_class=pl.col("scheduling_class").sort_by("time").first(),
        finished=(pl.col("type") == EVENT_FINISH).any().cast(pl.Int8),
        failed=pl.col("type").is_in(list(_TERMINAL_FAIL)).any().cast(pl.Int8),
        has_terminal=pl.col("type").is_in([EVENT_FINISH, EVENT_FAIL, EVENT_LOST]).any().cast(pl.Int8),
        evict_count=(pl.col("type") == EVENT_EVICT).sum(),
    )


def _machine_capacity(attrs_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per-machine CPU / memory capacity (max over the machine's history)."""
    return attrs_lf.group_by("machine_id").agg(
        capacity_cpus=pl.col("capacity_cpus").max(),
        capacity_memory=pl.col("capacity_memory").max(),
    )


# ---------------------------------------------------------------------------
# Strictly-prior history (the dominant failure signal, leakage-safe)
# ---------------------------------------------------------------------------
def prior_counts(events_lf: pl.LazyFrame, query_lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """Count each entity's earlier FAIL/LOST and SCHEDULE events strictly before a
    query timestamp.

    This is the leakage-safe way to attach the RQ1-dominant resubmission /
    prior-failure signal to a conflict episode: for every ``query_lf`` row (keyed
    by ``keys`` plus a ``qtime`` column) it returns the number of that entity's
    FAIL/LOST events (``prior_fail``) and SCHEDULE events (``prior_resub``) whose
    ``time`` is strictly less than ``qtime``.

    Implementation: union the entity's history events with the query rows, order
    each entity's combined stream by time (with the query placed before
    same-timestamp events so an event at exactly ``qtime`` is excluded), and read
    the cumulative event count at each query row. Near-linear, no per-row join
    blow-up. The caller joins the result back on ``keys`` plus ``qtime == time``.

    Args:
        events_lf: event stream carrying ``type``, ``time``, and ``keys``.
        query_lf: one row per conflict episode with ``keys`` and ``qtime``.
        keys: entity key columns (``["collection_id", "instance_index"]`` for
            instance history, ``["collection_id"]`` for collection history).
    """
    history = events_lf.filter(
        pl.col("type").is_in([EVENT_SCHEDULE, EVENT_FAIL, EVENT_LOST])
    ).select(
        *keys,
        t=pl.col("time"),
        dfail=pl.col("type").is_in(list(_TERMINAL_FAIL)).cast(pl.Int32),
        dsched=(pl.col("type") == EVENT_SCHEDULE).cast(pl.Int32),
        _q=pl.lit(0, dtype=pl.Int8),
    )
    query = query_lf.select(
        *keys,
        t=pl.col("qtime"),
        dfail=pl.lit(0, dtype=pl.Int32),
        dsched=pl.lit(0, dtype=pl.Int32),
        _q=pl.lit(1, dtype=pl.Int8),
    )
    # _ord places the query (0) before same-timestamp events (1), so the cumulative
    # count at a query row excludes events at exactly qtime (strict "<" semantics).
    combined = pl.concat([history, query]).with_columns(
        _ord=pl.when(pl.col("_q") == 1).then(0).otherwise(1)
    )
    counted = combined.with_columns(
        prior_fail=pl.col("dfail").cum_sum().over(keys, order_by=["t", "_ord"]),
        prior_resub=pl.col("dsched").cum_sum().over(keys, order_by=["t", "_ord"]),
    )
    return counted.filter(pl.col("_q") == 1).select(*keys, "t", "prior_fail", "prior_resub")


def _finalize(lf: pl.LazyFrame, feature_cols: list[str]) -> pl.LazyFrame:
    """Order columns into the shared envelope: meta, conflict_type, features,
    label. Keeps every labeler's output union-compatible."""
    ordered = [
        "conflict_id",
        "conflict_type",
        "machine_id",
        "start_time",
        "end_time",
        *feature_cols,
        LABEL_COLUMN,
    ]
    return lf.select(ordered)


# ---------------------------------------------------------------------------
# 1. Resource contention
# ---------------------------------------------------------------------------
def label_resource_contention(
    usage_lf: pl.LazyFrame,
    events_lf: pl.LazyFrame,
    attrs_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Label machine-time windows where resident instances' summed *requests*
    exceed machine capacity.

    Args:
        usage_lf: instance_usage rows establishing residency intervals; needs
            ``collection_id``, ``instance_index``, ``machine_id``,
            ``start_time``, ``end_time``.
        events_lf: instance_events for per-instance requests, submit-time
            priority / scheduling class, and terminal outcome (see
            :func:`_instance_summary`).
        attrs_lf: per-machine capacity with ``machine_id``, ``capacity_cpus``,
            ``capacity_memory``.

    Returns:
        One row per flagged ``(machine_id, window)`` in the shared envelope.
        ``resolution_outcome`` is 1 when no resident instance failed in the
        window (clean), 0 when at least one FAIL/LOST occurred (escalation).
        Still-running instances are treated as non-escalating, so the label is
        censoring-robust at the cost of some optimism for windows whose
        instances complete after the window.

    Detection-time features: ``n_concurrent``, ``sum_cpu_request``,
    ``sum_memory_request``, ``max_cpu_request``, ``max_memory_request``,
    ``cpu_oversub_ratio``, ``memory_oversub_ratio``, ``capacity_cpus``,
    ``capacity_memory``, ``mean_priority``, ``max_priority``, ``frac_production``,
    and the resident instances' strictly-prior history aggregated over the window
    (``prior_fail_total``, ``prior_fail_max``, ``frac_with_prior_fail``,
    ``prior_resub_mean``). All are computable from requests, counts, and prior
    events observed at the window start.
    """
    inst = _instance_summary(events_lf)
    cap = _machine_capacity(attrs_lf)

    # Residency interval per instance on its machine (usage establishes who is
    # resident when). Skip rows without a machine assignment.
    residency = (
        usage_lf.filter(pl.col("machine_id").is_not_null())
        .group_by("collection_id", "instance_index", "machine_id")
        .agg(
            resident_start=pl.col("start_time").min(),
            resident_end=pl.col("end_time").max(),
        )
        .join(inst, on=["collection_id", "instance_index"], how="inner")
        .with_columns(
            win_lo=(pl.col("resident_start") // WINDOW_US),
            win_hi=(pl.col("resident_end") // WINDOW_US),
        )
        .with_columns(
            # Truncate pathologically long residencies (documented limitation).
            win_hi=pl.min_horizontal(
                "win_hi", pl.col("win_lo") + MAX_RESIDENCY_WINDOWS
            )
        )
    )

    # Explode each residency into the windows it covers (interval-overlap
    # binning, deterministic and range-join-free).
    exploded = (
        residency.with_columns(
            window=pl.int_ranges(pl.col("win_lo"), pl.col("win_hi") + 1)
        )
        .explode("window")
        .with_columns(
            is_production=(pl.col("submit_priority") >= PRIORITY_MONITORING_LOW).cast(pl.Int8),
            window_start=pl.col("window") * WINDOW_US,
        )
    )

    # Per resident instance, its strictly-prior failure / resubmission history as
    # of the window start (counted before window_start, so leakage-safe).
    rc_query = exploded.select(
        "collection_id", "instance_index", qtime=pl.col("window_start")
    ).unique()
    rc_priors = prior_counts(events_lf, rc_query, ["collection_id", "instance_index"]).rename(
        {"t": "window_start"}
    )
    exploded = exploded.join(
        rc_priors, on=["collection_id", "instance_index", "window_start"], how="left"
    ).with_columns(
        prior_fail=pl.col("prior_fail").fill_null(0),
        prior_resub=pl.col("prior_resub").fill_null(0),
    )

    grouped = exploded.group_by("machine_id", "window").agg(
        n_concurrent=pl.len(),
        sum_cpu_request=pl.col("cpu_request").sum(),
        sum_memory_request=pl.col("memory_request").sum(),
        max_cpu_request=pl.col("cpu_request").max(),
        max_memory_request=pl.col("memory_request").max(),
        mean_priority=pl.col("submit_priority").mean(),
        max_priority=pl.col("submit_priority").max(),
        frac_production=pl.col("is_production").mean(),
        # Strictly-prior history of the resident instances (detection-time).
        prior_fail_total=pl.col("prior_fail").sum(),
        prior_fail_max=pl.col("prior_fail").max(),
        frac_with_prior_fail=(pl.col("prior_fail") > 0).mean(),
        prior_resub_mean=pl.col("prior_resub").mean(),
        # Label inputs: a window escalates if any resident instance failed.
        _n_failed=pl.col("failed").sum(),
        _n_finished=pl.col("finished").sum(),
        _n_terminal=pl.col("has_terminal").sum(),
    )

    flagged = (
        grouped.join(cap, on="machine_id", how="inner")
        .with_columns(
            cpu_oversub_ratio=pl.col("sum_cpu_request") / pl.col("capacity_cpus"),
            memory_oversub_ratio=pl.col("sum_memory_request") / pl.col("capacity_memory"),
        )
        .filter(
            (pl.col("n_concurrent") >= MIN_CONCURRENT)
            # Light censoring guard: require at least one observed terminal so the
            # window is not labeled purely from still-running instances. A strict
            # all-observed rule drops every busy window (busy machines, where
            # contention occurs, almost always carry a long-running instance that
            # does not terminate in-window), so coverage is partial by design.
            & (pl.col("_n_terminal") >= 1)
            & (
                (pl.col("cpu_oversub_ratio") > OVERSUBSCRIPTION_RATIO)
                | (pl.col("memory_oversub_ratio") > OVERSUBSCRIPTION_RATIO)
            )
        )
        .with_columns(
            conflict_id=pl.lit("rc_")
            + pl.col("machine_id").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("window").cast(pl.Utf8),
            conflict_type=pl.lit("resource_contention"),
            start_time=pl.col("window") * WINDOW_US,
            end_time=(pl.col("window") + 1) * WINDOW_US,
            # Escalate (0) if any resident instance failed; clean (1) otherwise.
            # Still-running (censored, non-failing) instances do not count as
            # escalation, which is what keeps busy contended windows in scope.
            resolution_outcome=(pl.col("_n_failed") == 0).cast(pl.Int8),
        )
    )

    feature_cols = [
        "n_concurrent",
        "sum_cpu_request",
        "sum_memory_request",
        "max_cpu_request",
        "max_memory_request",
        "cpu_oversub_ratio",
        "memory_oversub_ratio",
        "capacity_cpus",
        "capacity_memory",
        "mean_priority",
        "max_priority",
        "frac_production",
        "prior_fail_total",
        "prior_fail_max",
        "frac_with_prior_fail",
        "prior_resub_mean",
    ]
    return _finalize(flagged, feature_cols)


# ---------------------------------------------------------------------------
# 2. Priority inversion
# ---------------------------------------------------------------------------
def label_priority_inversion(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Label EVICTs where the evicted instance outranked the instance scheduled
    into its place on the same machine within :data:`INVERSION_MATCH_US`.

    Args:
        events_lf: instance_events with ``time``, ``type``, ``machine_id``,
            ``priority``, ``scheduling_class``, ``collection_id``,
            ``instance_index``.

    Returns:
        One row per inverted EVICT in the shared envelope.
        ``resolution_outcome`` is 1 when the evicted instance did not fail
        (finished or still progressing), 0 when it reached FAIL/LOST. Still-running
        instances count as non-escalating, matching the failure-based convention
        used across the three conflict types.

    Detection-time features: ``evicted_priority``, ``replacing_priority``,
    ``priority_gap``, ``evicted_scheduling_class``, ``evicted_is_production``, and
    the evicted instance's strictly-prior history (``evicted_prior_fail``,
    ``evicted_prior_resub``, ``evicted_has_prior_fail``). All are observable at the
    eviction instant.
    """
    inst = _instance_summary(events_lf)

    # Sort by time so the as-of join below can run; ``time`` is the shared key.
    evicts = events_lf.filter(
        (pl.col("type") == EVENT_EVICT) & pl.col("machine_id").is_not_null()
    ).select(
        "collection_id",
        "instance_index",
        "machine_id",
        "time",
        evicted_priority=pl.col("priority"),
        evicted_scheduling_class=pl.col("scheduling_class"),
    ).sort("time")

    schedules = events_lf.filter(
        (pl.col("type") == EVENT_SCHEDULE) & pl.col("machine_id").is_not_null()
    ).select(
        "machine_id",
        "time",
        replacing_priority=pl.col("priority"),
    ).sort("time")

    # Nearest replacing SCHEDULE on the same machine within the match window after
    # each EVICT, via an as-of join. This replaces the per-machine evict x schedule
    # cross join (which is quadratic in a machine's event count and blows up memory)
    # with a near-linear merge that keeps one replacer per evict.
    matched = evicts.join_asof(
        schedules,
        on="time",
        by="machine_id",
        strategy="forward",
        tolerance=INVERSION_MATCH_US,
    ).rename({"time": "evict_time"})

    # Strictly-prior history of the evicted instance, counted up to the eviction
    # instant (the RQ1-dominant resubmission / prior-failure signal, leakage-safe).
    inv_priors = prior_counts(
        events_lf,
        matched.select("collection_id", "instance_index", qtime=pl.col("evict_time")),
        ["collection_id", "instance_index"],
    ).rename({"t": "evict_time", "prior_fail": "evicted_prior_fail",
              "prior_resub": "evicted_prior_resub"})

    flagged = (
        matched.filter(
            pl.col("replacing_priority").is_not_null()
            & (pl.col("evicted_priority") > pl.col("replacing_priority"))
        )
        .join(
            inst.select("collection_id", "instance_index", "failed"),
            on=["collection_id", "instance_index"],
            how="left",
        )
        .join(
            inv_priors,
            on=["collection_id", "instance_index", "evict_time"],
            how="left",
        )
        .with_columns(
            conflict_id=pl.lit("pi_")
            + pl.col("machine_id").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("collection_id").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("instance_index").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("evict_time").cast(pl.Utf8),
            conflict_type=pl.lit("priority_inversion"),
            start_time=pl.col("evict_time"),
            end_time=pl.col("evict_time") + INVERSION_MATCH_US,
            priority_gap=pl.col("evicted_priority") - pl.col("replacing_priority"),
            evicted_is_production=(pl.col("evicted_priority") >= PRIORITY_MONITORING_LOW).cast(pl.Int8),
            evicted_prior_fail=pl.col("evicted_prior_fail").fill_null(0),
            evicted_prior_resub=pl.col("evicted_prior_resub").fill_null(0),
            evicted_has_prior_fail=(pl.col("evicted_prior_fail").fill_null(0) > 0).cast(pl.Int8),
            # Escalate (0) if the evicted instance failed; clean (1) otherwise
            # (finished or still progressing). Failure-based convention.
            resolution_outcome=(pl.col("failed").fill_null(0) == 0).cast(pl.Int8),
        )
    )

    feature_cols = [
        "evicted_priority",
        "replacing_priority",
        "priority_gap",
        "evicted_scheduling_class",
        "evicted_is_production",
        "evicted_prior_fail",
        "evicted_prior_resub",
        "evicted_has_prior_fail",
    ]
    return _finalize(flagged, feature_cols)


# ---------------------------------------------------------------------------
# 3. Scheduling violations
# ---------------------------------------------------------------------------
def label_scheduling_violations(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Label collection windows with abnormal SCHEDULE-to-EVICT or
    SCHEDULE-to-FAIL ratios.

    The SCHEDULE / EVICT / FAIL churn that signals a scheduling violation lives in
    the *instance* event stream, not in ``collection_events`` (collections rarely
    emit EVICT/FAIL events of their own). This labeler therefore aggregates the
    instance events of each collection into per-window counts.

    Args:
        events_lf: instance_events with ``time``, ``type``, ``collection_id``,
            ``scheduling_class`` (``priority`` optional). Should be scoped to a
            sample of *whole collections* so a collection's churn is observed in
            full, rather than the machine-scoped slice used for the other types.

    Returns:
        One row per flagged ``(collection_id, window)`` in the shared envelope.
        ``machine_id`` is null (collection-level). ``resolution_outcome`` is 1
        when the *next* window stabilizes (escalation ratios fall back below the
        flag thresholds), else 0.

    Detection-time features: ``n_schedule``, ``n_evict``, ``n_fail``,
    ``evict_ratio``, ``fail_ratio``, ``scheduling_class``, and the collection's
    strictly-prior churn history before the window (``coll_prior_fail``,
    ``coll_prior_resub``). The next-window stabilization used for the label is
    never exposed as a feature.
    """
    binned = (
        events_lf.with_columns(window=(pl.col("time") // WINDOW_US))
        .group_by("collection_id", "window")
        .agg(
            n_schedule=(pl.col("type") == EVENT_SCHEDULE).sum(),
            n_evict=(pl.col("type") == EVENT_EVICT).sum(),
            n_fail=pl.col("type").is_in(list(_TERMINAL_FAIL)).sum(),
            scheduling_class=pl.col("scheduling_class").max(),
        )
        .with_columns(
            evict_ratio=pl.col("n_evict") / pl.max_horizontal("n_schedule", pl.lit(1)),
            fail_ratio=pl.col("n_fail") / pl.max_horizontal("n_schedule", pl.lit(1)),
        )
    )

    # Collection-level strictly-prior churn before each window start (leakage-safe).
    sv_priors = prior_counts(
        events_lf,
        binned.select("collection_id", qtime=pl.col("window") * WINDOW_US).unique(),
        ["collection_id"],
    ).rename({"t": "_window_start", "prior_fail": "coll_prior_fail",
              "prior_resub": "coll_prior_resub"})
    binned = binned.with_columns(_window_start=pl.col("window") * WINDOW_US).join(
        sv_priors, on=["collection_id", "_window_start"], how="left"
    ).with_columns(
        coll_prior_fail=pl.col("coll_prior_fail").fill_null(0),
        coll_prior_resub=pl.col("coll_prior_resub").fill_null(0),
    ).drop("_window_start")

    # Next-window escalation per collection (for the label only).
    nxt = binned.select(
        "collection_id",
        prev_window=pl.col("window") - 1,
        next_evict_ratio=pl.col("evict_ratio"),
        next_fail_ratio=pl.col("fail_ratio"),
    )

    flagged = (
        binned.filter(
            (pl.col("n_schedule") >= MIN_SCHEDULE_EVENTS)
            & (
                (pl.col("evict_ratio") > VIOLATION_EVICT_RATIO)
                | (pl.col("fail_ratio") > VIOLATION_FAIL_RATIO)
            )
        )
        .join(
            nxt,
            left_on=["collection_id", "window"],
            right_on=["collection_id", "prev_window"],
            how="left",
        )
        .with_columns(
            conflict_id=pl.lit("sv_")
            + pl.col("collection_id").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("window").cast(pl.Utf8),
            conflict_type=pl.lit("scheduling_violation"),
            machine_id=pl.lit(None, dtype=pl.Int64),
            start_time=pl.col("window") * WINDOW_US,
            end_time=(pl.col("window") + 1) * WINDOW_US,
            # Stable next window (or no next window observed) = clean resolution.
            resolution_outcome=(
                pl.col("next_evict_ratio").fill_null(0.0).le(VIOLATION_EVICT_RATIO)
                & pl.col("next_fail_ratio").fill_null(0.0).le(VIOLATION_FAIL_RATIO)
            ).cast(pl.Int8),
        )
    )

    feature_cols = [
        "n_schedule",
        "n_evict",
        "n_fail",
        "evict_ratio",
        "fail_ratio",
        "scheduling_class",
        "coll_prior_fail",
        "coll_prior_resub",
    ]
    return _finalize(flagged, feature_cols)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def build_conflict_dataset(
    usage_lf: pl.LazyFrame,
    events_machine_lf: pl.LazyFrame,
    attrs_lf: pl.LazyFrame,
    events_collection_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Union the three conflict-type labelers into one pooled dataset.

    Two working-set scopes feed the union because the conflict types depend on
    different interaction structures:

    - ``usage_lf`` / ``events_machine_lf`` are *machine-scoped* (all instances on
      a sample of whole machines), so contention and priority inversion observe
      true co-residency on a machine. ``attrs_lf`` supplies that machine's
      capacity.
    - ``events_collection_lf`` is *collection-scoped* (all instances of a sample
      of whole collections), so a collection's scheduling churn is observed in
      full.

    Uses a diagonal concat so each labeler keeps its own detection-time feature
    columns; the per-type columns absent from a given row are filled with null
    (the notebook imputes them with a missing-indicator, mirroring RQ1). The
    pooled frame carries ``conflict_type`` for one-hot encoding and
    ``resolution_outcome`` as the supervised target.
    """
    parts = [
        label_resource_contention(usage_lf, events_machine_lf, attrs_lf),
        label_priority_inversion(events_machine_lf),
        label_scheduling_violations(events_collection_lf),
    ]
    return pl.concat(parts, how="diagonal_relaxed")
