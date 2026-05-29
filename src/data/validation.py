"""Post-preprocessing assertion helpers.

Functions here check single invariants the EDA established, with
tolerance bands tuned to the Phase 2 evidence. They are called from
notebook 08 Section 6 (and will move into a ``run_preprocessing_assertions``
wrapper in step 2.4) and from ``tests/test_preprocessing.py``.

Each assertion raises ``AssertionFailedError`` on tolerance violation
and returns the observed value on success, so callers can log it. Pure
LazyFrame functions, no I/O.
"""

from __future__ import annotations

import polars as pl

from src.data.schemas import (
    EVENT_EVICT,
    PRIORITY_MONITORING_LOW,
)


class AssertionFailedError(AssertionError):
    """Raised when a preprocessing invariant violates its tolerance band."""


def assert_class_balance(
    lf: pl.LazyFrame,
    expected_ratio: float,
    tolerance: float = 0.05,
    label_column: str = "failure_label",
) -> float:
    """Assert the negative:positive class ratio is within tolerance.

    For the primary Google failure label the V02 baseline is
    approximately 3.4:1 (FINISH : FAIL_LOST). Rows where the label is
    NULL are excluded before the ratio is computed.

    Args:
        lf: LazyFrame containing ``label_column``.
        expected_ratio: target ratio of negatives to positives.
        tolerance: fractional tolerance. ``0.05`` allows the observed
            ratio to fall in ``[0.95 * expected, 1.05 * expected]``.
        label_column: name of the binary failure label column.

    Returns:
        The observed ratio.

    Raises:
        AssertionFailedError: when the observed ratio falls outside the
            tolerance band.
    """
    counts = (
        lf.filter(pl.col(label_column).is_not_null())
          .group_by(label_column)
          .agg(pl.len().alias("n"))
          .collect()
    )
    by_label = dict(
        zip(counts[label_column].to_list(), counts["n"].to_list())
    )
    n_pos = int(by_label.get(1, 0))
    n_neg = int(by_label.get(0, 0))
    if n_pos == 0:
        raise AssertionFailedError(
            "assert_class_balance: no positives observed; cannot compute ratio."
        )
    observed = n_neg / n_pos
    lower = expected_ratio * (1.0 - tolerance)
    upper = expected_ratio * (1.0 + tolerance)
    if not (lower <= observed <= upper):
        raise AssertionFailedError(
            f"assert_class_balance: observed {observed:.4f} outside "
            f"[{lower:.4f}, {upper:.4f}] (target {expected_ratio:.4f}, "
            f"tolerance {tolerance:.1%})."
        )
    return observed


def assert_null_rate(
    lf: pl.LazyFrame,
    column: str,
    expected_rate: float,
    tolerance: float = 0.01,
) -> float:
    """Assert the null rate of ``column`` is within tolerance.

    Used for V04 ``cpu_request`` / ``memory_request`` post-drop checks
    and for V11 record-level CPI / MAPI null-rate verification when the
    caller has already restricted to a single outcome bucket.

    Args:
        lf: LazyFrame to check.
        column: column whose null rate is checked.
        expected_rate: target null rate as a fraction in ``[0, 1]``.
        tolerance: absolute tolerance band, in fractional units. ``0.01``
            allows the observed rate to land within +/- 1 percentage
            point of the target.

    Returns:
        The observed null rate as a fraction.

    Raises:
        AssertionFailedError: when the observed rate falls outside the
            tolerance band, or when the LazyFrame is empty.
    """
    stats = (
        lf.select([
            pl.len().alias("n"),
            pl.col(column).is_null().sum().alias("n_null"),
        ])
        .collect()
        .to_dicts()[0]
    )
    n = int(stats["n"])
    n_null = int(stats["n_null"])
    if n == 0:
        raise AssertionFailedError(
            f"assert_null_rate: '{column}' frame is empty; cannot compute rate."
        )
    observed = n_null / n
    lower = expected_rate - tolerance
    upper = expected_rate + tolerance
    if not (lower <= observed <= upper):
        raise AssertionFailedError(
            f"assert_null_rate: '{column}' observed {observed:.4f} outside "
            f"[{lower:.4f}, {upper:.4f}]."
        )
    return observed


def assert_tier3_inversion(
    lf: pl.LazyFrame,
    cpu_column: str = "avg_cpu",
    label_column: str = "failure_label",
) -> tuple[float, float]:
    """Confirm the V12 utilization inversion survives preprocessing.

    V12 documented that failing instances exhibit LOWER median absolute
    CPU utilization than successful ones (FAIL_LOST median 0.012 vs
    FINISH 0.081 in the Phase 2 EDA follow-up query). This assertion
    is the regression guard for that finding: if preprocessing or
    feature engineering accidentally washes out the inversion, the
    Chapter 4 Tier 3 ablation loses its empirical anchor.

    The LazyFrame must contain one row per usage observation joined to
    its instance's failure label. Callers that have not yet joined are
    responsible for doing so before invoking this assertion.

    Args:
        lf: LazyFrame with at minimum ``cpu_column`` and
            ``label_column``.
        cpu_column: name of the CPU utilization column.
        label_column: name of the binary failure label column.

    Returns:
        Tuple ``(median_cpu_when_fail, median_cpu_when_finish)``.

    Raises:
        AssertionFailedError: when either label group is missing from
            the input, or when the inversion does not hold.
    """
    medians = (
        lf.filter(pl.col(label_column).is_not_null())
          .group_by(label_column)
          .agg(pl.col(cpu_column).median().alias("median_cpu"))
          .collect()
    )
    by_label = dict(
        zip(medians[label_column].to_list(), medians["median_cpu"].to_list())
    )
    median_fail = by_label.get(1)
    median_finish = by_label.get(0)
    if median_fail is None or median_finish is None:
        raise AssertionFailedError(
            f"assert_tier3_inversion: missing a label group in '{cpu_column}'. "
            f"Saw labels {sorted(by_label.keys())}."
        )
    if not (median_fail < median_finish):
        raise AssertionFailedError(
            f"assert_tier3_inversion: V12 inversion broken on '{cpu_column}'. "
            f"median(fail) {median_fail:.4f} should be less than "
            f"median(finish) {median_finish:.4f}."
        )
    return float(median_fail), float(median_finish)


def assert_monitoring_evict_excluded(
    lf: pl.LazyFrame,
    label_columns: tuple[str, ...] = (
        "failure_label",
        "failure_label_sensitivity_prod_evict",
    ),
    type_column: str = "type",
    priority_column: str = "priority",
    monitoring_priority_low: int = PRIORITY_MONITORING_LOW,
    evict_type: int = EVENT_EVICT,
) -> None:
    """Assert monitoring-priority EVICT rows carry NULL labels (V27).

    Regression guard for the V27 exclusion decision. Any monitoring
    EVICT row that ends up with a non-NULL label in any column listed
    in ``label_columns`` means the labeling logic regressed and now
    treats canary / health-check preemptions as failures. The F3.2
    repeats distribution (notebook 07b Section 4) is the load-bearing
    evidence for excluding them.

    Args:
        lf: LazyFrame of event-level rows including ``type_column``,
            ``priority_column``, and every column in ``label_columns``.
        label_columns: label columns to check for leaks.
        type_column: column carrying the event-type code.
        priority_column: column carrying the priority.
        monitoring_priority_low: priority threshold at and above which
            an EVICT counts as monitoring (default 360 per
            ``src/data/schemas.PRIORITY_MONITORING_LOW``).
        evict_type: event-type code for EVICT (default 4 per
            ``src/data/schemas.EVENT_EVICT``).

    Raises:
        AssertionFailedError: when any monitoring EVICT row is labeled.
    """
    base = lf.filter(
        (pl.col(type_column) == evict_type)
        & (pl.col(priority_column) >= monitoring_priority_low)
    )
    select_exprs = [
        pl.col(col).is_not_null().sum().alias(f"n_{col}_labeled")
        for col in label_columns
    ]
    result = base.select(select_exprs).collect().to_dicts()[0]
    leaks = {k: int(v) for k, v in result.items() if int(v) > 0}
    if leaks:
        raise AssertionFailedError(
            f"assert_monitoring_evict_excluded: monitoring EVICTs were "
            f"labeled despite V27. Leaks: {leaks}."
        )


def assert_row_count(
    lf: pl.LazyFrame,
    expected_min: int,
    expected_max: int,
) -> int:
    """Assert the LazyFrame row count falls inside a range.

    Used by the lifecycle reconstruction guard (~74.5M unique
    ``(collection_id, instance_index)`` pairs per F4) and by section-
    level row-count checks in notebook 08.

    Args:
        lf: LazyFrame whose rows are counted.
        expected_min: lower bound (inclusive).
        expected_max: upper bound (inclusive).

    Returns:
        The observed row count.

    Raises:
        AssertionFailedError: when the count is outside the range.
    """
    observed = int(lf.select(pl.len()).collect().item())
    if not (expected_min <= observed <= expected_max):
        raise AssertionFailedError(
            f"assert_row_count: observed {observed:,} outside "
            f"[{expected_min:,}, {expected_max:,}]."
        )
    return observed
