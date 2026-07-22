"""Classification metrics with stratified bootstrap confidence intervals.

Every headline metric is reported as ``(point_estimate, ci_low, ci_high)`` with a
95% percentile bootstrap interval. MCC is the primary metric under the severe
class imbalance of both datasets; F1 and PR-AUC are co-reported; ROC-AUC and the
Brier score are complementary diagnostics (V23 metric strategy; P15 bootstrap
confidence intervals).

**Stratified bootstrap.** Positives and negatives are resampled independently,
each to its original count, then concatenated. This preserves the class balance in
every resample, which matters at these imbalance ratios: a naive bootstrap can
draw a resample with no positives, making PR-AUC and ROC-AUC undefined and
widening the interval artifactually. As a side benefit, every resample keeps both
classes, so the AUC metrics are always defined.

**Boundary.** Inputs may be Polars Series, NumPy arrays, or plain lists. Label
arguments (``y_true``, ``y_pred``) are coerced to integer 0/1; score arguments
(``y_score``) to float. ``calibration_table`` returns a Polars DataFrame.

The bootstrap point estimates are computed with the same closed-form / scikit-learn
routines used for the headline number, so the reported point always matches what a
reader would get from ``sklearn.metrics`` directly.

**Performance under drift.** The second half of the module holds the custom metrics
for the online-learning question: degradation rate, detection latency, retraining
effectiveness, sustainment window, and a fixed-prevalence MCC. The last of these
exists because MCC is prevalence-sensitive and the natural failure rate declines
across the evaluation years, so a raw year-over-year MCC series confounds prior
shift with the covariate drift it is meant to measure.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_SEED: int = 42
DEFAULT_N_BOOT: int = 1000
DEFAULT_ALPHA: float = 0.05

CI = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------
def _as_int(x: object) -> np.ndarray:
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    elif isinstance(x, pl.DataFrame):
        x = x.to_numpy().ravel()
    return np.asarray(x).ravel().astype(np.int64)


def _as_float(x: object) -> np.ndarray:
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    elif isinstance(x, pl.DataFrame):
        x = x.to_numpy().ravel()
    return np.asarray(x).ravel().astype(np.float64)


# ---------------------------------------------------------------------------
# Closed-form label metrics (fast inside the bootstrap loop)
# ---------------------------------------------------------------------------
def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` from a length-4 bincount of ``2*y_true + y_pred``."""
    counts = np.bincount(2 * y_true + y_pred, minlength=4)
    return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient (0 when the denominator vanishes, matching
    scikit-learn's degenerate-case convention)."""
    tn, fp, fn, tp = _confusion(y_true, y_pred)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(numerator / denominator) if denominator > 0 else 0.0


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 for the positive class (0 when there are no predicted or actual positives)."""
    _, fp, fn, tp = _confusion(y_true, y_pred)
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Stratified bootstrap engine
# ---------------------------------------------------------------------------
def _stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_other: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
    alpha: float,
    return_replicates: bool = False,
) -> CI | tuple[float, float, float, np.ndarray]:
    """Return ``(point, ci_low, ci_high)`` for ``metric_fn(y_true, y_other)`` with a
    stratified percentile bootstrap. ``y_other`` is ``y_pred`` for label metrics or
    ``y_score`` for score metrics.

    With ``return_replicates=True`` the resample distribution is appended as a
    fourth element ``(point, ci_low, ci_high, replicates)``. A one-sided threshold
    p-value is read off that same array (see :func:`bootstrap_threshold_pvalue`), so
    the interval and the p-value come from one resample distribution and cannot
    disagree. Single-class input yields an empty replicate array."""
    point = float(metric_fn(y_true, y_other))
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        # Single-class input: the interval is undefined; report the point only.
        if return_replicates:
            return point, float("nan"), float("nan"), np.empty(0, dtype=np.float64)
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = np.concatenate(
            (rng.choice(pos, size=pos.size, replace=True),
             rng.choice(neg, size=neg.size, replace=True))
        )
        boots[b] = metric_fn(y_true[idx], y_other[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if return_replicates:
        return point, float(lo), float(hi), boots
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Public metric-with-CI functions
# ---------------------------------------------------------------------------
def mcc_with_ci(y_true, y_pred, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                *, alpha: float = DEFAULT_ALPHA, return_replicates: bool = False) -> CI:
    """Matthews correlation coefficient with a stratified bootstrap CI.

    With ``return_replicates=True`` returns ``(point, ci_low, ci_high, replicates)``
    so a one-sided threshold p-value can be read off the same resamples."""
    return _stratified_bootstrap_ci(_as_int(y_true), _as_int(y_pred), _mcc,
                                    n_boot, seed, alpha, return_replicates)


def f1_with_ci(y_true, y_pred, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
               *, alpha: float = DEFAULT_ALPHA, return_replicates: bool = False) -> CI:
    """Positive-class F1 with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(_as_int(y_true), _as_int(y_pred), _f1,
                                    n_boot, seed, alpha, return_replicates)


def pr_auc_with_ci(y_true, y_score, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                   *, alpha: float = DEFAULT_ALPHA, return_replicates: bool = False) -> CI:
    """Average precision (PR-AUC) with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(
        _as_int(y_true), _as_float(y_score),
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_boot, seed, alpha, return_replicates,
    )


def roc_auc_with_ci(y_true, y_score, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                    *, alpha: float = DEFAULT_ALPHA, return_replicates: bool = False) -> CI:
    """ROC-AUC with a stratified bootstrap CI (complementary baseline)."""
    return _stratified_bootstrap_ci(
        _as_int(y_true), _as_float(y_score),
        lambda yt, ys: float(roc_auc_score(yt, ys)),
        n_boot, seed, alpha, return_replicates,
    )


def brier_score_with_ci(y_true, y_score, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                        *, alpha: float = DEFAULT_ALPHA, return_replicates: bool = False) -> CI:
    """Brier score (calibration; lower is better) with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(
        _as_int(y_true), _as_float(y_score),
        lambda yt, ys: float(brier_score_loss(yt, ys)),
        n_boot, seed, alpha, return_replicates,
    )


# ---------------------------------------------------------------------------
# One-sided threshold p-value from a bootstrap replicate distribution
# ---------------------------------------------------------------------------
def bootstrap_threshold_pvalue(replicates, threshold: float,
                               *, greater_is_better: bool = True) -> float:
    """One-sided bootstrap p-value: the share of resamples that fail to clear the target.

    This is the p-value companion to the CI-based decision rule. The threshold test
    rejects the null only when the whole CI sits on the favourable side of the
    target; this quantifies *how far* the resample distribution sits from that
    target, giving the ordering the family-wise correction needs. For a
    greater-is-better metric with the alternative "metric exceeds ``threshold``" it
    is ``mean(replicates <= threshold)``; for a less-is-better metric it is
    ``mean(replicates >= threshold)``.

    Pass the ``replicates`` array from a ``*_with_ci(..., return_replicates=True)``
    call so the p-value and the reported interval come from the same resamples. A
    return of ``0.0`` means every resample cleared the target; report it as
    ``< 1 / n_boot`` rather than as an exact zero, since the bootstrap can only
    resolve p-values down to its resample count. Non-finite replicates are dropped;
    an empty (single-class) distribution returns NaN.
    """
    reps = _as_float(replicates)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return float("nan")
    if greater_is_better:
        return float(np.mean(reps <= float(threshold)))
    return float(np.mean(reps >= float(threshold)))


# ---------------------------------------------------------------------------
# Custom metrics for performance under concept drift (RQ5)
# ---------------------------------------------------------------------------
def performance_degradation_rate(mcc_over_time, months) -> float:
    """Ordinary-least-squares slope of MCC against time, in MCC units per month.

    Negative means the model is going stale. ``months`` may be dates, datetimes, or
    a numeric month index; dates are converted to months elapsed from the first
    entry so the slope is per month regardless of input type. Returns NaN with fewer
    than two points.
    """
    y = _as_float(mcc_over_time)
    x = _months_elapsed(months)
    if y.size < 2 or x.size != y.size:
        return float("nan")
    if np.all(x == x[0]):
        return float("nan")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def drift_detection_latency(detection_time, drift_onset_estimate) -> int:
    """Days between an estimated drift onset and its detection.

    Negative when the detector fired before the onset estimate, which is a real
    outcome worth seeing rather than clipping: it means either the detector is
    firing on noise or the onset estimate is late. The schema-era boundary gives a
    ground-truth onset date, so this is measurable rather than notional there.
    """
    return int((_as_date(detection_time) - _as_date(drift_onset_estimate)).days)


def retraining_effectiveness(mcc_pre: float, mcc_post: float, mcc_reference: float) -> float:
    """Share of lost performance that retraining recovers.

    ``(mcc_post - mcc_pre) / (mcc_reference - mcc_pre)``, where ``mcc_reference`` is
    the level being recovered *toward*, normally the model's own initial (pre-drift)
    MCC. 1.0 means full recovery, 0.0 none, negative means retraining made it worse,
    above 1.0 means it overshot its starting point.

    ``mcc_reference`` must be a reachable level, not the 0.85 hypothesis threshold.
    Normalizing against an unreachable target would compress every retraining event
    into an indistinguishable sliver near zero and destroy the metric's ability to
    discriminate. Returns NaN when there was nothing to recover
    (``mcc_reference == mcc_pre``).
    """
    denominator = float(mcc_reference) - float(mcc_pre)
    if denominator == 0.0:
        return float("nan")
    return float((float(mcc_post) - float(mcc_pre)) / denominator)


def performance_sustainment_window(mcc_over_time, threshold: float, months=None) -> int:
    """Consecutive months of MCC at or above ``threshold`` before the first dip.

    Returns 0 when the series starts below the threshold, which is the expected
    result against the 0.85 target on this data and is reported as such. The
    informative version of the metric uses a reachable reference instead, for
    example ``sustainment_reference(initial_mcc)`` at 80% of the model's own
    starting MCC. ``months`` is accepted for interface symmetry and is unused: the
    series is assumed already ordered in time.
    """
    y = _as_float(mcc_over_time)
    below = np.flatnonzero(y < float(threshold))
    return int(below[0]) if below.size else int(y.size)


def sustainment_reference(initial_mcc: float, fraction: float = 0.80) -> float:
    """A reachable reference level for the sustainment window: a fraction of the
    model's own initial MCC.

    The 0.85 sustained-MCC target is tested and reported separately. It is not
    reachable on this data at natural prevalence, where the static baseline starts
    near 0.20, so a sustainment window defined against it is 0 for every strategy
    and cannot distinguish them. Defining the window against the model's own
    starting performance makes the metric informative about *degradation*, which is
    what the adaptive-versus-static comparison turns on.
    """
    return float(fraction) * float(initial_mcc)


def mcc_at_fixed_prevalence(y_true, y_pred, target_prevalence: float,
                            seed: int = DEFAULT_SEED) -> float:
    """MCC recomputed on a subsample rebalanced to a fixed positive rate.

    MCC is prevalence-sensitive, and the natural failure-day rate *declines* across
    the evaluation years (0.0048% in 2023 to 0.0037% in 2025). A falling base rate
    moves MCC on its own, with no change in the covariate distribution and no
    staleness in the model. Comparing raw per-window MCC across years therefore
    confounds prior shift with covariate drift.

    This holds prevalence fixed by discarding observations from whichever class is
    over-represented relative to ``target_prevalence`` (never duplicating any), so a
    change in the resulting series is attributable to something other than the base
    rate. Report it alongside the raw series, not instead of it: the raw series is
    what a deployed model would actually score, and the fixed-prevalence series is
    what isolates drift.
    """
    yt = _as_int(y_true)
    yp = _as_int(y_pred)
    pos = np.flatnonzero(yt == 1)
    neg = np.flatnonzero(yt == 0)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError("target_prevalence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    # Keep every observation of the limiting class; downsample the other to hit the
    # target rate exactly (as closely as integer counts allow).
    n_pos_if_neg_full = int(round(neg.size * target_prevalence / (1 - target_prevalence)))
    if n_pos_if_neg_full <= pos.size:
        keep_pos = rng.choice(pos, size=max(n_pos_if_neg_full, 1), replace=False)
        keep_neg = neg
    else:
        n_neg = int(round(pos.size * (1 - target_prevalence) / target_prevalence))
        keep_pos = pos
        keep_neg = rng.choice(neg, size=max(min(n_neg, neg.size), 1), replace=False)
    idx = np.concatenate((keep_pos, keep_neg))
    return _mcc(yt[idx], yp[idx])


def _months_elapsed(months) -> np.ndarray:
    """Months elapsed from the first entry. Accepts dates, datetimes, or numbers."""
    seq = list(months)
    if not seq:
        return np.asarray([], dtype=np.float64)
    first = seq[0]
    if hasattr(first, "year") and hasattr(first, "month"):
        base = seq[0]
        return np.asarray(
            [(d.year - base.year) * 12 + (d.month - base.month) for d in seq],
            dtype=np.float64,
        )
    return _as_float(seq)


def _as_date(value):
    """Coerce a date, datetime, or ISO date string to a ``datetime.date``."""
    from datetime import date, datetime

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"expected a date, datetime, or ISO date string; got {type(value)!r}")


def calibration_table(y_true, y_score, n_bins: int = 10) -> pl.DataFrame:
    """Reliability table over ``n_bins`` equal-width probability bins on [0, 1].

    Columns: ``bin`` (index), ``bin_low``, ``bin_high``, ``n`` (count),
    ``mean_predicted`` (mean predicted probability in the bin), and
    ``observed_rate`` (fraction of actual positives in the bin). Empty bins carry
    null ``mean_predicted`` / ``observed_rate``. A well-calibrated model has
    ``mean_predicted`` close to ``observed_rate`` in every populated bin.
    """
    yt = _as_int(y_true)
    ys = _as_float(y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each score to a bin in [0, n_bins-1]; the rightmost edge is inclusive.
    bin_idx = np.clip(np.digitize(ys, edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin": b,
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "n": n,
            "mean_predicted": float(ys[mask].mean()) if n else None,
            "observed_rate": float(yt[mask].mean()) if n else None,
        })
    return pl.DataFrame(rows)
