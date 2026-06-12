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
) -> CI:
    """Return ``(point, ci_low, ci_high)`` for ``metric_fn(y_true, y_other)`` with a
    stratified percentile bootstrap. ``y_other`` is ``y_pred`` for label metrics or
    ``y_score`` for score metrics."""
    point = float(metric_fn(y_true, y_other))
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        # Single-class input: the interval is undefined; report the point only.
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
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Public metric-with-CI functions
# ---------------------------------------------------------------------------
def mcc_with_ci(y_true, y_pred, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                *, alpha: float = DEFAULT_ALPHA) -> CI:
    """Matthews correlation coefficient with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(_as_int(y_true), _as_int(y_pred), _mcc, n_boot, seed, alpha)


def f1_with_ci(y_true, y_pred, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
               *, alpha: float = DEFAULT_ALPHA) -> CI:
    """Positive-class F1 with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(_as_int(y_true), _as_int(y_pred), _f1, n_boot, seed, alpha)


def pr_auc_with_ci(y_true, y_score, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                   *, alpha: float = DEFAULT_ALPHA) -> CI:
    """Average precision (PR-AUC) with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(
        _as_int(y_true), _as_float(y_score),
        lambda yt, ys: float(average_precision_score(yt, ys)), n_boot, seed, alpha,
    )


def roc_auc_with_ci(y_true, y_score, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                    *, alpha: float = DEFAULT_ALPHA) -> CI:
    """ROC-AUC with a stratified bootstrap CI (complementary baseline)."""
    return _stratified_bootstrap_ci(
        _as_int(y_true), _as_float(y_score),
        lambda yt, ys: float(roc_auc_score(yt, ys)), n_boot, seed, alpha,
    )


def brier_score_with_ci(y_true, y_score, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                        *, alpha: float = DEFAULT_ALPHA) -> CI:
    """Brier score (calibration; lower is better) with a stratified bootstrap CI."""
    return _stratified_bootstrap_ci(
        _as_int(y_true), _as_float(y_score),
        lambda yt, ys: float(brier_score_loss(yt, ys)), n_boot, seed, alpha,
    )


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
