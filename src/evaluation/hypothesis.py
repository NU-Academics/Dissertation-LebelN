"""Hypothesis testing for the per-RQ thresholds and cross-RQ error control.

Three building blocks plus the family-wise correction the five-RQ design needs:

- :func:`one_sample_threshold_test` decides whether a metric clears its locked
  target. The rule is the CI-based one-sided test committed in Chapter 3: reject
  the null only when the lower bound of the (1 - alpha) confidence interval
  exceeds the threshold (for a greater-is-better metric). This is deliberately
  conservative and pairs with the bootstrap CIs from ``metrics.py``.
- :func:`paired_wilcoxon_cv` compares two models across the same CV folds with a
  Wilcoxon signed-rank test on the paired fold differences.
- :func:`cohens_d` reports the standardized effect size between two samples.
- :func:`holm_bonferroni` and :func:`benjamini_hochberg` adjust a family of
  p-values (one per RQ) so the family-wise error rate / false-discovery rate is
  controlled at alpha across the five research questions (P16).

All functions take plain Python floats or lists; no Polars or NumPy types are
required at the boundary. Each returns a plain ``dict`` (or float) so results
serialize directly into ``outputs/tables/`` rows.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# One-sample threshold test (CI-based, one-sided)
# ---------------------------------------------------------------------------
def one_sample_threshold_test(
    metric_value: float,
    ci_low: float,
    ci_high: float,
    threshold: float,
    alpha: float = DEFAULT_ALPHA,
    *,
    metric_name: str = "metric",
    greater_is_better: bool = True,
) -> dict:
    """Decide whether ``metric_value`` clears ``threshold`` using its CI.

    The null hypothesis is that the metric does not beat the target. For a
    greater-is-better metric (MCC, F1, PR-AUC, success rate, efficiency gain,
    lead time) the null is rejected only when ``ci_low > threshold``; for a
    less-is-better metric (e.g. Brier) only when ``ci_high < threshold``. The CI
    passed in should be the (1 - ``alpha``) interval produced by ``metrics.py``.

    Returns a dict with the inputs, ``reject`` (bool), ``decision`` (str),
    ``margin`` (signed distance from the decisive CI bound to the threshold), and
    a plain-language ``narrative``.
    """
    if greater_is_better:
        reject = ci_low > threshold
        margin = ci_low - threshold
        bound_text = f"the CI lower bound {ci_low:.4f}"
        relation = "exceeds" if reject else "does not exceed"
        null_text = f"{metric_name} <= {threshold}"
    else:
        reject = ci_high < threshold
        margin = threshold - ci_high
        bound_text = f"the CI upper bound {ci_high:.4f}"
        relation = "falls below" if reject else "does not fall below"
        null_text = f"{metric_name} >= {threshold}"

    decision = "reject H0" if reject else "fail to reject H0"
    outcome = "is rejected" if reject else "is not rejected"
    narrative = (
        f"{metric_name} = {metric_value:.4f} (CI [{ci_low:.4f}, {ci_high:.4f}]); "
        f"{bound_text} {relation} the {threshold} target, so H0 ({null_text}) "
        f"{outcome} at alpha = {alpha} (one-sided)."
    )
    return {
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "threshold": float(threshold),
        "alpha": float(alpha),
        "greater_is_better": greater_is_better,
        "reject": bool(reject),
        "decision": decision,
        "margin": float(margin),
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Paired Wilcoxon across CV folds
# ---------------------------------------------------------------------------
def paired_wilcoxon_cv(
    fold_results_a: list[float],
    fold_results_b: list[float],
    alpha: float = DEFAULT_ALPHA,
    *,
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test on paired per-fold metric values (model A vs B).

    The two lists must be aligned: entry ``i`` is each model's metric on fold
    ``i``. With only a handful of walk-forward folds the test has low power and
    cannot reach small p-values, so interpret it alongside :func:`cohens_d` and
    the per-fold spread; it is a guard against over-reading a fold-mean
    difference, not a high-powered test.
    """
    a = np.asarray(fold_results_a, dtype=np.float64)
    b = np.asarray(fold_results_b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError("fold_results_a and fold_results_b must have the same length.")
    if a.size == 0:
        raise ValueError("Need at least one paired fold result.")

    diff = a - b
    median_difference = float(np.median(diff))
    mean_difference = float(diff.mean())

    if np.allclose(diff, 0.0):
        statistic, p_value = float("nan"), 1.0
    else:
        from scipy.stats import wilcoxon

        try:
            result = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, p_value = float("nan"), 1.0

    reject = p_value < alpha
    decision = "reject H0 (models differ)" if reject else "fail to reject H0 (no detected difference)"
    narrative = (
        f"Paired Wilcoxon over {a.size} folds: median difference (A - B) "
        f"{median_difference:+.4f}, p = {p_value:.4f} ({alternative}); {decision} "
        f"at alpha = {alpha}. With {a.size} folds, power is limited; read with the "
        f"effect size."
    )
    return {
        "n_folds": int(a.size),
        "statistic": statistic,
        "p_value": float(p_value),
        "alpha": float(alpha),
        "alternative": alternative,
        "reject": bool(reject),
        "decision": decision,
        "median_difference": median_difference,
        "mean_difference": mean_difference,
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------
def cohens_d(values_a: list[float], values_b: list[float]) -> float:
    """Pooled (independent-samples) Cohen's d for ``a`` versus ``b``.

    ``d = (mean_a - mean_b) / pooled_sd`` with the pooled standard deviation using
    the unbiased (ddof = 1) variances. Returns 0.0 when the pooled standard
    deviation is zero (both samples constant). Positive d means A scores higher.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("cohens_d needs at least two observations per group.")
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


# ---------------------------------------------------------------------------
# Family-wise / false-discovery correction across the RQs
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: list[float], alpha: float = DEFAULT_ALPHA) -> dict:
    """Holm step-down family-wise correction (primary method for the five RQs).

    Returns the input p-values, the Holm-adjusted p-values (in the original
    order), per-test ``reject`` flags (adjusted p < alpha), and the count rejected.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        raise ValueError("Need at least one p-value.")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for i, pv in enumerate(ranked):
        running = max(running, (m - i) * pv)  # step-down monotonicity
        adjusted_sorted[i] = min(running, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    reject = adjusted < alpha
    return {
        "method": "holm-bonferroni",
        "alpha": float(alpha),
        "p_values": [float(x) for x in p],
        "adjusted": [float(x) for x in adjusted],
        "reject": [bool(x) for x in reject],
        "n_reject": int(reject.sum()),
    }


def benjamini_hochberg(p_values: list[float], alpha: float = DEFAULT_ALPHA) -> dict:
    """Benjamini-Hochberg false-discovery-rate correction (supporting method).

    Returns the input p-values, the BH-adjusted p-values (in the original order),
    per-test ``reject`` flags (adjusted p <= alpha), and the count rejected.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        raise ValueError("Need at least one p-value.")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 1.0
    for i in range(m - 1, -1, -1):  # step-up from the largest p-value
        running = min(running, ranked[i] * m / (i + 1))
        adjusted_sorted[i] = min(running, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    reject = adjusted <= alpha
    return {
        "method": "benjamini-hochberg",
        "alpha": float(alpha),
        "p_values": [float(x) for x in p],
        "adjusted": [float(x) for x in adjusted],
        "reject": [bool(x) for x in reject],
        "n_reject": int(reject.sum()),
    }
