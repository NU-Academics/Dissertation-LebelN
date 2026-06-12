"""Unit tests for ``src/evaluation/hypothesis.py``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.hypothesis import (
    benjamini_hochberg,
    cohens_d,
    holm_bonferroni,
    one_sample_threshold_test,
    paired_wilcoxon_cv,
)


# ---------------------------------------------------------------------------
# one_sample_threshold_test
# ---------------------------------------------------------------------------
def test_threshold_reject_when_ci_low_exceeds() -> None:
    out = one_sample_threshold_test(0.95, 0.94, 0.96, 0.90, metric_name="MCC")
    assert out["reject"] is True
    assert out["decision"] == "reject H0"
    assert out["margin"] == pytest.approx(0.04)
    assert "rejected" in out["narrative"]


def test_threshold_fail_when_ci_low_below() -> None:
    out = one_sample_threshold_test(0.91, 0.89, 0.96, 0.90)
    assert out["reject"] is False
    assert out["decision"] == "fail to reject H0"
    assert out["margin"] == pytest.approx(-0.01)


def test_threshold_boundary_is_strict() -> None:
    # ci_low exactly equal to the threshold does not clear it.
    out = one_sample_threshold_test(0.93, 0.90, 0.95, 0.90)
    assert out["reject"] is False


def test_threshold_less_is_better() -> None:
    rej = one_sample_threshold_test(0.05, 0.03, 0.07, 0.10, greater_is_better=False)
    assert rej["reject"] is True and rej["margin"] == pytest.approx(0.03)
    fail = one_sample_threshold_test(0.08, 0.03, 0.12, 0.10, greater_is_better=False)
    assert fail["reject"] is False


# ---------------------------------------------------------------------------
# paired_wilcoxon_cv
# ---------------------------------------------------------------------------
def test_wilcoxon_identical_folds() -> None:
    folds = [0.90, 0.91, 0.92, 0.93]
    out = paired_wilcoxon_cv(folds, folds)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["reject"] is False
    assert out["median_difference"] == pytest.approx(0.0)
    assert math.isnan(out["statistic"])


def test_wilcoxon_consistent_difference() -> None:
    pytest.importorskip("scipy")
    a = [0.90, 0.92, 0.91, 0.93, 0.95, 0.94]
    b = [0.80, 0.82, 0.81, 0.83, 0.85, 0.84]
    out = paired_wilcoxon_cv(a, b)
    assert out["n_folds"] == 6
    assert out["median_difference"] > 0
    assert out["p_value"] < 0.05
    assert out["reject"] is True


def test_wilcoxon_length_mismatch() -> None:
    with pytest.raises(ValueError):
        paired_wilcoxon_cv([0.9, 0.8], [0.9])


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------
def test_cohens_d_known_value() -> None:
    # Both groups have unit variance; means differ by 1 -> d = 1.0.
    assert cohens_d([2, 3, 4], [1, 2, 3]) == pytest.approx(1.0)


def test_cohens_d_zero_when_identical() -> None:
    assert cohens_d([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(0.0)


def test_cohens_d_constant_groups() -> None:
    assert cohens_d([5, 5, 5], [5, 5, 5]) == 0.0


def test_cohens_d_requires_two_observations() -> None:
    with pytest.raises(ValueError):
        cohens_d([1.0], [2.0, 3.0])


# ---------------------------------------------------------------------------
# Family-wise / FDR corrections
# ---------------------------------------------------------------------------
def test_holm_all_reject_small_pvalues() -> None:
    out = holm_bonferroni([0.01, 0.02, 0.04], alpha=0.05)
    assert out["adjusted"] == pytest.approx([0.03, 0.04, 0.04])
    assert out["reject"] == [True, True, True]
    assert out["n_reject"] == 3


def test_holm_preserves_input_order() -> None:
    out = holm_bonferroni([0.04, 0.01, 0.02], alpha=0.05)
    # Adjusted values map back to the input positions.
    assert out["adjusted"] == pytest.approx([0.04, 0.03, 0.04])
    assert out["reject"] == [True, True, True]


def test_holm_no_reject_large_pvalues() -> None:
    out = holm_bonferroni([0.04, 0.5, 0.6], alpha=0.05)
    assert out["reject"] == [False, False, False]
    assert out["n_reject"] == 0


def test_bh_values() -> None:
    out = benjamini_hochberg([0.01, 0.02, 0.04], alpha=0.05)
    assert out["adjusted"] == pytest.approx([0.03, 0.03, 0.04])
    assert out["reject"] == [True, True, True]


def test_corrections_match_statsmodels() -> None:
    sm = pytest.importorskip("statsmodels.stats.multitest")
    p = [0.001, 0.013, 0.021, 0.04, 0.6]
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    _, holm_ref, _, _ = sm.multipletests(p, alpha=0.05, method="holm")
    _, bh_ref, _, _ = sm.multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(holm["adjusted"], holm_ref, atol=1e-12)
    np.testing.assert_allclose(bh["adjusted"], bh_ref, atol=1e-12)


def test_empty_pvalues_raises() -> None:
    with pytest.raises(ValueError):
        holm_bonferroni([])
    with pytest.raises(ValueError):
        benjamini_hochberg([])
