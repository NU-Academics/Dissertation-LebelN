"""Unit tests for ``src/evaluation/metrics.py``.

Anchored on analytically known cases (a perfect predictor, and a confusion matrix
with MCC = 0 / F1 = 0.5), plus parity with scikit-learn for the point estimates,
stratified-bootstrap behavior, determinism, and the calibration table.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from src.evaluation.metrics import (
    brier_score_with_ci,
    calibration_table,
    drift_detection_latency,
    f1_with_ci,
    mcc_at_fixed_prevalence,
    mcc_with_ci,
    performance_degradation_rate,
    performance_sustainment_window,
    pr_auc_with_ci,
    retraining_effectiveness,
    roc_auc_with_ci,
    sustainment_reference,
)


# ---------------------------------------------------------------------------
# Analytic cases
# ---------------------------------------------------------------------------
def test_mcc_f1_known_confusion() -> None:
    # TP=FP=FN=TN=1 -> MCC = 0 exactly, F1 = 0.5 exactly.
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    mcc_point, _, _ = mcc_with_ci(y_true, y_pred, n_boot=200)
    f1_point, _, _ = f1_with_ci(y_true, y_pred, n_boot=200)
    assert mcc_point == pytest.approx(0.0)
    assert f1_point == pytest.approx(0.5)


def test_perfect_predictor_degenerate_ci() -> None:
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0]
    y_score = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    for point, lo, hi in (mcc_with_ci(y_true, y_pred), f1_with_ci(y_true, y_pred),
                          roc_auc_with_ci(y_true, y_score), pr_auc_with_ci(y_true, y_score)):
        # Every stratified resample is still perfect -> point and CI all equal 1.0.
        assert point == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    b_point, b_lo, b_hi = brier_score_with_ci(y_true, y_score)
    assert b_point == pytest.approx(0.0)
    assert b_lo == pytest.approx(0.0) and b_hi == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Parity with scikit-learn point estimates
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(42)
    n = 400
    y = (rng.random(n) < 0.25).astype(int)          # ~25% positive
    score = np.clip(0.15 + 0.6 * y + rng.normal(scale=0.3, size=n), 0, 1)
    pred = (score >= 0.5).astype(int)
    return y, pred, score


def test_point_estimates_match_sklearn(scored) -> None:
    y, pred, score = scored
    assert mcc_with_ci(y, pred, n_boot=200)[0] == pytest.approx(matthews_corrcoef(y, pred))
    assert f1_with_ci(y, pred, n_boot=200)[0] == pytest.approx(f1_score(y, pred, zero_division=0))
    assert pr_auc_with_ci(y, score, n_boot=200)[0] == pytest.approx(average_precision_score(y, score))
    assert roc_auc_with_ci(y, score, n_boot=200)[0] == pytest.approx(roc_auc_score(y, score))
    assert brier_score_with_ci(y, score, n_boot=200)[0] == pytest.approx(brier_score_loss(y, score))


def test_ci_is_ordered_and_in_range(scored) -> None:
    y, pred, score = scored
    for point, lo, hi in (mcc_with_ci(y, pred, n_boot=300), f1_with_ci(y, pred, n_boot=300)):
        assert lo <= hi
        assert -1.0 <= lo and hi <= 1.0
    for fn in (pr_auc_with_ci, roc_auc_with_ci, brier_score_with_ci):
        point, lo, hi = fn(y, score, n_boot=300)
        assert lo <= hi
        assert 0.0 <= lo and hi <= 1.0
        assert math.isfinite(lo) and math.isfinite(hi)


# ---------------------------------------------------------------------------
# Stratified bootstrap behavior
# ---------------------------------------------------------------------------
def test_stratified_bootstrap_keeps_both_classes_on_imbalanced_data() -> None:
    # 15 positives in 300 rows: a naive bootstrap could draw a positive-free
    # resample and NaN the AUC; the stratified one cannot.
    rng = np.random.default_rng(7)
    y = np.zeros(300, dtype=int)
    y[:15] = 1
    rng.shuffle(y)
    score = np.clip(0.1 + 0.5 * y + rng.normal(scale=0.3, size=300), 0, 1)
    point, lo, hi = roc_auc_with_ci(y, score, n_boot=500)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= point <= hi or lo <= hi  # finite, ordered interval


def test_determinism_same_seed(scored) -> None:
    y, pred, _ = scored
    assert mcc_with_ci(y, pred, n_boot=300, seed=42) == mcc_with_ci(y, pred, n_boot=300, seed=42)


def test_single_class_returns_nan_ci() -> None:
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    point, lo, hi = mcc_with_ci(y_true, y_pred, n_boot=50)
    assert math.isnan(lo) and math.isnan(hi)


def test_accepts_polars_series(scored) -> None:
    y, pred, _ = scored
    np_point = mcc_with_ci(y, pred, n_boot=100, seed=3)
    pl_point = mcc_with_ci(pl.Series(y), pl.Series(pred), n_boot=100, seed=3)
    assert np_point == pl_point


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------
def test_calibration_table_structure_and_calibration() -> None:
    rng = np.random.default_rng(0)
    score = rng.random(20_000)
    y = (rng.random(20_000) < score).astype(int)  # perfectly calibrated by construction
    table = calibration_table(y, score, n_bins=10)

    assert isinstance(table, pl.DataFrame)
    assert table.columns == ["bin", "bin_low", "bin_high", "n", "mean_predicted", "observed_rate"]
    assert table.height == 10
    assert int(table["n"].sum()) == 20_000

    populated = table.filter(pl.col("n") > 0)
    gap = (populated["mean_predicted"] - populated["observed_rate"]).abs().max()
    assert gap < 0.06  # observed tracks predicted in every populated bin


def test_calibration_table_handles_empty_bins() -> None:
    # All scores in [0.0, 0.1): only the first bin is populated.
    y = [0, 1, 0, 1]
    score = [0.01, 0.02, 0.03, 0.04]
    table = calibration_table(y, score, n_bins=10)
    assert int(table["n"].sum()) == 4
    assert table.filter(pl.col("n") == 0).height == 9
    first = table.filter(pl.col("bin") == 0).row(0, named=True)
    assert first["n"] == 4
    assert first["observed_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Custom metrics for performance under drift
# ---------------------------------------------------------------------------
def test_degradation_rate_recovers_a_known_slope() -> None:
    months = [date(2023, m, 1) for m in range(1, 13)]
    mcc = [0.50 - 0.01 * i for i in range(12)]  # exactly -0.01 per month
    assert performance_degradation_rate(mcc, months) == pytest.approx(-0.01)
    # A flat series has no degradation; a rising one has a positive slope.
    assert performance_degradation_rate([0.3] * 12, months) == pytest.approx(0.0, abs=1e-12)
    assert performance_degradation_rate([0.1 * i for i in range(12)], months) > 0
    # Too few points is NaN, not a fabricated slope.
    assert math.isnan(performance_degradation_rate([0.4], [date(2023, 1, 1)]))


def test_degradation_rate_accepts_a_numeric_month_index() -> None:
    mcc = [0.5, 0.4, 0.3]
    assert performance_degradation_rate(mcc, [0, 1, 2]) == pytest.approx(-0.1)


def test_detection_latency_counts_days_and_keeps_its_sign() -> None:
    onset = date(2021, 4, 1)
    assert drift_detection_latency(date(2021, 4, 15), onset) == 14
    assert drift_detection_latency("2021-04-15", "2021-04-01") == 14
    # Firing before the onset estimate is a real outcome, reported negative.
    assert drift_detection_latency(date(2021, 3, 25), onset) == -7


def test_retraining_effectiveness_spans_no_recovery_to_full_recovery() -> None:
    # Dropped from 0.50 to 0.20; recovering to 0.50 is full recovery.
    assert retraining_effectiveness(0.20, 0.50, 0.50) == pytest.approx(1.0)
    assert retraining_effectiveness(0.20, 0.20, 0.50) == pytest.approx(0.0)
    assert retraining_effectiveness(0.20, 0.35, 0.50) == pytest.approx(0.5)
    # Retraining that hurt is negative; overshooting the reference exceeds 1.
    assert retraining_effectiveness(0.20, 0.10, 0.50) < 0
    assert retraining_effectiveness(0.20, 0.60, 0.50) > 1
    # Nothing to recover: undefined rather than divide-by-zero.
    assert math.isnan(retraining_effectiveness(0.50, 0.55, 0.50))


def test_sustainment_window_counts_months_before_the_first_dip() -> None:
    mcc = [0.9, 0.88, 0.86, 0.84, 0.9]  # dips below 0.85 at index 3
    assert performance_sustainment_window(mcc, 0.85) == 3
    # Never dips: the whole horizon is sustained.
    assert performance_sustainment_window([0.9] * 6, 0.85) == 6
    # Starts below the threshold: zero, which is the expected result against 0.85
    # on this data and is reported honestly rather than redefined away.
    assert performance_sustainment_window([0.20, 0.19, 0.18], 0.85) == 0


def test_sustainment_reference_makes_the_window_informative() -> None:
    # An 0.85 window cannot distinguish these strategies: both are 0.
    holding = [0.20, 0.20, 0.19, 0.20]
    decaying = [0.20, 0.15, 0.10, 0.05]
    assert performance_sustainment_window(holding, 0.85) == 0
    assert performance_sustainment_window(decaying, 0.85) == 0
    # A reference at 80% of the model's own starting MCC does distinguish them.
    ref = sustainment_reference(0.20, fraction=0.80)
    assert ref == pytest.approx(0.16)
    assert performance_sustainment_window(holding, ref) == 4
    assert performance_sustainment_window(decaying, ref) == 1


def test_fixed_prevalence_mcc_separates_prior_shift_from_covariate_drift() -> None:
    rng = np.random.default_rng(0)

    def window(n: int, prevalence: float, skill: float):
        """A predictor of constant skill at a given base rate."""
        y = (rng.random(n) < prevalence).astype(int)
        # Flip a fixed share of labels: discriminative power is identical in both
        # windows by construction, only the base rate differs.
        flip = rng.random(n) > skill
        return y, np.where(flip, 1 - y, y)

    y_hi, p_hi = window(200_000, 0.010, 0.90)
    y_lo, p_lo = window(200_000, 0.004, 0.90)

    raw_hi, _, _ = mcc_with_ci(y_hi, p_hi, n_boot=50)
    raw_lo, _, _ = mcc_with_ci(y_lo, p_lo, n_boot=50)
    # The raw MCC moves purely because the base rate fell, with no loss of skill.
    assert abs(raw_hi - raw_lo) > 0.02

    fixed_hi = mcc_at_fixed_prevalence(y_hi, p_hi, target_prevalence=0.004)
    fixed_lo = mcc_at_fixed_prevalence(y_lo, p_lo, target_prevalence=0.004)
    # Held at one prevalence, the two windows agree: no drift, only prior shift.
    assert abs(fixed_hi - fixed_lo) < 0.02


def test_fixed_prevalence_mcc_guards_its_inputs() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="target_prevalence"):
        mcc_at_fixed_prevalence(y, p, target_prevalence=1.5)
    # Single-class input has no defined MCC.
    assert math.isnan(mcc_at_fixed_prevalence(np.zeros(4), np.zeros(4), 0.1))


def test_fixed_prevalence_mcc_is_deterministic_under_a_seed() -> None:
    rng = np.random.default_rng(7)
    y = (rng.random(50_000) < 0.02).astype(int)
    p = np.where(rng.random(50_000) > 0.85, 1 - y, y)
    first = mcc_at_fixed_prevalence(y, p, 0.005, seed=42)
    second = mcc_at_fixed_prevalence(y, p, 0.005, seed=42)
    assert first == second
