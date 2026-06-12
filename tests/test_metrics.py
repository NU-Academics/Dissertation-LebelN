"""Unit tests for ``src/evaluation/metrics.py``.

Anchored on analytically known cases (a perfect predictor, and a confusion matrix
with MCC = 0 / F1 = 0.5), plus parity with scikit-learn for the point estimates,
stratified-bootstrap behavior, determinism, and the calibration table.
"""

from __future__ import annotations

import math

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
    f1_with_ci,
    mcc_with_ci,
    pr_auc_with_ci,
    roc_auc_with_ci,
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
