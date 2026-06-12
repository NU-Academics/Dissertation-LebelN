"""Unit tests for ``src/models/ensemble.py``.

Synthetic data with a known linear signal. The third-party learners
(imbalanced-learn, XGBoost, LightGBM) are skipped if their library is absent so
the scikit-learn wrappers always run. Tree counts are kept small for speed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from src.models.ensemble import (
    EnsembleWrapper,
    GradientBoostingWrapper,
    RandomForestWrapper,
    SoftVotingStack,
    WRAPPERS,
    XGBoostWrapper,
    build_wrapper,
)

SEED = 42
FEATURES = [
    "cpu_request", "memory_request", "request_ratio",
    "prior_fail_count", "queue_time", "scheduling_class",
]

# Small, fast estimator sizes per wrapper.
SMALL: dict[str, dict[str, object]] = {
    "random_forest": dict(n_estimators=16),
    "balanced_random_forest": dict(n_estimators=16),
    "xgboost": dict(n_estimators=24, max_depth=3),
    "lightgbm": dict(n_estimators=24),
    "gradient_boosting": dict(n_estimators=24, max_depth=2),
}
# Library each wrapper needs beyond scikit-learn (for importorskip).
_LIB = {"balanced_random_forest": "imblearn", "xgboost": "xgboost", "lightgbm": "lightgbm"}


def _make_data(n: int = 600) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    logits = 1.5 * x[:, 0] - 1.2 * x[:, 3] + 0.8 * x[:, 5]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(np.int8)
    frame = pl.DataFrame({c: x[:, i] for i, c in enumerate(FEATURES)})
    return frame, y


@pytest.fixture(scope="module")
def data() -> tuple[pl.DataFrame, np.ndarray]:
    return _make_data()


@pytest.mark.parametrize("name", list(WRAPPERS))
def test_fit_predict_importances(name: str, data) -> None:
    if name in _LIB:
        pytest.importorskip(_LIB[name])
    frame, y = data
    wrapper = build_wrapper(name, **SMALL[name])
    wrapper.fit(frame, y)

    proba = wrapper.predict_proba(frame)
    assert proba.shape == (frame.height,)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # Learns the planted signal better than chance on the training data.
    assert roc_auc_score(y, proba) > 0.7

    importances = wrapper.feature_importances()
    assert set(importances) == set(FEATURES)
    assert all(v >= 0.0 for v in importances.values())


@pytest.mark.parametrize("name", list(WRAPPERS))
def test_save_load_roundtrip(name: str, data, tmp_path) -> None:
    if name in _LIB:
        pytest.importorskip(_LIB[name])
    frame, y = data
    wrapper = build_wrapper(name, **SMALL[name]).fit(frame, y)
    before = wrapper.predict_proba(frame)

    path = tmp_path / f"{name}.pkl"
    wrapper.save(path)
    reloaded = type(wrapper).load(path)
    np.testing.assert_allclose(before, reloaded.predict_proba(frame))
    assert isinstance(reloaded, EnsembleWrapper)


def test_protocol_isinstance(data) -> None:
    frame, y = data
    wrapper = RandomForestWrapper(n_estimators=8).fit(frame, y)
    assert isinstance(wrapper, EnsembleWrapper)


def test_accepts_numpy_and_polars_equivalently(data) -> None:
    frame, y = data
    wrapper = RandomForestWrapper(n_estimators=16).fit(frame, y)
    from_polars = wrapper.predict_proba(frame)
    from_numpy = wrapper.predict_proba(frame.to_numpy())
    np.testing.assert_allclose(from_polars, from_numpy)


def test_soft_voting_stack(data, tmp_path) -> None:
    frame, y = data
    rf = RandomForestWrapper(n_estimators=16).fit(frame, y)
    gb = GradientBoostingWrapper(n_estimators=24, max_depth=2).fit(frame, y)
    p_rf, p_gb = rf.predict_proba(frame), gb.predict_proba(frame)

    equal = SoftVotingStack([rf, gb], weights="equal")
    p_eq = equal.predict_proba(frame)
    np.testing.assert_allclose(p_eq, 0.5 * (p_rf + p_gb), atol=1e-6)
    assert np.all(p_eq >= np.minimum(p_rf, p_gb) - 1e-6)
    assert np.all(p_eq <= np.maximum(p_rf, p_gb) + 1e-6)

    weighted = SoftVotingStack([rf, gb], weights=[3.0, 1.0])
    np.testing.assert_allclose(weighted.predict_proba(frame), 0.75 * p_rf + 0.25 * p_gb, atol=1e-6)

    importances = equal.feature_importances()
    assert set(importances) == set(FEATURES)

    path = tmp_path / "stack.pkl"
    equal.save(path)
    np.testing.assert_allclose(SoftVotingStack.load(path).predict_proba(frame), p_eq, atol=1e-6)


def test_xgboost_auto_scale_pos_weight(data) -> None:
    pytest.importorskip("xgboost")
    frame, y = data
    wrapper = XGBoostWrapper(n_estimators=10, max_depth=2).fit(frame, y)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    spw = wrapper.estimator.get_params()["scale_pos_weight"]
    assert spw == pytest.approx(n_neg / max(n_pos, 1))


def test_build_wrapper_unknown() -> None:
    with pytest.raises(KeyError):
        build_wrapper("not_a_model")
