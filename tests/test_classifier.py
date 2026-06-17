"""Unit tests for ``src/models/classifier.py``.

Synthetic data with a known linear signal. The scikit-learn wrappers (Decision
Tree, linear SVM) always run; the Keras ``SimpleNNWrapper`` is skipped if
TensorFlow is absent. Sizes are kept small for speed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from src.models.ensemble import EnsembleWrapper
from src.models.classifier import (
    CLASSIFIERS,
    DecisionTreeWrapper,
    SimpleNNWrapper,
    SVMWrapper,
    build_classifier,
)

SEED = 42
FEATURES = [
    "cpu_request", "memory_request", "request_ratio",
    "prior_fail_count", "queue_time", "scheduling_class",
]


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


# ---------------------------------------------------------------------------
# scikit-learn wrappers (Decision Tree, linear SVM)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["decision_tree", "linear_svm"])
def test_fit_predict_importances(name: str, data) -> None:
    frame, y = data
    wrapper = build_classifier(name, random_state=SEED)
    assert isinstance(wrapper, EnsembleWrapper)
    wrapper.fit(frame, y)

    proba = wrapper.predict_proba(frame)
    assert proba.shape == (frame.height,)          # 1-D positive-class score
    assert np.all(np.isfinite(proba))
    # The score should separate the classes well above chance on training data.
    assert roc_auc_score(y, proba) > 0.7

    importances = wrapper.feature_importances()
    assert set(importances) == set(FEATURES)
    assert all(v >= 0 for v in importances.values())


@pytest.mark.parametrize("name", ["decision_tree", "linear_svm"])
def test_save_load_roundtrip(name: str, data, tmp_path) -> None:
    frame, y = data
    wrapper = build_classifier(name, random_state=SEED).fit(frame, y)
    path = tmp_path / f"{name}.pkl"
    wrapper.save(path)
    reloaded = type(wrapper).load(path)
    np.testing.assert_allclose(reloaded.predict_proba(frame), wrapper.predict_proba(frame))


def test_numpy_and_pandas_inputs(data) -> None:
    """The Polars boundary also accepts NumPy and pandas frames."""
    frame, y = data
    wrapper = DecisionTreeWrapper(random_state=SEED).fit(frame, y)
    from_numpy = wrapper.predict_proba(frame.to_numpy())
    pd = pytest.importorskip("pandas")
    from_pandas = wrapper.predict_proba(pd.DataFrame(frame.to_numpy(), columns=FEATURES))
    np.testing.assert_allclose(from_numpy, from_pandas)


def test_svm_score_is_decision_function(data) -> None:
    """SVMWrapper.predict_proba returns the signed decision function (not [0, 1])."""
    frame, y = data
    wrapper = SVMWrapper(random_state=SEED).fit(frame, y)
    score = wrapper.predict_proba(frame)
    assert score.ndim == 1
    # Decision-function scores are unbounded; at least some lie outside [0, 1].
    assert (score < 0).any() or (score > 1).any()


# ---------------------------------------------------------------------------
# Keras SimpleNNWrapper (skipped without TensorFlow)
# ---------------------------------------------------------------------------
def test_simple_nn_fit_predict(data) -> None:
    pytest.importorskip("tensorflow")
    frame, y = data
    wrapper = SimpleNNWrapper(hidden_units=16, epochs=3, random_state=SEED).fit(frame, y)
    proba = wrapper.predict_proba(frame)
    assert proba.shape == (frame.height,)
    assert np.all((proba >= 0) & (proba <= 1))      # sigmoid output is a probability


def test_simple_nn_importances_raise(data) -> None:
    pytest.importorskip("tensorflow")
    frame, y = data
    wrapper = SimpleNNWrapper(hidden_units=16, epochs=1, random_state=SEED).fit(frame, y)
    with pytest.raises(NotImplementedError):
        wrapper.feature_importances()


def test_simple_nn_save_load(data, tmp_path) -> None:
    pytest.importorskip("tensorflow")
    frame, y = data
    wrapper = SimpleNNWrapper(hidden_units=16, epochs=3, random_state=SEED).fit(frame, y)
    path = tmp_path / "simple_nn.pkl"
    wrapper.save(path)
    reloaded = SimpleNNWrapper.load(path)
    np.testing.assert_allclose(reloaded.predict_proba(frame), wrapper.predict_proba(frame), atol=1e-5)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_build_classifier_unknown() -> None:
    with pytest.raises(KeyError):
        build_classifier("not_a_model")


def test_registry_complete() -> None:
    assert set(CLASSIFIERS) == {"decision_tree", "linear_svm", "simple_nn"}
