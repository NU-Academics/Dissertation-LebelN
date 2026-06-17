"""Conflict-resolution classifier wrappers (RQ2).

Thin wrappers around the non-ensemble learners validated in
``notebooks/13_rq2_conflict.py``: a scikit-learn Decision Tree, a linear SVM, and
a one-hidden-layer Keras neural network. They reuse the
:class:`~src.models.ensemble.EnsembleWrapper` protocol, the Polars-to-float32
boundary helpers, and the pickle save / load already defined in
``src/models/ensemble.py``, so the modeling notebook, the evaluation code, and the
hyperparameter search treat every learner identically. The Random Forest used in
RQ2 is :class:`~src.models.ensemble.RandomForestWrapper`; it is not duplicated
here.

**Common interface** (the :class:`~src.models.ensemble.EnsembleWrapper` protocol):

- ``fit(X, y, sample_weight=None)`` -> self
- ``predict_proba(X)`` -> 1-D ``np.ndarray`` of the positive-class score
- ``feature_importances()`` -> ``dict[feature_name, importance]``
- ``save(path)`` / ``load(path)`` -> pickle round-trip of the fitted wrapper

**Per-learner notes.**

- :class:`DecisionTreeWrapper` is a straight ``_BaseEnsembleWrapper`` subclass
  (Decision Tree exposes both ``predict_proba`` and ``feature_importances_``), with
  inverse-prior class weighting.
- :class:`SVMWrapper` wraps ``LinearSVC``, which has no ``predict_proba``. Its
  ``decision_function`` is the positive-class score: monotone in P(failure), so it
  is correct for ranking and threshold tuning (RQ2 tunes the operating threshold,
  never a fixed 0.5). ``feature_importances`` reports ``|coef_|``.
- :class:`SimpleNNWrapper` is a single hidden layer (default 32 units), ReLU,
  sigmoid output, Adam at the default learning rate, no grid search. The point of
  including it is to test whether a neural network categorically beats the tree
  methods on this task, which is itself informative for Chapter 4. Neural networks
  expose no per-feature importances, so ``feature_importances`` raises. The wrapper
  is picklable: ``__getstate__`` / ``__setstate__`` serialize the architecture and
  the trained weights (Keras estimators do not pickle natively).

**Division of labor (unchanged from ensemble.py).** Each wrapper sets
cost-sensitive class weighting to the inverse class prior where the learner
supports it. Resampling (SMOTE) and any per-entity cap are fold-level concerns
owned by the caller and applied only to training data; these wrappers do not
resample.

Cross-references (``outputs/tables/eda_decisions.csv``): V38-V40 (RQ2 conflict
labeling, history features, entity-grouped split, result); V23 (MCC-primary
metric under imbalance).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.models.ensemble import (
    DEFAULT_SEED,
    _BaseEnsembleWrapper,
    _to_features,
    _to_labels,
)

__all__ = [
    "DecisionTreeWrapper",
    "SVMWrapper",
    "SimpleNNWrapper",
    "CLASSIFIERS",
    "build_classifier",
]


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------
class DecisionTreeWrapper(_BaseEnsembleWrapper):
    """scikit-learn ``DecisionTreeClassifier`` with inverse-prior class weights.

    The interpretable baseline of the RQ2 zoo. Inherits ``fit`` / ``predict_proba``
    / ``feature_importances`` / ``save`` / ``load`` unchanged from the base, since a
    decision tree exposes both ``predict_proba`` and ``feature_importances_``.
    """

    name = "decision_tree"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from sklearn.tree import DecisionTreeClassifier

        opts: dict[str, object] = dict(class_weight="balanced", random_state=random_state)
        opts.update(params)
        super().__init__(DecisionTreeClassifier(**opts))


# ---------------------------------------------------------------------------
# Linear SVM
# ---------------------------------------------------------------------------
class SVMWrapper(_BaseEnsembleWrapper):
    """``LinearSVC`` with inverse-prior class weights.

    ``LinearSVC`` has no ``predict_proba``; the signed ``decision_function`` is the
    positive-class score, which is monotone in P(failure) and therefore correct for
    ranking and for the validation-tuned operating threshold. The score is not a
    probability (it is unbounded), so it must not be read as one. Feature
    importances are the absolute linear coefficients.
    """

    name = "linear_svm"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from sklearn.svm import LinearSVC

        opts: dict[str, object] = dict(class_weight="balanced", random_state=random_state)
        opts.update(params)
        super().__init__(LinearSVC(**opts))

    def predict_proba(self, X: object) -> np.ndarray:
        """Signed distance to the separating hyperplane (1-D), used as the
        positive-class score. Not a calibrated probability."""
        arr, _ = _to_features(X)
        return np.asarray(self._estimator.decision_function(arr)).ravel()

    def feature_importances(self) -> dict[str, float]:
        if self._feature_names is None:
            raise RuntimeError("Call fit() before feature_importances().")
        coef = np.abs(np.asarray(self._estimator.coef_)).ravel()
        return {name: float(v) for name, v in zip(self._feature_names, coef)}


# ---------------------------------------------------------------------------
# One-hidden-layer neural network (Keras)
# ---------------------------------------------------------------------------
class SimpleNNWrapper:
    """One-hidden-layer Keras classifier satisfying the ``EnsembleWrapper`` protocol.

    Architecture: ``Dense(hidden_units, relu)`` then ``Dense(1, sigmoid)``, compiled
    with Adam (default learning rate) and binary cross-entropy. No grid search. The
    sigmoid output is the positive-class score. The wrapper is picklable via
    ``__getstate__`` / ``__setstate__`` (Keras models are serialized as their JSON
    config plus weight arrays, since they do not pickle natively).
    """

    name = "simple_nn"

    def __init__(
        self,
        hidden_units: int = 32,
        epochs: int = 10,
        batch_size: int = 256,
        random_state: int = DEFAULT_SEED,
    ) -> None:
        self.hidden_units = int(hidden_units)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self._model = None
        self._feature_names: list[str] | None = None

    # -- construction ------------------------------------------------------
    def _build(self, n_features: int):
        import tensorflow as tf

        tf.random.set_seed(self.random_state)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(self.hidden_units, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    # -- interface ---------------------------------------------------------
    def fit(self, X: object, y: object, sample_weight: object = None) -> "SimpleNNWrapper":
        arr, names = _to_features(X)
        self._feature_names = names
        y_arr = _to_labels(y)
        self._model = self._build(arr.shape[1])
        kwargs: dict[str, object] = dict(epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        if sample_weight is not None:
            kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        self._model.fit(arr, y_arr, **kwargs)
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """P(failure) for the positive class, as a 1-D array (the sigmoid output)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        arr, _ = _to_features(X)
        return np.asarray(self._model.predict(arr, verbose=0)).ravel()

    def feature_importances(self) -> dict[str, float]:
        raise NotImplementedError(
            "SimpleNNWrapper (neural network) exposes no per-feature importances; "
            "use a tree learner or SHAP (Phase 5) for attribution."
        )

    # -- persistence (Keras models do not pickle natively) -----------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self._model is not None:
            state["_model"] = {
                "config": self._model.to_json(),
                "weights": self._model.get_weights(),
            }
        return state

    def __setstate__(self, state: dict) -> None:
        model_blob = state.get("_model")
        self.__dict__.update(state)
        if isinstance(model_blob, dict):
            import tensorflow as tf

            model = tf.keras.models.model_from_json(model_blob["config"])
            model.set_weights(model_blob["weights"])
            model.compile(optimizer="adam", loss="binary_crossentropy")
            self._model = model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleNNWrapper":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
CLASSIFIERS: dict[str, type] = {
    DecisionTreeWrapper.name: DecisionTreeWrapper,
    SVMWrapper.name: SVMWrapper,
    SimpleNNWrapper.name: SimpleNNWrapper,
}


def build_classifier(name: str, random_state: int = DEFAULT_SEED, **params: object):
    """Construct a classifier wrapper by name (one of ``CLASSIFIERS``).

    ``DecisionTreeWrapper`` and ``SVMWrapper`` forward ``**params`` to the
    underlying scikit-learn estimator; ``SimpleNNWrapper`` accepts ``hidden_units``
    / ``epochs`` / ``batch_size``.
    """
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier {name!r}; choose from {sorted(CLASSIFIERS)}.")
    return CLASSIFIERS[name](random_state=random_state, **params)
