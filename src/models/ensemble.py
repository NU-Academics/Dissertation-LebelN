"""Reusable ensemble model wrappers for failure prediction (RQ1).

Thin, uniform wrappers around the tree-ensemble learners validated in
``notebooks/12_rq1_ensemble_google.py``: scikit-learn Random Forest and Gradient
Boosting, imbalanced-learn Balanced Random Forest, XGBoost, and LightGBM, plus a
``SoftVotingStack`` that averages a set of fitted members. They expose one
interface so the modeling notebooks, the hyperparameter search, and the
evaluation code can treat every learner identically.

**Common interface** (the :class:`EnsembleWrapper` protocol):

- ``fit(X, y, sample_weight=None)`` -> self
- ``predict_proba(X)`` -> 1-D ``np.ndarray`` of P(failure) for the positive class
  (not the sklearn ``(n, 2)`` matrix; RQ1 is binary and downstream threshold
  tuning and PR-AUC want the positive-class score directly)
- ``feature_importances()`` -> ``dict[feature_name, importance]``
- ``save(path)`` / ``load(path)`` -> pickle round-trip of the fitted wrapper

**Polars at the boundary.** Every method accepts a Polars ``DataFrame`` (a NumPy
array or pandas frame also works). Conversion to a dense ``float32`` NumPy array
happens once per call via :meth:`polars.DataFrame.to_numpy`, which materializes a
C-contiguous copy: O(n_rows * n_cols) time and memory. Call ``fit`` / ``predict``
on whole frames, never row by row. Tier 2 nulls (rapid-onset crashes emit no early
usage, V09) pass through as NaN; LightGBM and XGBoost consume NaN natively, while
the sklearn and imbalanced-learn learners require the caller to impute first.

**Division of labor (do not duplicate the caller's work).** Each wrapper sets
cost-sensitive class weighting to the inverse class prior where the learner
supports it (``class_weight="balanced"``; XGBoost uses ``scale_pos_weight``
computed at fit; Gradient Boosting, which has no ``class_weight``, applies
balanced ``sample_weight`` at fit). Resampling (SMOTE) and the per-instance
negative cap are fold-level concerns owned by the caller and applied only to
training data (V02 / V17 imbalance handling; never on validation or test). These
wrappers do not resample.

Cross-references (``outputs/tables/eda_decisions.csv``): V13 (tier structure),
V02 / V17 / V23 (imbalance handling and metric choice), P06 (three-level
prediction architecture), P08 (hyperparameter tuning families).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------
def _to_features(x: object) -> tuple[np.ndarray, list[str]]:
    """Return ``(float32 ndarray, feature_names)`` from a Polars DataFrame,
    NumPy array, or pandas DataFrame.

    Conversion cost: a dense C-contiguous copy of the whole block, O(n * d).
    """
    if isinstance(x, pl.DataFrame):
        names = list(x.columns)
        arr = x.to_numpy()
    elif isinstance(x, np.ndarray):
        arr = x
        names = [f"f{i}" for i in range(arr.shape[1])]
    elif hasattr(x, "columns") and hasattr(x, "to_numpy"):  # pandas DataFrame
        names = [str(c) for c in x.columns]
        arr = x.to_numpy()
    else:
        raise TypeError(
            "X must be a polars.DataFrame, numpy.ndarray, or pandas.DataFrame; "
            f"got {type(x)!r}."
        )
    return np.asarray(arr, dtype=np.float32), names


def _to_labels(y: object) -> np.ndarray:
    """Return a 1-D ``int8`` label array from a Polars Series/DataFrame, NumPy
    array, or list."""
    if isinstance(y, pl.Series):
        arr = y.to_numpy()
    elif isinstance(y, pl.DataFrame):
        if y.width != 1:
            raise ValueError("y as a DataFrame must have exactly one column.")
        arr = y.to_numpy().ravel()
    else:
        arr = np.asarray(y)
    return arr.ravel().astype(np.int8)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class EnsembleWrapper(Protocol):
    """The uniform interface every wrapper (and the stack) satisfies."""

    def fit(self, X: object, y: object, sample_weight: object = None) -> "EnsembleWrapper": ...

    def predict_proba(self, X: object) -> np.ndarray: ...

    def feature_importances(self) -> dict[str, float]: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleWrapper": ...


# ---------------------------------------------------------------------------
# Base implementation for the single-estimator wrappers
# ---------------------------------------------------------------------------
class _BaseEnsembleWrapper:
    """Shared Polars-boundary fit / predict / importance / persistence logic.

    Subclasses build their concrete estimator in ``__init__`` and may override
    :meth:`_before_fit` (adjust the estimator from the labels, e.g. XGBoost's
    ``scale_pos_weight``) or :meth:`_auto_sample_weight` (supply balanced sample
    weights for learners without a ``class_weight`` option).
    """

    name: str = "base"

    def __init__(self, estimator: object) -> None:
        self._estimator = estimator
        self._feature_names: list[str] | None = None

    # -- hooks -------------------------------------------------------------
    def _before_fit(self, y: np.ndarray) -> None:
        """Adjust the estimator using the training labels. Default: no-op."""

    def _auto_sample_weight(self, y: np.ndarray) -> np.ndarray | None:
        """Balanced sample weights for learners lacking ``class_weight``.
        Default: ``None`` (the estimator handles weighting itself)."""
        return None

    # -- interface ---------------------------------------------------------
    def fit(self, X: object, y: object, sample_weight: object = None) -> "_BaseEnsembleWrapper":
        arr, names = _to_features(X)
        self._feature_names = names
        y_arr = _to_labels(y)
        self._before_fit(y_arr)
        if sample_weight is None:
            sample_weight = self._auto_sample_weight(y_arr)
        if sample_weight is not None:
            self._estimator.fit(arr, y_arr, sample_weight=np.asarray(sample_weight, dtype=np.float64))
        else:
            self._estimator.fit(arr, y_arr)
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """P(failure) for the positive class, as a 1-D array."""
        arr, _ = _to_features(X)
        proba = self._estimator.predict_proba(arr)
        return np.asarray(proba)[:, 1]

    def feature_importances(self) -> dict[str, float]:
        if self._feature_names is None:
            raise RuntimeError("Call fit() before feature_importances().")
        importances = getattr(self._estimator, "feature_importances_", None)
        if importances is None:
            raise AttributeError(
                f"{type(self._estimator).__name__} exposes no feature_importances_."
            )
        return {name: float(v) for name, v in zip(self._feature_names, importances)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "_BaseEnsembleWrapper":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    @property
    def estimator(self) -> object:
        """The underlying fitted estimator (for inspection / SHAP later)."""
        return self._estimator


# ---------------------------------------------------------------------------
# Concrete wrappers
# ---------------------------------------------------------------------------
class RandomForestWrapper(_BaseEnsembleWrapper):
    """scikit-learn ``RandomForestClassifier`` with inverse-prior class weights."""

    name = "random_forest"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from sklearn.ensemble import RandomForestClassifier

        opts: dict[str, object] = dict(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=random_state
        )
        opts.update(params)
        super().__init__(RandomForestClassifier(**opts))


class BalancedRandomForestWrapper(_BaseEnsembleWrapper):
    """imbalanced-learn ``BalancedRandomForestClassifier`` (under-samples each
    bootstrap to balance), which complements SMOTE as a resampling-aware baseline."""

    name = "balanced_random_forest"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from imblearn.ensemble import BalancedRandomForestClassifier

        opts: dict[str, object] = dict(
            n_estimators=300,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        )
        opts.update(params)
        super().__init__(BalancedRandomForestClassifier(**opts))


class XGBoostWrapper(_BaseEnsembleWrapper):
    """XGBoost ``XGBClassifier``. ``scale_pos_weight`` defaults to the inverse
    class prior (n_neg / n_pos), computed at fit unless the caller fixed it."""

    name = "xgboost"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from xgboost import XGBClassifier

        opts: dict[str, object] = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=1.0,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
        opts.update(params)
        # Auto-balance only when the caller did not pin scale_pos_weight.
        self._auto_scale_pos_weight = "scale_pos_weight" not in params
        super().__init__(XGBClassifier(**opts))

    def _before_fit(self, y: np.ndarray) -> None:
        if self._auto_scale_pos_weight:
            n_pos = int(y.sum())
            n_neg = int(y.size - n_pos)
            self._estimator.set_params(scale_pos_weight=(n_neg / max(n_pos, 1)))


class LightGBMWrapper(_BaseEnsembleWrapper):
    """LightGBM ``LGBMClassifier`` with inverse-prior class weights."""

    name = "lightgbm"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from lightgbm import LGBMClassifier

        opts: dict[str, object] = dict(
            n_estimators=400,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        )
        opts.update(params)
        super().__init__(LGBMClassifier(**opts))


class GradientBoostingWrapper(_BaseEnsembleWrapper):
    """scikit-learn ``GradientBoostingClassifier``. It has no ``class_weight``,
    so balanced ``sample_weight`` is applied at fit unless the caller passes its
    own weights."""

    name = "gradient_boosting"

    def __init__(self, random_state: int = DEFAULT_SEED, **params: object) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        opts: dict[str, object] = dict(random_state=random_state)
        opts.update(params)
        super().__init__(GradientBoostingClassifier(**opts))

    def _auto_sample_weight(self, y: np.ndarray) -> np.ndarray | None:
        from sklearn.utils.class_weight import compute_sample_weight

        return compute_sample_weight(class_weight="balanced", y=y)


# ---------------------------------------------------------------------------
# Soft-voting stack
# ---------------------------------------------------------------------------
class SoftVotingStack:
    """Probability-level soft-voting ensemble of fitted :class:`EnsembleWrapper`
    members.

    Built explicitly (rather than via ``sklearn.VotingClassifier``) so each member
    keeps its own boundary handling and any per-member preprocessing the caller
    attached. ``predict_proba`` is the (optionally weighted) mean of the members'
    positive-class probabilities; ``feature_importances`` is the weighted mean of
    the members' importances (members are assumed trained on the same columns).
    """

    name = "soft_voting_stack"

    def __init__(self, members: list[EnsembleWrapper], weights: str | list[float] = "equal") -> None:
        if not members:
            raise ValueError("SoftVotingStack needs at least one member.")
        self.members = list(members)
        self._weights = self._resolve_weights(weights, len(self.members))

    @staticmethod
    def _resolve_weights(weights: str | list[float], n: int) -> np.ndarray:
        if weights == "equal" or weights is None:
            return np.full(n, 1.0 / n)
        w = np.asarray(weights, dtype=np.float64)
        if w.size != n:
            raise ValueError(f"Expected {n} weights, got {w.size}.")
        if w.sum() <= 0:
            raise ValueError("Member weights must sum to a positive value.")
        return w / w.sum()

    def fit(self, X: object, y: object, sample_weight: object = None) -> "SoftVotingStack":
        """Fit every member on the same data. Typically the members are already
        fitted (the stack is assembled from the top performers), so this is
        provided mainly for interface completeness."""
        for member in self.members:
            member.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        stacked = np.vstack([member.predict_proba(X) for member in self.members])
        return np.average(stacked, axis=0, weights=self._weights)

    def feature_importances(self) -> dict[str, float]:
        per_member = [member.feature_importances() for member in self.members]
        keys = per_member[0].keys()
        return {
            key: float(sum(w * d.get(key, 0.0) for w, d in zip(self._weights, per_member)))
            for key in keys
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SoftVotingStack":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
WRAPPERS: dict[str, type[_BaseEnsembleWrapper]] = {
    RandomForestWrapper.name: RandomForestWrapper,
    BalancedRandomForestWrapper.name: BalancedRandomForestWrapper,
    XGBoostWrapper.name: XGBoostWrapper,
    LightGBMWrapper.name: LightGBMWrapper,
    GradientBoostingWrapper.name: GradientBoostingWrapper,
}


def build_wrapper(name: str, random_state: int = DEFAULT_SEED, **params: object) -> _BaseEnsembleWrapper:
    """Construct a wrapper by name (one of ``WRAPPERS``), forwarding overrides to
    the underlying estimator (used by the modeling notebook and the tuning search)."""
    if name not in WRAPPERS:
        raise KeyError(f"Unknown wrapper {name!r}; choose from {sorted(WRAPPERS)}.")
    return WRAPPERS[name](random_state=random_state, **params)
