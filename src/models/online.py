"""Incremental (online) learner wrappers for the concept-drift study (RQ5).

Thin, uniform wrappers around River's incremental learners, mirroring the shape of
``src/models/ensemble.py`` so the drift-simulation notebook can treat a static
batch ensemble and an adaptive online learner through comparable calls. River is
the committed online-learning library: each observation is processed once and
discarded, so the learner never holds the corpus in memory, which is what makes a
13-year stream tractable.

**Common interface** (the :class:`OnlineLearner` protocol):

- ``learn_one(x, y, w=1.0)`` -> self, where ``x`` is a ``dict[str, float]``
- ``predict_proba_one(x)`` -> P(failure), a single float for the positive class
- ``predict_proba(X)`` -> 1-D ``np.ndarray`` over a frame, row by row, provided so
  a window can be scored with the same call signature the batch wrappers use
- ``save(path)`` / ``load(path)`` -> pickle round-trip, for checkpoint-and-resume

**Prequential discipline.** In a drift study the learner must predict *before* it
learns from an observation, or the reported performance is contaminated by the
label it is being scored on. Both operations are exposed separately and the caller
orders them (predict, score, then learn). :meth:`learn_one` never returns a
prediction, precisely so the two cannot be silently fused.

**Row dicts, not matrices.** River consumes one ``dict`` per observation. Use
:func:`frame_to_dicts` once per window rather than building dicts in a Python loop
over a Polars frame, which is the difference between a tolerable and an
intolerable stream time. Nulls arrive as ``float('nan')``; the tree learners
tolerate them, and any feature can be dropped from the dict entirely to represent
absence instead, which is the more idiomatic River encoding.

**Class imbalance.** These learners see the undersampled training stream, where the
imbalance is bounded, but the residual skew still swamps an unweighted incremental
tree. Each wrapper accepts ``class_weights`` and passes a per-observation weight to
River when the underlying learner supports one. Whether the learner accepts a
weight is detected once at construction rather than assumed, and
:attr:`supports_weights` reports the answer, so an unweighted learner is visible
rather than silently ignoring the argument.

**Evaluation prevalence.** These wrappers do not resample and do not calibrate.
Training-side streaming may run at the working-set prevalence, but every reported
metric is scored on natural-prevalence data (V48), and that separation is the
caller's responsibility.

A note on the boosting learner. River implements online bagging and online boosting
(AdaBoost, in the incremental formulation of Freund and Schapire, 1997) but has no
online *gradient* boosting. :class:`OnlineBoostingOnline` therefore operationalizes
the boosting arm of the online-learning family with River's incremental AdaBoost
over Hoeffding trees, and :class:`LogisticSGDOnline` provides an incremental linear
baseline. The naming reflects what the code actually does rather than the family
label.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------
def frame_to_dicts(
    X: object,
    feature_names: list[str] | None = None,
    drop_nulls: bool = False,
) -> list[dict[str, float]]:
    """Convert a frame to River's row-dict format, one dict per observation.

    ``drop_nulls=True`` omits null features from the row entirely, which River reads
    as "not observed" rather than as a numeric value. That is the more faithful
    encoding for the hardware-counter and schema-era missingness in this study,
    where absence is itself informative, but it is off by default because it makes
    rows ragged and some learners prefer a stable feature set.
    """
    if isinstance(X, pl.DataFrame):
        names = feature_names or list(X.columns)
        arr = X.select(names).to_numpy().astype(np.float64)
    elif isinstance(X, np.ndarray):
        arr = np.asarray(X, dtype=np.float64)
        names = feature_names or [f"f{i}" for i in range(arr.shape[1])]
    elif hasattr(X, "columns") and hasattr(X, "to_numpy"):  # pandas DataFrame
        names = feature_names or [str(c) for c in X.columns]
        arr = np.asarray(X[names].to_numpy(), dtype=np.float64)
    else:
        raise TypeError(
            "X must be a polars.DataFrame, numpy.ndarray, or pandas.DataFrame; "
            f"got {type(X)!r}."
        )
    if arr.shape[1] != len(names):
        raise ValueError("feature_names length does not match the frame width")

    if drop_nulls:
        return [
            {n: float(v) for n, v in zip(names, row) if not np.isnan(v)}
            for row in arr
        ]
    return [dict(zip(names, (float(v) for v in row))) for row in arr]


def _to_labels(y: object) -> np.ndarray:
    if isinstance(y, pl.Series):
        arr = y.to_numpy()
    elif isinstance(y, pl.DataFrame):
        if y.width != 1:
            raise ValueError("y as a DataFrame must have exactly one column.")
        arr = y.to_numpy().ravel()
    else:
        arr = np.asarray(y)
    return arr.ravel().astype(np.int8)


def _river():
    try:
        import river
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "river is required for the online learners. Install it with "
            "`pip install 'river>=0.21,<0.23'`."
        ) from exc
    return river


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class OnlineLearner(Protocol):
    """The uniform interface every online wrapper satisfies."""

    def learn_one(self, x: dict[str, float], y: int, w: float = 1.0) -> "OnlineLearner": ...

    def predict_proba_one(self, x: dict[str, float]) -> float: ...

    def predict_proba(self, X: object) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "OnlineLearner": ...


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class _BaseOnlineWrapper:
    """Shared plumbing: weighted learning, positive-class scoring, checkpointing.

    Subclasses set ``self._model`` to a River classifier and ``self._params`` to the
    constructor arguments needed to rebuild it on :meth:`reset`.
    """

    def __init__(
        self,
        model: object,
        class_weights: dict[int, float] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._model = model
        self.class_weights = class_weights or {}
        self.feature_names = feature_names
        self._n_learned: int = 0
        self._supports_weights: bool = self._detect_weight_support(model)

    @staticmethod
    def _detect_weight_support(model: object) -> bool:
        """Ask the learner whether ``learn_one`` takes a sample weight.

        Detected rather than assumed: River's learners are inconsistent about it
        across versions, and silently dropping the weight on a severely imbalanced
        stream would produce an all-negative learner that still looked well-formed.
        """
        try:
            sig = inspect.signature(model.learn_one)
        except (TypeError, ValueError):  # pragma: no cover
            return False
        if "w" in sig.parameters:
            return True
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    @property
    def supports_weights(self) -> bool:
        """True when per-observation weights reach the underlying learner."""
        return self._supports_weights

    @property
    def n_learned(self) -> int:
        """Observations consumed since construction or the last :meth:`reset`."""
        return self._n_learned

    @property
    def model(self) -> object:
        """The wrapped River learner (for introspection, not for mutation)."""
        return self._model

    def learn_one(self, x: dict[str, float], y: int, w: float = 1.0) -> "_BaseOnlineWrapper":
        """Update the model with one observation. Returns self, never a prediction."""
        label = int(y)
        weight = float(w) * float(self.class_weights.get(label, 1.0))
        if self._supports_weights:
            self._model.learn_one(x, label, w=weight)
        else:
            self._model.learn_one(x, label)
        self._n_learned += 1
        return self

    def predict_proba_one(self, x: dict[str, float]) -> float:
        """P(failure) for one observation. Returns 0.0 before the first update."""
        proba = self._model.predict_proba_one(x)
        return float(proba.get(1, 0.0)) if proba else 0.0

    def predict_proba(self, X: object) -> np.ndarray:
        """P(failure) over a frame, row by row (River has no batch path)."""
        rows = frame_to_dicts(X, self.feature_names)
        return np.fromiter(
            (self.predict_proba_one(row) for row in rows), dtype=np.float64, count=len(rows)
        )

    def learn_many(self, X: object, y: object, w: object = None) -> "_BaseOnlineWrapper":
        """Stream a frame in row order. The caller owns the ordering.

        This is a convenience for warm-up windows, not a batch fit: rows are
        consumed one at a time in the order given, so the frame must already be
        sorted the way the stream should arrive.
        """
        rows = frame_to_dicts(X, self.feature_names)
        labels = _to_labels(y)
        if len(rows) != labels.size:
            raise ValueError("X and y have different lengths")
        weights = np.ones(labels.size) if w is None else np.asarray(w, dtype=np.float64).ravel()
        for row, label, weight in zip(rows, labels, weights):
            self.learn_one(row, int(label), float(weight))
        return self

    def save(self, path: str | Path) -> None:
        """Pickle the fitted wrapper (checkpoint-and-resume across sessions)."""
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "_BaseOnlineWrapper":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    def reset(self) -> "_BaseOnlineWrapper":
        """Rebuild the underlying learner from scratch (a one-shot retraining step)."""
        self._model = self._build()
        self._n_learned = 0
        return self

    def _build(self) -> object:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------
class AdaptiveRandomForestOnline(_BaseOnlineWrapper):
    """Adaptive Random Forest (wraps ``river.forest.ARFClassifier``).

    The headline adaptive learner. Each tree carries its own drift detector and is
    replaced by a background tree when its detector fires, so the forest adapts
    without an external retraining trigger. That internal adaptation is the point of
    the comparison against the static baseline.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = DEFAULT_SEED,
        class_weights: dict[int, float] | None = None,
        feature_names: list[str] | None = None,
        **params: object,
    ) -> None:
        self._params = {"n_models": n_models, "seed": seed, **params}
        super().__init__(self._build(), class_weights, feature_names)

    def _build(self) -> object:
        from river import forest

        return forest.ARFClassifier(**self._params)


class HoeffdingAdaptiveTreeOnline(_BaseOnlineWrapper):
    """Hoeffding Adaptive Tree (wraps ``river.tree.HoeffdingAdaptiveTreeClassifier``).

    A single incremental tree that grows branches on statistical evidence (the
    Hoeffding bound) and swaps a subtree for an alternate when ADWIN signals that
    the subtree has gone stale. The cheap adaptive learner: one tree rather than a
    forest, so it is the fallback when the stream budget binds.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        class_weights: dict[int, float] | None = None,
        feature_names: list[str] | None = None,
        **params: object,
    ) -> None:
        self._params = {"seed": seed, **params}
        super().__init__(self._build(), class_weights, feature_names)

    def _build(self) -> object:
        from river import tree

        return tree.HoeffdingAdaptiveTreeClassifier(**self._params)


class OnlineBoostingOnline(_BaseOnlineWrapper):
    """Online boosting (wraps ``river.ensemble.AdaBoostClassifier``).

    River implements incremental bagging and boosting but no online *gradient*
    boosting, so the boosting arm of the online family is operationalized here as
    incremental AdaBoost (Freund and Schapire, 1997) over Hoeffding trees. Named for
    what it does rather than for the family label, so the deviation is visible in
    the code and not only in the prose.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = DEFAULT_SEED,
        class_weights: dict[int, float] | None = None,
        feature_names: list[str] | None = None,
        **params: object,
    ) -> None:
        self._params = {"n_models": n_models, "seed": seed, **params}
        super().__init__(self._build(), class_weights, feature_names)

    def _build(self) -> object:
        from river import ensemble, tree

        params = dict(self._params)
        model = params.pop("model", None) or tree.HoeffdingTreeClassifier()
        return ensemble.AdaBoostClassifier(model=model, **params)


class LogisticSGDOnline(_BaseOnlineWrapper):
    """Incremental logistic regression (wraps ``river.linear_model.LogisticRegression``).

    The linear incremental baseline, and the reference point for whether the
    adaptive tree learners are earning their cost. Features are standardized inside
    the pipeline, because an unscaled SMART raw count would otherwise dominate the
    gradient.
    """

    def __init__(
        self,
        class_weights: dict[int, float] | None = None,
        feature_names: list[str] | None = None,
        **params: object,
    ) -> None:
        self._params = dict(params)
        super().__init__(self._build(), class_weights, feature_names)

    def _build(self) -> object:
        from river import compose, linear_model, preprocessing

        return compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(**self._params),
        )


# ---------------------------------------------------------------------------
# Ensemble of online learners
# ---------------------------------------------------------------------------
class OnlineSoftVotingEnsemble:
    """Weighted average of online learners, with optional dynamic reweighting.

    Static mode (``dynamic=False``) averages the members' probabilities under fixed
    weights, mirroring the batch ``SoftVotingStack``.

    Dynamic mode tracks each member's recent loss with an exponentially weighted
    moving average and reweights toward the members that are currently performing,
    which is the ensemble-library mitigation for recurring drift: a member that
    suited an earlier regime is down-weighted rather than discarded, and recovers
    weight if that regime returns.

    Member losses update only on :meth:`learn_one`, using the prediction each member
    makes *before* it sees the label, so the weights are prequential and carry no
    peek at the outcome.
    """

    def __init__(
        self,
        members: dict[str, OnlineLearner],
        dynamic: bool = True,
        decay: float = 0.99,
        min_weight: float = 0.01,
    ) -> None:
        if not members:
            raise ValueError("at least one member is required")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.members = dict(members)
        self.dynamic = dynamic
        self.decay = decay
        self.min_weight = min_weight
        self._loss: dict[str, float] = {name: 0.0 for name in self.members}
        self._n_learned: int = 0

    @property
    def n_learned(self) -> int:
        return self._n_learned

    def weights(self) -> dict[str, float]:
        """Current member weights, normalized to sum to 1."""
        if not self.dynamic or self._n_learned == 0:
            equal = 1.0 / len(self.members)
            return {name: equal for name in self.members}
        # Inverse recent loss, floored so a member can always recover.
        raw = {name: max(1.0 - loss, self.min_weight) for name, loss in self._loss.items()}
        total = sum(raw.values())
        return {name: w / total for name, w in raw.items()}

    def predict_proba_one(self, x: dict[str, float]) -> float:
        weights = self.weights()
        return float(
            sum(w * self.members[name].predict_proba_one(x) for name, w in weights.items())
        )

    def predict_proba(self, X: object) -> np.ndarray:
        rows = frame_to_dicts(X)
        return np.fromiter(
            (self.predict_proba_one(row) for row in rows), dtype=np.float64, count=len(rows)
        )

    def learn_one(self, x: dict[str, float], y: int, w: float = 1.0) -> "OnlineSoftVotingEnsemble":
        label = int(y)
        for name, member in self.members.items():
            if self.dynamic:
                # Score before learning: the member's loss on an observation it has
                # not yet seen the label for.
                pred = member.predict_proba_one(x)
                loss = abs(label - pred)
                self._loss[name] = self.decay * self._loss[name] + (1.0 - self.decay) * loss
            member.learn_one(x, label, w)
        self._n_learned += 1
        return self

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "OnlineSoftVotingEnsemble":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ONLINE_LEARNERS: dict[str, type[_BaseOnlineWrapper]] = {
    "adaptive_random_forest": AdaptiveRandomForestOnline,
    "hoeffding_adaptive_tree": HoeffdingAdaptiveTreeOnline,
    "online_boosting": OnlineBoostingOnline,
    "logistic_sgd": LogisticSGDOnline,
}


def build_online_learner(name: str, **params: object) -> _BaseOnlineWrapper:
    """Build an online learner by name. Raises on an unknown name."""
    if name not in ONLINE_LEARNERS:
        raise ValueError(
            f"unknown online learner {name!r}; available: {sorted(ONLINE_LEARNERS)}"
        )
    _river()  # fail early and clearly when the library is missing
    return ONLINE_LEARNERS[name](**params)


def balanced_class_weights(y: object) -> dict[int, float]:
    """Inverse-prior class weights from a label array, matching the batch wrappers.

    Returns ``{0: n / (2 * n_neg), 1: n / (2 * n_pos)}``, the scikit-learn
    ``class_weight="balanced"`` convention, so a streamed learner carries the same
    cost sensitivity as its batch counterpart.
    """
    labels = _to_labels(y)
    n = labels.size
    n_pos = int((labels == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}
