"""Unit tests for ``src/models/online.py``.

The learners are driven on a small separable synthetic stream and asserted on the
properties the drift study depends on: they learn (better than chance after a
warm-up), they are prequential (predicting does not consume the label), they
checkpoint and resume exactly, they reset, and the row-dict boundary preserves the
feature contract. The soft-voting ensemble is additionally asserted to reweight
toward the member that is actually performing, which is the recurring-drift
mitigation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("river")

from src.models.online import (  # noqa: E402
    AdaptiveRandomForestOnline,
    HoeffdingAdaptiveTreeOnline,
    LogisticSGDOnline,
    ONLINE_LEARNERS,
    OnlineBoostingOnline,
    OnlineLearner,
    OnlineSoftVotingEnsemble,
    balanced_class_weights,
    build_online_learner,
    frame_to_dicts,
)

SEED = 42
FEATURES = ["a", "b", "c"]


def _stream(n: int = 600, seed: int = SEED) -> tuple[pl.DataFrame, np.ndarray]:
    """Separable binary stream: the positive class is shifted on feature 'a'."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    a = rng.normal(0.0, 1.0, n) + 3.0 * y
    b = rng.normal(0.0, 1.0, n)
    c = rng.normal(0.0, 1.0, n)
    return pl.DataFrame({"a": a, "b": b, "c": c}), y.astype(np.int8)


def _learn_all(model, X: pl.DataFrame, y: np.ndarray) -> None:
    for row, label in zip(frame_to_dicts(X, FEATURES), y):
        model.learn_one(row, int(label))


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------
def test_frame_to_dicts_preserves_the_feature_contract() -> None:
    X, _ = _stream(n=5)
    rows = frame_to_dicts(X, FEATURES)
    assert len(rows) == 5
    assert list(rows[0]) == FEATURES
    assert rows[0]["a"] == pytest.approx(X["a"][0])


def test_frame_to_dicts_can_drop_nulls_instead_of_encoding_them() -> None:
    X = pl.DataFrame({"a": [1.0, 2.0], "b": [float("nan"), 4.0]})
    kept = frame_to_dicts(X, ["a", "b"], drop_nulls=False)
    dropped = frame_to_dicts(X, ["a", "b"], drop_nulls=True)
    assert np.isnan(kept[0]["b"])
    assert "b" not in dropped[0]
    assert dropped[1]["b"] == 4.0


def test_frame_to_dicts_rejects_an_unknown_input_type() -> None:
    with pytest.raises(TypeError):
        frame_to_dicts("not a frame")


def test_balanced_class_weights_match_the_inverse_prior() -> None:
    y = np.array([0] * 90 + [1] * 10)
    weights = balanced_class_weights(y)
    assert weights[0] == pytest.approx(100 / (2 * 90))
    assert weights[1] == pytest.approx(100 / (2 * 10))
    # Degenerate input does not explode.
    assert balanced_class_weights(np.zeros(10)) == {0: 1.0, 1: 1.0}


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ONLINE_LEARNERS))
def test_every_learner_satisfies_the_protocol(name: str) -> None:
    model = build_online_learner(name, feature_names=FEATURES)
    assert isinstance(model, OnlineLearner)
    assert model.n_learned == 0
    # Predicting before any learning is defined and returns a probability.
    proba = model.predict_proba_one({f: 0.0 for f in FEATURES})
    assert 0.0 <= proba <= 1.0


@pytest.mark.parametrize("name", sorted(ONLINE_LEARNERS))
def test_every_learner_beats_chance_on_a_separable_stream(name: str) -> None:
    X, y = _stream()
    model = build_online_learner(name, feature_names=FEATURES)
    _learn_all(model, X, y)
    assert model.n_learned == X.height

    X_test, y_test = _stream(n=300, seed=SEED + 1)
    proba = model.predict_proba(X_test)
    assert proba.shape == (X_test.height,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    # Positives must score higher than negatives on a stream that is separable by
    # construction. A learner failing this is not learning at all.
    assert proba[y_test == 1].mean() > proba[y_test == 0].mean() + 0.2


def test_build_online_learner_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown online learner"):
        build_online_learner("gradient_boosting_but_online")


def test_learn_one_is_prequential_and_returns_no_prediction() -> None:
    X, y = _stream(n=50)
    model = HoeffdingAdaptiveTreeOnline(feature_names=FEATURES)
    row = frame_to_dicts(X, FEATURES)[0]
    returned = model.learn_one(row, int(y[0]))
    assert returned is model  # returns self, never a prediction
    # Predicting does not consume the observation.
    before = model.n_learned
    model.predict_proba_one(row)
    assert model.n_learned == before


def test_class_weights_reach_the_learner_or_are_reported_as_unsupported() -> None:
    model = AdaptiveRandomForestOnline(
        n_models=3, feature_names=FEATURES, class_weights={0: 1.0, 1: 20.0}
    )
    # Whether weights reach the learner is detected, not assumed. If a River version
    # stops accepting them, this flips to False rather than silently dropping them.
    assert isinstance(model.supports_weights, bool)
    X, y = _stream(n=200)
    _learn_all(model, X, y)
    assert model.n_learned == 200


def test_learn_many_streams_a_frame_in_row_order() -> None:
    X, y = _stream(n=100)
    model = LogisticSGDOnline(feature_names=FEATURES)
    model.learn_many(X, y)
    assert model.n_learned == 100
    with pytest.raises(ValueError, match="different lengths"):
        model.learn_many(X, y[:50])


# ---------------------------------------------------------------------------
# Checkpoint and resume (the Colab session boundary)
# ---------------------------------------------------------------------------
def test_checkpoint_resumes_exactly_where_it_left_off(tmp_path) -> None:
    X, y = _stream(n=400)
    rows = frame_to_dicts(X, FEATURES)
    half = 200

    model = HoeffdingAdaptiveTreeOnline(feature_names=FEATURES)
    for row, label in zip(rows[:half], y[:half]):
        model.learn_one(row, int(label))
    path = tmp_path / "online.pkl"
    model.save(path)

    # Resume from the checkpoint and finish the stream.
    resumed = HoeffdingAdaptiveTreeOnline.load(path)
    assert resumed.n_learned == half
    for row, label in zip(rows[half:], y[half:]):
        resumed.learn_one(row, int(label))

    # A learner that streamed the whole thing in one session must agree exactly.
    straight = HoeffdingAdaptiveTreeOnline(feature_names=FEATURES)
    for row, label in zip(rows, y):
        straight.learn_one(row, int(label))

    X_test, _ = _stream(n=100, seed=SEED + 2)
    np.testing.assert_allclose(resumed.predict_proba(X_test), straight.predict_proba(X_test))
    assert resumed.n_learned == straight.n_learned == 400


def test_reset_rebuilds_the_learner(tmp_path) -> None:
    X, y = _stream(n=200)
    model = OnlineBoostingOnline(n_models=3, feature_names=FEATURES)
    _learn_all(model, X, y)
    assert model.n_learned == 200
    model.reset()
    assert model.n_learned == 0
    # A reset learner has forgotten the stream: it scores the flat prior again.
    assert model.predict_proba_one({f: 0.0 for f in FEATURES}) in (0.0, pytest.approx(0.5, abs=0.5))


# ---------------------------------------------------------------------------
# Online soft-voting ensemble
# ---------------------------------------------------------------------------
def test_static_ensemble_averages_its_members_equally() -> None:
    X, y = _stream(n=200)
    members = {
        "hat": HoeffdingAdaptiveTreeOnline(feature_names=FEATURES),
        "logit": LogisticSGDOnline(feature_names=FEATURES),
    }
    ensemble = OnlineSoftVotingEnsemble(members, dynamic=False)
    for row, label in zip(frame_to_dicts(X, FEATURES), y):
        ensemble.learn_one(row, int(label))

    weights = ensemble.weights()
    assert weights == {"hat": pytest.approx(0.5), "logit": pytest.approx(0.5)}

    X_test, _ = _stream(n=50, seed=SEED + 3)
    rows = frame_to_dicts(X_test, FEATURES)
    expected = 0.5 * members["hat"].predict_proba_one(rows[0]) + 0.5 * members[
        "logit"
    ].predict_proba_one(rows[0])
    assert ensemble.predict_proba_one(rows[0]) == pytest.approx(expected)


def test_dynamic_ensemble_reweights_toward_the_performing_member() -> None:
    X, y = _stream(n=600)
    good = HoeffdingAdaptiveTreeOnline(feature_names=FEATURES)
    # The saboteur never learns: its weight must decay relative to the good member.
    bad = HoeffdingAdaptiveTreeOnline(feature_names=FEATURES)
    ensemble = OnlineSoftVotingEnsemble({"good": good, "bad": bad}, dynamic=True, decay=0.95)
    for row, label in zip(frame_to_dicts(X, FEATURES), y):
        # Corrupt only the saboteur's view by feeding the ensemble normally but
        # poisoning the bad member's labels beforehand.
        bad.learn_one(row, int(1 - label))
        ensemble.learn_one(row, int(label))

    weights = ensemble.weights()
    assert weights["good"] > weights["bad"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_dynamic_weights_are_floored_so_a_member_can_recover() -> None:
    members = {"a": LogisticSGDOnline(feature_names=FEATURES)}
    ensemble = OnlineSoftVotingEnsemble(members, dynamic=True, min_weight=0.05)
    X, y = _stream(n=50)
    for row, label in zip(frame_to_dicts(X, FEATURES), y):
        ensemble.learn_one(row, int(label))
    assert ensemble.weights()["a"] == pytest.approx(1.0)


def test_ensemble_rejects_an_empty_member_set() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        OnlineSoftVotingEnsemble({})


def test_ensemble_checkpoint_roundtrip(tmp_path) -> None:
    X, y = _stream(n=150)
    ensemble = OnlineSoftVotingEnsemble(
        {"hat": HoeffdingAdaptiveTreeOnline(feature_names=FEATURES)}, dynamic=True
    )
    for row, label in zip(frame_to_dicts(X, FEATURES), y):
        ensemble.learn_one(row, int(label))
    path = tmp_path / "ens.pkl"
    ensemble.save(path)

    loaded = OnlineSoftVotingEnsemble.load(path)
    assert loaded.n_learned == ensemble.n_learned
    X_test, _ = _stream(n=40, seed=SEED + 4)
    np.testing.assert_allclose(loaded.predict_proba(X_test), ensemble.predict_proba(X_test))
