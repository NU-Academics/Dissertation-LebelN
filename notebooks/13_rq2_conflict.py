# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 13. RQ2 - Conflict Resolution (Google Cluster Traces)
#
# **Research question.** RQ2: how can supervised classifiers predict, at the
# moment a scheduling conflict is detected, whether that conflict will resolve
# cleanly (all affected work finishes) or escalate (failure / persistent eviction
# churn)? The target is a >0.80 resolution-success result. Consistent with the
# study's imbalance-robust convention (Chicco & Jurman, 2023), MCC is the primary
# hypothesis-testing metric; F1 and PR-AUC are co-reported (Saito & Rehmsmeier,
# 2015). Reporting follows TRIPOD+AI (Collins et al., 2024).
#
# **Conflict episodes.** Built by `src.features.conflict_labels` over three
# conflict types (resource contention, priority inversion, and scheduling
# violations), each carrying detection-time features and a `resolution_outcome`
# label. The three are pooled (one-hot `conflict_type`).
#
# **Leakage discipline.** Every feature is knowable at the conflict's detection
# time (`start_time`); the `resolution_outcome` label is derived only from how the
# conflict ends and never appears as a feature (the RQ1 feature-source lesson,
# V35). The train / val / test split is by `conflict_id`, so no conflict episode
# straddles the boundary. SMOTE and class weighting are applied inside the
# training portion only; the decision threshold is tuned on validation, never
# fixed at 0.5; and every metric is reported with a stratified bootstrap CI.
#
# **Module reuse.** The model wrappers (`src.models.ensemble`), the metrics with
# CIs (`src.evaluation.metrics`), and the one-sample threshold test
# (`src.evaluation.hypothesis`) are imported, not re-inlined. Random Forest is the
# `RandomForestWrapper`. The Decision Tree, Linear SVM, and one-hidden-layer Keras
# classifiers are built inline here and extracted to `src.models.classifier` once
# this notebook validates them.
#
# **Output.** `outputs/tables/rq2_results.csv`
# (`model, conflict_scope, metric, value, ci_low, ci_high`) and
# `outputs/tables/rq2_hypothesis_test.csv`.
#
# > **Starting structure.** Detection thresholds, the working-set scope, the model
# > set, and the operating thresholds are expected to be revised as runs come back.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars lightgbm xgboost scikit-learn imbalanced-learn tensorflow pandas pyarrow google-cloud-bigquery-storage optuna pyyaml matplotlib

# %%
import os
import sys
from pathlib import Path

from google.colab import userdata

GITHUB_PAT = userdata.get('GITHUB_PAT')
REPO_OWNER = 'NU-Academics'
REPO_NAME = 'Dissertation-LebelN'
REPO_DIR = f'/content/{REPO_NAME}'
REPO_URL = f'https://{GITHUB_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git'

if os.path.exists(REPO_DIR):
    # !cd {REPO_DIR} && git pull --quiet
    print(f"Repo already present at {REPO_DIR}; pulled latest.")
else:
    # !git clone --quiet {REPO_URL} {REPO_DIR}
    print(f"Cloned repo to {REPO_DIR}.")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Colab caches imported modules in sys.modules, so a later `git pull` has no
# effect until the runtime restarts. Drop any previously imported repo modules
# here, so the freshly pulled source is what gets imported in the cells below.
for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from utils.colab_setup import setup_drive, OUTPUT_DIR

setup_drive()

# %%
import gc
import warnings

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from src.features.conflict_labels import (
    CONFLICT_TYPES,
    LABEL_COLUMN,
    META_COLUMNS,
    build_conflict_dataset,
)
from src.models.ensemble import RandomForestWrapper
from src.evaluation.metrics import (
    calibration_table,
    f1_with_ci,
    mcc_with_ci,
    pr_auc_with_ci,
)
from src.evaluation.hypothesis import one_sample_threshold_test

RANDOM_SEED = 42  # P14 random-seed convention.
rng = np.random.default_rng(RANDOM_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'
CACHE_DIR = OUTPUT_DIR / 'cache'
for directory in (TABLES_DIR, FIGURES_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RQ2_RESULTS_CSV = TABLES_DIR / 'rq2_results.csv'
RQ2_HYPOTHESIS_CSV = TABLES_DIR / 'rq2_hypothesis_test.csv'

# %% [markdown]
# ## 1. Build the labeled conflict working set
#
# The three labelers are pure-Polars transforms, but the source tables are far too
# large to materialize in Colab, so the heavy reduction happens BigQuery-side. Two
# working-set scopes are pulled because the conflict types depend on different
# interaction structures, and an independent per-instance sample would erase them:
#
# - **Machine scope** (resource contention + priority inversion): all instances on
#   a sample of whole machines, so two instances that collide on a machine are both
#   retained. Supplies the machine-bearing event stream and the residency pull
#   (`instance_usage` pre-aggregated BigQuery-side to one `MIN(start)`,
#   `MAX(end)`, `machine_id` row per instance, so the 7.5B-row usage table never
#   lands here). Machine capacity comes from `machine_events`.
# - **Collection scope** (scheduling violations): all instance events of a sample
#   of whole collections, so a collection's SCHEDULE / EVICT / FAIL churn is
#   observed in full (that churn lives in the instance stream, not in
#   `collection_events`).
#
# Both scopes are bounded to a schedule-day window. The resulting conflict rate and
# the collection-type concentration are the basis for calibrating the detection
# thresholds and for the RQ2 collection-type-concentration finding. Widen
# `DAY_LO/DAY_HI`, `MACHINE_SAMPLE_NUM`, or `COLL_SAMPLE_NUM` for tighter CIs once
# the pipeline runs clean.

# %%
from google.colab import auth

from utils.bq_client import get_client, table_ref

auth.authenticate_user()
_bq = get_client()

DAY_US = 86_400_000_000
# Trace time is microseconds offset; anchor at 0 and window by schedule day.
DAY_LO = 10                     # inclusive start day of the working-set window
DAY_HI = 12                     # inclusive end day (a 2-day slice, as the exploration queries used)
SAMPLE_DEN = 1000               # hash denominator for both entity samples

# Machine scope (resource contention + priority inversion): keep ALL instances on
# a sample of whole machines, so co-residency on a machine survives the sample. An
# independent per-instance sample would split most colliding pairs and erase the
# very interaction these two conflict types depend on.
MACHINE_SAMPLE_NUM = 25         # ~2.5% of the ~10,005 machines (raise once the run is stable)
# Collection scope (scheduling violations): keep ALL instances of a sample of
# whole collections, so a collection's scheduling churn is observed in full.
COLL_SAMPLE_NUM = 20            # ~2% of collections

_T_LO = DAY_LO * DAY_US
_T_HI = (DAY_HI + 1) * DAY_US
_machine_keep = f"MOD(ABS(FARM_FINGERPRINT(CAST(machine_id AS STRING))), {SAMPLE_DEN}) < {MACHINE_SAMPLE_NUM}"
_coll_keep = f"MOD(ABS(FARM_FINGERPRINT(CAST(collection_id AS STRING))), {SAMPLE_DEN}) < {COLL_SAMPLE_NUM}"


def _events_machine_sql() -> str:
    """Machine-bearing instance events on the sampled machines in the window,
    restricted to the event types the labelers use: SCHEDULE (3), EVICT (4), FAIL
    (5), FINISH (6), LOST (8). These carry everything needed (requests, priority,
    scheduling class, terminal outcome); dropping the high-volume UPDATE_RUNNING /
    UPDATE_PENDING events at the source is what keeps the pull inside Colab RAM."""
    return f"""
SELECT time, type, collection_id, instance_index, machine_id,
       priority, scheduling_class, cpu_request, memory_request
FROM {table_ref('instance_events_full')}
WHERE time >= {_T_LO} AND time < {_T_HI}
  AND machine_id IS NOT NULL AND {_machine_keep}
  AND type IN (3, 4, 5, 6, 8)
"""


def _residency_sql() -> str:
    """One row per instance on the sampled machines: its machine and residency
    interval (the labeler's own min/max is idempotent on this)."""
    return f"""
SELECT collection_id, instance_index, machine_id,
       MIN(start_time) AS start_time, MAX(end_time) AS end_time
FROM {table_ref('instance_usage_full')}
WHERE start_time >= {_T_LO} AND start_time < {_T_HI}
  AND machine_id IS NOT NULL AND {_machine_keep}
GROUP BY collection_id, instance_index, machine_id
"""


def _events_collection_sql() -> str:
    """All instance events of the sampled collections in the window, for the
    collection-level scheduling-churn aggregation (the SCHEDULE / EVICT / FAIL
    signal lives in the instance stream, not in collection_events)."""
    return f"""
SELECT time, type, collection_id, scheduling_class
FROM {table_ref('instance_events_full')}
WHERE time >= {_T_LO} AND time < {_T_HI} AND {_coll_keep}
  AND type IN (3, 4, 5, 8)
"""


def _capacity_sql() -> str:
    """Per-machine capacity from machine_events (ADD events carry capacity)."""
    return f"""
SELECT machine_id,
       MAX(capacity_cpus) AS capacity_cpus,
       MAX(capacity_memory) AS capacity_memory
FROM {table_ref('machine_events_full')}
WHERE capacity_cpus IS NOT NULL
GROUP BY machine_id
"""


def _pull(sql: str) -> pl.DataFrame:
    arrow = _bq.query(sql).to_arrow(create_bqstorage_client=True)
    df = pl.from_arrow(arrow)
    del arrow
    gc.collect()
    return df


CONFLICT_CACHE = CACHE_DIR / f"rq2_conflicts_d{DAY_LO}-{DAY_HI}_m{MACHINE_SAMPLE_NUM}_c{COLL_SAMPLE_NUM}.parquet"
REBUILD_CONFLICTS = True   # set False to reload the labeled conflict frame from Drive

if REBUILD_CONFLICTS or not CONFLICT_CACHE.exists():
    print(f"Pulling working-set tables (days {DAY_LO}-{DAY_HI}, "
          f"~{MACHINE_SAMPLE_NUM / 10:.1f}% machines / ~{COLL_SAMPLE_NUM / 10:.1f}% collections) ...")
    events_machine_df = _pull(_events_machine_sql())
    residency_df = _pull(_residency_sql())
    events_collection_df = _pull(_events_collection_sql())
    cap_df = _pull(_capacity_sql())
    for _n, _d in (("events (machine)", events_machine_df), ("residency", residency_df),
                   ("events (collection)", events_collection_df), ("capacity", cap_df)):
        print(f"  {_n:20s} {_d.height:>10,} rows")

    # Stream the labeling (explode + as-of join + group-by) so Polars spills to
    # disk rather than holding every intermediate in RAM.
    conflicts = build_conflict_dataset(
        residency_df.lazy(), events_machine_df.lazy(), cap_df.lazy(), events_collection_df.lazy()
    ).collect(engine="streaming")
    conflicts.write_parquet(CONFLICT_CACHE)
    print(f"Labeled {conflicts.height:,} conflict episodes -> {CONFLICT_CACHE.name}")
    del events_machine_df, residency_df, events_collection_df, cap_df
    gc.collect()
else:
    conflicts = pl.read_parquet(CONFLICT_CACHE)
    print(f"Loaded {conflicts.height:,} conflict episodes <- {CONFLICT_CACHE.name}")

# Guard: the strictly-prior history features must be present. If they are not, an
# old conflict_labels module is loaded (restart the runtime) or a pre-history cache
# was reloaded (set REBUILD_CONFLICTS = True).
_expected_history = {"prior_fail_total", "evicted_prior_fail", "coll_prior_fail"}
assert _expected_history <= set(conflicts.columns), (
    "History features missing from the conflict frame: "
    f"{sorted(_expected_history - set(conflicts.columns))}. Restart the Colab "
    "runtime so the pulled conflict_labels is imported, and rebuild "
    "(REBUILD_CONFLICTS = True)."
)

# Conflict-rate and class balance per type (the calibration / concentration view).
print("\nConflict episodes by type (positive = clean resolution):")
for ct in CONFLICT_TYPES:
    sub = conflicts.filter(pl.col("conflict_type") == ct)
    if sub.height == 0:
        print(f"  {ct:22s} 0 episodes (widen the window / sample or relax thresholds)")
        continue
    pos = int(sub.filter(pl.col(LABEL_COLUMN) == 1).height)
    print(f"  {ct:22s} {sub.height:>8,} episodes | clean {pos:>7,} "
          f"({pos / sub.height:.3f}) | escalate:clean {(sub.height - pos) / max(pos, 1):.2f}:1")

# %% [markdown]
# ## 2. Feature assembly and the conflict-keyed split
#
# `conflict_type` is one-hot encoded and retained as a feature; the remaining
# per-type detection-time columns are kept (the pooled union carries nulls where a
# column does not apply to a type, imputed with a missing-indicator below). The
# identity columns in `META_COLUMNS` are dropped. The split is by a hash of
# `conflict_id`, so no episode straddles train / val / test.

# %%
conflicts = conflicts.with_columns(
    [(pl.col("conflict_type") == ct).cast(pl.Int8).alias(f"conflict_type_{ct}")
     for ct in CONFLICT_TYPES]
)

_DROP = set(META_COLUMNS) | {"conflict_type", LABEL_COLUMN}
FEATURE_COLS = [c for c in conflicts.columns if c not in _DROP]
print(f"{len(FEATURE_COLS)} feature columns (incl. {len(CONFLICT_TYPES)} conflict-type indicators).")


def _split_mask(df: pl.DataFrame) -> pl.Series:
    """Deterministic [0, 1) hash of the entity `group_key` for the group split.

    Splitting on `group_key` (the machine for contention, the instance for
    inversion, the collection for violation) rather than `conflict_id` (per
    episode) is what stops a recurring entity's episodes leaking across train /
    test: an instance that contributes many inversion episodes, or a machine with
    many contended windows, lands entirely on one side."""
    return (df["group_key"].hash(seed=RANDOM_SEED) % 1_000_000) / 1_000_000


_u = _split_mask(conflicts)
train_df = conflicts.filter(_u >= 0.30)
val_df = conflicts.filter((_u >= 0.15) & (_u < 0.30))
test_df = conflicts.filter(_u < 0.15)

# No entity may straddle the split (the leakage guard).
_ids = [set(d["group_key"].to_list()) for d in (train_df, val_df, test_df)]
assert not (_ids[0] & _ids[1]) and not (_ids[0] & _ids[2]) and not (_ids[1] & _ids[2]), \
    "group_key straddles the split; hashing is wrong."
for _name, _d in (("train", train_df), ("val", val_df), ("test", test_df)):
    _p = int(_d.filter(pl.col(LABEL_COLUMN) == 1).height)
    print(f"  {_name:5s} {_d.height:>8,} | clean {_p:>7,} ({_p / max(_d.height, 1):.3f})")

# %% [markdown]
# ## 3. Modeling helpers
#
# Imputation is fitted on training data only (median + missing-indicator, which
# preserves the per-type null-as-signal of the pooled union), SMOTE resamples the
# imputed training matrix only, and the decision threshold is tuned on validation
# by maximizing MCC. Linear SVM has no `predict_proba`, so its `decision_function`
# scores are used for thresholding and PR-AUC. Random Forest is the
# `RandomForestWrapper`; the Decision Tree, Linear SVM, and Keras NN are inline
# pending extraction to `src.models.classifier`.

# %%
NATIVE_TREE = {"random_forest"}  # wrapper handles its own class weighting


def make_xy(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = df.select(LABEL_COLUMN).to_numpy().ravel().astype(np.int8)
    return x, y


def _build_keras_nn(n_features: int):
    """One hidden layer (32 units), ReLU, sigmoid output, Adam at defaults. No
    grid search; the point is to test whether the NN categorically beats the tree
    methods, which is itself informative for Chapter 4."""
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


class _KerasAdapter:
    """Minimal fit / predict_proba adapter so the Keras NN matches the other
    learners' interface (1-D positive-class score)."""

    def __init__(self, n_features: int) -> None:
        self._model = _build_keras_nn(n_features)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_KerasAdapter":
        self._model.fit(x, y, epochs=10, batch_size=256, verbose=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x, verbose=0).ravel()


MODEL_NAMES = [
    "logistic_regression", "most_frequent",   # baselines
    "decision_tree", "linear_svm", "random_forest", "keras_nn",  # the four classifiers
]


def make_estimator(name: str, n_features: int):
    """Construct a learner exposing `fit(X, y)` and a 1-D positive-class score via
    `predict_proba` or `decision_function`. Class weighting is inverse-prior where
    the learner supports it; SMOTE (applied by the harness) covers the rest."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    if name == "most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)
    if name == "random_forest":
        return RandomForestWrapper(random_state=RANDOM_SEED)  # reused, not rebuilt
    if name == "keras_nn":
        return _KerasAdapter(n_features)
    raise ValueError(f"Unknown model: {name}")


def _scores(est, x: np.ndarray) -> np.ndarray:
    """1-D positive-class score from whichever interface the learner exposes."""
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(x)
        proba = np.asarray(proba)
        return proba if proba.ndim == 1 else proba[:, 1]
    if hasattr(est, "decision_function"):
        return np.asarray(est.decision_function(x)).ravel()
    return est.predict(x).astype(np.float64)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Decision threshold maximizing MCC on validation scores."""
    grid = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best_t, best_m = float(np.median(y_score)), -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, (y_score >= t).astype(np.int8))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def _resample_train(x_tr: np.ndarray, y_tr: np.ndarray, do_smote: bool):
    """Fit the imputer on train, optionally SMOTE the imputed train matrix. Return
    the fitted imputer and the (resampled) training arrays."""
    imp = SimpleImputer(strategy="median", add_indicator=True).fit(x_tr)
    x_tr_i = imp.transform(x_tr)
    if do_smote and int(y_tr.sum()) > 5 and int((y_tr == 0).sum()) > 5:
        x_tr_i, y_tr = SMOTE(random_state=RANDOM_SEED).fit_resample(x_tr_i, y_tr)
    return imp, x_tr_i, y_tr


def fit_score(name: str, cols: list[str], tr: pl.DataFrame, va: pl.DataFrame,
              te: pl.DataFrame, est_override=None) -> dict:
    """Fit on train (impute + SMOTE training-only), tune the threshold on val,
    score test with stratified-bootstrap CIs from `src.evaluation.metrics`. The
    train / val / test frames are passed in so the same routine serves the pooled
    set and each per-conflict-type subset."""
    x_tr, y_tr = make_xy(tr, cols)
    x_val, y_val = make_xy(va, cols)
    x_te, y_te = make_xy(te, cols)

    do_smote = name != "most_frequent"
    imp, x_tr_r, y_tr_r = _resample_train(x_tr, y_tr, do_smote)
    x_val_i, x_te_i = imp.transform(x_val), imp.transform(x_te)

    est = est_override if est_override is not None else make_estimator(name, x_tr_r.shape[1])
    est.fit(x_tr_r, y_tr_r)

    thr = best_threshold(y_val, _scores(est, x_val_i))
    test_score = _scores(est, x_te_i)
    y_pred = (test_score >= thr).astype(np.int8)

    mcc = mcc_with_ci(y_te, y_pred, seed=RANDOM_SEED)
    f1 = f1_with_ci(y_te, y_pred, seed=RANDOM_SEED)
    pr = pr_auc_with_ci(y_te, test_score, seed=RANDOM_SEED)
    val_mcc = float(matthews_corrcoef(y_val, (_scores(est, x_val_i) >= thr).astype(np.int8)))

    del x_tr, x_val, x_te, x_tr_r
    gc.collect()
    return {"model": name, "threshold": thr, "val_mcc": val_mcc,
            "metrics": {"mcc": mcc, "f1": f1, "pr_auc": pr},
            "test_score": test_score, "y_pred": y_pred}


# %% [markdown]
# ## 4. Time-blocked cross-validation (stability diagnostic)
#
# Expanding-window folds ordered by `start_time` within the training split, used
# as a stability check (mean / min / max fold MCC). SMOTE runs inside each fold's
# training portion only.

# %%
def time_block_cv(name: str, cols: list[str], train_frame: pl.DataFrame, n_folds: int = 3) -> dict:
    tr = train_frame.sort("start_time")
    edges = np.linspace(0, tr.height, n_folds + 2, dtype=int)
    fold_mccs: list[float] = []
    for k in range(1, n_folds + 1):
        tr_part = tr[: edges[k]]
        te_part = tr[edges[k]: edges[k + 1]]
        if te_part.height == 0 or tr_part.filter(pl.col(LABEL_COLUMN) == 1).height < 5:
            continue
        x_tr, y_tr = make_xy(tr_part, cols)
        x_te, y_te = make_xy(te_part, cols)
        imp, x_tr_r, y_tr_r = _resample_train(x_tr, y_tr, name != "most_frequent")
        est = make_estimator(name, x_tr_r.shape[1])
        est.fit(x_tr_r, y_tr_r)
        score = _scores(est, imp.transform(x_te))
        thr = best_threshold(y_te, score)
        fold_mccs.append(float(matthews_corrcoef(y_te, (score >= thr).astype(np.int8))))
        del est, x_tr, x_te, x_tr_r
        gc.collect()
    if not fold_mccs:
        return {"cv_mcc_mean": float("nan"), "cv_mcc_min": float("nan"), "cv_mcc_max": float("nan")}
    arr = np.array(fold_mccs)
    return {"cv_mcc_mean": float(arr.mean()), "cv_mcc_min": float(arr.min()), "cv_mcc_max": float(arr.max())}


# %% [markdown]
# ## 5. Run the model zoo per conflict scope
#
# The three conflict types have very different clean-resolution prevalence
# (contention ~0.05, scheduling violation ~0.20, priority inversion ~0.95). A
# single pooled model with the `conflict_type` one-hot would let the type
# indicator carry much of the separation, inflating the pooled MCC without
# demonstrating genuine within-type discrimination. So the zoo runs once per
# conflict type and once pooled; the per-type scopes are the honest result and the
# >0.80 target is judged on them, with pooled reported as context. Each scope is
# the global conflict-keyed split filtered to that type, so the no-straddle
# property is preserved (a conflict_id belongs to exactly one type).
#
# Two baselines (logistic regression, most-frequent) and the four classifiers
# (Decision Tree, Linear SVM, Random Forest, one-hidden-layer Keras NN). The best
# model in a scope is the highest validation MCC among the non-dummy learners.

# %%
SCOPES = ["pooled", *CONFLICT_TYPES]
RUN_MODEL_ZOO = True
RECORDS_CACHE = CACHE_DIR / "rq2_records.pkl"

import pickle


def _scope_frames(scope: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Train / val / test for a scope: the global split, filtered to one conflict
    type, or all three when ``scope == 'pooled'``."""
    if scope == "pooled":
        return train_df, val_df, test_df
    f = lambda d: d.filter(pl.col("conflict_type") == scope)  # noqa: E731
    return f(train_df), f(val_df), f(test_df)


def _both_classes(*frames: pl.DataFrame) -> bool:
    """True only if every split carries both classes (else metrics are undefined)."""
    for d in frames:
        p = int(d.filter(pl.col(LABEL_COLUMN) == 1).height)
        if d.height == 0 or p == 0 or p == d.height:
            return False
    return True


if RUN_MODEL_ZOO:
    records: list[dict] = []
    for scope in SCOPES:
        tr, va, te = _scope_frames(scope)
        if not _both_classes(tr, va, te):
            print(f"\n=== scope {scope}: skipped (a split lacks one class) ===")
            continue
        print(f"\n=== scope {scope} ({tr.height:,} train / {te.height:,} test) ===")
        for model_name in MODEL_NAMES:
            res = fit_score(model_name, FEATURE_COLS, tr, va, te)
            res["cv"] = time_block_cv(model_name, FEATURE_COLS, tr)
            res["scope"] = scope
            records.append(res)
            m = res["metrics"]
            print(f"  {res['model']:20s} val_MCC={res['val_mcc']:.4f}  "
                  f"test_MCC={m['mcc'][0]:.4f} [{m['mcc'][1]:.4f},{m['mcc'][2]:.4f}]  "
                  f"F1={m['f1'][0]:.4f}  PR-AUC={m['pr_auc'][0]:.4f}")
    with open(RECORDS_CACHE, "wb") as _fh:
        pickle.dump(records, _fh)
    print(f"\nCached {len(records)} records -> {RECORDS_CACHE.name}")
else:
    with open(RECORDS_CACHE, "rb") as _fh:
        records = pickle.load(_fh)
    print(f"Loaded {len(records)} records <- {RECORDS_CACHE.name}")


def _best_record(scope: str) -> dict:
    cand = [r for r in records if r["scope"] == scope and r["model"] != "most_frequent"]
    return max(cand, key=lambda r: r["val_mcc"])

# %% [markdown]
# ## 6. Reporting cell
#
# One row per `(model, conflict_scope, metric, value, ci_low, ci_high)` to
# `outputs/tables/rq2_results.csv`, where `conflict_scope` is `pooled` or one of
# the three conflict types. The walk-forward CV MCC travels as the `cv_mcc_mean`
# row (CI columns carry the fold min / max).

# %%
rows: list[dict] = []
for rec in records:
    scope = rec["scope"]
    for metric, (val, lo, hi) in rec["metrics"].items():
        rows.append({"model": rec["model"], "conflict_scope": scope, "metric": metric,
                     "value": round(val, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    cv = rec.get("cv") or {}
    if "cv_mcc_mean" in cv and cv["cv_mcc_mean"] == cv["cv_mcc_mean"]:  # not NaN
        rows.append({"model": rec["model"], "conflict_scope": scope, "metric": "cv_mcc_mean",
                     "value": round(cv["cv_mcc_mean"], 4), "ci_low": round(cv["cv_mcc_min"], 4),
                     "ci_high": round(cv["cv_mcc_max"], 4)})

rq2_df = pl.DataFrame(rows).sort(["conflict_scope", "model", "metric"])
rq2_df.write_csv(str(RQ2_RESULTS_CSV))
print(f"Wrote {rq2_df.height} rows -> {RQ2_RESULTS_CSV}")

# Best test MCC per scope (the headline view).
print("\nBest test MCC by scope (non-dummy):")
for scope in SCOPES:
    cand = [r for r in records if r["scope"] == scope and r["model"] != "most_frequent"]
    if not cand:
        continue
    b = max(cand, key=lambda r: r["metrics"]["mcc"][0])
    mcc = b["metrics"]["mcc"]
    print(f"  {scope:22s} {b['model']:18s} MCC {mcc[0]:.4f} [{mcc[1]:.4f}, {mcc[2]:.4f}]")

# %% [markdown]
# ## 7. Hyperparameter tuning (Optuna + time-blocked CV)
#
# A Bayesian search for the Random Forest and a compact grid for the Decision Tree
# and Linear SVM, each scored by mean time-blocked-CV MCC on the pooled set. Best
# params write to `configs/models/rq2_google_{model}.yaml`. Heavy step; set
# `RUN_TUNING = False` to reuse saved configs (Section 8 reads them when present).

# %%
import time

import optuna
import yaml

CONFIG_DIR = Path(REPO_DIR) / "configs" / "models"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR = OUTPUT_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 40
RUN_TUNING = True


def _cv_mcc_for(est_factory, cols: list[str], n_folds: int = 3) -> float:
    tr = train_df.sort("start_time")
    edges = np.linspace(0, tr.height, n_folds + 2, dtype=int)
    fold_mccs: list[float] = []
    for k in range(1, n_folds + 1):
        tr_part, te_part = tr[: edges[k]], tr[edges[k]: edges[k + 1]]
        if te_part.height == 0 or tr_part.filter(pl.col(LABEL_COLUMN) == 1).height < 5:
            continue
        x_tr, y_tr = make_xy(tr_part, cols)
        x_te, y_te = make_xy(te_part, cols)
        imp, x_tr_r, y_tr_r = _resample_train(x_tr, y_tr, True)
        est = est_factory()
        est.fit(x_tr_r, y_tr_r)
        score = _scores(est, imp.transform(x_te))
        thr = best_threshold(y_te, score)
        fold_mccs.append(float(matthews_corrcoef(y_te, (score >= thr).astype(np.int8))))
        del est, x_tr, x_te, x_tr_r
        gc.collect()
    return float(np.mean(fold_mccs)) if fold_mccs else float("nan")


def _rf_objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200),
        max_depth=trial.suggest_categorical("max_depth", [None, 16, 24, 32]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    )
    return _cv_mcc_for(lambda: RandomForestWrapper(random_state=RANDOM_SEED, **params), FEATURE_COLS)


def _write_config(name: str, params: dict, cv_mcc: float) -> None:
    payload = {"model": name, "scope": "pooled", "random_state": RANDOM_SEED,
               "cv_mcc_mean": float(cv_mcc), "params": params}
    with open(CONFIG_DIR / f"rq2_google_{name}.yaml", "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    print(f"  wrote rq2_google_{name}.yaml (CV MCC {cv_mcc:.4f})")


if RUN_TUNING:
    _t0 = time.perf_counter()
    _rf_study = optuna.create_study(
        study_name="rq2_google_rf", direction="maximize",
        storage=f"sqlite:///{OPTUNA_DIR / 'rq2_google_rf.db'}", load_if_exists=True)
    _done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in _rf_study.trials)
    if _done < N_TRIALS:
        _rf_study.optimize(_rf_objective, n_trials=N_TRIALS - _done, gc_after_trial=True)
    _write_config("random_forest", _rf_study.best_params, _rf_study.best_value)

    _dt_best, _dt_mcc = {}, -1.0
    for md in (8, 16, 24, None):
        for ml in (1, 5, 20):
            m = _cv_mcc_for(lambda md=md, ml=ml: DecisionTreeClassifier(
                class_weight="balanced", max_depth=md, min_samples_leaf=ml, random_state=RANDOM_SEED),
                FEATURE_COLS)
            if m == m and m > _dt_mcc:
                _dt_mcc, _dt_best = m, {"max_depth": md, "min_samples_leaf": ml}
    _write_config("decision_tree", _dt_best, _dt_mcc)

    _svm_best, _svm_mcc = {}, -1.0
    for c in (0.01, 0.1, 1.0, 10.0):
        m = _cv_mcc_for(lambda c=c: LinearSVC(class_weight="balanced", C=c, random_state=RANDOM_SEED),
                        FEATURE_COLS)
        if m == m and m > _svm_mcc:
            _svm_mcc, _svm_best = m, {"C": c}
    _write_config("linear_svm", _svm_best, _svm_mcc)
    print(f"Tuning complete in {(time.perf_counter() - _t0) / 60:.1f} min.")
else:
    print("RUN_TUNING is False; Section 8 will reuse existing configs if present.")

# %% [markdown]
# ## 8. Checkpoint the best model
#
# The best single classifier by validation MCC is refit on the training split
# (with its tuned config if present) and pickled with a metadata sidecar (library
# versions, a content hash of the training data, the threshold, and the
# validation / test metrics), mirroring the RQ1 checkpoint.

# %%
import hashlib
import json
from importlib import metadata as _ilmd

from utils.colab_setup import CHECKPOINT_DIR

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _data_sha(df: pl.DataFrame) -> str:
    return hashlib.sha256(np.sort(df.hash_rows().to_numpy()).tobytes()).hexdigest()


def _library_versions() -> dict:
    out = {}
    for pkg in ("scikit-learn", "imbalanced-learn", "tensorflow", "polars", "numpy", "optuna"):
        try:
            out[pkg] = _ilmd.version(pkg)
        except _ilmd.PackageNotFoundError:
            out[pkg] = None
    return out


def _tuned_estimator(name: str):
    """Best-config estimator for a tunable family, else None (Section 3 defaults)."""
    path = CONFIG_DIR / f"rq2_google_{name}.yaml"
    if not path.exists():
        return None
    params = yaml.safe_load(path.read_text()).get("params") or {}
    if name == "random_forest":
        return RandomForestWrapper(random_state=RANDOM_SEED, **params)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED, **params)
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, **params)
    return None


_best = _best_record("pooled")
_best_name = _best["model"]
_override = _tuned_estimator(_best_name)
_ck = fit_score(_best_name, FEATURE_COLS, train_df, val_df, test_df, est_override=_override)

_model_path = CHECKPOINT_DIR / "rq2_google_best.pkl"
# Refit cleanly for persistence (imputer + estimator), then pickle the pair.
_x_tr, _y_tr = make_xy(train_df, FEATURE_COLS)
_imp, _x_tr_r, _y_tr_r = _resample_train(_x_tr, _y_tr, _best_name != "most_frequent")
_est = _tuned_estimator(_best_name) or make_estimator(_best_name, _x_tr_r.shape[1])
_est.fit(_x_tr_r, _y_tr_r)
with open(_model_path, "wb") as fh:
    pickle.dump({"imputer": _imp, "estimator": _est, "threshold": _ck["threshold"],
                 "feature_columns": FEATURE_COLS}, fh, protocol=pickle.HIGHEST_PROTOCOL)

_sidecar = {
    "artifact": "rq2_google_best",
    "model": _best_name,
    "scope": "pooled",
    "tuned": _override is not None,
    "threshold": float(_ck["threshold"]),
    "feature_columns": FEATURE_COLS,
    "n_train_rows": int(train_df.height),
    "random_seed": RANDOM_SEED,
    "train_data_sha256": _data_sha(train_df.select(FEATURE_COLS + [LABEL_COLUMN])),
    "val_mcc": _ck["val_mcc"],
    "test_metrics": {k: [round(x, 4) for x in v] for k, v in _ck["metrics"].items()},
    "library_versions": _library_versions(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "note": "Imputer + classifier pair; SMOTE is training-only and not part of the saved object.",
}
with open(CHECKPOINT_DIR / "rq2_google_best.json", "w") as fh:
    json.dump(_sidecar, fh, indent=2)
print(f"Checkpointed {_best_name} -> {_model_path} (test MCC {_ck['metrics']['mcc'][0]:.4f})")
del _x_tr, _x_tr_r
gc.collect()

# %% [markdown]
# ## 9. Artifacts and the >0.80 hypothesis test (per scope)
#
# For each scope (the three conflict types and pooled): PR / ROC / calibration /
# confusion figures under `figures/rq2_google/{scope}/`, and the one-sample test of
# the best model's MCC against the 0.80 RQ2 target. The test uses the canonical
# stratified-bootstrap CI (`src.evaluation.metrics.mcc_with_ci`) and the CI-based
# one-sided rule (`src.evaluation.hypothesis.one_sample_threshold_test`): the target
# is met only when the CI lower bound clears 0.80. The per-type rows are the
# headline RQ2 result; the pooled row is reported as context (its MCC is partly the
# trivial separation between the differing per-type prevalences).
#
# > **Operationalization.** The >0.80 "resolution-success" target is tested on MCC,
# > the study's primary imbalance-robust metric (as RQ1 was). F1 and PR-AUC are
# > reported alongside for context. Confirm this is the intended operationalization
# > before the result is written into Chapter 4.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

FIG_DIR = FIGURES_DIR / "rq2_google"
RQ2_TARGET = 0.80
hyp_rows: list[dict] = []

for scope in SCOPES:
    cand = [r for r in records if r["scope"] == scope and r["model"] != "most_frequent"]
    if not cand:
        continue
    best = max(cand, key=lambda r: r["val_mcc"])
    _, _, te = _scope_frames(scope)
    y_test = te[LABEL_COLUMN].to_numpy().astype(np.int8)
    sdir = FIG_DIR / scope
    sdir.mkdir(parents=True, exist_ok=True)

    # PR / ROC across the non-dummy learners in this scope.
    for kind, curve_fn, fname, xlab, ylab in (
        ("PR", precision_recall_curve, "pr_curves.png", "Recall", "Precision"),
        ("ROC", roc_curve, "roc_curves.png", "False positive rate", "True positive rate"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for r in sorted(cand, key=lambda r: r["val_mcc"], reverse=True):
            if kind == "PR":
                precision, recall, _ = curve_fn(y_test, r["test_score"])
                ax.plot(recall, precision, label=f"{r['model']} (AP={r['metrics']['pr_auc'][0]:.3f})")
            else:
                fpr, tpr, _ = curve_fn(y_test, r["test_score"])
                ax.plot(fpr, tpr, label=r["model"])
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(f"{kind} - {scope}")
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout(); fig.savefig(sdir / fname, dpi=150); plt.close(fig)

    # Reliability + confusion for the scope's best model.
    _cal = calibration_table(y_test, best["test_score"], n_bins=10).filter(pl.col("n") > 0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(_cal["mean_predicted"], _cal["observed_rate"], "o-", label=best["model"])
    ax.set_xlabel("Mean predicted score"); ax.set_ylabel("Observed clean-resolution rate")
    ax.set_title(f"Reliability - {scope} ({best['model']})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(sdir / "calibration.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(y_test, best["y_pred"])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Purples")
    ax.set_title(f"{scope} - {best['model']} (MCC {best['metrics']['mcc'][0]:.3f})", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred escalate", "pred clean"]); ax.set_yticklabels(["true escalate", "true clean"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout(); fig.savefig(sdir / "confusion_matrix.png", dpi=150); plt.close(fig)

    # One-sample MCC test vs 0.80 for this scope's best model.
    mcc_point, ci_lo, ci_hi = mcc_with_ci(y_test, best["y_pred"], n_boot=1000, seed=RANDOM_SEED)
    test = one_sample_threshold_test(mcc_point, ci_lo, ci_hi, RQ2_TARGET, metric_name="MCC")
    hyp_rows.append({
        "scope": scope, "model": best["model"], "metric": "mcc",
        "value": round(mcc_point, 4), "ci_low": round(ci_lo, 4), "ci_high": round(ci_hi, 4),
        "threshold_target": RQ2_TARGET, "reject_h0": test["reject"],
        "decision": test["decision"], "margin": round(test["margin"], 4),
    })
    print(f"  {scope:22s} {best['model']:18s} MCC {mcc_point:.4f} "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]  {test['decision']} vs {RQ2_TARGET}")

hyp_df = pl.DataFrame(hyp_rows)
hyp_df.write_csv(str(RQ2_HYPOTHESIS_CSV))
print(f"\nWrote {RQ2_HYPOTHESIS_CSV} ({hyp_df.height} scope rows); figures under {FIG_DIR}")
