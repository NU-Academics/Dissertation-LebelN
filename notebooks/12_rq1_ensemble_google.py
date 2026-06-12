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
# # 12. RQ1 — Ensemble Failure Prediction (Google Cluster Traces, episode grain)
#
# **Research question.** RQ1: what ensemble learning algorithms can achieve
# >0.90 predictive performance (MCC / F1 / PR-AUC) in predicting system failures
# from multi-modal telemetry? MCC is the primary hypothesis-testing metric under
# class imbalance (Chicco & Jurman, 2023); PR-AUC is the complementary
# imbalance-robust diagnostic (Saito & Rehmsmeier, 2015). Reporting follows
# TRIPOD+AI (Collins et al., 2024).
#
# **Grain.** One row per *scheduled attempt* (episode), the leakage-free grain
# established in notebook 10 Section 11-12 and re-validated in notebook 11
# Section 3.8. History is strictly prior, so submission-time counts cannot peek
# at the terminal label.
#
# **Prediction points.** The RQ1 curve is reported at three points whose feature
# masks were validated in notebook 11 Section 3.8 (all leakage-free at the
# episode grain): at-submission, at-scheduling, and early-runtime. The >0.90 MCC
# target is tested at early-runtime, where the Tier 2 slope/ramp/counter signal
# becomes available (V09, V33).
#
# **Imbalance handling.** Per fold: class weights set to the inverse class prior,
# plus SMOTE applied only inside the training portion of each fold (never the
# validation or test portion). The recurring-instance negative tail is
# de-concentrated up front with a per-instance negative cap on the training split
# only (default 5, applied in the Section 1 extraction); positives are never
# capped, and validation / test keep the natural distribution.
#
# **Validation protocol.** Walk-forward (expanding-window) cross-validation
# inside the training period (Cerqueira et al., 2020; Bergmeir & Benitez, 2012),
# a held-out validation block for model selection / threshold tuning, and a final
# held-out test block. Uncertainty is quantified with percentile bootstrap CIs.
#
# **Output.** `outputs/tables/rq1_google.csv` with one row per
# `(prediction_point, model, metric, value, ci_low, ci_high)`.
#
# > **Starting structure.** This is the first runnable skeleton; thresholds,
# > hyperparameters, the SMOTE/NaN handling at early-runtime, and the model set
# > are expected to be revised as runs come back.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars lightgbm xgboost scikit-learn imbalanced-learn pandas pyarrow google-cloud-bigquery-storage optuna pyyaml matplotlib

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

# %%
from utils.colab_setup import setup_drive, OUTPUT_DIR

setup_drive()

# %%
import gc
import warnings

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features.episodes import DEFAULT_NEG_CAP

RANDOM_SEED = 42  # P14 random-seed convention.
rng = np.random.default_rng(RANDOM_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# Persistent Drive output store (mirrors notebooks 10 / 11), not the ephemeral
# cloned repo, which is recreated each runtime and lost on termination.
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'
for directory in (TABLES_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RQ1_TABLE_CSV = TABLES_DIR / 'rq1_google.csv'
LABEL_COL = "failure_label"

# %% [markdown]
# ## 1. Load the episode-grain feature matrix (instance-grouped splits)
#
# Source is the durable BigQuery `episode_features` table (notebook 10
# Section 12): one row per scheduled attempt, Tier 1 + Tier 2 + Tier 3, with
# strictly-prior history (no instance-grain leakage).
#
# The train / val / test split is done in-warehouse, mirroring notebook 11
# Section 3.8's validated extraction, so this notebook's numbers land on the
# validated RQ1 curve (early-runtime ~0.89, notebook 11 Section 3.9) rather than
# ~0.04 above it. Each split's SQL:
# 1. **Instance-keyed group split** (V32): instances are bucketed by a
#    FARM_FINGERPRINT of the instance key into train / val / test, so no instance
#    straddles the boundary. This is what stops the recurring-instance history
#    bleed a temporal split admits (an instance's strictly-prior history leaking
#    from its early attempts into its later ones across the split).
# 2. **Train-only negative cap:** a ROW_NUMBER over (instance, failure_label)
#    keeps every positive plus the first `NEG_CAP` real negatives per instance, on
#    the train split only; validation and test keep the natural distribution.
# 3. **`schedule_time` / `sched_day`:** joined from `episode_schedule_intervals`
#    (Section 12.1) for the walk-forward CV folds.
# 4. **Per-split subsample:** an episode-key hash thins each split to a permille,
#    so only a few hundred thousand rows reach Colab; the early-runtime learning
#    curve (notebook 11 Section 3.9) saturates well below that.

# %%
from google.colab import auth

from utils.bq_client import get_client, table_ref

auth.authenticate_user()  # idempotent; ensures ADC for the BigQuery extract.
_bq = get_client()

EPISODE_MATRIX_TABLE = 'episode_features'
EPISODE_INTERVALS_TABLE = 'episode_schedule_intervals'
NEG_CAP = DEFAULT_NEG_CAP            # per-instance negative cap (V32; default 5).
DAY_US = 86_400_000_000             # microseconds per trace day.
N_TRACE_DAYS = 31                   # documented span of cell-a (May 2019).

# Instance-keyed group split (V32; the notebook 11 Section 3.8 method) into
# ~70/15/15 by a FARM_FINGERPRINT of the instance key, so an instance's episodes
# never straddle the split. This is what prevents the recurring-instance bleed a
# temporal split admits: the strictly-prior history of an instance's early
# attempts would otherwise leak into its later attempts across the boundary,
# inflating the at-submission MCC. Buckets 0..N_BUCKETS-1: train below
# TRAIN_BUCKET_MAX, then val, then test. Matching the method behind the validated
# RQ1 curve (early-runtime ~0.89, notebook 11 Section 3.9) keeps this notebook's
# numbers on that curve instead of ~0.04 above it.
N_BUCKETS = 20
TRAIN_BUCKET_MAX = 14               # buckets [0, 14)  -> train (~70% of instances)
VAL_BUCKET_MAX = 17                 # buckets [14, 17) -> val (~15%); [17, 20) -> test

# Per-split subsample (permille of the episode-key hash) to bound Colab memory.
# The early-runtime learning curve (notebook 11 Section 3.9) saturates by ~110K
# train rows, so a few hundred thousand is ample; raise for tighter CIs.
TRAIN_PERMILLE = 15                 # ~575K capped train rows. (Permille 60 pulled
                                    # 2.29M, which is overkill and slows the zoo:
                                    # the curve saturates by ~110K train rows.)
EVAL_PERMILLE = 20                  # ~270K each for the uncapped val / test sets

# Walk-forward (expanding-window) CV folds by schedule day inside the training
# split (Cerqueira et al., 2020). The group-split train spans all days, so these
# probe temporal robustness over the train instances; (train_day_max, lo, hi).
WALK_FORWARD_FOLDS = [
    (10, 11, 13),
    (13, 14, 16),
    (16, 17, 19),
    (19, 20, 22),
]


def _episode_split_sql(split: str) -> str:
    """SQL pulling one instance-grouped split (``train`` / ``val`` / ``test``) of
    the episode matrix, mirroring notebook 11 Section 3.8's validated extraction.

    - **Group split:** instances are bucketed 0..N_BUCKETS-1 by a FARM_FINGERPRINT
      of the instance key, so no instance straddles train / val / test (V32).
    - **Train cap:** a ROW_NUMBER over (instance, failure_label) keeps every
      positive plus the first ``NEG_CAP`` real negatives per instance. Train only.
    - **Eval untouched:** val / test keep the natural class distribution.
    - **Subsample:** a hash of the episode key thins each split to a permille.
    - **schedule_time / sched_day:** joined from the intervals table for the
      walk-forward CV folds.
    """
    key = "CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING))"
    epkey = f"CONCAT({key},'_',CAST(sched_seq AS STRING))"
    if split == "train":
        grp_pred = f"_grp < {TRAIN_BUCKET_MAX}"
        permille = TRAIN_PERMILLE
        rn_col = (",\n        ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index, "
                  "failure_label ORDER BY _ephash) AS _rn")
        cap_clause = f"WHERE {LABEL_COL} = 1 OR _rn <= {NEG_CAP}"
        drop_cols = "_grp, _ephash, _rn"
    elif split == "val":
        grp_pred = f"_grp >= {TRAIN_BUCKET_MAX} AND _grp < {VAL_BUCKET_MAX}"
        permille, rn_col, cap_clause, drop_cols = EVAL_PERMILLE, "", "", "_grp, _ephash"
    else:  # test
        grp_pred = f"_grp >= {VAL_BUCKET_MAX}"
        permille, rn_col, cap_clause, drop_cols = EVAL_PERMILLE, "", "", "_grp, _ephash"
    return f"""
WITH anchored AS (
    SELECT MIN(schedule_time) AS t0 FROM {table_ref(EPISODE_INTERVALS_TABLE)}
),
split AS (
    SELECT
        b.*,
        i.schedule_time,
        1 + CAST(FLOOR(SAFE_DIVIDE(i.schedule_time - a.t0, {DAY_US})) AS INT64) AS sched_day,
        MOD(ABS(FARM_FINGERPRINT({key})), {N_BUCKETS}) AS _grp,
        ABS(FARM_FINGERPRINT({epkey})) AS _ephash
    FROM {table_ref(EPISODE_MATRIX_TABLE)} b
    JOIN {table_ref(EPISODE_INTERVALS_TABLE)} i
      USING (collection_id, instance_index, sched_seq)
    CROSS JOIN anchored a
),
sided AS (
    SELECT *{rn_col}
    FROM split
    WHERE {grp_pred}
),
capped AS (
    SELECT * FROM sided
    {cap_clause}
)
SELECT * EXCEPT ({drop_cols})
FROM capped
WHERE MOD(_ephash, 1000) < {permille}
"""


def _pull_split(split: str) -> pl.DataFrame:
    arrow = _bq.query(_episode_split_sql(split)).to_arrow(create_bqstorage_client=True)
    df = pl.from_arrow(arrow)
    del arrow
    gc.collect()
    return df


CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
USE_SPLIT_CACHE = True   # reload splits from the Drive Parquet cache if present; set False to force a fresh BigQuery pull


def _load_split(split: str) -> pl.DataFrame:
    """Pull a split from BigQuery, caching the result to Parquet on Drive so a
    later Colab session reloads it and skips the (slow) BigQuery pull.

    Cache invalidation: the cached Parquet reflects the extraction at write time.
    After the upstream ``episode_features`` table is rebuilt, or any extraction
    constant changes (split buckets, permille, the per-instance negative cap),
    set ``USE_SPLIT_CACHE = False`` once to refresh, or delete the files in
    ``CACHE_DIR``."""
    cache_path = CACHE_DIR / f"rq1_google_split_{split}.parquet"
    if USE_SPLIT_CACHE and cache_path.exists():
        df = pl.read_parquet(cache_path)
        print(f"  {split:5s} loaded from cache  ({df.height:>9,} rows) <- {cache_path.name}")
        return df
    df = _pull_split(split)
    df.write_parquet(cache_path)
    print(f"  {split:5s} pulled from BigQuery ({df.height:>9,} rows), cached -> {cache_path.name}")
    return df


print("Loading instance-grouped train / val / test splits (Drive cache, else BigQuery Arrow stream) ...")
train_df = _load_split("train")
val_df = _load_split("val")
test_df = _load_split("test")
for _name, _df in (("train (capped)", train_df), ("val (natural)", val_df),
                   ("test (natural)", test_df)):
    _p = int(_df.filter(pl.col(LABEL_COL) == 1).height)
    print(f"  {_name:16s} {_df.height:>9,} rows | positive {_p:>8,} "
          f"({_p / max(_df.height, 1):.3f}) | neg:pos {(_df.height - _p) / max(_p, 1):.2f}:1")

# %% [markdown]
# ## 2. Prediction-point feature masks
#
# Derived by name from the loaded columns (robust to one-hot suffixes), matching
# the groups validated in notebook 11 Section 3.8. All three points are
# leakage-free at the episode grain.
#
# - **At-submission:** requests, request ratio, priority-tier one-hots,
#   scheduling class, the submit-time (schedule-wall-clock) temporal features,
#   and the strictly-prior history.
# - **At-scheduling:** adds queue time and the episode's assigned-machine
#   platform one-hots.
# - **Early-runtime:** adds the Tier 2 slope / ramp / counter features and the
#   hardware-counter availability flag. Tier 2 availability is ~19.4% and is
#   label-correlated (V09), so missingness is itself signal.

# %%
_cols = set(train_df.columns)

EP_SUBMIT_TEMPORAL = sorted(c for c in _cols if c.startswith("submit_"))
EP_PRIORITY_ONEHOT = sorted(c for c in _cols if c.startswith("priority_tier_"))
EP_PLATFORM_ONEHOT = sorted(c for c in _cols if c.startswith("platform_"))
EP_REQUEST = [c for c in ("cpu_request", "memory_request", "request_ratio") if c in _cols]
EP_HISTORY = [c for c in ("prior_fail_count", "has_prior_fail", "resubmission_count",
                          "prior_evict_count", "first_resubmission") if c in _cols]
EP_TIER2 = [c for c in (
    "cpu_slope_5s", "cpu_slope_15s", "cpu_slope_30s",
    "memory_slope_5s", "memory_slope_15s", "memory_slope_30s",
    "initial_cpu_ramp", "initial_memory_ramp", "first_interval_util_ratio",
    "cpi_value", "mapi_value",
) if c in _cols]

at_submission_cols = (
    EP_REQUEST
    + EP_PRIORITY_ONEHOT
    + (["scheduling_class"] if "scheduling_class" in _cols else [])
    + EP_SUBMIT_TEMPORAL
    + EP_HISTORY
)
at_scheduling_cols = (
    at_submission_cols
    + (["queue_time"] if "queue_time" in _cols else [])
    + EP_PLATFORM_ONEHOT
)
early_runtime_cols = (
    at_scheduling_cols
    + EP_TIER2
    + (["has_hardware_counters"] if "has_hardware_counters" in _cols else [])
)

PREDICTION_POINTS: dict[str, list[str]] = {
    "at_submission": at_submission_cols,
    "at_scheduling": at_scheduling_cols,
    "early_runtime": early_runtime_cols,
}
for _pp, _cols_pp in PREDICTION_POINTS.items():
    print(f"  {_pp:14s} {len(_cols_pp):3d} features")

# Tier 2 features carry NaN where no in-band usage exists (rapid-onset crashes).
# Tree learners that consume NaN natively (LightGBM, XGBoost) can read the
# missingness as signal; the others (and SMOTE's distance computation) need a
# dense matrix, so the pipeline below imputes with an added missing-indicator,
# which preserves the label-correlated availability signal (V09). Revisit at
# early-runtime: a NaN-preserving LightGBM/XGBoost variant is the natural ablation.
NATIVE_NAN_MODELS = {"xgboost", "lightgbm"}

# %% [markdown]
# ## 3. Split verification
#
# The split itself was done in-warehouse (Section 1): instance-keyed group
# buckets, the train-only negative cap, and per-split subsampling, mirroring
# notebook 11 Section 3.8 so this notebook's numbers sit on the validated RQ1
# curve (early-runtime ~0.89, notebook 11 Section 3.9) rather than the inflated
# values a temporal split produced (recurring-instance bleed through the
# strictly-prior history). Here we only verify the two properties that matter: no
# instance straddles the splits, and `sched_day` is present for the walk-forward
# folds. Train is capped and val/test are natural, as printed in Section 1.

# %%
def _instances(df: pl.DataFrame) -> set:
    """Set of (collection_id, instance_index) keys present in a split."""
    return set(zip(df["collection_id"].to_list(), df["instance_index"].to_list()))


_tr_inst, _va_inst, _te_inst = _instances(train_df), _instances(val_df), _instances(test_df)
_overlap = (_tr_inst & _va_inst) | (_tr_inst & _te_inst) | (_va_inst & _te_inst)
assert not _overlap, f"{len(_overlap)} instances straddle the group split; bucketing is wrong."
print(f"Group split clean: no instance straddles train / val / test "
      f"({len(_tr_inst):,} / {len(_va_inst):,} / {len(_te_inst):,} instances).")

assert "sched_day" in train_df.columns, "sched_day missing; the intervals join did not attach it."
print(f"Train spans schedule days {int(train_df['sched_day'].min())}-"
      f"{int(train_df['sched_day'].max())}; the walk-forward folds probe temporal "
      f"robustness within the group-split train.")

# %% [markdown]
# *(The leak-localization diagnostics that lived here, former Sections 3.1-3.4,
# did their job and were removed once the at-submission leak was traced to
# terminal-derived `priority` / `scheduling_class` (fixed in notebooks 8 and 10,
# decision V35) and the remaining inflation to the temporal split's strictly-prior
# history bleed (resolved by the instance-keyed group split adopted in Section 1).
# The evidence is preserved in `outputs/tables/rq1_google_leakage_diagnostic.csv`
# and `rq1_google_single_feature_mcc.csv`, and in the V35 row of
# `eda_decisions.csv`. The train-only negative cap they referenced is now applied
# in-warehouse in the Section 1 extraction.)*

# %% [markdown]
# ## 4. Modeling helpers
#
# A single matrix builder, a per-model factory (class weights = inverse class
# prior + SMOTE inside the training portion only), an evaluation block (MCC, F1,
# PR-AUC), a bootstrap-CI helper, and a validation-tuned decision threshold.

# %%
def make_xy(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X float32, y int8) for the given feature columns. Tier 2 nulls
    pass through as NaN; the pipeline handles imputation where required."""
    x = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = df.select(LABEL_COL).to_numpy().ravel().astype(np.int8)
    return x, y


def build_estimator(name: str, y_train: np.ndarray):
    """Construct the named estimator wrapped in an SMOTE pipeline.

    Class weights use the inverse class prior (`class_weight='balanced'` or, for
    XGBoost, `scale_pos_weight = n_neg / n_pos`). SMOTE is the first fitted step
    so it only ever sees the training portion of a fold. A `SimpleImputer` with
    an added missing-indicator precedes SMOTE for dense-input learners; it is a
    documented revision point for the NaN-native learners at early-runtime.
    """
    n_pos_ = int(y_train.sum())
    n_neg_ = int(y_train.size - n_pos_)
    spw = n_neg_ / max(n_pos_, 1)

    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    elif name == "decision_tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    elif name == "most_frequent":
        clf = DummyClassifier(strategy="most_frequent")
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        )
    elif name == "balanced_random_forest":
        clf = BalancedRandomForestClassifier(
            n_estimators=300, sampling_strategy="all", replacement=True,
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    elif name == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    elif name == "lightgbm":
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED, verbosity=-1,
        )
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown model: {name}")

    # The dummy baseline ignores X, so it needs no imputation / resampling.
    if name == "most_frequent":
        return ImbPipeline([("model", clf)])
    return ImbPipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", clf),
    ])


MODEL_NAMES = [
    "logistic_regression", "decision_tree", "most_frequent",
    "random_forest", "balanced_random_forest",
    "xgboost", "lightgbm", "gradient_boosting",
]


def predict_proba(estimator, x: np.ndarray) -> np.ndarray:
    """Positive-class probability, falling back to the label for degenerate
    estimators (the most-frequent dummy exposes a constant proba)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    return estimator.predict(x).astype(np.float64)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Decision threshold maximizing MCC on the validation probabilities."""
    grid = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_t, best_m = 0.5, -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, (y_prob >= t).astype(np.int8))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """RQ1 metric block: MCC (primary), F1, PR-AUC."""
    y_pred = (y_prob >= threshold).astype(np.int8)
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
    n_boot: int = 1000, alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for each metric over paired resamples of the
    test arrays (alpha = 0.05)."""
    local = np.random.default_rng(RANDOM_SEED)
    n = y_true.size
    acc = {m: np.empty(n_boot) for m in ("mcc", "f1", "pr_auc")}
    for b in range(n_boot):
        idx = local.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        pred = (yp >= threshold).astype(np.int8)
        acc["mcc"][b] = matthews_corrcoef(yt, pred)
        acc["f1"][b] = f1_score(yt, pred, zero_division=0)
        acc["pr_auc"][b] = average_precision_score(yt, yp) if yt.sum() > 0 else 0.0
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for m, v in acc.items()}


# %% [markdown]
# ## 5. Walk-forward cross-validation
#
# Expanding-window folds inside the training period. SMOTE runs inside each
# fold's training portion only (it is the first fitted pipeline step), and the
# fold's test portion is never resampled. Returns the mean / min / max fold MCC
# per model, used both as a stability diagnostic and to break ties in the
# top-three selection for the voting stack.

# %%
def walk_forward_cv(model_name: str, cols: list[str]) -> dict[str, float]:
    """Mean / min / max validation-fold MCC for one model over the four
    expanding-window folds."""
    fold_mccs: list[float] = []
    for tr_max, te_lo, te_hi in WALK_FORWARD_FOLDS:
        tr = train_df.filter(pl.col("sched_day") <= tr_max)
        te = train_df.filter(
            (pl.col("sched_day") >= te_lo) & (pl.col("sched_day") <= te_hi)
        )
        if te.height == 0 or tr.filter(pl.col(LABEL_COL) == 1).height == 0:
            continue
        x_tr, y_tr = make_xy(tr, cols)
        x_te, y_te = make_xy(te, cols)
        est = build_estimator(model_name, y_tr)
        est.fit(x_tr, y_tr)
        prob = predict_proba(est, x_te)
        thr = best_threshold(y_te, prob)
        fold_mccs.append(float(matthews_corrcoef(y_te, (prob >= thr).astype(np.int8))))
        del est, x_tr, x_te
        gc.collect()
    if not fold_mccs:
        return {"cv_mcc_mean": float("nan"), "cv_mcc_min": float("nan"), "cv_mcc_max": float("nan")}
    arr = np.array(fold_mccs)
    return {"cv_mcc_mean": float(arr.mean()), "cv_mcc_min": float(arr.min()), "cv_mcc_max": float(arr.max())}


# %% [markdown]
# ## 6. Per-prediction-point run
#
# For each prediction point and each model: run walk-forward CV, fit on the full
# training block, tune the threshold on the validation block, then score the
# held-out test block with bootstrap CIs. After the individual models, build a
# soft-voting stack of the top three models by validation MCC and score it the
# same way.

# %%
def fit_select_score(model_name: str, cols: list[str]) -> dict:
    """Fit on train, tune threshold on val, score test. Returns the assembled
    record including CV summary, the validation MCC (for stack selection), the
    tuned threshold, the test metrics, and their bootstrap CIs."""
    x_tr, y_tr = make_xy(train_df, cols)
    x_val, y_val = make_xy(val_df, cols)
    x_te, y_te = make_xy(test_df, cols)

    cv = walk_forward_cv(model_name, cols)
    est = build_estimator(model_name, y_tr)
    est.fit(x_tr, y_tr)

    val_prob = predict_proba(est, x_val)
    thr = best_threshold(y_val, val_prob)
    val_mcc = float(matthews_corrcoef(y_val, (val_prob >= thr).astype(np.int8)))

    test_prob = predict_proba(est, x_te)
    metrics = compute_metrics(y_te, test_prob, thr)
    cis = bootstrap_ci(y_te, test_prob, thr)
    del est, x_tr, x_val, x_te
    gc.collect()
    return {"model": model_name, "cv": cv, "val_mcc": val_mcc,
            "threshold": thr, "metrics": metrics, "cis": cis, "test_prob": test_prob}


def soft_voting_stack(top_models: list[str], cols: list[str]) -> dict:
    """Equal-weight soft-voting ensemble of the top-three models, averaged at the
    probability level (built explicitly rather than via VotingClassifier so each
    member keeps its own SMOTE pipeline)."""
    x_tr, y_tr = make_xy(train_df, cols)
    x_val, y_val = make_xy(val_df, cols)
    x_te, y_te = make_xy(test_df, cols)

    val_probs, test_probs = [], []
    for name in top_models:
        est = build_estimator(name, y_tr)
        est.fit(x_tr, y_tr)
        val_probs.append(predict_proba(est, x_val))
        test_probs.append(predict_proba(est, x_te))
        del est
        gc.collect()
    val_prob = np.mean(val_probs, axis=0)
    test_prob = np.mean(test_probs, axis=0)
    thr = best_threshold(y_val, val_prob)
    val_mcc = float(matthews_corrcoef(y_val, (val_prob >= thr).astype(np.int8)))
    metrics = compute_metrics(y_te, test_prob, thr)
    cis = bootstrap_ci(y_te, test_prob, thr)
    del x_tr, x_val, x_te
    gc.collect()
    return {"model": f"soft_voting_top3[{'+'.join(top_models)}]", "cv": {},
            "val_mcc": val_mcc, "threshold": thr, "metrics": metrics, "cis": cis,
            "test_prob": test_prob}


# %% [markdown]
# **Session economy.** The zoo fit below is the multi-hour step. On completion it
# pickles the assembled `records` to Drive; set `RUN_MODEL_ZOO = False` on a later
# session to reload them and run Sections 9-10 (which consume `records`) without
# refitting. Re-run the zoo (`RUN_MODEL_ZOO = True`) after any change to the
# splits, feature masks, or estimators. Section 1 must still run either way, since
# the decomposition cells and the Section 9 checkpoint refit use the loaded splits.

# %%
import pickle

RUN_MODEL_ZOO = True   # set False to skip the multi-hour zoo fit and reload records from the Drive cache
RECORDS_CACHE = CACHE_DIR / "rq1_google_records.pkl"

if RUN_MODEL_ZOO:
    records: list[dict] = []
    for pp_name, pp_cols in PREDICTION_POINTS.items():
        print(f"\n=== Prediction point: {pp_name} ({len(pp_cols)} features) ===")
        pp_results = []
        for model_name in MODEL_NAMES:
            res = fit_select_score(model_name, pp_cols)
            pp_results.append(res)
            print(f"  {res['model']:24s} val_MCC={res['val_mcc']:.4f}  "
                  f"test_MCC={res['metrics']['mcc']:.4f}  "
                  f"F1={res['metrics']['f1']:.4f}  PR-AUC={res['metrics']['pr_auc']:.4f}")

        # Soft-voting stack of the top three by validation MCC (exclude the dummy).
        ranked = sorted(
            (r for r in pp_results if r["model"] != "most_frequent"),
            key=lambda r: r["val_mcc"], reverse=True,
        )
        top3 = [r["model"] for r in ranked[:3]]
        stack = soft_voting_stack(top3, pp_cols)
        pp_results.append(stack)
        print(f"  {stack['model']:24s} val_MCC={stack['val_mcc']:.4f}  "
              f"test_MCC={stack['metrics']['mcc']:.4f}  "
              f"F1={stack['metrics']['f1']:.4f}  PR-AUC={stack['metrics']['pr_auc']:.4f}")

        for r in pp_results:
            records.append({"prediction_point": pp_name, **r})

    with open(RECORDS_CACHE, "wb") as _fh:
        pickle.dump(records, _fh)
    print(f"\nModel-zoo records cached ({len(records)} rows) -> {RECORDS_CACHE.name}")
else:
    with open(RECORDS_CACHE, "rb") as _fh:
        records = pickle.load(_fh)
    print(f"Loaded {len(records)} model-zoo records from cache -> {RECORDS_CACHE.name} "
          "(RUN_MODEL_ZOO=False; Sections 9-10 will use these)")

# %% [markdown]
# ### 6.5 Submission history decomposition
#
# At-submission came in around 0.89-0.90 for the tree ensembles, far above the
# validated baseline curve (submission 0.58-0.69) and flattening the "submission
# is hard, you need runtime" story. The guide treats a high at-submission MCC as a
# leakage red flag, so before trusting it we isolate where the signal lives. This
# cell fits the SAME tuned pipeline (LightGBM and Random Forest) at submission
# WITHOUT the strictly-prior history (`submission_conservative`, 15 features)
# versus WITH it (`at_submission`, 20 features), on the same group split.
#
# Read: if conservative drops to ~0.6 and +history jumps to ~0.89, the
# strictly-prior resubmission history (V10) is the legitimate driver and the RQ1
# narrative needs updating (submission is more predictive than the baseline
# implied). If conservative is already ~0.89, a static submission feature is still
# carrying it and there is another leak to hunt despite V35.

# %%
submission_conservative_cols = [c for c in at_submission_cols if c not in set(EP_HISTORY)]
_decomp_points = {
    "submission_conservative": submission_conservative_cols,
    "submission_plus_history": at_submission_cols,
}
print(f"Submission decomposition (tuned models, test MCC; "
      f"conservative={len(submission_conservative_cols)} feat vs "
      f"+history={len(at_submission_cols)} feat):")
_decomp_rows = []
for _mname in ("lightgbm", "random_forest"):
    for _pt, _cols2 in _decomp_points.items():
        _xtr, _ytr = make_xy(train_df, _cols2)
        _xval, _yval = make_xy(val_df, _cols2)
        _xte, _yte = make_xy(test_df, _cols2)
        _est = build_estimator(_mname, _ytr)
        _est.fit(_xtr, _ytr)
        _thr = best_threshold(_yval, predict_proba(_est, _xval))
        _mcc = float(matthews_corrcoef(_yte, (predict_proba(_est, _xte) >= _thr).astype(np.int8)))
        _decomp_rows.append({"model": _mname, "features": _pt,
                             "n_features": len(_cols2), "test_mcc": _mcc})
        print(f"  {_mname:16s} {_pt:24s} n={len(_cols2):2d}  test_MCC={_mcc:.4f}")
        del _est, _xtr, _xval, _xte
        gc.collect()
pl.DataFrame(_decomp_rows).write_csv(str(TABLES_DIR / "rq1_google_submission_decomposition.csv"))
print("\nIf conservative is far below +history, the strictly-prior resubmission "
      "history (V10) drives at-submission (legitimate, narrative-changing); if "
      "conservative is already high, a static feature is still carrying it.")

# %% [markdown]
# ### 6.6 First-episode confirmation (concentration vs genuine static signal)
#
# The static-only submission MCC came in around 0.76, above the baseline. Two
# explanations: (a) an episode-grain concentration effect, where recurring failers
# contribute many static-identical failure episodes so static attributes
# effectively flag failer-instances; or (b) static submit-time features genuinely
# separate failure even on novel instances. This cell restricts to each instance's
# FIRST attempt (`resubmission_count == 0`), where no prior history exists and each
# instance contributes exactly one row, then fits the same tuned models on the
# static features.
#
# Read: if first-episode static MCC is well below ~0.76, the all-episode number is
# the (legitimate) recurring-failer concentration and novel instances are genuinely
# harder; if it is still ~0.76, static features separate even first attempts and
# `scheduling_class` / priority warrant a closer confound check.

# %%
def _first_ep(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("resubmission_count") == 0)


_fe_tr, _fe_val, _fe_te = _first_ep(train_df), _first_ep(val_df), _first_ep(test_df)
for _n, _d in (("train", _fe_tr), ("val", _fe_val), ("test", _fe_te)):
    _p = int(_d.filter(pl.col(LABEL_COL) == 1).height)
    print(f"  first-episode {_n}: {_d.height:>9,} rows | positive {_p:>8,} "
          f"({_p / max(_d.height, 1):.3f})")

print("\nStatic-feature MCC on first episodes only (no history available), test set:")
_fe_rows = []
for _mname in ("lightgbm", "random_forest"):
    _xtr, _ytr = make_xy(_fe_tr, submission_conservative_cols)
    _xval, _yval = make_xy(_fe_val, submission_conservative_cols)
    _xte, _yte = make_xy(_fe_te, submission_conservative_cols)
    _est = build_estimator(_mname, _ytr)
    _est.fit(_xtr, _ytr)
    _thr = best_threshold(_yval, predict_proba(_est, _xval))
    _mcc = float(matthews_corrcoef(_yte, (predict_proba(_est, _xte) >= _thr).astype(np.int8)))
    _fe_rows.append({"model": _mname, "subset": "first_episode_static", "test_mcc": _mcc})
    print(f"  {_mname:16s} first-episode static  test_MCC={_mcc:.4f}  "
          f"(all-episode static was ~0.76)")
    del _est, _xtr, _xval, _xte
    gc.collect()
pl.DataFrame(_fe_rows).write_csv(str(TABLES_DIR / "rq1_google_first_episode_static.csv"))
print("\nFar below ~0.76 -> the all-episode static signal is recurring-failer "
      "concentration (legitimate; novel instances are harder). Still ~0.76 -> "
      "static features separate even novel first attempts (confound check warranted).")

# %% [markdown]
# ## 7. Reporting cell
#
# Flatten to one row per `(prediction_point, model, metric, value, ci_low,
# ci_high)` and write `outputs/tables/rq1_google.csv`. The walk-forward CV MCC is
# emitted as the `cv_mcc_mean` metric (CI columns carry the fold min/max) so the
# stability diagnostic travels with the headline test metrics. The >0.90 MCC
# target is flagged at early-runtime.

# %%
rows: list[dict] = []
for rec in records:
    pp = rec["prediction_point"]
    model = rec["model"]
    for metric, value in rec["metrics"].items():
        lo, hi = rec["cis"][metric]
        rows.append({"prediction_point": pp, "model": model, "metric": metric,
                     "value": value, "ci_low": lo, "ci_high": hi})
    cv = rec.get("cv") or {}
    if "cv_mcc_mean" in cv:
        rows.append({"prediction_point": pp, "model": model, "metric": "cv_mcc_mean",
                     "value": cv["cv_mcc_mean"], "ci_low": cv["cv_mcc_min"], "ci_high": cv["cv_mcc_max"]})

rq1_df = pl.DataFrame(rows).sort(["prediction_point", "model", "metric"])
rq1_df.write_csv(str(RQ1_TABLE_CSV))
print(f"Wrote {rq1_df.height} rows -> {RQ1_TABLE_CSV}")

# RQ1 headline: best test MCC at early-runtime against the >0.90 target.
_er = rq1_df.filter(
    (pl.col("prediction_point") == "early_runtime") & (pl.col("metric") == "mcc")
)
if _er.height:
    best = _er.sort("value", descending=True).row(0, named=True)
    print(f"\nBest early-runtime test MCC: {best['value']:.4f} "
          f"95% CI [{best['ci_low']:.4f}, {best['ci_high']:.4f}] ({best['model']})")
    print(f"RQ1 >0.90 MCC target: {'MET' if best['value'] > 0.90 else 'NOT MET'} "
          f"(interpret alongside Tier 2/3 null fractions; V09 missingness is label-correlated).")

# %%
print(rq1_df.head(24))

# %% [markdown]
# ## 8. Hyperparameter tuning (Optuna + walk-forward CV)
#
# Bayesian optimization (Optuna) for LightGBM and XGBoost, and a shared
# walk-forward grid for Random Forest and Balanced Random Forest, each scored by
# the mean walk-forward-CV MCC at the RQ1 headline (early-runtime) point using the
# same per-fold impute + SMOTE + threshold-tuning pipeline as Section 6. Balanced
# Random Forest is tuned because RQ1 is an imbalance problem and it attacks that
# directly; the three baselines (logistic regression, decision tree, dummy) and
# sklearn Gradient Boosting (dominated by LightGBM / XGBoost) are left untuned.
# Best params are written to `configs/models/rq1_google_{lgbm,xgb,rf,brf}_{point}.yaml`.
# The filename is point-aware, so tuning a second point (set `TUNE_POINT`) never
# clobbers the first: run once at `early_runtime` (the RQ1 headline) and once at
# `at_submission` to give that checkpoint, which feeds RQ3 / RQ4, its own tuned
# config. The Optuna studies persist to a SQLite database on Drive and are topped
# up to `N_TRIALS` (not re-run from scratch), so re-running a tuned point only
# rewrites its config rather than burning another `N_TRIALS` of search.
#
# This is the heavy step of the week. Set `RUN_TUNING = False` to skip it and
# reuse the saved configs (the Section 9 checkpoint reads them when present).

# %%
import json
import time

import optuna
import yaml

CONFIG_DIR = Path(REPO_DIR) / "configs" / "models"   # repo working copy; commit the YAMLs after
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR = OUTPUT_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 50                       # per Bayesian family
TUNE_POINT = "early_runtime"        # the RQ1 headline; set "at_submission" to retune that point
TUNE_COLS = PREDICTION_POINTS[TUNE_POINT]
RUN_TUNING = True                   # set False to reuse existing configs/models/*.yaml
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Tunable family -> sklearn-native estimator param names.
_TUNABLE = {"lightgbm", "xgboost", "random_forest", "balanced_random_forest"}


def _tuned_pipeline(family: str, params: dict, y_train: np.ndarray):
    """Build the impute + SMOTE + classifier pipeline for a tunable family with
    explicit hyperparameters (the inverse-prior weighting is kept fixed)."""
    n_pos_ = int(y_train.sum())
    spw = (y_train.size - n_pos_) / max(n_pos_, 1)
    if family == "lightgbm":
        clf = lgb.LGBMClassifier(class_weight="balanced", subsample_freq=1, n_jobs=-1,
                                 random_state=RANDOM_SEED, verbosity=-1, **params)
    elif family == "xgboost":
        clf = xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
                                n_jobs=-1, random_state=RANDOM_SEED, **params)
    elif family == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                     random_state=RANDOM_SEED, **params)
    elif family == "balanced_random_forest":
        clf = BalancedRandomForestClassifier(sampling_strategy="all", replacement=True,
                                             n_jobs=-1, random_state=RANDOM_SEED, **params)
    else:
        raise ValueError(f"Not a tunable family: {family}")
    return ImbPipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", clf),
    ])


def _cv_mcc(params: dict, family: str, cols: list[str]) -> float:
    """Mean walk-forward-CV MCC for one parameter set (NaN if no usable fold)."""
    fold_mccs: list[float] = []
    for tr_max, te_lo, te_hi in WALK_FORWARD_FOLDS:
        tr = train_df.filter(pl.col("sched_day") <= tr_max)
        te = train_df.filter((pl.col("sched_day") >= te_lo) & (pl.col("sched_day") <= te_hi))
        if te.height == 0 or tr.filter(pl.col(LABEL_COL) == 1).height == 0:
            continue
        x_tr, y_tr = make_xy(tr, cols)
        x_te, y_te = make_xy(te, cols)
        est = _tuned_pipeline(family, params, y_tr)
        est.fit(x_tr, y_tr)
        prob = predict_proba(est, x_te)
        thr = best_threshold(y_te, prob)
        fold_mccs.append(float(matthews_corrcoef(y_te, (prob >= thr).astype(np.int8))))
        del est, x_tr, x_te
        gc.collect()
    return float(np.mean(fold_mccs)) if fold_mccs else float("nan")


def _lgbm_objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 2000),
        num_leaves=trial.suggest_int("num_leaves", 31, 511),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),   # feature_fraction
        subsample=trial.suggest_float("subsample", 0.5, 1.0),                 # bagging_fraction
        min_child_samples=trial.suggest_int("min_child_samples", 100, 10000, log=True),  # min_data_in_leaf
    )
    return _cv_mcc(params, "lightgbm", TUNE_COLS)


def _xgb_objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 100, log=True),
    )
    return _cv_mcc(params, "xgboost", TUNE_COLS)


def _run_optuna(name: str, objective) -> optuna.Study:
    """Resume (or create) the per-family study for the current TUNE_POINT and top it
    up to N_TRIALS completed trials. Topping up rather than always adding N_TRIALS
    makes a re-run idempotent: re-running a point that already has N_TRIALS done
    just rewrites the config (e.g. under a new point-aware name) without burning
    another N_TRIALS of search; a fresh point still gets the full N_TRIALS."""
    study = optuna.create_study(
        study_name=f"rq1_google_{name}_{TUNE_POINT}",
        direction="maximize",
        storage=f"sqlite:///{OPTUNA_DIR / f'rq1_google_{name}.db'}",
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(N_TRIALS - done, 0)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    else:
        print(f"  {name}: {done} trials already complete for {TUNE_POINT}; reusing best.")
    return study


def _forest_walkforward_grid(family: str) -> tuple[dict, float]:
    """Walk-forward-CV grid shared by the random-forest families (n_estimators,
    max_depth, min_samples_leaf), used for both `random_forest` and
    `balanced_random_forest`. Widened past the original corner (the first pass
    pinned both families at n_estimators 800 / max_depth 35 / min_samples_leaf 20,
    i.e. the edge of the old grid) so the search brackets the optimum on every
    axis: more trees, deeper including no cap, and a smaller leaf floor."""
    grid = [
        {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml}
        for ne in (800, 1200)
        for md in (35, 50, None)
        for ml in (5, 20)
    ]
    best, best_mcc = {}, -1.0
    for params in grid:
        m = _cv_mcc(params, family, TUNE_COLS)
        print(f"  {family} {params} -> CV MCC {m:.4f}")
        if m > best_mcc:
            best_mcc, best = m, params
    return best, best_mcc


def _write_config(name: str, params: dict, cv_mcc: float) -> Path:
    """Write the tuned config, keyed by family AND prediction point so tuning a
    second point never clobbers the first (`rq1_google_{family}_{point}.yaml`)."""
    payload = {"model": name, "prediction_point": TUNE_POINT, "random_state": RANDOM_SEED,
               "cv_mcc_mean": float(cv_mcc), "params": params}
    path = CONFIG_DIR / f"rq1_google_{name}_{TUNE_POINT}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    print(f"  wrote {path.name} (CV MCC {cv_mcc:.4f})")
    return path


if RUN_TUNING:
    print(f"Tuning at the {TUNE_POINT} point, {N_TRIALS} trials per Bayesian family ...")
    _t0 = time.perf_counter()
    _lgbm_study = _run_optuna("lgbm", _lgbm_objective)
    _write_config("lgbm", _lgbm_study.best_params, _lgbm_study.best_value)
    _xgb_study = _run_optuna("xgb", _xgb_objective)
    _write_config("xgb", _xgb_study.best_params, _xgb_study.best_value)
    _rf_best, _rf_mcc = _forest_walkforward_grid("random_forest")
    _write_config("rf", _rf_best, _rf_mcc)
    _brf_best, _brf_mcc = _forest_walkforward_grid("balanced_random_forest")
    _write_config("brf", _brf_best, _brf_mcc)
    print(f"Tuning complete in {(time.perf_counter() - _t0) / 60:.1f} min; "
          f"configs in {CONFIG_DIR}, studies in {OPTUNA_DIR}.")
else:
    print("RUN_TUNING is False; Section 9 will reuse existing configs/models/*.yaml if present.")

# %% [markdown]
# ## 9. Checkpoint the best ensembles
#
# Persist the deployable model at the two points the downstream work consumes: the
# best at-submission ensemble (the input to RQ4 admission control) and the best
# early-runtime ensemble (the RQ1 headline). For each point the best single
# ensemble learner by validation MCC is refit on the full training split (with its
# tuned config from Section 8 if present, else the Section 6 defaults), then saved
# with a metadata sidecar (library versions, a content hash of the training data,
# fit timing, the decision threshold, and the validation/test metrics) so the
# checkpoint is reproducible and auditable.

# %%
import hashlib
import pickle
from importlib import metadata as _ilmd

from utils.colab_setup import CHECKPOINT_DIR

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
_FAMILY_CONFIG = {"lightgbm": "lgbm", "xgboost": "xgb", "random_forest": "rf",
                  "balanced_random_forest": "brf"}


def _data_sha(df: pl.DataFrame) -> str:
    """Order-invariant content hash of a frame (sorted per-row hashes)."""
    row_hashes = np.sort(df.hash_rows().to_numpy())
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _library_versions() -> dict:
    out = {}
    for pkg in ("scikit-learn", "xgboost", "lightgbm", "imbalanced-learn",
                "polars", "numpy", "optuna"):
        try:
            out[pkg] = _ilmd.version(pkg)
        except _ilmd.PackageNotFoundError:
            out[pkg] = None
    return out


def _load_tuned_params(model_name: str, prediction_point: str) -> dict | None:
    """Load the tuned params for a family at a specific prediction point. Configs
    are point-aware (`rq1_google_{family}_{point}.yaml`), so the at-submission
    checkpoint never picks up early-runtime params or vice versa; falls back to
    None (Section 6 defaults) when the point has not been tuned."""
    key = _FAMILY_CONFIG.get(model_name)
    if key is None:
        return None
    path = CONFIG_DIR / f"rq1_google_{key}_{prediction_point}.yaml"
    if not path.exists():
        return None
    with open(path) as fh:
        return yaml.safe_load(fh).get("params")


def _best_single(prediction_point: str) -> str:
    """Name of the best single ensemble learner (excludes the dummy and the
    soft-voting stack) at a prediction point, by validation MCC."""
    candidates = [
        r for r in records
        if r["prediction_point"] == prediction_point
        and not r["model"].startswith("soft_voting")
        and r["model"] != "most_frequent"
    ]
    return max(candidates, key=lambda r: r["val_mcc"])["model"]


def checkpoint_best(prediction_point: str, filename: str) -> dict:
    cols = PREDICTION_POINTS[prediction_point]
    model_name = _best_single(prediction_point)
    tuned = _load_tuned_params(model_name, prediction_point)

    x_tr, y_tr = make_xy(train_df, cols)
    x_val, y_val = make_xy(val_df, cols)
    x_te, y_te = make_xy(test_df, cols)

    if tuned is not None:
        est = _tuned_pipeline(model_name, tuned, y_tr)
        tuned_flag = True
    else:
        est = build_estimator(model_name, y_tr)
        tuned_flag = False

    _t0 = time.perf_counter()
    est.fit(x_tr, y_tr)
    fit_seconds = time.perf_counter() - _t0

    thr = best_threshold(y_val, predict_proba(est, x_val))
    val_mcc = float(matthews_corrcoef(y_val, (predict_proba(est, x_val) >= thr).astype(np.int8)))
    test_metrics = compute_metrics(y_te, predict_proba(est, x_te), thr)

    model_path = CHECKPOINT_DIR / f"{filename}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(est, fh, protocol=pickle.HIGHEST_PROTOCOL)

    sidecar = {
        "artifact": filename,
        "model": model_name,
        "prediction_point": prediction_point,
        "tuned": tuned_flag,
        "tuned_params": tuned,
        "threshold": float(thr),
        "feature_columns": cols,
        "n_train_rows": int(train_df.height),
        "random_seed": RANDOM_SEED,
        "train_data_sha256": _data_sha(train_df.select(cols + [LABEL_COL])),
        "val_mcc": val_mcc,
        "test_metrics": test_metrics,
        "fit_seconds": round(fit_seconds, 2),
        "library_versions": _library_versions(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "Fitted imblearn pipeline (impute + SMOTE-at-fit + classifier); "
                "SMOTE is a no-op at predict, so predict_proba reproduces deployment.",
    }
    with open(CHECKPOINT_DIR / f"{filename}.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)

    print(f"Checkpointed {model_name} ({prediction_point}) -> {model_path}")
    print(f"  threshold {thr:.4f} | test MCC {test_metrics['mcc']:.4f} | "
          f"tuned={tuned_flag} | train SHA {sidecar['train_data_sha256'][:12]}...")
    del x_tr, x_val, x_te
    gc.collect()
    return sidecar


_ckpt_atsub = checkpoint_best("at_submission", "rq1_google_best_atsubmission")
_ckpt_early = checkpoint_best("early_runtime", "rq1_google_best_earlyruntime")

# %% [markdown]
# ## 10. Artifacts: curves, calibration, confusion, and the hypothesis test
#
# PR / ROC curves for the model zoo at the early-runtime point, a reliability plot
# and confusion matrices for the best model at each point, and the one-sample
# hypothesis test of the best-ensemble MCC against the 0.90 RQ1 target. The test
# uses the canonical stratified-bootstrap CI (`src/evaluation/metrics.py`) and the
# CI-based one-sided decision rule (`src/evaluation/hypothesis.py`).

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.evaluation.hypothesis import one_sample_threshold_test
from src.evaluation.metrics import calibration_table, mcc_with_ci

FIG_DIR = FIGURES_DIR / "rq1_google"
FIG_DIR.mkdir(parents=True, exist_ok=True)
y_test = test_df[LABEL_COL].to_numpy().astype(np.int8)


def _records_for(point: str) -> list[dict]:
    return [r for r in records if r["prediction_point"] == point and r["model"] != "most_frequent"]


def _best_record(point: str) -> dict:
    return max(_records_for(point), key=lambda r: r["val_mcc"])


# --- PR and ROC curves at the early-runtime point ---
for kind, curve_fn, fname, xlab, ylab in (
    ("PR", precision_recall_curve, "pr_curves.png", "Recall", "Precision"),
    ("ROC", roc_curve, "roc_curves.png", "False positive rate", "True positive rate"),
):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in sorted(_records_for("early_runtime"), key=lambda r: r["val_mcc"], reverse=True):
        if kind == "PR":
            precision, recall, _ = curve_fn(y_test, r["test_prob"])
            ax.plot(recall, precision, label=f"{r['model']} (AP={r['metrics']['pr_auc']:.3f})")
        else:
            fpr, tpr, _ = curve_fn(y_test, r["test_prob"])
            ax.plot(fpr, tpr, label=r["model"])
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"{kind} curves - early-runtime (RQ1)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150); plt.close(fig)
    print(f"Wrote {FIG_DIR / fname}")

# --- Reliability (calibration) plot for the best early-runtime model ---
_best_er = _best_record("early_runtime")
_cal = calibration_table(y_test, _best_er["test_prob"], n_bins=10).filter(pl.col("n") > 0)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
ax.plot(_cal["mean_predicted"], _cal["observed_rate"], "o-", label=_best_er["model"])
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed failure rate")
ax.set_title("Reliability - best early-runtime model"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "calibration.png", dpi=150); plt.close(fig)
print(f"Wrote {FIG_DIR / 'calibration.png'}")

# --- Confusion matrices for the best model at each prediction point ---
points = list(PREDICTION_POINTS)
fig, axes = plt.subplots(1, len(points), figsize=(4.2 * len(points), 4))
for ax, point in zip(np.atleast_1d(axes), points):
    rec = _best_record(point)
    pred = (rec["test_prob"] >= rec["threshold"]).astype(np.int8)
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap="Purples")
    ax.set_title(f"{point}\n{rec['model']} (MCC {rec['metrics']['mcc']:.3f})", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150); plt.close(fig)
print(f"Wrote {FIG_DIR / 'confusion_matrices.png'}")

# --- One-sample hypothesis test of the best-ensemble MCC vs the 0.90 target ---
RQ1_TARGET = 0.90
hyp_rows = []
for point in points:
    rec = _best_record(point)
    y_pred = (rec["test_prob"] >= rec["threshold"]).astype(np.int8)
    mcc_point, ci_lo, ci_hi = mcc_with_ci(y_test, y_pred, n_boot=1000, seed=RANDOM_SEED)
    test = one_sample_threshold_test(mcc_point, ci_lo, ci_hi, RQ1_TARGET, metric_name="MCC")
    hyp_rows.append({
        "prediction_point": point, "model": rec["model"], "metric": "mcc",
        "value": round(mcc_point, 4), "ci_low": round(ci_lo, 4), "ci_high": round(ci_hi, 4),
        "threshold_target": RQ1_TARGET, "reject_h0": test["reject"],
        "decision": test["decision"], "margin": round(test["margin"], 4),
        "narrative": test["narrative"],
    })
    print(f"  {point:14s} {rec['model']:20s} MCC {mcc_point:.4f} "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]  {test['decision']} vs {RQ1_TARGET}")

hyp_df = pl.DataFrame(hyp_rows)
hyp_df.write_csv(str(TABLES_DIR / "rq1_google_hypothesis_test.csv"))
print(f"Wrote {TABLES_DIR / 'rq1_google_hypothesis_test.csv'}")
