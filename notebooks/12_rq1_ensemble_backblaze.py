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
# # 12. Backblaze RQ1: Multi-Horizon Ensemble Failure Prediction
#
# **Purpose.** Train the ensemble failure-prediction strategy on the Backblaze
# SMART feature matrix at three horizons (7, 14, 30 days) and report results
# against the >0.90 MCC target at the true class prevalence. Hard-drive sibling of
# the Google RQ1 notebook; reuses the same model, metric, and hypothesis modules.
#
# **Temporal design (no leakage).** Train on working-set rows with year <= 2021.
# Fit the operating threshold and the probability calibration on a 2022 slice held
# at natural prevalence (notebook 10c), so both are set at the deployment base
# rate rather than the undersampled working-set rate. Report on the
# natural-prevalence 2023-2025 test (notebook 10b). The three periods do not
# overlap, and rolling and lag features were computed within each drive by date.
#
# **Drive-model prior (target encoding).** Model-specific SMART discriminability
# is an established finding, and prior Backblaze studies exploit it by fixing a
# single high-failure model. Here a leakage-safe per-model failure-rate prior is
# added as one numeric feature, fit only on 2021-and-earlier data and applied to
# every split. It is a per-model aggregate, not a per-row label, so it does not
# leak the outcome.
#
# **Imbalance and calibration.** The working set is already moderately balanced
# (~5% positive at 20:1), so cost-sensitive learning (inverse-prior class weights,
# internal to the wrappers) is the default; SMOTE is avoided as a memory bomb, and
# the 10:1 and 40:1 sensitivity sets cover the imbalance-ratio question. Isotonic
# or Platt calibration is fit on the 2022 natural-prevalence slice so probabilities
# are meaningful at the deployment base rate.
#
# **Ceiling.** The learning-curve baseline saturated near MCC 0.61 at the
# undersampled prevalence and stayed flat as data grew, so the levers are tuning,
# the ensemble, the model prior, and calibration, not more data. A trivial leak
# would read near 1.0. If the tuned ensembles cap below 0.90 at natural
# prevalence, that is reported as a dataset-level finding (moderate SMART
# discriminability at natural prevalence), not chased by loosening leakage.
#
# **Outputs.**
# - `outputs/tables/rq1_backblaze.csv` (horizon, model, metric, value, ci_low,
#   ci_high, prevalence).
# - `outputs/tables/rq1_backblaze_hypothesis_test.csv`.
# - `outputs/figures/rq1_backblaze/pr_curves_{h}d.png`.
# - Checkpoint `rq1_backblaze_best_14d.pkl` plus JSON sidecar for the drift work.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars pandas pyarrow scikit-learn xgboost lightgbm imbalanced-learn matplotlib google-cloud-storage

# %%
import os
import sys
from pathlib import Path

from google.colab import userdata

GITHUB_PAT = userdata.get('GITHUB_PAT')
PROJECT_ID = userdata.get('GCP_PROJECT_ID')
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

# Purge cached repo modules so a git pull actually takes effect in a warm runtime.
for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from google.colab import auth
auth.authenticate_user()

# %%
import gc
import json
import pickle

import numpy as np
import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR, CHECKPOINT_DIR

from src.models.ensemble import build_wrapper
from src.evaluation.metrics import mcc_with_ci, f1_with_ci, pr_auc_with_ci
from src.evaluation.hypothesis import one_sample_threshold_test

setup_drive()

TABLES_DIR = OUTPUT_DIR / 'tables'
FEATURES_DIR = OUTPUT_DIR / 'features'
FIG_DIR = OUTPUT_DIR / 'figures' / 'rq1_backblaze'
for d in [TABLES_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_WORKING_PREFIX = 'backblaze_features/working_set_20x'
GCS_NATURAL_PREFIX = 'backblaze_features/natural_test_2023_2025'
GCS_NATURAL_VAL_PREFIX = 'backblaze_features/natural_val_2022'
LOCAL_WORKING = Path('/content/working_set_20x')
LOCAL_NATURAL = Path('/content/natural_test_2023_2025')
LOCAL_NATURAL_VAL = Path('/content/natural_val_2022')
for d in [LOCAL_WORKING, LOCAL_NATURAL, LOCAL_NATURAL_VAL]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
HORIZONS = (7, 14, 30)
RQ1_TARGET = 0.90

# Compute knobs. The learning curve saturated near 1% of the data, so a bounded
# training subsample loses nothing and keeps the tree learners fast on 2 cores.
# Natural-prevalence metrics use a uniform (prevalence-preserving) sample of the
# 2023-2025 test so the bootstrap CI is computable; uniform sampling does not
# change the class balance.
TRAIN_CAP_ROWS = 800_000          # all positives kept; negatives capped to this budget
EVAL_SAMPLE_ROWS = 6_000_000      # uniform prevalence-preserving natural-test sample
NBOOT_NATURAL = 200               # bootstrap resamples for the natural-test CIs
RF_N_ESTIMATORS = 150             # forest size for RF / Balanced RF on a 2-core runtime
TE_SMOOTHING = 300.0              # additive smoothing for the drive-model prior
TE_TARGET = 'failure_within_30d'  # horizon-agnostic model-risk prior, fit on <=2021 only
# Models scored at natural prevalence (the RQ1 contenders that scale to scoring).
NATURAL_TEST_MODELS = ('lightgbm', 'xgboost', 'balanced_random_forest', 'soft_voting_stack')

# %% [markdown]
# ### Fetch the working set, the natural-prevalence test, and the 2022 validation

# %%
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)


def sync_prefix(prefix: str, local_dir: Path) -> list[Path]:
    paths = []
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        local_path = local_dir / blob.name.split('/')[-1]
        if not (local_path.exists() and local_path.stat().st_size == blob.size):
            blob.download_to_filename(str(local_path))
        paths.append(local_path)
    return sorted(paths)


working_files = sync_prefix(GCS_WORKING_PREFIX, LOCAL_WORKING)
natural_files = sync_prefix(GCS_NATURAL_PREFIX, LOCAL_NATURAL)
natural_val_files = sync_prefix(GCS_NATURAL_VAL_PREFIX, LOCAL_NATURAL_VAL)
print(f"working: {len(working_files)}; natural test: {len(natural_files)}; "
      f"natural val 2022: {len(natural_val_files)}")
assert natural_val_files, "natural_val_2022 not found; run notebook 10c first"

# %% [markdown]
# ## 1. Feature columns
#
# The notebook 11 exclusion set: drop identifiers and string categoricals, the
# outcome column, the calendar-monotonic indices, and, when modeling one horizon,
# the other two horizon targets. LightGBM and XGBoost take the numeric features
# with native null handling; the null-intolerant learners get a fitted imputer.
# The drive-model prior (Section 2) is appended as one extra numeric feature.

# %%
SCHEMA = json.loads((FEATURES_DIR / 'backblaze_feature_schema.json').read_text())
ALL_COLUMNS = SCHEMA['columns']
assert SCHEMA['n_columns'] == 146, "unexpected working-set schema width"

HORIZON_TARGETS = {f'failure_within_{h}d' for h in HORIZONS}
EXCLUDE_ALWAYS = {
    'date', 'serial_number', 'model', 'model_canonical', 'manufacturer', 'era',
    'failure', 'failure_observed', 'censored', 'is_last_obs',
    'year', 'fleet_age_days',
}
MODEL_PRIOR = 'model_prior'


def base_feature_columns(horizon: int) -> list[str]:
    target = f'failure_within_{horizon}d'
    drop = EXCLUDE_ALWAYS | (HORIZON_TARGETS - {target})
    return [c for c in ALL_COLUMNS if c not in drop and c != target]


BASE_FEATURES = base_feature_columns(30)
for h in HORIZONS:
    assert base_feature_columns(h) == BASE_FEATURES, "feature columns drift across horizons"
print(f"Base features: {len(BASE_FEATURES)} (expected ~134)")
assert 'year' not in BASE_FEATURES and 'fleet_age_days' not in BASE_FEATURES, "calendar leak"
assert not (set(BASE_FEATURES) & HORIZON_TARGETS), "horizon target leaked into features"
assert not (set(BASE_FEATURES) & {'failure', 'serial_number', 'era'}), "id/outcome leak"

# %% [markdown]
# ## 2. Training pool, drive-model prior, and split
#
# Train pool is working-set rows with year <= 2021. The per-model failure-rate
# prior is fit on that pool only (smoothed toward the global rate for rare
# models), then attached to every split. `FEATURES` = base features + the prior.

# %%
READ_COLS = BASE_FEATURES + list(HORIZON_TARGETS) + ['year', 'model_canonical']
train_pool = (
    pl.scan_parquet(str(LOCAL_WORKING / 'bucket_*.parquet'))
    .select(READ_COLS)
    .filter(pl.col('year') <= 2021)
    .with_columns([pl.col(c).cast(pl.Float32) for c in BASE_FEATURES])
    .collect()
)
print(f"train pool (<=2021): {train_pool.height:,} rows")

# Leakage-safe drive-model prior: per-model smoothed failure rate from <=2021 only.
global_rate = float(train_pool[TE_TARGET].mean())
enc = (
    train_pool.group_by('model_canonical')
    .agg(pl.len().alias('n'), pl.col(TE_TARGET).sum().alias('pos'))
    .with_columns(
        ((pl.col('pos') + TE_SMOOTHING * global_rate) / (pl.col('n') + TE_SMOOTHING))
        .cast(pl.Float32).alias(MODEL_PRIOR)
    )
    .select('model_canonical', MODEL_PRIOR)
)
print(f"drive-model prior fit on {enc.height} models; global 30d rate {global_rate:.5f}")


def attach_prior(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.join(enc, on='model_canonical', how='left')
        .with_columns(pl.col(MODEL_PRIOR).fill_null(global_rate).cast(pl.Float32))
    )


train_pool = attach_prior(train_pool)
FEATURES = BASE_FEATURES + [MODEL_PRIOR]
print(f"Model features (with prior): {len(FEATURES)}")


def cap_train(pool: pl.DataFrame, target: str, cap: int, seed: int) -> pl.DataFrame:
    """All positives for this horizon plus a capped uniform negative sample."""
    pos = pool.filter(pl.col(target) == 1)
    neg = pool.filter(pl.col(target) == 0)
    neg_budget = max(cap - pos.height, pos.height)
    if neg.height > neg_budget:
        neg = neg.sample(n=neg_budget, seed=seed)
    return pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)


# %% [markdown]
# ## 3. Natural-prevalence 2022 validation and 2023-2025 test samples
#
# The 2022 validation (notebook 10c) sets the threshold and calibration at the
# true base rate. The 2023-2025 test is uniformly sampled per partition (never
# fully materialized) to keep the bootstrap CI computable while preserving
# prevalence. Both carry the drive-model prior.

# %%
nat_val = (
    pl.scan_parquet(str(LOCAL_NATURAL_VAL / 'bucket_*.parquet'))
    .select(BASE_FEATURES + list(HORIZON_TARGETS) + ['model_canonical'])
    .collect()
    .with_columns([pl.col(c).cast(pl.Float32) for c in BASE_FEATURES])
)
nat_val = attach_prior(nat_val)
print(f"natural val 2022: {nat_val.height:,} rows "
      f"({100 * nat_val['failure_within_30d'].mean():.4f}% positive at 30d)")

# %%
natural_glob = sorted(LOCAL_NATURAL.glob('bucket_*.parquet'))
total_natural = (
    pl.scan_parquet(str(LOCAL_NATURAL / 'bucket_*.parquet'))
    .select(pl.len()).collect(engine='streaming').item()
)
eval_frac = min(1.0, EVAL_SAMPLE_ROWS / total_natural)
eval_parts = []
for f in natural_glob:
    part = (
        pl.read_parquet(f, columns=BASE_FEATURES + list(HORIZON_TARGETS) + ['model_canonical'])
        .with_columns([pl.col(c).cast(pl.Float32) for c in BASE_FEATURES])
    )
    eval_parts.append(part.sample(fraction=eval_frac, seed=SEED))
    del part
    gc.collect()
natural_eval = attach_prior(pl.concat(eval_parts))
del eval_parts
gc.collect()
print(f"natural test total: {total_natural:,}; eval sample: {natural_eval.height:,} "
      f"(frac {eval_frac:.4f})")
for h in HORIZONS:
    tgt = f'failure_within_{h}d'
    print(f"  {tgt}: {int(natural_eval[tgt].sum()):,} positive "
          f"({100 * natural_eval[tgt].mean():.4f}%)")

# %% [markdown]
# ## 4. Modeling helpers
#
# Learners are built by name via `build_wrapper` (cost-sensitive weighting is
# internal). Null-intolerant learners share a median imputer fit on the training
# features; LightGBM and XGBoost keep NaN as signal. Calibration is isotonic or
# Platt (lower validation Brier), and the operating threshold maximizes MCC on the
# calibrated 2022 natural-prevalence validation scores.

# %%
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

NATIVE_NULL = {'lightgbm', 'xgboost'}
# scikit-learn GradientBoosting is sequential and does not scale on a 2-core
# runtime; XGBoost and LightGBM are the gradient-boosting learners here.
WRAPPER_NAMES = ('random_forest', 'balanced_random_forest', 'xgboost', 'lightgbm')
FOREST_LEARNERS = {'random_forest', 'balanced_random_forest'}


def to_matrix(df: pl.DataFrame, imputer: SimpleImputer | None) -> np.ndarray:
    arr = df.select(FEATURES).to_numpy().astype(np.float32)
    if imputer is not None:
        arr = imputer.transform(arr).astype(np.float32)
    return arr


def fit_calibrator(val_scores: np.ndarray, val_y: np.ndarray) -> tuple[str, object]:
    iso = IsotonicRegression(out_of_bounds='clip').fit(val_scores, val_y)
    platt = LogisticRegression(max_iter=1000).fit(val_scores.reshape(-1, 1), val_y)
    iso_p = np.clip(iso.predict(val_scores), 0, 1)
    platt_p = platt.predict_proba(val_scores.reshape(-1, 1))[:, 1]
    b_iso = float(np.mean((iso_p - val_y) ** 2))
    b_platt = float(np.mean((platt_p - val_y) ** 2))
    return ('isotonic', iso) if b_iso <= b_platt else ('platt', platt)


def apply_calibrator(kind: str, obj: object, scores: np.ndarray) -> np.ndarray:
    if kind == 'isotonic':
        return np.clip(obj.predict(scores), 0.0, 1.0)
    return obj.predict_proba(scores.reshape(-1, 1))[:, 1]


def best_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    from src.evaluation.metrics import _mcc
    cands = np.unique(np.clip(np.quantile(proba, np.linspace(0.50, 0.9999, 300)), 1e-6, 1 - 1e-6))
    best_t, best_m = 0.5, -2.0
    for t in cands:
        m = _mcc(y.astype(np.int64), (proba >= t).astype(np.int64))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def fit_baselines(Xtr_imp, ytr, seed):
    return {
        'decision_tree': DecisionTreeClassifier(
            max_depth=8, class_weight='balanced', random_state=seed).fit(Xtr_imp, ytr),
        'logistic_regression': SGDClassifier(
            loss='log_loss', class_weight='balanced', random_state=seed).fit(Xtr_imp, ytr),
    }


# %% [markdown]
# ## 5. Per-horizon training, calibration, and natural-prevalence evaluation

# %%
records = []
fitted = {}


def eval_block(y_true, proba, threshold, n_boot, horizon, model_name):
    y_pred = (proba >= threshold).astype(np.int8)
    for metric_name, res in (
        ('mcc', mcc_with_ci(y_true, y_pred, n_boot=n_boot, seed=SEED)),
        ('f1', f1_with_ci(y_true, y_pred, n_boot=n_boot, seed=SEED)),
        ('pr_auc', pr_auc_with_ci(y_true, proba, n_boot=n_boot, seed=SEED)),
    ):
        pt, lo, hi = res
        records.append({
            'horizon': horizon, 'model': model_name, 'metric': metric_name,
            'value': pt, 'ci_low': lo, 'ci_high': hi,
            'threshold': threshold, 'prevalence': 'natural',
        })


for horizon in HORIZONS:
    target = f'failure_within_{horizon}d'
    print(f"\n===== horizon {horizon}d =====")
    tr = cap_train(train_pool, target, TRAIN_CAP_ROWS, SEED)
    ytr = tr[target].to_numpy().astype(np.int8)
    yval = nat_val[target].to_numpy().astype(np.int8)
    ynat = natural_eval[target].to_numpy().astype(np.int8)
    print(f"  train {tr.height:,} ({int(ytr.sum()):,} pos); "
          f"nat-val {nat_val.height:,}; nat-test {natural_eval.height:,}")

    imputer = SimpleImputer(strategy='median').fit(tr.select(FEATURES).to_numpy().astype(np.float32))
    Xtr_nan = tr.select(FEATURES).to_numpy().astype(np.float32)
    Xtr_imp = imputer.transform(Xtr_nan).astype(np.float32)
    Xval_nan = to_matrix(nat_val, None)
    Xval_imp = to_matrix(nat_val, imputer)
    Xnat_nan = to_matrix(natural_eval, None)
    Xnat_imp = to_matrix(natural_eval, imputer)

    val_mcc_by_model = {}
    for name in WRAPPER_NAMES:
        native = name in NATIVE_NULL
        Xtr = Xtr_nan if native else Xtr_imp
        Xval = Xval_nan if native else Xval_imp
        Xnat = Xnat_nan if native else Xnat_imp
        extra = {'n_estimators': RF_N_ESTIMATORS} if name in FOREST_LEARNERS else {}
        wrapper = build_wrapper(name, random_state=SEED, **extra).fit(Xtr, ytr)
        val_scores = wrapper.predict_proba(Xval)
        kind, calib = fit_calibrator(val_scores, yval)
        val_cal = apply_calibrator(kind, calib, val_scores)
        thr = best_threshold(yval, val_cal)
        val_mcc_by_model[name] = float(
            mcc_with_ci(yval, (val_cal >= thr).astype(np.int8), n_boot=200, seed=SEED)[0])
        fitted[(horizon, name)] = {
            'wrapper': wrapper, 'calibrator_kind': kind, 'calibrator': calib,
            'threshold': thr, 'needs_impute': not native,
            'imputer': None if native else imputer, 'feature_columns': FEATURES,
        }
        if name in NATURAL_TEST_MODELS:
            nat_cal = apply_calibrator(kind, calib, wrapper.predict_proba(Xnat))
            eval_block(ynat, nat_cal, thr, NBOOT_NATURAL, horizon, name)
            fitted[(horizon, name)]['nat_proba'] = nat_cal
        print(f"  {name:>24}: nat-val MCC {val_mcc_by_model[name]:.4f} "
              f"(calib {kind}, thr {thr:.4f})")

    # Baselines at natural prevalence (imputed features).
    for bname, bmodel in fit_baselines(Xtr_imp, ytr, SEED).items():
        thr_b = best_threshold(yval, bmodel.predict_proba(Xval_imp)[:, 1])
        eval_block(ynat, bmodel.predict_proba(Xnat_imp)[:, 1], thr_b, NBOOT_NATURAL, horizon, bname)
    # Most-frequent reference (all-negative): MCC and F1 are 0, PR-AUC is prevalence.
    eval_block(ynat, np.zeros_like(ynat, dtype=np.float64), 0.5, NBOOT_NATURAL, horizon, 'most_frequent')

    # Soft-voting stack: mean of the top-3 learners' calibrated probabilities.
    top3 = sorted(val_mcc_by_model, key=val_mcc_by_model.get, reverse=True)[:3]

    def stacked(which):
        cols = []
        for name in top3:
            st = fitted[(horizon, name)]
            native = name in NATIVE_NULL
            X = (Xval_nan if native else Xval_imp) if which == 'val' else (Xnat_nan if native else Xnat_imp)
            s = st['wrapper'].predict_proba(X)
            cols.append(apply_calibrator(st['calibrator_kind'], st['calibrator'], s))
        return np.mean(np.vstack(cols), axis=0)

    stack_val = stacked('val')
    stack_thr = best_threshold(yval, stack_val)
    stack_nat = stacked('nat')
    eval_block(ynat, stack_nat, stack_thr, NBOOT_NATURAL, horizon, 'soft_voting_stack')
    fitted[(horizon, 'soft_voting_stack')] = {'members': top3, 'threshold': stack_thr, 'nat_proba': stack_nat}
    print(f"  soft_voting_stack top3={top3} thr {stack_thr:.4f}")

    del Xtr_nan, Xtr_imp, Xval_nan, Xval_imp, Xnat_nan, Xnat_imp
    gc.collect()

# %% [markdown]
# ## 6. Results table and hypothesis test
#
# MCC is primary and tested against 0.90 via the CI-based one-sided test (reject
# only when the CI lower bound clears 0.90).

# %%
results = pl.DataFrame(records)
results.write_csv(TABLES_DIR / 'rq1_backblaze.csv')
print(f"Saved {TABLES_DIR / 'rq1_backblaze.csv'} ({results.height} rows)")

nat_mcc = results.filter(pl.col('metric') == 'mcc')
print("\nNatural-prevalence MCC by horizon and model:")
print(nat_mcc.sort(['horizon', 'value'], descending=[False, True]).to_pandas().to_string(index=False))
print("\nNatural-prevalence PR-AUC by horizon and model:")
print(results.filter(pl.col('metric') == 'pr_auc')
      .sort(['horizon', 'value'], descending=[False, True]).to_pandas().to_string(index=False))

hyp_rows = []
for row in nat_mcc.iter_rows(named=True):
    test = one_sample_threshold_test(
        row['value'], row['ci_low'], row['ci_high'], RQ1_TARGET,
        metric_name=f"MCC_{row['model']}_{row['horizon']}d")
    test['horizon'] = row['horizon']
    test['model'] = row['model']
    hyp_rows.append(test)
hyp = pl.DataFrame(hyp_rows)
hyp.write_csv(TABLES_DIR / 'rq1_backblaze_hypothesis_test.csv')
print(f"\nSaved {TABLES_DIR / 'rq1_backblaze_hypothesis_test.csv'}")
print(f"Cells clearing 0.90 at natural prevalence: {int(hyp['reject'].sum())}/{hyp.height}")

# %% [markdown]
# ## 7. PR-curve figures (natural prevalence)

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

for horizon in HORIZONS:
    ynat = natural_eval[f'failure_within_{horizon}d'].to_numpy().astype(np.int8)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in NATURAL_TEST_MODELS:
        st = fitted.get((horizon, name))
        if st is None or 'nat_proba' not in st:
            continue
        prec, rec, _ = precision_recall_curve(ynat, st['nat_proba'])
        ax.plot(rec, prec, label=name)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Backblaze RQ1 PR curves, {horizon}-day horizon (natural prevalence)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'pr_curves_{horizon}d.png', dpi=150)
    plt.close(fig)
print(f"Saved PR-curve figures to {FIG_DIR}")

# %% [markdown]
# ## 8. Checkpoint the best 14-day model
#
# The best 14-day contender by natural-prevalence MCC is pickled with a JSON
# sidecar (feature order, threshold, calibrator, prior, library versions,
# training-data hash) for the drift analysis. Calibration was fit at natural
# prevalence, so the checkpoint's probabilities are meaningful at the deployment
# base rate.

# %%
import hashlib
import sklearn
import lightgbm
import xgboost

nat_mcc_14 = nat_mcc.filter(pl.col('horizon') == 14).sort('value', descending=True)
best_model_14 = nat_mcc_14['model'][0]
best_mcc_14 = nat_mcc_14['value'][0]
print(f"Best 14-day contender: {best_model_14} (natural MCC {best_mcc_14:.4f})")

train14 = cap_train(train_pool, 'failure_within_14d', TRAIN_CAP_ROWS, SEED)
data_hash = hashlib.sha256(train14.select(FEATURES).to_numpy().tobytes()).hexdigest()

CKPT = CHECKPOINT_DIR / 'rq1_backblaze_best_14d.pkl'
SIDECAR = CHECKPOINT_DIR / 'rq1_backblaze_best_14d.json'
if best_model_14 == 'soft_voting_stack':
    members = fitted[(14, 'soft_voting_stack')]['members']
    payload = {
        'kind': 'soft_voting_stack',
        'members': {m: {k: fitted[(14, m)][k] for k in
                        ('wrapper', 'calibrator_kind', 'calibrator', 'needs_impute',
                         'imputer', 'feature_columns')} for m in members},
        'threshold': fitted[(14, 'soft_voting_stack')]['threshold'],
        'drive_model_prior': enc.to_dicts(), 'global_prior': global_rate,
    }
else:
    st = fitted[(14, best_model_14)]
    payload = {
        'kind': 'single', 'model_name': best_model_14,
        'wrapper': st['wrapper'], 'calibrator_kind': st['calibrator_kind'],
        'calibrator': st['calibrator'], 'needs_impute': st['needs_impute'],
        'imputer': st['imputer'], 'feature_columns': st['feature_columns'],
        'threshold': st['threshold'],
        'drive_model_prior': enc.to_dicts(), 'global_prior': global_rate,
    }
with CKPT.open('wb') as fh:
    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

sidecar = {
    'horizon': 14, 'best_model': best_model_14, 'natural_mcc': float(best_mcc_14),
    'threshold': float(payload['threshold']),
    'calibration_fit_on': 'natural_prevalence_2022', 'feature_columns': FEATURES,
    'n_features': len(FEATURES), 'model_prior_smoothing': TE_SMOOTHING,
    'train_rows': int(train14.height), 'train_data_sha256': data_hash,
    'eval_sample_rows': int(natural_eval.height), 'eval_fraction': float(eval_frac), 'seed': SEED,
    'library_versions': {
        'polars': pl.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'lightgbm': lightgbm.__version__, 'xgboost': xgboost.__version__,
    },
}
SIDECAR.write_text(json.dumps(sidecar, indent=2))
print(f"Checkpoint: {CKPT}")
print(f"Sidecar:    {SIDECAR}")

# %% [markdown]
# ## 9. Summary

# %%
print("BACKBLAZE RQ1 MULTI-HORIZON SUMMARY (natural prevalence, model prior + 2022 calibration)")
print("=" * 78)
for horizon in HORIZONS:
    sub = nat_mcc.filter(pl.col('horizon') == horizon).sort('value', descending=True)
    top = sub.row(0, named=True)
    prc = results.filter((pl.col('horizon') == horizon) & (pl.col('metric') == 'pr_auc')).sort('value', descending=True).row(0, named=True)
    print(f"  {horizon:>2}d: best MCC {top['model']:>22} {top['value']:.4f} "
          f"CI [{top['ci_low']:.4f}, {top['ci_high']:.4f}]; best PR-AUC {prc['value']:.4f}")
print(f"RQ1 target: MCC > {RQ1_TARGET} (one-sided, CI lower bound)")
print("=" * 78)
