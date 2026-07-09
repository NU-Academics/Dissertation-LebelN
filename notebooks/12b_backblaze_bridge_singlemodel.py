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
# # 12b. Backblaze RQ1 Bridge: Single-Model, Balanced-Test Protocol
#
# **Purpose.** Quantify how much of prior reported Backblaze failure-prediction
# performance is an artifact of evaluation design rather than modeling. Several
# published studies report MCC in the 0.94 to 0.98 range by (1) fixing a single
# high-failure drive model and (2) evaluating on a heavily undersampled test set,
# often around a 14:1 healthy-to-failed ratio. This notebook applies that same
# protocol with the same pipeline used for the fleet-wide analysis, then contrasts
# it with the natural class prevalence for the identical model. The gap between the
# two isolates the effect of test-set balancing, and the gap to the fleet-wide
# natural result (notebook 12) isolates the single-model selection effect.
#
# **Design.** One high-volume model is selected. The model is trained on
# 2022-and-earlier working-set rows, positives being the union of the 14-day
# pre-failure window. The operating threshold is tuned on a 2022 slice held at the
# 14:1 protocol ratio. Two test sets are then scored, both drawn from the
# 2023-2025 period for the same model: one undersampled to 14:1 (the prevailing
# protocol) and one at natural prevalence. PR-AUC is the threshold-free headline
# because it is prevalence-sensitive and needs no operating point.
#
# **Reuse.** No new feature engineering: the working set and the natural test are
# filtered to the chosen model. Learners, metrics, and the soft-voting stack come
# from the shared modules.
#
# **Output.** `outputs/tables/rq1_backblaze_bridge.csv`
# (protocol, model, metric, value, ci_low, ci_high, prevalence_ratio).

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars pandas pyarrow scikit-learn xgboost lightgbm imbalanced-learn google-cloud-storage

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

for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from google.colab import auth
auth.authenticate_user()

# %%
import gc
import json

import numpy as np
import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR

from src.models.ensemble import build_wrapper
from src.evaluation.metrics import mcc_with_ci, f1_with_ci, pr_auc_with_ci

setup_drive()

TABLES_DIR = OUTPUT_DIR / 'tables'
FEATURES_DIR = OUTPUT_DIR / 'features'
LOCAL_WORKING = Path('/content/working_set_20x')
LOCAL_NATURAL = Path('/content/natural_test_2023_2025')
for d in [TABLES_DIR, LOCAL_WORKING, LOCAL_NATURAL]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_WORKING_PREFIX = 'backblaze_features/working_set_20x'
GCS_NATURAL_PREFIX = 'backblaze_features/natural_test_2023_2025'

SEED = 42
np.random.seed(SEED)
BRIDGE_MODEL = 'ST4000DM000'   # a high-volume, high-failure Seagate model; fallback below
PROTOCOL_RATIO = 14            # healthy-to-failed ratio of the prevailing protocol
HORIZON = 14                   # positives = union of the 14-day pre-failure window
NBOOT = 300

# %% [markdown]
# ### Fetch the working set and natural test from GCS

# %%
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)


def sync_prefix(prefix, local_dir):
    paths = []
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        p = local_dir / blob.name.split('/')[-1]
        if not (p.exists() and p.stat().st_size == blob.size):
            blob.download_to_filename(str(p))
        paths.append(p)
    return sorted(paths)


sync_prefix(GCS_WORKING_PREFIX, LOCAL_WORKING)
sync_prefix(GCS_NATURAL_PREFIX, LOCAL_NATURAL)

# %% [markdown]
# ## 1. Feature columns and model selection
#
# Same numeric feature set as the fleet analysis, minus identifiers, outcome, and
# calendar-monotonic indices. The drive-model prior is not used here (a single
# model makes it constant). If the preferred model is absent, fall back to the
# model with the most 14-day positives in the working set.

# %%
SCHEMA = json.loads((FEATURES_DIR / 'backblaze_feature_schema.json').read_text())
ALL_COLUMNS = SCHEMA['columns']
HORIZON_TARGETS = {f'failure_within_{h}d' for h in (7, 14, 30)}
EXCLUDE = {
    'date', 'serial_number', 'model', 'model_canonical', 'manufacturer', 'era',
    'failure', 'failure_observed', 'censored', 'is_last_obs', 'year', 'fleet_age_days',
}
TARGET = f'failure_within_{HORIZON}d'
FEATURES = [c for c in ALL_COLUMNS if c not in (EXCLUDE | HORIZON_TARGETS)]
print(f"features: {len(FEATURES)}")

# The bridge model must have failures both in the training years (working set)
# and in the 2023-2025 test period, or one of the two test sets would be empty.
# ST4000DM000 was largely retired before the test period, so the fallback picks
# the model with the most test-period positives among those with enough training
# positives.
ws_pos = (
    pl.scan_parquet(str(LOCAL_WORKING / 'bucket_*.parquet'))
    .filter(pl.col('year') <= 2021)
    .group_by('model_canonical').agg(pl.col(TARGET).sum().alias('train_pos')).collect()
)
test_pos = (
    pl.scan_parquet(str(LOCAL_NATURAL / 'bucket_*.parquet'))
    .group_by('model_canonical').agg(pl.col(TARGET).sum().alias('test_pos')).collect()
)
cand = (
    ws_pos.join(test_pos, on='model_canonical', how='inner')
    .filter((pl.col('train_pos') >= 500) & (pl.col('test_pos') >= 200))
    .sort('test_pos', descending=True)
)
assert cand.height > 0, "no model has enough train and test positives for the bridge"
pref = cand.filter(pl.col('model_canonical') == BRIDGE_MODEL)
chosen = pref if pref.height else cand.head(1)
BRIDGE_MODEL = chosen['model_canonical'][0]
print(f"bridge model: {BRIDGE_MODEL} "
      f"(train 14d positives {int(chosen['train_pos'][0]):,}, "
      f"test 14d positives {int(chosen['test_pos'][0]):,})")
print("candidates by test positives:")
print(cand.head(6).to_pandas().to_string(index=False))

# %% [markdown]
# ## 2. Build the single-model train, validation, and two test sets
#
# Train on the model's 2021-and-earlier rows; tune the threshold on its 2022 rows
# at the 14:1 protocol ratio; test on its 2023-2025 rows at both the 14:1 protocol
# ratio and natural prevalence.

# %%
READ = FEATURES + [TARGET, 'year', 'model_canonical']
ws_model = (
    pl.scan_parquet(str(LOCAL_WORKING / 'bucket_*.parquet'))
    .select(READ).filter(pl.col('model_canonical') == BRIDGE_MODEL)
    .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURES]).collect()
)
nat_model = (
    pl.scan_parquet(str(LOCAL_NATURAL / 'bucket_*.parquet'))
    .select(READ).filter(pl.col('model_canonical') == BRIDGE_MODEL)
    .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURES]).collect()
)
# Cap the natural test at a manageable size (uniform sampling preserves prevalence)
# so a very high-volume model does not exhaust memory when its matrices are built.
NAT_MODEL_CAP = 8_000_000
if nat_model.height > NAT_MODEL_CAP:
    nat_model = nat_model.sample(n=NAT_MODEL_CAP, seed=SEED)
print(f"{BRIDGE_MODEL}: working-set rows {ws_model.height:,}; natural-test rows {nat_model.height:,}")


def rebalance(df, target, ratio, seed):
    """Undersample the negative class to `ratio`:1 healthy-to-failed."""
    pos = df.filter(pl.col(target) == 1)
    neg = df.filter(pl.col(target) == 0)
    n_neg = min(neg.height, ratio * pos.height)
    return pl.concat([pos, neg.sample(n=n_neg, seed=seed)]).sample(fraction=1.0, shuffle=True, seed=seed)


# Train on the model's 2021-and-earlier rows; carve a 20% validation for the
# operating threshold (rebalanced to the protocol ratio). A within-train split is
# used rather than a 2022 slice so the bridge does not depend on the model having
# 2022 coverage.
model_train = ws_model.filter(pl.col('year') <= 2021).sample(fraction=1.0, shuffle=True, seed=SEED)
n_val = int(model_train.height * 0.2)
train = model_train.tail(model_train.height - n_val)
val_1441 = rebalance(model_train.head(n_val), TARGET, PROTOCOL_RATIO, SEED)
test_natural = nat_model  # 2023-2025, natural prevalence
test_1441 = rebalance(nat_model, TARGET, PROTOCOL_RATIO, SEED)
print(f"  train {train.height:,} ({int(train[TARGET].sum()):,} pos); "
      f"val@{PROTOCOL_RATIO}:1 {val_1441.height:,}; "
      f"test@{PROTOCOL_RATIO}:1 {test_1441.height:,} "
      f"({100 * test_1441[TARGET].mean():.2f}% pos); "
      f"test natural {test_natural.height:,} ({100 * test_natural[TARGET].mean():.4f}% pos)")

# %% [markdown]
# ## 3. Train the ensemble and score both test sets

# %%
from sklearn.impute import SimpleImputer

NATIVE_NULL = {'lightgbm', 'xgboost'}
ZOO = ('random_forest', 'balanced_random_forest', 'xgboost', 'lightgbm')

ytr = train[TARGET].to_numpy().astype(np.int8)
imputer = SimpleImputer(strategy='median').fit(train.select(FEATURES).to_numpy().astype(np.float32))


def mats(df):
    nan = df.select(FEATURES).to_numpy().astype(np.float32)
    return nan, imputer.transform(nan).astype(np.float32)


Xtr_nan, Xtr_imp = mats(train)
Xval_nan, Xval_imp = mats(val_1441)
Xn_nan, Xn_imp = mats(test_natural)
Xb_nan, Xb_imp = mats(test_1441)
yval = val_1441[TARGET].to_numpy().astype(np.int8)
y_nat = test_natural[TARGET].to_numpy().astype(np.int8)
y_bal = test_1441[TARGET].to_numpy().astype(np.int8)


def best_threshold(y, p):
    from src.evaluation.metrics import _mcc
    cands = np.unique(np.clip(np.quantile(p, np.linspace(0.5, 0.999, 200)), 1e-6, 1 - 1e-6))
    bt, bm = 0.5, -2.0
    for t in cands:
        m = _mcc(y.astype(np.int64), (p >= t).astype(np.int64))
        if m > bm:
            bm, bt = m, float(t)
    return bt


preds = {}   # model -> dict(val, nat, bal)
for name in ZOO:
    native = name in NATIVE_NULL
    extra = {'n_estimators': 150} if name in {'random_forest', 'balanced_random_forest'} else {}
    w = build_wrapper(name, random_state=SEED, **extra).fit(Xtr_nan if native else Xtr_imp, ytr)
    preds[name] = {
        'val': w.predict_proba(Xval_nan if native else Xval_imp),
        'nat': w.predict_proba(Xn_nan if native else Xn_imp),
        'bal': w.predict_proba(Xb_nan if native else Xb_imp),
    }
    print(f"  fitted {name}")

# Soft-voting stack of all four learners (simple mean of probabilities).
preds['soft_voting_stack'] = {
    k: np.mean(np.vstack([preds[n][k] for n in ZOO]), axis=0) for k in ('val', 'nat', 'bal')
}

# %% [markdown]
# ## 4. Report MCC / PR-AUC / F1 at both prevalences
#
# The threshold is tuned once per model on the 2022 14:1 validation. The 14:1 test
# reproduces the prevailing protocol; the natural test shows the same model and
# pipeline at the deployment base rate. PR-AUC is threshold-free.

# %%
rows = []
for name, p in preds.items():
    thr = best_threshold(yval, p['val'])
    for ratio_tag, y, score in (
        (f'{PROTOCOL_RATIO}:1', y_bal, p['bal']),
        ('natural', y_nat, p['nat']),
    ):
        ypred = (score >= thr).astype(np.int8)
        for metric, res in (
            ('mcc', mcc_with_ci(y, ypred, n_boot=NBOOT, seed=SEED)),
            ('f1', f1_with_ci(y, ypred, n_boot=NBOOT, seed=SEED)),
            ('pr_auc', pr_auc_with_ci(y, score, n_boot=NBOOT, seed=SEED)),
        ):
            pt, lo, hi = res
            rows.append({
                'model': BRIDGE_MODEL, 'learner': name, 'metric': metric,
                'value': pt, 'ci_low': lo, 'ci_high': hi,
                'prevalence_ratio': ratio_tag, 'threshold': thr,
            })

# %% [markdown]
# ## 4b. In-era random-split 14:1 (the prevailing literature protocol)
#
# The evaluation above keeps a strict temporal split (train 2021-and-earlier,
# test 2023-2025), which for a retired model means testing on an aged, shifted
# survivor population. Prior single-model studies instead use a random split
# within the model's active period at the same 14:1 ratio. This block reproduces
# that protocol on the same pipeline, so the residual gap to published numbers is
# attributable to in-era random splitting rather than to the model or the code.
# The random split is deliberately not leakage-strict; it exists only to
# characterize the prevailing protocol, which the fleet analysis otherwise avoids.

# %%
ie = ws_model.sample(fraction=1.0, shuffle=True, seed=SEED)
n = ie.height
i_tr, i_val = int(n * 0.70), int(n * 0.85)
ie_train = rebalance(ie.head(i_tr), TARGET, PROTOCOL_RATIO, SEED)
ie_val = rebalance(ie.slice(i_tr, i_val - i_tr), TARGET, PROTOCOL_RATIO, SEED)
ie_test = rebalance(ie.slice(i_val, n - i_val), TARGET, PROTOCOL_RATIO, SEED)
print(f"  in-era train {ie_train.height:,}, val {ie_val.height:,}, test {ie_test.height:,} "
      f"({100 * ie_test[TARGET].mean():.2f}% pos)")

ie_imputer = SimpleImputer(strategy='median').fit(ie_train.select(FEATURES).to_numpy().astype(np.float32))


def ie_mats(df):
    nan = df.select(FEATURES).to_numpy().astype(np.float32)
    return nan, ie_imputer.transform(nan).astype(np.float32)


IEtr_nan, IEtr_imp = ie_mats(ie_train)
IEval_nan, IEval_imp = ie_mats(ie_val)
IEte_nan, IEte_imp = ie_mats(ie_test)
ie_ytr = ie_train[TARGET].to_numpy().astype(np.int8)
ie_yval = ie_val[TARGET].to_numpy().astype(np.int8)
ie_yte = ie_test[TARGET].to_numpy().astype(np.int8)

ie_preds = {}
for name in ZOO:
    native = name in NATIVE_NULL
    extra = {'n_estimators': 150} if name in {'random_forest', 'balanced_random_forest'} else {}
    w = build_wrapper(name, random_state=SEED, **extra).fit(IEtr_nan if native else IEtr_imp, ie_ytr)
    ie_preds[name] = {
        'val': w.predict_proba(IEval_nan if native else IEval_imp),
        'te': w.predict_proba(IEte_nan if native else IEte_imp),
    }
ie_preds['soft_voting_stack'] = {
    k: np.mean(np.vstack([ie_preds[nm][k] for nm in ZOO]), axis=0) for k in ('val', 'te')
}

for name, p in ie_preds.items():
    thr = best_threshold(ie_yval, p['val'])
    ypred = (p['te'] >= thr).astype(np.int8)
    for metric, res in (
        ('mcc', mcc_with_ci(ie_yte, ypred, n_boot=NBOOT, seed=SEED)),
        ('f1', f1_with_ci(ie_yte, ypred, n_boot=NBOOT, seed=SEED)),
        ('pr_auc', pr_auc_with_ci(ie_yte, p['te'], n_boot=NBOOT, seed=SEED)),
    ):
        pt, lo, hi = res
        rows.append({
            'model': BRIDGE_MODEL, 'learner': name, 'metric': metric,
            'value': pt, 'ci_low': lo, 'ci_high': hi,
            'prevalence_ratio': f'{PROTOCOL_RATIO}:1_in_era', 'threshold': thr,
        })

# %% [markdown]
# ## 4c. Save and report all protocols

# %%
bridge = pl.DataFrame(rows)
bridge.write_csv(TABLES_DIR / 'rq1_backblaze_bridge.csv')
print(f"Saved {TABLES_DIR / 'rq1_backblaze_bridge.csv'}")

print(f"\nSingle model {BRIDGE_MODEL}, {HORIZON}-day horizon (MCC and PR-AUC):")
print(bridge.filter(pl.col('metric').is_in(['mcc', 'pr_auc']))
      .sort(['metric', 'prevalence_ratio', 'value'], descending=[False, False, True])
      .to_pandas().to_string(index=False))

# %% [markdown]
# ## 5. Summary
#
# The contrast to read: MCC and PR-AUC at the 14:1 protocol versus at natural
# prevalence for the identical model and pipeline, and the further drop to the
# fleet-wide natural result in notebook 12. Together these separate the two
# sources of apparent performance in prior work: test-set balancing and
# single-model selection.

# %%
stack = bridge.filter(pl.col('learner') == 'soft_voting_stack')
print("BACKBLAZE RQ1 BRIDGE SUMMARY (soft-voting stack)")
print("=" * 68)
for ratio_tag, label in (
    (f'{PROTOCOL_RATIO}:1_in_era', 'in-era random split, 14:1 test (literature protocol)'),
    (f'{PROTOCOL_RATIO}:1', 'temporal split, 14:1 test'),
    ('natural', 'temporal split, natural prevalence'),
):
    sub = stack.filter(pl.col('prevalence_ratio') == ratio_tag)
    mcc = sub.filter(pl.col('metric') == 'mcc')['value'][0]
    prc = sub.filter(pl.col('metric') == 'pr_auc')['value'][0]
    print(f"  MCC {mcc:.4f}  PR-AUC {prc:.4f}  {label}")
print("  (compare to fleet-wide natural prevalence in notebook 12: best MCC ~0.20)")
print("=" * 68)
print("=" * 60)
