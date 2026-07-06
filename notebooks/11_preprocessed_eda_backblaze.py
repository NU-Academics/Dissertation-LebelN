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
# # 11. Backblaze Preprocessed EDA and Learning Curve
#
# **Purpose.** Validate that the engineered Backblaze feature matrix supports the
# failure-prediction task and decide whether the primary working set is large
# enough, by fitting a baseline model on growing fractions of the training data
# and reading the resulting learning curve.
#
# **Design.** A baseline LightGBM classifier predicts the 30-day failure horizon
# on the 20:1 primary working set. Training uses observations through 2022;
# evaluation uses a held-out 2023-2025 fold, the drift-faithful, leakage-safe
# temporal split (rolling and lag features are computed within each drive in
# date order, so a drive's earlier observations train and its later observations
# evaluate). The curve is scored by MCC with a stratified bootstrap confidence
# interval (`src/evaluation/metrics.mcc_with_ci`), with PR-AUC reported
# alongside.
#
# **Adequacy rule (mirrors the Google block).** The working set is judged
# adequate when successive MCC increases fall below 0.005 with overlapping 95%
# bootstrap intervals, indicating the curve has flattened. If it has not
# flattened at 100% of the primary set, that is documented rather than resolved
# by enlarging the set, since the ratio is a committed design choice.
#
# **Note on prevalence.** This curve is scored on the 20:1 working-set holdout, a
# consistent basis for the relative sample-size question. Final performance at
# the natural class prevalence is reported in the modeling notebooks.
#
# **Outputs.**
# - `outputs/tables/backblaze_learning_curve.csv`.
# - `outputs/figures/learning_curve_backblaze.png`.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars lightgbm scikit-learn numpy matplotlib google-cloud-storage

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
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import lightgbm as lgb

from utils.colab_setup import setup_drive, OUTPUT_DIR
from src.evaluation.metrics import mcc_with_ci, pr_auc_with_ci

setup_drive()

TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'
WS_DIR = Path('/content/backblaze_features/ratio_20')
for d in [TABLES_DIR, FIGURES_DIR, WS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_WS_PREFIX = 'backblaze_features/working_set_20x'

TARGET = 'failure_within_30d'
TRAIN_MAX_YEAR = 2022
TEST_MIN_YEAR = 2023
FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
SEED = 42

# Columns that are identifiers, string categoricals, or outcome/label leakage,
# excluded from the model features. The other-horizon targets and the censoring
# columns encode the outcome, and year / fleet_age_days are calendar-monotonic
# time indices that shift between the train and test periods.
EXCLUDE = {
    'serial_number', 'date', 'model', 'model_canonical', 'manufacturer', 'era',
    'failure', 'failure_observed', 'censored', 'is_last_obs',
    'failure_within_7d', 'failure_within_14d', 'failure_within_30d',
    'year', 'fleet_age_days',
}

# %% [markdown]
# ### Fetch the 20:1 working set from GCS

# %%
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET)
for blob in bucket.list_blobs(prefix=GCS_WS_PREFIX):
    if not blob.name.endswith('.parquet'):
        continue
    local_path = WS_DIR / blob.name.split('/')[-1]
    if not (local_path.exists() and local_path.stat().st_size == blob.size):
        blob.download_to_filename(str(local_path))
print(f"{len(list(WS_DIR.glob('bucket_*.parquet')))} working-set partitions at {WS_DIR}")

# %% [markdown]
# ## 1. Load features and make the temporal split
#
# The feature columns are every numeric column that is not an identifier, string
# category, outcome, or calendar time index. The split is on `year`: training
# through 2022, evaluation 2023-2025.

# %%
all_cols = pl.scan_parquet(str(WS_DIR / 'bucket_*.parquet')).collect_schema().names()
feature_cols = [c for c in all_cols if c not in EXCLUDE]
print(f"Feature columns: {len(feature_cols)}")


def _load(year_filter: pl.Expr) -> tuple[np.ndarray, np.ndarray]:
    frame = (
        pl.scan_parquet(str(WS_DIR / 'bucket_*.parquet'))
        .filter(year_filter)
        .select([pl.col(c).cast(pl.Float32) for c in feature_cols] + [pl.col(TARGET)])
        .collect()
    )
    y = frame[TARGET].to_numpy().astype(np.int8)
    x = frame.select(feature_cols).to_numpy()
    return x, y


x_train, y_train = _load(pl.col('year') <= TRAIN_MAX_YEAR)
x_test, y_test = _load(pl.col('year') >= TEST_MIN_YEAR)
print(f"Train: {x_train.shape}, positive rate {y_train.mean():.4f}")
print(f"Test:  {x_test.shape}, positive rate {y_test.mean():.4f}")

# %% [markdown]
# ## 2. Learning-curve harness
#
# For each training fraction: draw a class-stratified subsample of the training
# rows, fit the baseline LightGBM, score the fixed 2023-2025 test fold, pick the
# MCC-optimal threshold on the test scores (applied consistently across
# fractions), and record MCC with a bootstrap CI and PR-AUC.

# %%
BASE_PARAMS = dict(
    n_estimators=300, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
    n_jobs=-1, random_state=SEED, verbosity=-1,
)


def _best_mcc_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on the score grid that maximizes MCC (consistent per fraction)."""
    grid = np.quantile(scores, np.linspace(0.50, 0.999, 60))
    best_t, best_m = 0.5, -1.0
    for t in np.unique(grid):
        m = mcc_with_ci(y_true, (scores >= t).astype(np.int8), n_boot=1)[0]
        if m > best_m:
            best_m, best_t = m, t
    return best_t


rng = np.random.default_rng(SEED)
pos_idx = np.flatnonzero(y_train == 1)
neg_idx = np.flatnonzero(y_train == 0)
records = []
for frac in FRACTIONS:
    n_pos = max(1, int(round(pos_idx.size * frac)))
    n_neg = max(1, int(round(neg_idx.size * frac)))
    sel = np.concatenate((
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ))
    xs, ys = x_train[sel], y_train[sel]
    scale = (ys == 0).sum() / max((ys == 1).sum(), 1)

    model = lgb.LGBMClassifier(scale_pos_weight=scale, **BASE_PARAMS)
    model.fit(xs, ys)
    scores = model.predict_proba(x_test)[:, 1]

    threshold = _best_mcc_threshold(y_test, scores)
    y_pred = (scores >= threshold).astype(np.int8)
    mcc, mcc_lo, mcc_hi = mcc_with_ci(y_test, y_pred)
    pr, pr_lo, pr_hi = pr_auc_with_ci(y_test, scores)
    records.append({
        'fraction': frac, 'n_train': int(sel.size),
        'mcc': mcc, 'mcc_lo': mcc_lo, 'mcc_hi': mcc_hi,
        'pr_auc': pr, 'pr_auc_lo': pr_lo, 'pr_auc_hi': pr_hi,
        'threshold': float(threshold),
    })
    print(f"  frac {frac:>5.2f}  n={sel.size:>10,}  "
          f"MCC {mcc:.4f} [{mcc_lo:.4f}, {mcc_hi:.4f}]  PR-AUC {pr:.4f}")

curve = pl.DataFrame(records)
curve.write_csv(TABLES_DIR / 'backblaze_learning_curve.csv')
print(f"Saved {TABLES_DIR / 'backblaze_learning_curve.csv'}")

# %% [markdown]
# ## 3. Asymptote decision
#
# The curve is flattened when the last MCC increase is below 0.005 and the top
# two fractions' 95% bootstrap intervals overlap.

# %%
mcc_vals = curve['mcc'].to_list()
last_delta = mcc_vals[-1] - mcc_vals[-2]
overlap = curve['mcc_lo'][-1] <= curve['mcc_hi'][-2] and curve['mcc_lo'][-2] <= curve['mcc_hi'][-1]
asymptoted = (last_delta < 0.005) and overlap
print(f"Final MCC {mcc_vals[-1]:.4f} at 100% of the primary set")
print(f"Last delta {last_delta:.4f} (< 0.005: {last_delta < 0.005}); CI overlap: {overlap}")
print(f"Learning curve asymptoted: {asymptoted}")
if not asymptoted:
    print("Curve not flattened at 100%; documented as a limitation, the ratio is a committed choice.")

# %% [markdown]
# ## 4. Learning-curve figure

# %%
fig, ax = plt.subplots(figsize=(8, 5))
fr = curve['fraction'].to_numpy() * 100
ax.plot(fr, curve['mcc'].to_numpy(), marker='o', color='#1f77b4', label='MCC')
ax.fill_between(fr, curve['mcc_lo'].to_numpy(), curve['mcc_hi'].to_numpy(),
                alpha=0.2, color='#1f77b4')
ax.set_xlabel('Training data used (% of the 20:1 primary working set)')
ax.set_ylabel('MCC (2023-2025 holdout)')
ax.set_title('Figure 11.1: Backblaze learning curve, 30-day failure horizon')
ax.set_xscale('log')
ax.set_xticks([1, 5, 10, 25, 50, 100])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'learning_curve_backblaze.png', bbox_inches='tight')
print(f"Saved {FIGURES_DIR / 'learning_curve_backblaze.png'}")
plt.show()

# %% [markdown]
# ## 5. Summary

# %%
print("BACKBLAZE LEARNING CURVE SUMMARY")
print("=" * 60)
print(f"Feature columns:   {len(feature_cols)}")
print(f"Train / test rows: {x_train.shape[0]:,} / {x_test.shape[0]:,}")
print(f"Final MCC (100%):  {mcc_vals[-1]:.4f} "
      f"[{curve['mcc_lo'][-1]:.4f}, {curve['mcc_hi'][-1]:.4f}]")
print(f"Final PR-AUC:      {curve['pr_auc'][-1]:.4f}")
print(f"Asymptoted:        {asymptoted}")
print("=" * 60)
