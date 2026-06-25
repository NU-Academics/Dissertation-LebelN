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
# # 18. Feature Ablation (Google Cluster Traces)
#
# **Purpose.** Quantify how much each feature tier contributes to failure
# prediction at the scheduled-episode grain, holding the learner and its
# hyperparameters fixed and varying only the feature set. This confirms the
# predictive-value hierarchy (V13) empirically and isolates how much of the
# early-runtime lift is the label-correlated missingness of Tier 2 (V09) rather than
# the slope magnitudes themselves.
#
# **Feature sets.**
# 1. *Tier 1 only* (strictly-prior history + scheduling + temporal): the
#    at-submission feature set carried by the RQ1 checkpoint sidecar.
# 2. *Tier 1 + Tier 2* (adds early-runtime slopes, ramps, first-interval ratio,
#    CPI/MAPI values).
# 3. *All tiers* (adds Tier 3 windowed utilization).
# 4. *Tier 2 missingness only* (the per-feature null indicators of the Tier 2
#    columns, nothing else), to isolate null-as-signal.
#
# **Protocol.** One LightGBM learner with identical hyperparameters
# (`LightGBMWrapper`, inverse-prior class weights). The instance-keyed group split
# matches the RQ1 notebook (buckets of a FARM_FINGERPRINT of the instance key: train
# below 14, validation [14, 17), test >= 17 of 20), so no instance straddles and the
# ablation sits on the validated RQ1 curve. The decision threshold is tuned for MCC
# on the validation split and applied to the test split; MCC, F1, and PR-AUC are
# reported with 95% stratified bootstrap CIs.
#
# **Output.** `outputs/tables/google_feature_ablation.csv`.

# %% [markdown]
# ## 0. Colab session setup
#
# The `sys.modules` purge drops previously imported repo modules so a `git pull`
# takes effect inside a running runtime. Restart the runtime if anything still looks
# stale.

# %%
# !pip install -q polars lightgbm scikit-learn pandas pyarrow google-cloud-bigquery-storage matplotlib

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

for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from utils.colab_setup import setup_drive, OUTPUT_DIR, CHECKPOINT_DIR

setup_drive()

# %%
import gc
import json
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef

from src.models.ensemble import LightGBMWrapper
from src.evaluation.metrics import f1_with_ci, mcc_with_ci, pr_auc_with_ci

RANDOM_SEED = 42
warnings.filterwarnings("ignore", category=UserWarning)

TABLES_DIR = OUTPUT_DIR / 'tables'
CACHE_DIR = OUTPUT_DIR / 'cache'
FIG_DIR = OUTPUT_DIR / 'figures' / 'google_feature_ablation'
for directory in (TABLES_DIR, CACHE_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ABLATION_CSV = TABLES_DIR / 'google_feature_ablation.csv'
LABEL_COL = "failure_label"

# %% [markdown]
# ## 1. Define the feature tiers
#
# Tier 1 is read from the RQ1 at-submission checkpoint sidecar so the ablation's
# Tier 1 is exactly the feature set behind the RQ1 at-submission result. Tier 2 and
# Tier 3 are the early-runtime and windowed-utilization columns of
# `episode_features`. The asserts fail loudly if a tier list drifts from the table
# schema or overlaps Tier 1 (a stale module or schema change would surface here).

# %%
with open(CHECKPOINT_DIR / "rq1_google_best_atsubmission.json") as fh:
    TIER1 = list(json.load(fh)["feature_columns"])

TIER2 = [
    "cpu_slope_5s", "cpu_slope_15s", "cpu_slope_30s",
    "memory_slope_5s", "memory_slope_15s", "memory_slope_30s",
    "initial_cpu_ramp", "initial_memory_ramp", "first_interval_util_ratio",
    "cpi_value", "mapi_value",
]
TIER3 = [
    "avg_cpu_5min", "max_cpu_5min", "std_cpu_5min",
    "avg_cpu_15min", "max_cpu_15min", "std_cpu_15min",
    "avg_cpu_60min", "max_cpu_60min", "std_cpu_60min",
    "avg_memory_5min", "max_memory_5min", "std_memory_5min",
    "avg_memory_15min", "max_memory_15min", "std_memory_15min",
    "avg_memory_60min", "max_memory_60min", "std_memory_60min",
]

# Tier 1 must be disjoint from Tier 2 / Tier 3 (Tier 1 is at-submission only).
assert not (set(TIER1) & set(TIER2 + TIER3)), "Tier 1 overlaps Tier 2/3; check the sidecar."
print(f"Tier 1: {len(TIER1)} | Tier 2: {len(TIER2)} | Tier 3: {len(TIER3)} features")

# Strictly-prior resubmission-history block within Tier 1 (the "failure begets
# failure" lever, V10/V31/V36). A history-free Tier 1 isolates the scheduling +
# temporal floor so the lift can be attributed to the history features.
HISTORY_COLS = [c for c in
                ("prior_fail_count", "prior_evict_count", "resubmission_count",
                 "has_prior_fail", "first_resubmission")
                if c in TIER1]
TIER1_NO_HISTORY = [c for c in TIER1 if c not in HISTORY_COLS]
assert HISTORY_COLS, "no history columns found in Tier 1; check the sidecar feature set."
print(f"History block: {len(HISTORY_COLS)} | Tier 1 without history: {len(TIER1_NO_HISTORY)}")

# %% [markdown]
# ## 2. Pull the instance-keyed group splits
#
# The split mirrors the RQ1 notebook: instances are bucketed by a FARM_FINGERPRINT
# of the instance key, train is buckets below 14, validation [14, 17), test >= 17 of
# 20. The train split keeps every positive plus a per-instance cap of real negatives
# (the recurring-failer tail control); validation and test keep the natural class
# distribution. A per-episode-key hash thins each split to bound Colab memory.

# %%
from google.colab import auth

from utils.bq_client import get_client, table_ref

auth.authenticate_user()
_bq = get_client()

N_BUCKETS = 20
TRAIN_BUCKET_MAX = 14
VAL_BUCKET_MAX = 17
NEG_CAP = 5
TRAIN_PERMILLE = 15
EVAL_PERMILLE = 20

_KEY = "CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING))"
_EPKEY = f"CONCAT({_KEY},'_',CAST(sched_seq AS STRING))"


def _episode_split_sql(split: str) -> str:
    if split == "train":
        grp_pred = f"_grp < {TRAIN_BUCKET_MAX}"
        permille = TRAIN_PERMILLE
        rn_col = (f",\n        ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index, "
                  f"{LABEL_COL} ORDER BY _ephash) AS _rn")
        cap_clause = f"WHERE {LABEL_COL} = 1 OR _rn <= {NEG_CAP}"
        drop_cols = "_grp, _ephash, _rn"
    elif split == "val":
        grp_pred = f"_grp >= {TRAIN_BUCKET_MAX} AND _grp < {VAL_BUCKET_MAX}"
        permille, rn_col, cap_clause, drop_cols = EVAL_PERMILLE, "", "", "_grp, _ephash"
    else:  # test
        grp_pred = f"_grp >= {VAL_BUCKET_MAX}"
        permille, rn_col, cap_clause, drop_cols = EVAL_PERMILLE, "", "", "_grp, _ephash"
    return f"""
WITH split AS (
    SELECT
        b.*,
        MOD(ABS(FARM_FINGERPRINT({_KEY})), {N_BUCKETS}) AS _grp,
        ABS(FARM_FINGERPRINT({_EPKEY})) AS _ephash
    FROM {table_ref('episode_features')} b
),
sided AS (
    SELECT *{rn_col}
    FROM split
    WHERE {grp_pred}
      AND MOD(_ephash, 1000) < {permille}
)
SELECT * EXCEPT({drop_cols})
FROM sided
{cap_clause}
"""


def _load_split(split: str) -> pl.DataFrame:
    cache = CACHE_DIR / f"ablation_{split}_p{TRAIN_PERMILLE if split == 'train' else EVAL_PERMILLE}.parquet"
    if cache.exists() and not REBUILD_SPLITS:
        df = pl.read_parquet(cache)
        print(f"Loaded {split}: {df.height:,} rows <- {cache.name}")
        return df
    arrow = _bq.query(_episode_split_sql(split)).to_arrow(create_bqstorage_client=True)
    df = pl.from_arrow(arrow)
    del arrow
    gc.collect()
    df.write_parquet(cache)
    print(f"Pulled {split}: {df.height:,} rows -> {cache.name}")
    return df


REBUILD_SPLITS = True   # set False to reload the splits from Drive
train_df = _load_split("train")
val_df = _load_split("val")
test_df = _load_split("test")

# Fail-loud guards: the label and every tier column must be present, and the group
# split must not let an instance straddle train and test.
for _name, _df in (("train", train_df), ("val", val_df), ("test", test_df)):
    missing = [c for c in (TIER1 + TIER2 + TIER3 + [LABEL_COL]) if c not in _df.columns]
    assert not missing, f"{_name} missing columns: {missing[:8]}"
_tr_keys = train_df.select(_inst := ["collection_id", "instance_index"]).unique()
_te_keys = test_df.select(_inst).unique()
_overlap = _tr_keys.join(_te_keys, on=_inst, how="inner").height
assert _overlap == 0, f"{_overlap} instances straddle train and test; split is leaking."
for _name, _df in (("train", train_df), ("val", val_df), ("test", test_df)):
    _p = float((_df[LABEL_COL] == 1).mean())
    print(f"  {_name}: {_df.height:,} rows | positive rate {_p:.4f}")

# %% [markdown]
# ## 3. Ablation
#
# Each feature set trains a fresh LightGBM with identical hyperparameters. LightGBM
# consumes Tier 2 / Tier 3 nulls natively (rapid-onset crashes emit no early usage,
# V09), so no imputation is applied. The threshold is tuned for MCC on the
# validation split and applied to the test split; PR-AUC is threshold-free.

# %%
FEATURE_SETS = {
    "tier1_no_history": TIER1_NO_HISTORY,
    "tier1": TIER1,
    "tier1_tier2": TIER1 + TIER2,
    "all_tiers": TIER1 + TIER2 + TIER3,
    "tier2_missingness_only": None,   # built as null indicators below
}


def _matrix(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Feature matrix for a feature set. ``tier2_missingness_only`` is the per-Tier-2
    null-indicator block; the others select the listed columns as-is (NaN passes
    through to LightGBM)."""
    if name == "tier2_missingness_only":
        return df.select([pl.col(c).is_null().cast(pl.Int8).alias(f"{c}__isnull") for c in TIER2])
    return df.select(FEATURE_SETS[name])


def _best_mcc_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_m = 0.5, -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, (scores >= t).astype(np.int8))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


val_y = val_df[LABEL_COL].to_numpy().astype(np.int64)
test_y = test_df[LABEL_COL].to_numpy().astype(np.int64)
train_y = train_df[LABEL_COL]

rows: list[dict] = []
for name in FEATURE_SETS:
    model = LightGBMWrapper(random_state=RANDOM_SEED)
    model.fit(_matrix(train_df, name), train_y)
    val_scores = model.predict_proba(_matrix(val_df, name))
    test_scores = model.predict_proba(_matrix(test_df, name))
    thr = _best_mcc_threshold(val_y, val_scores)
    test_pred = (test_scores >= thr).astype(np.int8)

    mcc = mcc_with_ci(test_y, test_pred, seed=RANDOM_SEED)
    f1 = f1_with_ci(test_y, test_pred, seed=RANDOM_SEED)
    prauc = pr_auc_with_ci(test_y, test_scores, seed=RANDOM_SEED)
    n_feat = test_df.pipe(_matrix, name).width

    print(f"\n[{name}] {n_feat} features | val-tuned threshold {thr:.4f}")
    print(f"  MCC    {mcc[0]:.4f} [{mcc[1]:.4f}, {mcc[2]:.4f}]")
    print(f"  F1     {f1[0]:.4f} [{f1[1]:.4f}, {f1[2]:.4f}]")
    print(f"  PR-AUC {prauc[0]:.4f} [{prauc[1]:.4f}, {prauc[2]:.4f}]")

    for metric, (v, lo, hi) in (("mcc", mcc), ("f1", f1), ("pr_auc", prauc)):
        rows.append({
            "feature_set": name, "n_features": n_feat, "metric": metric,
            "value": round(v, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "threshold": round(thr, 4),
        })
    del model
    gc.collect()

# %% [markdown]
# ## 4. Reporting
#
# One row per (feature set, metric) to `google_feature_ablation.csv`, plus an MCC
# bar chart. Reading (V10/V13/V31/V36): the history-free Tier 1 is the scheduling +
# temporal floor; adding the strictly-prior resubmission-history block is the
# dominant lever; early-runtime (Tier 2) and windowed utilization (Tier 3) then add
# incremental, monotonic value; the Tier 2 missingness-only model shows the null
# pattern alone is weakly predictive (V09), so most of the Tier 2 increment is the
# slope and ramp magnitudes rather than the mere presence of early usage.

# %%
ablation_df = pl.DataFrame(rows)
ablation_df.write_csv(str(ABLATION_CSV))
print(f"Wrote {ablation_df.height} rows -> {ABLATION_CSV}")
print(ablation_df.pivot(values="value", index="feature_set", on="metric"))

# %%
import matplotlib.pyplot as plt

mcc_rows = ablation_df.filter(pl.col("metric") == "mcc")
order = ["tier1_no_history", "tier1", "tier1_tier2", "all_tiers", "tier2_missingness_only"]
mcc_rows = mcc_rows.with_columns(
    _ord=pl.col("feature_set").replace_strict({n: i for i, n in enumerate(order)}, default=99)
).sort("_ord")
labels = mcc_rows["feature_set"].to_list()
vals = mcc_rows["value"].to_numpy()
err_lo = vals - mcc_rows["ci_low"].to_numpy()
err_hi = mcc_rows["ci_high"].to_numpy() - vals

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(range(len(labels)), vals, yerr=[err_lo, err_hi], capsize=4, color="#3a7ca5")
ax.axhline(0.90, color="#c44", lw=1, ls="--", label="RQ1 0.90 reference")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("test MCC (95% CI)")
ax.set_title("Feature-tier ablation (episode grain, RQ1 group split)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "mcc_by_feature_set.png", dpi=150)
plt.close(fig)
print(f"Saved {FIG_DIR / 'mcc_by_feature_set.png'}")
