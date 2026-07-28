# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Feature ablation and working-set sensitivity (Backblaze)
#
# **Purpose.** Two committed robustness deliverables for the Backblaze failure
# prediction at the 14-day horizon, both scored at natural prevalence on the same
# 2023-2025 test.
#
# 1. **Feature ablation.** Hold the learner and its hyperparameters fixed and vary
#    only the feature set, nesting the tiers from `src/features/backblaze_smart.py`:
#    the raw SMART readings plus the Tier 1 zero-inflation, onset-timing,
#    manufacturer and capacity signals; then adding the Tier 2 rolling, quantile and
#    rate-of-change dynamics; then adding the Tier 3 drive and fleet age and era
#    features; then adding the leakage-safe drive-model prior. The last step isolates
#    the prior as its own arm, since it was found to lift MCC through the operating
#    point while leaving PR-AUC flat.
# 2. **Working-set sensitivity.** Re-train the frozen 14-day soft-voting stack on the
#    10x, 20x and 40x training working sets and score each on the identical
#    natural-prevalence test. The working-set ratio is a training-side knob only, so
#    the robustness question is whether the choice moves the headline. The decision
#    rule is a paired bootstrap: the 95 percent interval of the (branch minus 20x)
#    MCC difference straddling zero means the choice does not bias the result.
#
# Everything is scored at natural prevalence, consistent with the evaluation
# protocol. The undersampled working set is a training-side stream only.
#
# **Outputs.** `outputs/tables/backblaze_feature_ablation.csv` and
# `outputs/tables/sensitivity_analyses.csv`.

# %% [markdown]
# ## 0. Session setup

# %%
# !pip install -q "numpy<2.3" polars pandas pyarrow scikit-learn xgboost lightgbm imbalanced-learn matplotlib google-cloud-storage

# %%
import os
import sys
from pathlib import Path

from google.colab import userdata

GITHUB_PAT = userdata.get("GITHUB_PAT")
PROJECT_ID = userdata.get("GCP_PROJECT_ID")
REPO_OWNER = "NU-Academics"
REPO_NAME = "Dissertation-LebelN"
REPO_DIR = f"/content/{REPO_NAME}"
REPO_URL = f"https://{GITHUB_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git"

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
import gc
import json
import time
import warnings

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from google.colab import auth

from utils.colab_setup import setup_drive, OUTPUT_DIR
from src.models.ensemble import build_wrapper
from src.evaluation.metrics import _mcc, mcc_with_ci, f1_with_ci, pr_auc_with_ci

warnings.filterwarnings("ignore", category=UserWarning)
auth.authenticate_user()
setup_drive()

# Stale-module guard: mcc_with_ci must expose the replicate array added for the
# paired difference CI, or the sensitivity decision cannot be computed.
import inspect
assert "return_replicates" in inspect.signature(mcc_with_ci).parameters, \
    "stale src.evaluation.metrics; restart the runtime after git pull"

TABLES_DIR = OUTPUT_DIR / "tables"
FEATURES_DIR = OUTPUT_DIR / "features"
FIG_DIR = OUTPUT_DIR / "figures" / "backblaze_sensitivity"
for d in (TABLES_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
HORIZON = 14
TARGET = f"failure_within_{HORIZON}d"
HORIZON_TARGETS = [f"failure_within_{h}d" for h in (7, 14, 30)]
TE_TARGET = "failure_within_30d"          # horizon-agnostic drive-model prior
TE_SMOOTHING = 300.0
TRAIN_CAP_ROWS = 800_000
RF_N_ESTIMATORS = 150
NBOOT = 300
EVAL_SAMPLE_ROWS = 3_000_000              # shared natural-test sample (paired scoring)
CAL_SAMPLE_ROWS = 2_000_000               # 2022 natural-prevalence calibration sample
STACK_MEMBERS = ("xgboost", "lightgbm", "random_forest")   # the frozen 14d stack
NATIVE_NULL = {"lightgbm", "xgboost"}
MODEL_PRIOR = "model_prior"

GCS_BUCKET = f"{PROJECT_ID}-dissertation-data"
WORKING_PREFIXES = {"10x": "backblaze_features/working_set_10x",
                    "20x": "backblaze_features/working_set_20x",
                    "40x": "backblaze_features/working_set_40x"}
GCS_NATURAL_PREFIX = "backblaze_features/natural_test_2023_2025"
GCS_NATURAL_VAL_PREFIX = "backblaze_features/natural_val_2022"

# %% [markdown]
# ## 1. Feature columns and tier partition
#
# The feature list matches the RQ1 model (base features plus the drive-model prior).
# Each feature is assigned a tier by the same rules used for the SHAP tier analysis,
# grounded in `src/features/backblaze_smart.py`. Anything the rules do not recognize
# is folded into the base arm rather than dropped, and printed so it can be checked.

# %%
SCHEMA = json.loads((FEATURES_DIR / "backblaze_feature_schema.json").read_text())
ALL_COLUMNS = SCHEMA["columns"]
EXCLUDE_ALWAYS = {
    "date", "serial_number", "model", "model_canonical", "manufacturer", "era",
    "failure", "failure_observed", "censored", "is_last_obs", "year", "fleet_age_days",
}
BASE_FEATURES = [c for c in ALL_COLUMNS
                 if c not in EXCLUDE_ALWAYS and c not in HORIZON_TARGETS]
FEATURES = BASE_FEATURES + [MODEL_PRIOR]
assert MODEL_PRIOR not in BASE_FEATURES
print(f"BASE_FEATURES: {len(BASE_FEATURES)} | FEATURES (with prior): {len(FEATURES)}")


def backblaze_tier(feature: str) -> str:
    low = feature.lower()
    if low == MODEL_PRIOR:
        return "prior"
    if ("days_since_first_nonzero" in low or low.startswith("has_nonzero_smart")
            or low.startswith("is_mfr_") or low.startswith("capacity")):
        return "tier1"
    if any(k in low for k in ("rollmean", "rollp95", "rollp99", "rollstd", "_delta_")):
        return "tier2"
    if low in ("drive_age_days", "month", "quarter") or low.startswith("era_smart"):
        return "tier3"
    if low.startswith("smart_") and low.endswith("_raw"):
        return "raw_smart"
    return "other"


tier_of = {c: backblaze_tier(c) for c in FEATURES}
by_tier = {t: [c for c in FEATURES if tier_of[c] == t]
           for t in ("raw_smart", "tier1", "tier2", "tier3", "prior", "other")}
for t, cols in by_tier.items():
    print(f"  {t:9s}: {len(cols)}")
if by_tier["other"]:
    print(f"  unrecognized (folded into base arm): {by_tier['other']}")

# Nested arms. The base arm is raw readings plus Tier 1 and anything unrecognized;
# each later arm adds one tier; the last adds the drive-model prior.
BASE_ARM = by_tier["raw_smart"] + by_tier["tier1"] + by_tier["other"]
ABLATION_SETS = {
    "tier1": BASE_ARM,
    "tier1_tier2": BASE_ARM + by_tier["tier2"],
    "all_tiers": BASE_ARM + by_tier["tier2"] + by_tier["tier3"],
    "all_tiers_plus_prior": BASE_ARM + by_tier["tier2"] + by_tier["tier3"] + by_tier["prior"],
}
assert set(ABLATION_SETS["all_tiers_plus_prior"]) == set(FEATURES), "arm partition lost a feature"
for name, cols in ABLATION_SETS.items():
    print(f"  arm {name:22s} {len(cols)} features")

# %% [markdown]
# ## 2. Data loaders
#
# The natural-prevalence test and the 2022 calibration set are each uniformly
# sampled once and reused, so every arm and every working set scores the identical
# rows and the paired difference is well defined. A training pool is built one
# bucket file at a time (all positives up to the cap budget, negatives sampled to
# the budget), so the 40x working set never fully materializes.

# %%
from google.cloud import storage

gcs = storage.Client(project=PROJECT_ID)
bucket = gcs.bucket(GCS_BUCKET)


def sync_prefix(prefix: str, local_dir: Path) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".parquet"):
            continue
        dest = local_dir / blob.name.split("/")[-1]
        if not (dest.exists() and dest.stat().st_size == blob.size):
            blob.download_to_filename(str(dest))
        out.append(dest)
    return sorted(out)


def uniform_sample(files: list[Path], cols: list[str], n_rows: int, seed: int) -> pl.DataFrame:
    """Uniform prevalence-preserving sample across parquet files, memory-bounded by
    reading one file at a time and keeping a per-file fraction."""
    total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    frac = min(1.0, n_rows / max(total, 1))
    parts = [pl.read_parquet(f, columns=cols).sample(fraction=frac, seed=seed) for f in files]
    return pl.concat(parts)


def build_train_pool(files: list[Path], cap: int, seed: int) -> pl.DataFrame:
    """All year<=2021 positives (14d) plus a RANDOM negative sample to the cap budget,
    matching cap_train in the RQ1 notebook (neg_budget = max(cap - n_pos, n_pos)).
    Read one bucket at a time so the full working set never materializes; negatives
    are sampled per file at the pooled fraction rather than taken by head(), so the
    draw is uniform over the whole pool and not biased by bucket order."""
    read_cols = [c for c in BASE_FEATURES] + HORIZON_TARGETS + ["year", "model_canonical"]
    keep_2021 = pl.col("year") <= 2021
    pos = (pl.scan_parquet(files).select(read_cols)
           .filter(keep_2021 & (pl.col(TARGET) == 1)).collect())
    n_pos = pos.height
    neg_budget = max(cap - n_pos, n_pos)
    n_neg_total = (pl.scan_parquet(files).filter(keep_2021 & (pl.col(TARGET) == 0))
                   .select(pl.len()).collect().item())
    frac = min(1.0, neg_budget / max(n_neg_total, 1))
    neg_parts = [
        (pl.read_parquet(f, columns=read_cols)
         .filter(keep_2021 & (pl.col(TARGET) == 0)).sample(fraction=frac, seed=seed))
        for f in files
    ]
    neg = pl.concat(neg_parts)
    if neg.height > neg_budget:
        neg = neg.sample(n=neg_budget, seed=seed)
    pool = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)
    print(f"    train pool: {n_pos:,} pos + {neg.height:,} neg = {pool.height:,} rows "
          f"(neg pool {n_neg_total:,}, frac {frac:.4f})")
    return pool


def fit_prior_full(files: list[Path]) -> tuple[pl.DataFrame, float]:
    """Leakage-safe smoothed drive-model failure-rate prior, fit on the FULL year<=2021
    working set (a cheap two-column scan), matching the RQ1 notebook. Fitting on the
    class-capped training pool instead would distort the base rate and, since the
    prior is the strongest feature, materially depress the model."""
    agg = (pl.scan_parquet(files).select(["model_canonical", TE_TARGET, "year"])
           .filter(pl.col("year") <= 2021)
           .group_by("model_canonical")
           .agg(pl.len().alias("n"), pl.col(TE_TARGET).sum().alias("pos"))
           .collect())
    global_rate = float(agg["pos"].sum() / max(int(agg["n"].sum()), 1))
    enc = (agg.with_columns(((pl.col("pos") + TE_SMOOTHING * global_rate)
                             / (pl.col("n") + TE_SMOOTHING)).cast(pl.Float32).alias(MODEL_PRIOR))
           .select("model_canonical", MODEL_PRIOR))
    print(f"    prior: {enc.height} models, global rate {global_rate:.5f}")
    return enc, global_rate


def attach_prior(df: pl.DataFrame, enc: pl.DataFrame, global_rate: float) -> pl.DataFrame:
    return (df.join(enc, on="model_canonical", how="left")
            .with_columns(pl.col(MODEL_PRIOR).fill_null(global_rate).cast(pl.Float32)))


# %%
# The shared evaluation and calibration samples (built once).
natural_files = sync_prefix(GCS_NATURAL_PREFIX, Path("/content/natural_test"))
val_files = sync_prefix(GCS_NATURAL_VAL_PREFIX, Path("/content/natural_val"))
assert natural_files and val_files, "natural test or 2022 validation missing"

read_cols_eval = BASE_FEATURES + HORIZON_TARGETS + ["model_canonical"]
natural_eval = uniform_sample(natural_files, read_cols_eval, EVAL_SAMPLE_ROWS, SEED)
nat_val = uniform_sample(val_files, read_cols_eval, CAL_SAMPLE_ROWS, SEED)
y_test = natural_eval[TARGET].to_numpy().astype(np.int8)
y_val = nat_val[TARGET].to_numpy().astype(np.int8)
print(f"natural test: {natural_eval.height:,} rows, {int(y_test.sum()):,} positive "
      f"({y_test.mean():.5f}); 2022 val: {nat_val.height:,} rows, {int(y_val.sum()):,} positive")

# %% [markdown]
# ## 3. Modeling helpers (matched to the RQ1 recipe)

# %%
def to_matrix(df: pl.DataFrame, cols: list[str], imputer) -> np.ndarray:
    arr = df.select(cols).to_numpy().astype(np.float32)
    return imputer.transform(arr).astype(np.float32) if imputer is not None else arr


def fit_calibrator(scores: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
    platt = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), y)
    b_iso = float(np.mean((np.clip(iso.predict(scores), 0, 1) - y) ** 2))
    b_platt = float(np.mean((platt.predict_proba(scores.reshape(-1, 1))[:, 1] - y) ** 2))
    return ("isotonic", iso) if b_iso <= b_platt else ("platt", platt)


def apply_calibrator(kind, obj, scores: np.ndarray) -> np.ndarray:
    if kind == "isotonic":
        return np.clip(obj.predict(scores), 0.0, 1.0)
    return obj.predict_proba(scores.reshape(-1, 1))[:, 1]


def best_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    cands = np.unique(np.clip(np.quantile(proba, np.linspace(0.50, 0.9999, 300)), 1e-6, 1 - 1e-6))
    best_t, best_m = 0.5, -2.0
    for t in cands:
        m = _mcc(y.astype(np.int64), (proba >= t).astype(np.int64))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def train_and_score_single(name, train_pool, enc, global_rate, cols):
    """Fit one calibrated learner on a feature set and return calibrated test proba
    plus the val-tuned threshold. Null-intolerant learners get a median imputer."""
    tr = attach_prior(train_pool, enc, global_rate) if MODEL_PRIOR in cols else train_pool
    va = attach_prior(nat_val, enc, global_rate) if MODEL_PRIOR in cols else nat_val
    te = attach_prior(natural_eval, enc, global_rate) if MODEL_PRIOR in cols else natural_eval
    ytr = tr[TARGET].to_numpy().astype(np.int8)
    native = name in NATIVE_NULL
    imputer = None if native else SimpleImputer(strategy="median").fit(
        tr.select(cols).to_numpy().astype(np.float32))
    extra = {"n_estimators": RF_N_ESTIMATORS} if name in ("random_forest", "balanced_random_forest") else {}
    wrapper = build_wrapper(name, random_state=SEED, **extra).fit(to_matrix(tr, cols, imputer), ytr)
    kind, calib = fit_calibrator(wrapper.predict_proba(to_matrix(va, cols, imputer)), y_val)
    thr = best_threshold(y_val, apply_calibrator(kind, calib, wrapper.predict_proba(to_matrix(va, cols, imputer))))
    test_cal = apply_calibrator(kind, calib, wrapper.predict_proba(to_matrix(te, cols, imputer)))
    return test_cal, thr


def train_and_score_stack(train_pool, enc, global_rate, cols):
    """Fit the fixed 3-member stack and return the mean calibrated test proba and the
    stack threshold tuned on the calibrated validation mean."""
    va = attach_prior(nat_val, enc, global_rate)
    val_cols, test_cols = [], []
    for name in STACK_MEMBERS:
        tr = attach_prior(train_pool, enc, global_rate)
        ytr = tr[TARGET].to_numpy().astype(np.int8)
        native = name in NATIVE_NULL
        imputer = None if native else SimpleImputer(strategy="median").fit(
            tr.select(cols).to_numpy().astype(np.float32))
        extra = {"n_estimators": RF_N_ESTIMATORS} if name == "random_forest" else {}
        wrapper = build_wrapper(name, random_state=SEED, **extra).fit(to_matrix(tr, cols, imputer), ytr)
        kind, calib = fit_calibrator(wrapper.predict_proba(to_matrix(va, cols, imputer)), y_val)
        val_cols.append(apply_calibrator(kind, calib, wrapper.predict_proba(to_matrix(va, cols, imputer))))
        te = attach_prior(natural_eval, enc, global_rate)
        test_cols.append(apply_calibrator(kind, calib, wrapper.predict_proba(to_matrix(te, cols, imputer))))
        del wrapper
        gc.collect()
    val_mean = np.mean(np.vstack(val_cols), axis=0)
    test_mean = np.mean(np.vstack(test_cols), axis=0)
    return test_mean, best_threshold(y_val, val_mean)


# %% [markdown]
# ## 4. Feature ablation on the 20x working set
#
# One LightGBM learner with identical hyperparameters per arm, NaN kept as signal,
# calibrated and thresholded on the 2022 natural-prevalence validation, scored on the
# natural test. The drive-model prior is added only in the last arm, so the
# all_tiers to all_tiers_plus_prior step is the prior's isolated contribution.

# %%
ws20_files = sync_prefix(WORKING_PREFIXES["20x"], Path("/content/working_20x"))
enc20, gr20 = fit_prior_full(ws20_files)
pool20 = build_train_pool(ws20_files, TRAIN_CAP_ROWS, SEED)

abl_rows = []
for arm, cols in ABLATION_SETS.items():
    test_cal, thr = train_and_score_single("lightgbm", pool20, enc20, gr20, cols)
    pred = (test_cal >= thr).astype(np.int8)
    mcc = mcc_with_ci(y_test, pred, n_boot=NBOOT, seed=SEED)
    f1 = f1_with_ci(y_test, pred, n_boot=NBOOT, seed=SEED)
    prauc = pr_auc_with_ci(y_test, test_cal, n_boot=NBOOT, seed=SEED)
    print(f"[{arm:22s}] {len(cols):3d} feat | MCC {mcc[0]:.4f} [{mcc[1]:.4f}, {mcc[2]:.4f}]"
          f" | PR-AUC {prauc[0]:.4f}")
    for metric, (v, lo, hi) in (("mcc", mcc), ("f1", f1), ("pr_auc", prauc)):
        abl_rows.append({"feature_set": arm, "n_features": len(cols), "metric": metric,
                         "value": round(v, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "threshold": round(thr, 4)})

ablation_df = pl.DataFrame(abl_rows)
ablation_df.write_csv(TABLES_DIR / "backblaze_feature_ablation.csv")
print("\nWrote backblaze_feature_ablation.csv")
print(ablation_df.pivot(values="value", index="feature_set", on="metric"))

# %% [markdown]
# ## 5. Working-set sensitivity
#
# The fixed 3-member stack is retrained on each working set over the full feature set
# and scored on the identical natural test. The 20x branch is the anchor; the delta
# and its paired bootstrap interval come from the same resample indices (the same
# seed over the same test labels), so the difference is the within-resample MCC gap.
# The choice does not bias the result when that interval straddles zero.

# %%
Z = 1.959964  # two-sided 95%


def mcc_reps(pred):
    _, _, _, reps = mcc_with_ci(y_test, pred, n_boot=NBOOT, seed=SEED, return_replicates=True)
    return reps


branch_pred, branch_point = {}, {}
for ratio, prefix in WORKING_PREFIXES.items():
    files = sync_prefix(prefix, Path(f"/content/working_{ratio}"))
    enc, gr = fit_prior_full(files)
    pool = build_train_pool(files, TRAIN_CAP_ROWS, SEED)
    test_cal, thr = train_and_score_stack(pool, enc, gr, FEATURES)
    pred = (test_cal >= thr).astype(np.int8)
    branch_pred[ratio] = pred
    branch_point[ratio] = float(_mcc(y_test.astype(np.int64), pred.astype(np.int64)))
    print(f"[working_set_{ratio}] stack MCC {branch_point[ratio]:.4f} (thr {thr:.4f})")
    del pool
    gc.collect()

anchor_reps = mcc_reps(branch_pred["20x"])
sens_rows = []
for ratio in WORKING_PREFIXES:
    pt, lo, hi = mcc_with_ci(y_test, branch_pred[ratio], n_boot=NBOOT, seed=SEED)
    if ratio == "20x":
        d, dlo, dhi, straddles = 0.0, 0.0, 0.0, True
    else:
        diff = mcc_reps(branch_pred[ratio]) - anchor_reps      # paired, same resamples
        d = branch_point[ratio] - branch_point["20x"]
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        straddles = bool(dlo <= 0.0 <= dhi)
    sens_rows.append({
        "branch": f"working_set_{ratio}", "dataset": "backblaze", "model": "soft_voting_stack",
        "horizon": HORIZON, "mcc": round(pt, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "delta_from_primary": round(d, 4), "delta_ci_low": round(float(dlo), 4),
        "delta_ci_high": round(float(dhi), 4), "difference_straddles_zero": straddles,
    })

sensitivity_df = pl.DataFrame(sens_rows)
sensitivity_df.write_csv(TABLES_DIR / "sensitivity_analyses.csv")
print("\nWrote sensitivity_analyses.csv")
print(sensitivity_df)
print("\nWorking-set choice does not bias the 14d result: "
      f"{all(r['difference_straddles_zero'] for r in sens_rows)}")

# %% [markdown]
# ## 6. Figures and run metadata

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAMP = time.strftime("%Y%m%dT%H%M%S")

mcc_arm = ablation_df.filter(pl.col("metric") == "mcc")
order = ["tier1", "tier1_tier2", "all_tiers", "all_tiers_plus_prior"]
mcc_arm = mcc_arm.with_columns(
    _o=pl.col("feature_set").replace_strict({n: i for i, n in enumerate(order)}, default=99)).sort("_o")
prauc_arm = (ablation_df.filter(pl.col("metric") == "pr_auc")
             .with_columns(_o=pl.col("feature_set").replace_strict(
                 {n: i for i, n in enumerate(order)}, default=99)).sort("_o"))
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - 0.2, mcc_arm["value"], 0.4, label="MCC",
       yerr=[mcc_arm["value"] - mcc_arm["ci_low"], mcc_arm["ci_high"] - mcc_arm["value"]], capsize=3)
ax.bar(x + 0.2, prauc_arm["value"], 0.4, label="PR-AUC",
       yerr=[prauc_arm["value"] - prauc_arm["ci_low"], prauc_arm["ci_high"] - prauc_arm["value"]], capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("test metric (95% CI)")
ax.set_title("Backblaze 14d feature-tier ablation (natural prevalence)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / f"ablation_{STAMP}.png", dpi=150)
plt.close(fig)

meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "horizon": HORIZON,
        "train_cap_rows": TRAIN_CAP_ROWS, "eval_sample_rows": int(natural_eval.height),
        "stack_members": list(STACK_MEMBERS), "n_boot": NBOOT,
        "ablation_arms": {k: len(v) for k, v in ABLATION_SETS.items()}}
(TABLES_DIR / f"backblaze_sensitivity_metadata_{STAMP}.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
