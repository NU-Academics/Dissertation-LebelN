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
# # SHAP feature attribution for the failure-prediction ensembles
#
# **Purpose.** Explain the two failure-prediction models that carry the primary
# contribution: the Google early-runtime ensemble (the designated prediction point)
# and the Backblaze 14-day model. Produce a global beeswarm, a top-20 mean absolute
# SHAP bar chart, and dependence plots for the leading features, then test whether
# the empirical ranking matches the theoretical feature-tier ordering.
#
# **This loads frozen checkpoints; it does not refit.** Both checkpoints were
# written by their modeling notebooks. Nothing here changes a fitted model or a
# reported metric; SHAP is computed on the frozen estimators over a sample of the
# same test data.
#
# **Two checkpoint shapes, handled explicitly.**
#
# - The Google early-runtime checkpoint is a single fitted pipeline: a median
#   imputer with an added missing-indicator, a fit-time resampler that is inert at
#   predict, and one tree classifier. SHAP runs on the classifier over the imputed
#   matrix, and the missing-indicator columns are carried as their own features
#   because null-as-signal is a legitimate observable at this prediction point.
# - The Backblaze 14-day checkpoint is a soft-voting stack whose payload holds three
#   fitted members (xgboost, lightgbm, random_forest), each with its own imputer and
#   calibrator. A tree explainer takes one model, so SHAP is run per member. The
#   global bar is the stack view, formed by normalizing each member's mean absolute
#   SHAP to sum to one and averaging with the stack's equal weights, which makes the
#   three members comparable despite their different native output scales. The
#   beeswarms are per member and labeled as such, since a single beeswarm across
#   three output scales is not coherent. No member is presented as the stack.
#
# **Watch the drive-model prior.** The Backblaze feature matrix includes the
# leakage-safe drive-model failure-rate prior. If it ranks at or near the top, much
# of the discrimination is drive-model identity rather than SMART degradation, which
# is a substantive point and connects to the single-model-study decomposition.

# %% [markdown]
# ## 0. Session setup
#
# Install pins before loading any checkpoint: a newer numpy can break the unpickle
# of the saved ensembles, so the numpy ceiling the ensembles were saved under is
# held here as well.

# %%
# !pip install -q "numpy<2.3" polars pandas pyarrow scikit-learn "xgboost" "lightgbm" imbalanced-learn shap matplotlib google-cloud-storage

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IN_COLAB:
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
else:
    REPO_DIR = str(Path(__file__).resolve().parents[1]) if "__file__" in globals() else "."
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %%
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

# Assert on a symbol the current environment must have, so a stale runtime fails
# loudly rather than mid-run.
assert hasattr(shap, "TreeExplainer"), "shap missing TreeExplainer; check the install"

# The wrapper class must import so the pickled Backblaze members unpickle.
from src.models.ensemble import SoftVotingStack, _BaseEnsembleWrapper  # noqa: F401

if IN_COLAB:
    from google.colab import auth

    auth.authenticate_user()
    from utils.colab_setup import CHECKPOINT_DIR, OUTPUT_DIR, setup_drive

    setup_drive()
else:
    OUTPUT_DIR = Path(REPO_DIR) / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

TABLES_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"
FIG_G = OUTPUT_DIR / "figures" / "shap" / "rq1_google"
FIG_B = OUTPUT_DIR / "figures" / "shap" / "rq1_backblaze"
for d in (TABLES_DIR, FIG_G, FIG_B):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
STAMP = time.strftime("%Y%m%dT%H%M%S")
# Exact TreeSHAP on a 300-tree unpruned forest scales with trees x leaves x depth^2,
# so the sample size below is kept modest: a global mean(|SHAP|) ranking stabilizes
# at a few thousand rows, and the Saabas approximation (SHAP_APPROX) avoids the exact
# blow-up on the random-forest members while leaving the ranking essentially unchanged.
SHAP_ROWS = 3_000             # explanation sample size (global ranking is stable here)
SHAP_APPROX = True            # fast Saabas path for the tree explainers
run_meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seed": SEED, "shap_rows_target": SHAP_ROWS, "shap_approximate": SHAP_APPROX,
            "shap_version": shap.__version__, "numpy_version": np.__version__}
print(f"Checkpoints: {CHECKPOINT_DIR}\nFigures: {FIG_G.parent}")

# %% [markdown]
# ## 1. Shared helpers

# %%
def class1_shap(sv) -> np.ndarray:
    """Positive-class SHAP values as a 2-D (n, n_features) array, across shap
    versions: a list per class, a 3-D (n, features, classes) array, or already 2-D."""
    if isinstance(sv, list):
        return np.asarray(sv[1] if len(sv) > 1 else sv[0])
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def explain_tree(est, X: np.ndarray, label: str) -> np.ndarray:
    """Positive-class TreeSHAP values with per-model timing.

    Uses the fast Saabas approximation when ``SHAP_APPROX`` is set, which is
    adequate for a global ranking and avoids the exact TreeSHAP cost on deep
    unpruned forests. Falls back to exact if a model rejects the approximate path."""
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(est)
    try:
        sv = explainer.shap_values(X, approximate=SHAP_APPROX)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] approximate path unavailable ({type(exc).__name__}); using exact")
        sv = explainer.shap_values(X)
    out = class1_shap(sv)
    print(f"  [{label}] SHAP on {X.shape[0]:,} x {X.shape[1]} features in "
          f"{time.perf_counter() - t0:.1f}s")
    return out


def mean_abs_importance(shap_2d: np.ndarray, names: list[str]) -> pl.DataFrame:
    """Global mean(|SHAP|) per feature, sorted descending."""
    m = np.abs(shap_2d).mean(axis=0)
    return (pl.DataFrame({"feature": names, "mean_abs_shap": m.astype(np.float64)})
            .sort("mean_abs_shap", descending=True))


def save_bar(imp: pl.DataFrame, title: str, path: Path, top: int = 20) -> None:
    top_df = imp.head(top).reverse()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_df["feature"].to_list(), top_df["mean_abs_shap"].to_list())
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_beeswarm(shap_2d: np.ndarray, X: np.ndarray, names: list[str],
                  title: str, path: Path, top: int = 20) -> None:
    plt.figure()
    shap.summary_plot(shap_2d, X, feature_names=names, max_display=top, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_dependence(shap_2d: np.ndarray, X: np.ndarray, names: list[str],
                    feature: str, path: Path) -> None:
    plt.figure()
    shap.dependence_plot(feature, shap_2d, X, feature_names=names,
                         interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def stratified_indices(y: np.ndarray, n: int, seed: int, proportional: bool = True,
                       min_pos: int = 1000) -> np.ndarray:
    """Row indices for a SHAP sample. Proportional preserves the label balance
    (used where positives are common). Non-proportional enriches the positive class
    up to ``min_pos`` (used where positives are too rare for a proportional sample
    to carry any), which is documented as a class-enriched explanation sample."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if n >= y.size:
        return np.arange(y.size)
    if proportional:
        k_pos = max(int(round(n * pos.size / y.size)), min(min_pos, pos.size))
    else:
        k_pos = min(pos.size, max(min_pos, int(round(n * 0.2))))
    k_pos = min(k_pos, pos.size)
    k_neg = min(neg.size, n - k_pos)
    idx = np.concatenate([rng.choice(pos, k_pos, replace=False),
                          rng.choice(neg, k_neg, replace=False)])
    rng.shuffle(idx)
    return idx


# %% [markdown]
# ## 2. Google early-runtime ensemble
#
# The checkpoint pipeline is loaded, the test split comes from the Drive Parquet
# cache written by the modeling notebook, and SHAP runs on the classifier over the
# imputed matrix. The missing-indicator columns the imputer adds are named after
# their source feature so they can be read as null-as-signal in the tier analysis.

# %%
G_CKPT = CHECKPOINT_DIR / "rq1_google_best_earlyruntime.pkl"
G_SIDE = CHECKPOINT_DIR / "rq1_google_best_earlyruntime.json"
G_TEST = CACHE_DIR / "rq1_google_split_test.parquet"
for p in (G_CKPT, G_SIDE, G_TEST):
    assert p.exists(), f"missing Google input: {p}"

import pickle

with G_CKPT.open("rb") as fh:
    g_pipe = pickle.load(fh)
g_side = json.loads(G_SIDE.read_text())
G_COLS = g_side["feature_columns"]
G_LABEL = "failure_label"
print(f"Google model: {g_side['model']} | features: {len(G_COLS)} | "
      f"pipeline steps: {list(g_pipe.named_steps)}")

# %%
g_test = pl.read_parquet(G_TEST)
gy = g_test[G_LABEL].to_numpy().astype(np.int8)
g_idx = stratified_indices(gy, SHAP_ROWS, SEED, proportional=True)
g_sample = g_test[g_idx]
gX_raw = g_sample.select([pl.col(c).cast(pl.Float32) for c in G_COLS]).to_numpy()
gy_sample = gy[g_idx]
print(f"Google SHAP sample: {gX_raw.shape[0]:,} rows, "
      f"{int(gy_sample.sum()):,} positive ({gy_sample.mean():.3f})")

# Impute exactly as the pipeline does at predict, then expand feature names to
# include the added missing indicators.
g_imputer = g_pipe.named_steps["impute"]
g_clf = g_pipe.named_steps["model"]
gX = g_imputer.transform(gX_raw)
g_names = list(G_COLS)
if getattr(g_imputer, "add_indicator", False) and g_imputer.indicator_ is not None:
    g_names += [f"{G_COLS[i]}__missing" for i in g_imputer.indicator_.features_]
assert gX.shape[1] == len(g_names), (gX.shape, len(g_names))

# %%
g_shap = explain_tree(g_clf, gX, "google_early_runtime")
g_imp = mean_abs_importance(g_shap, g_names)
g_imp.write_csv(TABLES_DIR / f"shap_importance_rq1_google_{STAMP}.csv")
save_bar(g_imp, "RQ1 Google early-runtime: top-20 mean(|SHAP|)",
         FIG_G / f"top20_bar_{STAMP}.png")
save_beeswarm(g_shap, gX, g_names,
              "RQ1 Google early-runtime: SHAP beeswarm", FIG_G / f"beeswarm_{STAMP}.png")
for feat in g_imp.head(5)["feature"].to_list():
    safe = feat.replace("/", "_")
    save_dependence(g_shap, gX, g_names, feat, FIG_G / f"dependence_{safe}_{STAMP}.png")
print("Google top-15 by mean(|SHAP|):")
print(g_imp.head(15))

# %% [markdown]
# ## 3. Backblaze 14-day stack
#
# The payload carries three fitted members. SHAP runs per member on that member's
# own imputed matrix; the drive-model prior is reattached from the payload so the
# feature matrix matches training. The stack bar is the equal-weight mean of the
# members' normalized mean absolute SHAP.

# %%
B_CKPT = CHECKPOINT_DIR / "rq1_backblaze_best_14d.pkl"
B_SIDE = CHECKPOINT_DIR / "rq1_backblaze_best_14d.json"
assert B_CKPT.exists() and B_SIDE.exists(), "missing Backblaze checkpoint or sidecar"
with B_CKPT.open("rb") as fh:
    b_payload = pickle.load(fh)
b_side = json.loads(B_SIDE.read_text())
assert b_payload["kind"] == "soft_voting_stack", f"unexpected payload kind {b_payload['kind']}"
B_FEATURES = b_side["feature_columns"]
B_TARGET = "failure_within_14d"
member_names = list(b_payload["members"])
print(f"Backblaze stack members: {member_names} | features: {len(B_FEATURES)}")

# %%
# Class-enriched explanation sample from the natural-prevalence 2023-2025 test.
# Positives are too rare there for a proportional sample to carry any, so all
# available positives are kept and negatives are subsampled. This is stated as a
# class-enriched explanation sample, not a natural-prevalence one.
GCS_NATURAL_PREFIX = "backblaze_features/natural_test_2023_2025"
LOCAL_NATURAL = Path("/content/natural_test_2023_2025")
POS_CAP = 3_000               # positives are rare; harvest up to this many
NEG_ROWS = 12_000             # negatives, spread across buckets

if IN_COLAB:
    from google.cloud import storage

    LOCAL_NATURAL.mkdir(parents=True, exist_ok=True)
    gcs = storage.Client(project=PROJECT_ID)
    bucket = gcs.bucket(f"{PROJECT_ID}-dissertation-data")
    for blob in gcs.list_blobs(bucket, prefix=GCS_NATURAL_PREFIX):
        if blob.name.endswith(".parquet"):
            dest = LOCAL_NATURAL / Path(blob.name).name
            if not dest.exists():
                blob.download_to_filename(dest)
    natural_files = sorted(LOCAL_NATURAL.glob("bucket_*.parquet"))
else:
    natural_files = sorted(CACHE_DIR.glob("backblaze_natural_test_*.parquet"))
assert natural_files, "no natural-test parquet files found"

# Build the class-enriched sample one bucket at a time so the full negative set is
# never materialized (it is tens of millions of rows). Each bucket contributes all
# of its positives, up to the cap, and a bounded negative quota; the lazy head()
# stops reading each file once its quota is met, so memory stays flat.
read_cols = [c for c in B_FEATURES if c != "model_prior"] + [B_TARGET, "model_canonical"]
neg_per_file = max(1, -(-NEG_ROWS // len(natural_files)))   # ceil division
pos_parts, neg_parts, n_pos, n_neg = [], [], 0, 0
for f in natural_files:
    lf = pl.scan_parquet(f).select(read_cols)
    if n_pos < POS_CAP:
        p = lf.filter(pl.col(B_TARGET) == 1).head(POS_CAP - n_pos).collect()
        if p.height:
            pos_parts.append(p)
            n_pos += p.height
    if n_neg < NEG_ROWS:
        take = min(neg_per_file, NEG_ROWS - n_neg)
        ng = lf.filter(pl.col(B_TARGET) == 0).head(take).collect()
        if ng.height:
            neg_parts.append(ng)
            n_neg += ng.height
    if n_pos >= POS_CAP and n_neg >= NEG_ROWS:
        break
b_sample = (pl.concat(pos_parts + neg_parts)
            .sample(fraction=1.0, shuffle=True, seed=SEED))
print(f"harvested {n_pos:,} positives, {n_neg:,} negatives from "
      f"{len(pos_parts)} / {len(natural_files)} buckets")

# Reattach the leakage-safe drive-model prior exactly as training did.
enc = pl.DataFrame(b_payload["drive_model_prior"])
b_sample = (b_sample.join(enc, on="model_canonical", how="left")
            .with_columns(pl.col("model_prior").fill_null(b_payload["global_prior"]).cast(pl.Float32)))
by = b_sample[B_TARGET].to_numpy().astype(np.int8)
print(f"Backblaze SHAP sample: {b_sample.height:,} rows, "
      f"{int(by.sum()):,} positive ({by.mean():.4f})")
bX_nan = b_sample.select([pl.col(c).cast(pl.Float32) for c in B_FEATURES]).to_numpy()

# %%
# Per-member SHAP, then the stack view as the equal-weight mean of normalized rankings.
member_imp = {}
norm_cols = []
for name in member_names:
    m = b_payload["members"][name]
    est = m["wrapper"].estimator
    imputer = m["imputer"]
    Xm = imputer.transform(bX_nan) if imputer is not None else bX_nan
    sv = explain_tree(est, Xm, f"backblaze_{name}")
    assert sv.shape[1] == len(B_FEATURES), (name, sv.shape, len(B_FEATURES))
    imp = mean_abs_importance(sv, B_FEATURES)
    member_imp[name] = imp
    save_beeswarm(sv, Xm, B_FEATURES,
                  f"RQ1 Backblaze 14d, member {name}: SHAP beeswarm (member-level)",
                  FIG_B / f"beeswarm_{name}_{STAMP}.png")
    m_abs = np.abs(sv).mean(axis=0)
    norm_cols.append(m_abs / m_abs.sum() if m_abs.sum() > 0 else m_abs)
    print(f"  {name}: top feature {imp['feature'][0]} ({imp['mean_abs_shap'][0]:.4g})")

stack_norm = np.mean(np.vstack(norm_cols), axis=0)
b_imp = (pl.DataFrame({"feature": B_FEATURES, "stack_norm_mean_abs_shap": stack_norm.astype(np.float64)})
         .sort("stack_norm_mean_abs_shap", descending=True))
b_imp.write_csv(TABLES_DIR / f"shap_importance_rq1_backblaze_{STAMP}.csv")
save_bar(b_imp.rename({"stack_norm_mean_abs_shap": "mean_abs_shap"}),
         "RQ1 Backblaze 14d stack: top-20 normalized mean(|SHAP|)",
         FIG_B / f"top20_bar_stack_{STAMP}.png")
print("Backblaze stack top-15 (normalized mean|SHAP|):")
print(b_imp.head(15))

# The drive-model prior check: where does it rank in the stack and each member?
prior_rank_stack = b_imp["feature"].to_list().index("model_prior") + 1
print(f"\nmodel_prior stack rank: {prior_rank_stack} / {len(B_FEATURES)}")
for name, imp in member_imp.items():
    r = imp["feature"].to_list().index("model_prior") + 1
    print(f"  model_prior rank in {name}: {r} / {len(B_FEATURES)}")

# %% [markdown]
# ## 4. Tier alignment
#
# Each top-15 feature is classified against the availability-tier scheme its dataset
# was engineered under. For Google the theoretical ordering is that pre-scheduling
# and historical features dominate, early-runtime slope and ramp features are
# moderate, and traditional utilization features are low or absent; a missing
# indicator inherits its source feature's tier, since null-as-signal is a
# pre-scheduling observable here, and submission-time temporal encodings are
# pre-scheduling (tier1). For Backblaze the tiers follow the SMART feature module:
# tier1 is zero-inflation indicators and degradation-onset timing and manufacturer
# and capacity, tier2 is the rolling and quantile and rate-of-change dynamics, tier3
# is drive and fleet age and calendar and the era-gated attributes. Two families sit
# outside those engineered tiers and are reported on their own: the raw current SMART
# readings, and the leakage-safe drive-model prior.

# %%
G_TIER1 = ("prior_fail", "has_prior", "resubmission", "prior_evict", "lifecycle",
           "has_hardware_counters", "workload", "priority", "scheduling_class",
           "platform", "cpu_request", "memory_request", "request_ratio",
           "queue_time", "submit")
G_TIER2 = ("slope", "ramp", "first_interval", "cpi", "mapi", "sequence_complexity",
           "running_duration", "initial_")
G_TIER3 = ("avg_cpu", "avg_memory", "max_cpu", "max_memory", "_util", "util_")


def google_tier(feature: str) -> str:
    base = feature[:-len("__missing")] if feature.endswith("__missing") else feature
    low = base.lower()
    if any(k in low for k in G_TIER1):
        return "tier1"
    if any(k in low for k in G_TIER2):
        return "tier2"
    if any(k in low for k in G_TIER3):
        return "tier3"
    return "unclassified"


def backblaze_tier(feature: str) -> str:
    """Classify a Backblaze feature by the tiers in src/features/backblaze_smart.py,
    plus two buckets outside the engineered tiers: the raw SMART readings and the
    drive-model prior."""
    low = feature.lower()
    if low == "model_prior":
        return "drive_model_prior"
    if ("days_since_first_nonzero" in low or low.startswith("has_nonzero_smart")
            or low.startswith("is_mfr_") or low.startswith("capacity")):
        return "tier1"
    if any(k in low for k in ("rollmean", "rollp95", "rollp99", "rollstd", "_delta_")):
        return "tier2"
    if low in ("fleet_age_days", "drive_age_days", "year", "month", "quarter") \
            or low.startswith("era_smart"):
        return "tier3"
    if low.startswith("smart_") and low.endswith("_raw"):
        return "raw_smart"
    return "unclassified"


def classify_top(imp: pl.DataFrame, classifier, dataset: str, scheme: str, top: int = 15):
    value_col = imp.columns[1]
    top_df = imp.head(top).with_columns(
        pl.col("feature").map_elements(classifier, return_dtype=pl.Utf8).alias("class"))
    prop = (top_df.group_by("class").len()
            .with_columns((pl.col("len") / top_df.height).alias("proportion"))
            .sort("proportion", descending=True))
    print(f"\n{dataset} top-{top} classification:")
    print(top_df.select(["feature", value_col, "class"]))
    print(prop)
    return (prop.rename({"len": "n_in_top15"})
            .with_columns(dataset=pl.lit(dataset), scheme=pl.lit(scheme))
            .select(["dataset", "scheme", "class", "n_in_top15", "proportion"]))


g_rows = classify_top(g_imp, google_tier, "rq1_google", "google_tier")
b_rows = classify_top(b_imp, backblaze_tier, "rq1_backblaze", "backblaze_tier")
tier_alignment = pl.concat([g_rows, b_rows], how="vertical")
tier_alignment.write_csv(TABLES_DIR / "tier_alignment.csv")
print("\nWrote tier_alignment.csv")
print(tier_alignment)

# %% [markdown]
# ## 5. Feature-importance rows for the stack members
#
# One row per (dataset, model, feature) mean absolute SHAP, for the record. Model
# feature importances for the other questions are added in the ablation notebook,
# where their estimators are already loaded.

# %%
fi_parts = [g_imp.head(30).with_columns(dataset=pl.lit("rq1_google"),
                                        model=pl.lit(g_side["model"]))
            .rename({"mean_abs_shap": "importance"})]
for name, imp in member_imp.items():
    fi_parts.append(imp.head(30).with_columns(dataset=pl.lit("rq1_backblaze"),
                                              model=pl.lit(name))
                    .rename({"mean_abs_shap": "importance"}))
feature_importances = pl.concat(
    [p.select(["dataset", "model", "feature", "importance"]) for p in fi_parts],
    how="vertical")
feature_importances.write_csv(TABLES_DIR / "feature_importances.csv")
print(f"Wrote feature_importances.csv ({feature_importances.height} rows)")

# %% [markdown]
# ## 6. Run metadata

# %%
run_meta.update({
    "google_model": g_side["model"],
    "google_shap_rows": int(gX.shape[0]),
    "google_n_features_expanded": len(g_names),
    "backblaze_members": member_names,
    "backblaze_shap_rows": int(b_sample.height),
    "backblaze_model_prior_stack_rank": int(prior_rank_stack),
    "figures": {"rq1_google": str(FIG_G), "rq1_backblaze": str(FIG_B)},
})
(TABLES_DIR / f"shap_run_metadata_{STAMP}.json").write_text(json.dumps(run_meta, indent=2))
print(json.dumps(run_meta, indent=2))
