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
# # 11. Google Cluster Traces Post-Preprocessing EDA and Working-Set Adequacy
#
# **Purpose.** Validate the engineered feature matrix produced by
# `notebooks/10_feature_engineering_google.py` and decide whether the locked
# working set is large enough for Chapter 4 modeling.
#
# **Inputs.**
# - `{OUTPUT_DIR}/features/google/instance_features.parquet` (the three-tier
#   feature matrix; one row per working-set instance).
# - `{OUTPUT_DIR}/features/google/feature_schema.json` (column -> tier map,
#   written by notebook 10 Section 10).
#
# **Outputs.**
# - `outputs/tables/google_preprocessing_verification.csv` (Section 1
#   distribution checks; committed to the repo).
# - `outputs/tables/tier3_inversion_check.csv` (Section 2 V12 guardrail;
#   overwrites the notebook-10 preview with the post-feature-engineering
#   medians).
# - `outputs/figures/learning_curve_google.png` (Section 3-4 curve with
#   bootstrap CI ribbon).
#
# **Sections.**
# 0. Colab session setup.
# 1. Post-preprocessing distribution verification (class imbalance, key feature
#    distributions, tier composition, per-tier null rates) vs Chapter 3
#    commitments. -> `google_preprocessing_verification.csv`.
# 2. Tier 3 inversion regression check (V12 guardrail). ->
#    `tier3_inversion_check.csv`.
# 3. Learning-curve harness (baseline LightGBM, MCC vs working-set fraction)
#    with a 95% bootstrap CI.
# 4. P05 working-set adequacy decision rule.
# 5. Learning-curve figure. -> `learning_curve_google.png`.
#
# **Chapter 3 commitments checked.**
# - V02 class imbalance: FINISH:FAIL_LOST approximately 3.39:1 (moderate;
#   manageable with cost-sensitive learning + SMOTE).
# - V10 resubmission dominance: nearly all FAIL_LOST instances have been
#   resubmitted at least once.
# - V12 utilization inversion: failing instances retain LOWER median absolute
#   CPU at every Tier 3 window.
# - V13 tier composition: 29 Tier 1, 11 Tier 2, 18 Tier 3 features.
# - P05 sample-size adequacy: consecutive MCC deltas below 0.005 with the 95%
#   bootstrap CI straddling zero.
# - P14 random seed convention: seed 42 throughout.
# - MCC is the primary RQ1 metric (Chicco & Jurman, 2023).

# %% [markdown]
# ---
# ## 0. Colab Session Setup

# %%
# !pip install -q polars lightgbm scikit-learn matplotlib pyarrow

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
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from src.data.validation import AssertionFailedError, assert_tier3_inversion

SEED = 42  # P14 random-seed convention.
rng = np.random.default_rng(SEED)

# %% [markdown]
# ---
# ## 1. Post-preprocessing distribution verification
#
# Confirm the engineered matrix matches the Chapter 3 commitments. Every check
# appends a `(check, expected, observed, ok)` row to the verification log,
# written to `outputs/tables/google_preprocessing_verification.csv`. All
# aggregations run with the Polars streaming engine so the 35M-row matrix is
# never fully materialized.

# %%
FEATURES_DIR = OUTPUT_DIR / 'features' / 'google'
MATRIX_PATH = FEATURES_DIR / 'instance_features.parquet'
SCHEMA_PATH = FEATURES_DIR / 'feature_schema.json'

# Persistent Drive output store (mirrors notebook 10), not the ephemeral cloned
# repo, which is recreated each runtime and lost on termination.
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'
for directory in (TABLES_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

VERIFICATION_CSV = TABLES_DIR / 'google_preprocessing_verification.csv'
TIER3_INVERSION_CSV = TABLES_DIR / 'tier3_inversion_check.csv'
# The main learning-curve figure is now the episode-grain RQ1 adequacy curve
# (Section 5). The instance-grain curve is retained as the leakage baseline only
# (Section 3.5b), with its own file so it is not mistaken for adequacy evidence.
LEARNING_CURVE_PNG = FIGURES_DIR / 'learning_curve_google.png'
LEAKED_BASELINE_PNG = FIGURES_DIR / 'learning_curve_google_instance_leaked.png'

LABEL_COL = "failure_label"

# Tier column lists straight from the notebook-10 schema manifest (authoritative).
with open(SCHEMA_PATH) as f:
    feature_schema = json.load(f)

TIER1_COLS = [c["name"] for c in feature_schema["columns"] if c["tier"] == "tier1"]
TIER2_COLS = [c["name"] for c in feature_schema["columns"] if c["tier"] == "tier2"]
TIER3_COLS = [c["name"] for c in feature_schema["columns"] if c["tier"] == "tier3"]
FEATURE_COLS = TIER1_COLS + TIER2_COLS + TIER3_COLS

print(f"Matrix:   {MATRIX_PATH}")
print(f"Features: {len(FEATURE_COLS)} ({len(TIER1_COLS)} T1 / {len(TIER2_COLS)} T2 / {len(TIER3_COLS)} T3)")


def _resolve_matrix_source() -> str:
    """Return the feature-matrix path to scan.

    Notebook 10 writes the matrix to Drive, but a large Drive-FUSE write can
    fail to flush before a Colab runtime recycles (the small feature_schema.json
    persists while the multi-GB Parquet may not). The BigQuery `instance_features`
    table and its GCS export are durable, so fall back to the GCS export when the
    Drive copy is missing or empty.
    """
    if MATRIX_PATH.exists() and MATRIX_PATH.stat().st_size > 0:
        print(f"Reading feature matrix from Drive: {MATRIX_PATH}")
        return str(MATRIX_PATH)
    from google.colab import auth
    from utils.bq_client import PROJECT_ID
    auth.authenticate_user()  # ADC for the gs:// scan.
    gcs_uri = f"gs://{PROJECT_ID}-dissertation-data/google_features/instance_features/"
    print(f"Drive matrix not found; reading the durable GCS export: {gcs_uri}")
    print("Tip: re-run notebook 10 Section 9.3 to refresh the Drive cache if you want a local copy.")
    return gcs_uri


matrix_source = _resolve_matrix_source()
matrix_lf = pl.scan_parquet(matrix_source)

# %%
verification_rows: list[dict] = []


def record_check(check: str, expected: object, observed: object, ok: bool, notes: str = "") -> None:
    """Append a verification row and print a one-line summary."""
    status = "PASS" if ok else "FAIL"
    verification_rows.append({
        "check": check,
        "expected": str(expected),
        "observed": str(observed),
        "ok": bool(ok),
        "notes": notes,
    })
    suffix = f" ({notes})" if notes else ""
    print(f"  [{status}] {check}: expected {expected}, observed {observed}{suffix}")


# %% [markdown]
# ### 1.1 Class imbalance (V02)
#
# The ~3.39:1 figure from the sentinel analysis is the EVENT-LEVEL FAIL_LOST
# class balance. This working set is labelled by per-instance TERMINAL outcome
# and restricted to scheduled instances, so its imbalance is necessarily more
# extreme: an instance that fails, is resubmitted, and ultimately FINISHes
# counts as a negative (V10 resubmission dominance), and failures that crash
# before scheduling are excluded. The per-instance ratio was left "to be
# reported in Chapter 4", so it is recorded here rather than asserted against
# the event-level number. The Chapter 3 commitment for V02 is the RELATIVE
# claim that Google imbalance is moderate next to Backblaze's extreme (~3
# orders of magnitude more severe), which is what the second check tests.

# %%
label_counts = (
    matrix_lf
    .filter(pl.col(LABEL_COL).is_not_null())
    .group_by(LABEL_COL)
    .agg(pl.len().alias("n"))
    .collect(engine="streaming")
)
by_label = dict(zip(label_counts[LABEL_COL].to_list(), label_counts["n"].to_list()))
n_pos = int(by_label.get(1, 0))
n_neg = int(by_label.get(0, 0))
n_labeled = n_pos + n_neg
imbalance_ratio = (n_neg / n_pos) if n_pos else float("nan")
pos_fraction = (n_pos / n_labeled) if n_labeled else float("nan")

print(f"Positives (FAIL_LOST): {n_pos:,}")
print(f"Negatives (FINISH):    {n_neg:,}")
print(f"Imbalance ratio (neg:pos): {imbalance_ratio:.3f}:1")
print(f"Positive fraction:         {pos_fraction:.4f}")

# Backblaze grand daily failure rate (0.0046%) is the extreme-imbalance anchor.
BACKBLAZE_POS_FRACTION = 0.000046

# Record the per-instance working-set imbalance (the Chapter 4 figure). No
# committed instance-level target exists, so this passes as long as the ratio
# is well-defined and sane; the value itself feeds the imbalance strategy.
record_check(
    "Section 1.1: working-set instance-level imbalance recorded for Chapter 4",
    expected="reported (no committed instance-level target)",
    observed=f"{imbalance_ratio:.1f}:1 (positive fraction {pos_fraction:.4f})",
    ok=(n_pos > 0 and 0.0 < pos_fraction < 0.5),
    notes="Per-instance terminal-outcome labelling on scheduled instances; feeds the Ch.4 imbalance strategy.",
)
# V02 relative claim: Google is moderate vs Backblaze's extreme. Test that
# Google's positive fraction is at least ~2 orders of magnitude above
# Backblaze's rather than against a fixed band.
severity_vs_backblaze = pos_fraction / BACKBLAZE_POS_FRACTION
record_check(
    "Section 1.1: Google imbalance far less extreme than Backblaze (V02 relative claim)",
    expected="Google positive fraction >= 100x Backblaze (~>=2 orders less severe)",
    observed=f"{severity_vs_backblaze:.0f}x Backblaze positive fraction",
    ok=(severity_vs_backblaze >= 100),
    notes="Ch.3 frames Google as moderate vs Backblaze extreme; relative, not a fixed 3.39:1.",
)

# %% [markdown]
# ### 1.2 Key feature distributions (V10, priority mix)
#
# Conditional means by label confirm the EDA signals survived feature
# engineering: V10 resubmission dominance (failing instances are
# overwhelmingly resubmitted) and a sensible priority mix.

# %%
cond_means = (
    matrix_lf
    .filter(pl.col(LABEL_COL).is_not_null())
    .group_by(LABEL_COL)
    .agg(
        pl.col("first_resubmission").mean().alias("p_first_resubmission"),
        pl.col("has_prior_fail").mean().alias("p_has_prior_fail"),
        pl.col("has_hardware_counters").mean().alias("p_has_hw_counters"),
        pl.col("priority_tier_production").mean().alias("p_production"),
        pl.col("queue_time").median().alias("median_queue_time"),
    )
    .collect(engine="streaming")
)
cond = {int(r[LABEL_COL]): r for r in cond_means.to_dicts()}
print(cond_means)

# V10: failing instances should be far more likely to have been resubmitted.
p_resub_fail = cond[1]["p_first_resubmission"]
p_resub_ok = cond[0]["p_first_resubmission"]
record_check(
    "Section 1.2: V10 resubmission dominance (P(resubmitted | fail) high)",
    expected="P(first_resubmission | fail) >= 0.70 and > P(.. | success)",
    observed=f"fail={p_resub_fail:.3f}, success={p_resub_ok:.3f}",
    ok=(p_resub_fail >= 0.70 and p_resub_fail > p_resub_ok),
    notes="V10: nearly all FAIL_LOST instances were resubmitted at least once.",
)
record_check(
    "Section 1.2: prior-failure signal stronger for failures",
    expected="P(has_prior_fail | fail) > P(has_prior_fail | success)",
    observed=f"fail={cond[1]['p_has_prior_fail']:.3f}, success={cond[0]['p_has_prior_fail']:.3f}",
    ok=(cond[1]["p_has_prior_fail"] > cond[0]["p_has_prior_fail"]),
)

# %% [markdown]
# ### 1.3 Tier composition (V13)

# %%
record_check(
    "Section 1.3: Tier composition matches V13 (29 / 11 / 18)",
    expected="29 Tier 1, 11 Tier 2, 18 Tier 3",
    observed=f"{len(TIER1_COLS)} / {len(TIER2_COLS)} / {len(TIER3_COLS)}",
    ok=(len(TIER1_COLS) == 29 and len(TIER2_COLS) == 11 and len(TIER3_COLS) == 18),
)

# %% [markdown]
# ### 1.4 Per-tier null rates
#
# Tier 2 / Tier 3 nulls are expected (instances with no in-band usage; rapid-
# onset crashes), while Tier 1 should be essentially complete. One streaming
# pass computes every feature's null count.

# %%
null_counts = (
    matrix_lf
    .select([pl.col(c).null_count().alias(c) for c in FEATURE_COLS] + [pl.len().alias("_n")])
    .collect(engine="streaming")
    .to_dicts()[0]
)
n_rows = int(null_counts.pop("_n"))
null_fracs = {c: null_counts[c] / n_rows for c in FEATURE_COLS}

tier1_max_null = max(null_fracs[c] for c in TIER1_COLS)
tier2_max_null = max(null_fracs[c] for c in TIER2_COLS)
tier3_max_null = max(null_fracs[c] for c in TIER3_COLS)
print(f"Max null fraction  Tier 1: {tier1_max_null:.4f}  Tier 2: {tier2_max_null:.4f}  Tier 3: {tier3_max_null:.4f}")

record_check(
    "Section 1.4: Tier 1 features essentially complete",
    expected="max Tier 1 null fraction <= 0.01",
    observed=f"{tier1_max_null:.4f}",
    ok=(tier1_max_null <= 0.01),
    notes="Tier 1 derives from the lifecycle backbone; nulls should be negligible.",
)
record_check(
    "Section 1.4: Tier 2/3 nulls present but bounded (expected; rapid-onset)",
    expected="max Tier 2/3 null fraction in (0, 1)",
    observed=f"T2={tier2_max_null:.4f}, T3={tier3_max_null:.4f}",
    ok=(0.0 <= tier2_max_null < 1.0 and 0.0 <= tier3_max_null < 1.0),
    notes="Instances without in-band usage observations carry null early-runtime/windowed features.",
)

# %% [markdown]
# ### 1.5 Write the distribution-verification table

# %%
verification_df = pl.DataFrame(verification_rows)
verification_df.write_csv(str(VERIFICATION_CSV))
print(f"Verification log: {VERIFICATION_CSV}")
print(verification_df.select(["check", "ok"]))
n_dist_failed = verification_df.filter(~pl.col("ok")).height
print(f"\n{n_dist_failed} distribution check(s) failed." if n_dist_failed
      else "\nAll distribution checks passed.")

# %% [markdown]
# ---
# ## 2. Tier 3 inversion regression check (V12 guardrail)
#
# The Chapter 4 Tier 3 ablation depends on the V12 utilization inversion:
# failing instances must retain LOWER median absolute CPU than successful ones
# at every Tier 3 window. This re-runs the inversion on the **post-feature-
# engineering** matrix (notebook 10 Section 10.1 used an approximate BigQuery
# median; here we use the exact Polars median via the shared guardrail
# `src.data.validation.assert_tier3_inversion`). A broken inversion means
# preprocessing or feature engineering washed out the signal -> loop back to
# debug before modeling.

# %%
TIER3_CPU_WINDOWS = ["avg_cpu_5min", "avg_cpu_15min", "avg_cpu_60min"]

inversion_rows = []
inversion_ok = True
for col in TIER3_CPU_WINDOWS:
    # Minimal 2-column lazy frame keeps the exact-median collect bounded.
    win_lf = matrix_lf.select([LABEL_COL, col])
    try:
        median_fail, median_finish = assert_tier3_inversion(
            win_lf, cpu_column=col, label_column=LABEL_COL
        )
        holds = True
    except AssertionFailedError as exc:
        # Recompute the medians for the record even when the guard fails.
        med = (
            win_lf.filter(pl.col(LABEL_COL).is_not_null())
            .group_by(LABEL_COL).agg(pl.col(col).median().alias("m"))
            .collect()
        )
        mm = dict(zip(med[LABEL_COL].to_list(), med["m"].to_list()))
        median_fail, median_finish = mm.get(1), mm.get(0)
        holds = False
        inversion_ok = False
        print(f"  [WARN] {exc}")

    inversion_rows.append({
        "window": col.replace("avg_cpu_", ""),
        "median_cpu_fail": median_fail,
        "median_cpu_success": median_finish,
        "inversion_holds": holds,
    })
    record_check(
        f"Section 2: V12 inversion holds at {col} (median CPU fail < success)",
        expected="fail < success",
        observed=f"fail={median_fail}, success={median_finish}",
        ok=holds,
        notes="regression guard for the V12 inversion finding; exact Polars median on the post-FE matrix.",
    )

pl.DataFrame(inversion_rows).write_csv(str(TIER3_INVERSION_CSV))
print(f"Tier 3 inversion check: {TIER3_INVERSION_CSV}")

# Hard stop: the ablation has no anchor if the inversion is gone.
assert inversion_ok, (
    "V12 Tier 3 inversion broken on the post-feature-engineering matrix. "
    "Debug preprocessing/feature engineering before modeling."
)

# %% [markdown]
# ---
# ## 3. Learning-curve harness (baseline LightGBM, MCC)
#
# Train a baseline LightGBM (default hyperparameters) on 1, 5, 10, 25, 50, and
# 100% of the working-set training pool and evaluate MCC (the primary RQ1
# metric) on a fixed held-out validation fold. MCC is reported with a 95%
# bootstrap CI; the consecutive-delta CIs drive the P05 adequacy rule
# (Section 4).
#
# **Memory note.** Training LightGBM on the full 35M-row working set in a free
# T4 Colab kernel (12.7 GB) is impractical, so the curve is computed on a
# representative base capped at `MAX_BASE_ROWS`. The cap is drawn by a
# deterministic hash subsample of the matrix (uniform over rows, reproducible
# via `SEED`); the learning-curve fractions are fractions of this base. This is
# exactly the P05 design: if MCC has not asymptoted at the cap, raise
# `MAX_BASE_ROWS` (or switch to a High-RAM runtime) and re-run; if it has, the
# full 35M working set is more than adequate. Features keep float32 and the
# native `lgb.Dataset(free_raw_data=True)` path is used to bound peak memory.
#
# **Grain note.** Sections 3.1-3.7 fit on the instance-grain matrix. The
# prediction-point ablation (3.7) and the episode re-check (3.8) show that matrix
# leaks lifecycle history into the label, so this curve is retained only as the
# leaked baseline (figure in 3.5b). The RQ1 adequacy curve is fit at the episode
# grain in Section 3.9, and the P05 decision (Section 4) and the main figure
# (Section 5) read from it.

# %%
MAX_BASE_ROWS = 12_000_000          # base cap for the curve (raise on High-RAM).
VAL_FRACTION = 0.20                 # held-out validation fold (stratified).
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
N_BOOTSTRAP = 500                   # bootstrap resamples for the CIs.
BOOTSTRAP_VAL_CAP = 200_000         # val subsample size used for the CIs.
P05_DELTA_THRESHOLD = 0.005         # |MCC delta| convergence threshold (P05).

LGBM_PARAMS = {
    "objective": "binary",
    "seed": SEED,
    "verbosity": -1,
    "n_jobs": -1,
}
LGBM_NUM_ROUNDS = 100               # LightGBM default boosting rounds.

# %% [markdown]
# ### 3.1 Load a capped, reproducible base into memory (float32)
#
# A deterministic per-row hash subsample keeps only ~`MAX_BASE_ROWS` rows, so
# the full matrix is never materialized. Nulls in Tier 2/3 features are left as
# NaN; LightGBM handles missing values natively.

# %%
n_labeled_total = (
    matrix_lf.filter(pl.col(LABEL_COL).is_not_null())
    .select(pl.len()).collect(engine="streaming").item()
)
keep_every = max(1, int(np.ceil(n_labeled_total / MAX_BASE_ROWS)))
print(f"Labeled rows: {n_labeled_total:,}; keeping ~1/{keep_every} for the curve base.")

base_lf = (
    matrix_lf
    .filter(pl.col(LABEL_COL).is_not_null())
    .with_row_index("_rid")
    # Uniform, deterministic subsample: hash the row index and bucket it.
    .filter((pl.col("_rid").hash(seed=SEED) % keep_every) == 0)
    .select([*FEATURE_COLS, LABEL_COL])
)

base_df = base_lf.collect(engine="streaming")
print(f"Base for learning curve: {base_df.height:,} rows x {len(FEATURE_COLS)} features.")

# Materialize once as float32 features + int8 label, then free the Polars frame.
# Cast to Float32 *inside* Polars first so to_numpy() does not transiently
# upcast the mixed int8/float columns to float64 (halves peak memory). Nulls in
# Tier 2/3 features become NaN, which LightGBM handles natively.
X_all = (
    base_df.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
    .to_numpy()
)
y_all = base_df.select(LABEL_COL).to_numpy().ravel().astype(np.int8)
del base_df
gc.collect()
print(f"X_all: {X_all.shape}, {X_all.nbytes / 1e9:.2f} GB; positives: {int(y_all.sum()):,}")

# %% [markdown]
# ### 3.2 Fixed stratified train/validation split
#
# One held-out validation fold is shared by every fraction so the MCCs are
# directly comparable. A capped, stratified bootstrap subset of the validation
# fold is drawn once for the CIs.

# %%
train_idx, val_idx = train_test_split(
    np.arange(X_all.shape[0]),
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=y_all,
)
X_val = X_all[val_idx]
y_val = y_all[val_idx]
print(f"Train pool: {train_idx.size:,}   Validation fold: {val_idx.size:,}")

# Stratified bootstrap subset of the validation fold (for CI speed).
if y_val.size > BOOTSTRAP_VAL_CAP:
    boot_idx, _ = train_test_split(
        np.arange(y_val.size), train_size=BOOTSTRAP_VAL_CAP,
        random_state=SEED, stratify=y_val,
    )
else:
    boot_idx = np.arange(y_val.size)
y_boot = y_val[boot_idx]
print(f"Bootstrap validation subset: {y_boot.size:,} rows.")

# %% [markdown]
# ### 3.3 Fast MCC and the bootstrap helper
#
# MCC is computed from the 2x2 confusion counts so the bootstrap resamples are
# cheap (a single `bincount` per resample). The point estimate is validated
# against `sklearn.metrics.matthews_corrcoef` in `tests` / the notebook smoke
# check.

# %%
def mcc_from_counts(tn: float, fp: float, fn: float, tp: float) -> float:
    """Matthews correlation coefficient from confusion counts (0 when the
    denominator vanishes, matching sklearn's degenerate-case convention)."""
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else 0.0


def mcc_of(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MCC for binary 0/1 arrays via a length-4 bincount (TN, FP, FN, TP)."""
    counts = np.bincount(2 * y_true + y_pred, minlength=4)
    return mcc_from_counts(*counts.astype(np.float64))


def bootstrap_mcc(
    y_true: np.ndarray, preds: dict[float, np.ndarray], n_boot: int
) -> dict[float, np.ndarray]:
    """Return, per fraction, an array of ``n_boot`` MCCs over resamples of the
    (shared) validation subset. The same resample indices are reused across
    fractions so consecutive-delta CIs are paired."""
    n = y_true.size
    out = {f: np.empty(n_boot, dtype=np.float64) for f in preds}
    local = np.random.default_rng(SEED)
    for b in range(n_boot):
        idx = local.integers(0, n, size=n)
        yt = y_true[idx]
        for f, pred in preds.items():
            out[f][b] = mcc_of(yt, pred[idx])
    return out


# %% [markdown]
# ### 3.4 Fit the curve
#
# For each fraction, draw a stratified subsample of the training pool, fit the
# baseline LightGBM, and record the validation MCC plus the predicted labels on
# the bootstrap subset.

# %%
curve_rows = []
boot_preds: dict[float, np.ndarray] = {}   # fraction -> predicted labels on y_boot.

for frac in FRACTIONS:
    n_frac = max(2, int(round(frac * train_idx.size)))
    if n_frac >= train_idx.size:
        sub_idx = train_idx
    else:
        sub_idx, _ = train_test_split(
            train_idx, train_size=n_frac, random_state=SEED, stratify=y_all[train_idx]
        )

    # Native Dataset with free_raw_data=True so the float copy is released after
    # histogram construction.
    X_train = X_all[sub_idx]
    y_train = y_all[sub_idx]
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_NUM_ROUNDS)
    del X_train, y_train, dtrain
    gc.collect()

    # Validation MCC at the 0.5 threshold (primary RQ1 metric).
    val_proba = booster.predict(X_val)
    val_pred = (val_proba >= 0.5).astype(np.int8)
    mcc_point = matthews_corrcoef(y_val, val_pred)

    # Predicted labels on the bootstrap subset for the CI.
    boot_proba = booster.predict(X_val[boot_idx])
    boot_preds[frac] = (boot_proba >= 0.5).astype(np.int8)

    n_pos_train = int(y_all[sub_idx].sum())
    curve_rows.append({
        "fraction": frac,
        "n_train": int(sub_idx.size),
        "n_pos_train": n_pos_train,
        "mcc": float(mcc_point),
    })
    print(f"  frac={frac:>4.0%}  n_train={sub_idx.size:>10,}  MCC={mcc_point:.4f}")
    del booster, val_proba, val_pred, boot_proba
    gc.collect()

curve_df = pl.DataFrame(curve_rows)
print(curve_df)

# %% [markdown]
# ### 3.5 Bootstrap CIs (per-fraction ribbon + consecutive deltas)

# %%
boot_mcc = bootstrap_mcc(y_boot, boot_preds, N_BOOTSTRAP)

# Per-fraction 95% CI (the ribbon).
ci_lo, ci_hi = {}, {}
for frac in FRACTIONS:
    ci_lo[frac], ci_hi[frac] = np.percentile(boot_mcc[frac], [2.5, 97.5])

# Consecutive-delta point estimates and 95% CIs (paired resamples).
delta_rows = []
for i in range(len(FRACTIONS) - 1):
    f0, f1 = FRACTIONS[i], FRACTIONS[i + 1]
    mcc0 = curve_df.filter(pl.col("fraction") == f0)["mcc"].item()
    mcc1 = curve_df.filter(pl.col("fraction") == f1)["mcc"].item()
    delta_point = mcc1 - mcc0
    delta_boot = boot_mcc[f1] - boot_mcc[f0]
    d_lo, d_hi = np.percentile(delta_boot, [2.5, 97.5])
    straddles_zero = bool(d_lo <= 0.0 <= d_hi)
    converged = bool(abs(delta_point) < P05_DELTA_THRESHOLD and straddles_zero)
    delta_rows.append({
        "from_fraction": f0, "to_fraction": f1,
        "delta_mcc": float(delta_point),
        "delta_ci_low": float(d_lo), "delta_ci_high": float(d_hi),
        "ci_straddles_zero": straddles_zero,
        "converged": converged,
    })

delta_df = pl.DataFrame(delta_rows)
print(delta_df)

# %% [markdown]
# ### 3.5b Leaked instance-grain baseline figure
#
# This figure plots the instance-grain curve above. It is retained only as the
# leakage baseline: the prediction-point ablation (3.7) and the episode re-check
# (3.8) show that the instance-grain matrix leaks lifecycle history into the
# label, so its near-0.97 MCC is inflated and is NOT the RQ1 adequacy evidence.
# The honest RQ1 adequacy curve is the episode-grain curve in Section 3.9, and it
# is the one the P05 decision (Section 4) and the main figure (Section 5) use.

# %%
_xs = np.array([r["n_train"] for r in curve_rows], dtype=float)
_ys = np.array([curve_df.filter(pl.col("fraction") == f)["mcc"].item() for f in FRACTIONS])
_lo = np.array([ci_lo[f] for f in FRACTIONS])
_hi = np.array([ci_hi[f] for f in FRACTIONS])
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(_xs, _lo, _hi, alpha=0.20, color="#9467bd", label="95% bootstrap CI")
ax.plot(_xs, _ys, marker="o", color="#9467bd", label="Validation MCC (leaked)")
for f, x, y in zip(FRACTIONS, _xs, _ys):
    ax.annotate(f"{f:.0%}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Training instances (log scale)")
ax.set_ylabel("Matthews correlation coefficient (MCC)")
ax.set_title("Google Cluster Traces - instance-grain LEAKED baseline (not RQ1 adequacy)")
ax.text(0.98, 0.04, "Leakage baseline only - see Section 3.9 for RQ1 adequacy",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="grey", alpha=0.9))
ax.legend(loc="lower right")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(str(LEAKED_BASELINE_PNG), dpi=150)
print(f"Leaked-baseline figure saved: {LEAKED_BASELINE_PNG}")
plt.show()

# %% [markdown]
# ### 3.6 Leakage diagnostics (feature importance + tier ablation)
#
# The learning curve saturates near MCC 0.97 from 1% of the data, which can
# indicate label leakage rather than a genuinely easy target. Two diagnostics
# localize the cause. (a) Gain-based feature importance for a full-feature
# model shows whether one or two columns dominate. (b) MCC under feature-subset
# ablations shows where the signal lives: if MCC stays high on Tier 1 alone,
# the lifecycle-derived history carries it; if it stays high on Tier 2/3 alone,
# label-correlated missingness does (failures crash before emitting usage, so
# their early-runtime/windowed features are null). Dropping the lifecycle
# history columns isolates whether those counts reach past the intended
# submission/scheduling prediction point. These also seed the Chapter 4
# ablation and the construct-validity check.

# %%
# Column index map for subsetting X_all (columns follow FEATURE_COLS order).
COL_INDEX = {c: i for i, c in enumerate(FEATURE_COLS)}

# Lifecycle-derived Tier 1 history: counts/positions that span the instance's
# trajectory up to its terminal event (the primary leakage suspects).
HISTORY_COLS = [
    "prior_fail_count", "has_prior_fail", "resubmission_count",
    "prior_evict_count", "first_resubmission", "lifecycle_position",
]

# Cap the diagnostic training size for speed; the curve shows MCC saturates well
# before this, so a capped fit reproduces the full-data MCC closely.
DIAG_TRAIN_ROWS = min(train_idx.size, 2_000_000)
if DIAG_TRAIN_ROWS < train_idx.size:
    diag_train_idx, _ = train_test_split(
        train_idx, train_size=DIAG_TRAIN_ROWS, random_state=SEED, stratify=y_all[train_idx]
    )
else:
    diag_train_idx = train_idx
y_diag = y_all[diag_train_idx]


def _fit_eval(cols: list[str]) -> tuple[float, lgb.Booster]:
    """Train a baseline LightGBM on the given feature columns; return
    (validation MCC, booster). Columns are selected positionally from X_all."""
    idx = [COL_INDEX[c] for c in cols]
    dtrain = lgb.Dataset(X_all[np.ix_(diag_train_idx, idx)], label=y_diag, free_raw_data=True)
    bst = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_NUM_ROUNDS)
    pred = (bst.predict(X_val[:, idx]) >= 0.5).astype(np.int8)
    return float(matthews_corrcoef(y_val, pred)), bst


# %%
# Tier ablation: localize where the high MCC comes from.
ablations = {
    "all_tiers": FEATURE_COLS,
    "tier1_only": TIER1_COLS,
    "tier2_3_only": TIER2_COLS + TIER3_COLS,
    "all_minus_history": [c for c in FEATURE_COLS if c not in HISTORY_COLS],
    "tier1_minus_history": [c for c in TIER1_COLS if c not in HISTORY_COLS],
}

ablation_rows = []
full_booster = None
print(f"Tier ablation (train on {DIAG_TRAIN_ROWS:,} rows, eval on the {y_val.size:,}-row val fold):")
for name, cols in ablations.items():
    mcc_a, bst = _fit_eval(cols)
    ablation_rows.append({"subset": name, "n_features": len(cols), "val_mcc": mcc_a})
    print(f"  {name:22s} n_feat={len(cols):3d}  MCC={mcc_a:.4f}")
    if name == "all_tiers":
        full_booster = bst
    else:
        del bst
    gc.collect()

ablation_df = pl.DataFrame(ablation_rows)
ablation_df.write_csv(str(TABLES_DIR / "google_leakage_ablation.csv"))

mcc_by_subset = {r["subset"]: r["val_mcc"] for r in ablation_rows}

# %%
# Gain-based feature importance for the full-feature model. The Dataset was
# built in FEATURE_COLS order, so the importance vector aligns with it.
gain = full_booster.feature_importance(importance_type="gain")
split = full_booster.feature_importance(importance_type="split")
importance_df = (
    pl.DataFrame({"feature": FEATURE_COLS, "gain": gain, "split": split})
    .with_columns(
        (pl.col("gain") / max(int(gain.sum()), 1)).alias("gain_frac"),
        pl.Series("tier", [
            "tier1" if c in TIER1_COLS else "tier2" if c in TIER2_COLS else "tier3"
            for c in FEATURE_COLS
        ]),
    )
    .sort("gain", descending=True)
)
importance_df.write_csv(str(TABLES_DIR / "google_feature_importance.csv"))
print("\nTop 15 features by gain:")
print(importance_df.head(15))
top1_frac = importance_df["gain_frac"][0]
top3_frac = float(importance_df["gain_frac"][:3].sum())
print(f"\nTop feature gain share: {top1_frac:.1%}; top 3: {top3_frac:.1%}")
del full_booster
gc.collect()

# %%
# Interpretive screens (informational; surfaced in the verification log).
mcc_all = mcc_by_subset["all_tiers"]
mcc_t1 = mcc_by_subset["tier1_only"]
mcc_t23 = mcc_by_subset["tier2_3_only"]
mcc_no_hist = mcc_by_subset["all_minus_history"]
history_drop = mcc_all - mcc_no_hist
print(f"\nMCC all={mcc_all:.4f} | tier1_only={mcc_t1:.4f} | tier2_3_only={mcc_t23:.4f} "
      f"| all_minus_history={mcc_no_hist:.4f} (drop {history_drop:+.4f})")

# Premature-saturation screen: a curve flat from 1% at high MCC is a leakage
# signature. Fails the screen (ok=False) when the 1% and 100% MCC are within
# 0.02 and the ceiling is above 0.90 -> investigate before trusting the result.
mcc_1pct = curve_df.filter(pl.col("fraction") == FRACTIONS[0])["mcc"].item()
mcc_100pct = curve_df.filter(pl.col("fraction") == 1.0)["mcc"].item()
premature_saturation = (abs(mcc_100pct - mcc_1pct) < 0.02 and mcc_100pct > 0.90)
record_check(
    "Section 3.6: no premature saturation (leakage screen)",
    expected="MCC at 1% and 100% differ by >= 0.02, or ceiling <= 0.90",
    observed=f"1%={mcc_1pct:.4f}, 100%={mcc_100pct:.4f}",
    ok=(not premature_saturation),
    notes="Flat-from-1% high MCC suggests leakage; inspect importance + ablation below.",
)
record_check(
    "Section 3.6: lifecycle history is not the sole driver",
    expected="MCC drop when history removed is modest (< 0.10), or tier1-only is not near-perfect",
    observed=f"all_minus_history drop={history_drop:+.4f}, tier1_only={mcc_t1:.4f}",
    ok=(history_drop < 0.10 or mcc_t1 < 0.90),
    notes="Large drop with near-perfect tier1-only points at the lifecycle counts as the leak.",
)
record_check(
    "Section 3.6: Tier 2/3 missingness is not the sole driver",
    expected="tier2_3-only MCC well below the full-feature MCC",
    observed=f"tier2_3_only={mcc_t23:.4f} vs all={mcc_all:.4f}",
    ok=(mcc_t23 < mcc_all - 0.05 or mcc_all <= 0.90),
    notes="Near-full MCC from Tier 2/3 alone implicates label-correlated missingness.",
)

# %% [markdown]
# ### 3.7 Prediction-point ablation (honest MCC per prediction point)
#
# The full matrix mixes features from different points in an instance's life
# and scores them against the terminal outcome, which inflates MCC: the Tier 3
# long windows, has_hardware_counters, and the post-hoc collection_size_at_submit
# / lifecycle_position only exist once the instance has run, so they partly
# encode the label. This ablation scores only the features admissible at each
# prediction point, giving an honest RQ1 baseline. Tier 3 is held out of every
# point (it is the confounded ablation tier); collection_size_at_submit,
# lifecycle_position, and the resubmission-history counts are excluded from the
# conservative submission set pending their submit-time / strictly-prior
# recomputation, so that number is a floor. A second submission variant adds
# the history back to quantify how much the (currently leaky) V10 signal
# contributes.

# %%
# Admissible feature groups, derived by name from FEATURE_COLS (robust to the
# platform / priority one-hot suffixes).
SUBMIT_TEMPORAL_COLS = [c for c in FEATURE_COLS if c.startswith("submit_")]
PRIORITY_ONEHOT_COLS = [c for c in FEATURE_COLS if c.startswith("priority_tier_")]
PLATFORM_ONEHOT_COLS = [c for c in FEATURE_COLS if c.startswith("platform_")]
REQUEST_COLS = [c for c in ("cpu_request", "memory_request", "request_ratio") if c in FEATURE_COLS]

# Conservative submission set: known at first submission; no post-hoc or
# runtime-derived columns.
submission_cols = (
    REQUEST_COLS
    + PRIORITY_ONEHOT_COLS
    + (["scheduling_class"] if "scheduling_class" in FEATURE_COLS else [])
    + SUBMIT_TEMPORAL_COLS
)
# Submission + the (currently leaky) resubmission history, to size the V10
# contribution before it is recomputed as strictly-prior history.
submission_plus_history_cols = submission_cols + [c for c in HISTORY_COLS if c in FEATURE_COLS]
# At-scheduling adds queue time and the assigned-machine platform.
at_scheduling_cols = (
    submission_cols
    + (["queue_time"] if "queue_time" in FEATURE_COLS else [])
    + PLATFORM_ONEHOT_COLS
)
# Early-runtime adds the Tier 2 first-30-60s slopes/ramps and the counter values.
early_runtime_cols = (
    at_scheduling_cols
    + [c for c in TIER2_COLS if c in FEATURE_COLS]
    + (["has_hardware_counters"] if "has_hardware_counters" in FEATURE_COLS else [])
)

prediction_points = {
    "submission_conservative": submission_cols,
    "submission_plus_history": submission_plus_history_cols,
    "at_scheduling": at_scheduling_cols,
    "early_runtime": early_runtime_cols,
    "all_features_reference": FEATURE_COLS,
}

pp_rows = []
print(f"Prediction-point ablation (train on {DIAG_TRAIN_ROWS:,} rows, eval on the {y_val.size:,}-row val fold):")
for name, cols in prediction_points.items():
    mcc_pp, bst = _fit_eval(cols)
    pp_rows.append({"prediction_point": name, "n_features": len(cols), "val_mcc": mcc_pp})
    print(f"  {name:24s} n_feat={len(cols):3d}  MCC={mcc_pp:.4f}")
    del bst
    gc.collect()

pp_df = pl.DataFrame(pp_rows)
pp_df.write_csv(str(TABLES_DIR / "google_prediction_point_ablation.csv"))
pp_mcc = {r["prediction_point"]: r["val_mcc"] for r in pp_rows}

# %%
mcc_submission = pp_mcc["submission_conservative"]
mcc_all_ref = pp_mcc["all_features_reference"]
leakage_gap = mcc_all_ref - mcc_submission
print(f"\nHonest at-submission MCC (conservative): {mcc_submission:.4f}")
print(f"All-features MCC:                        {mcc_all_ref:.4f}")
print(f"Inflation from non-submission / leaky features: {leakage_gap:+.4f}")
record_check(
    "Section 3.7: honest at-submission MCC recorded (RQ1 baseline)",
    expected="reported (conservative submission-only feature set)",
    observed=f"submission={mcc_submission:.4f}, all={mcc_all_ref:.4f}, gap={leakage_gap:+.4f}",
    ok=True,
    notes="Conservative floor: history, post-hoc collection features, and Tier 3 excluded.",
)

# %% [markdown]
# ### 3.8 Episode-grain leakage re-check (per-attempt redesign)
#
# Sections 3.6-3.7 localized the saturation to the lifecycle history reaching
# past the submission prediction point: at the instance grain,
# `submission_plus_history` scored an inflated MCC because `prior_fail_count` /
# `resubmission_count` are computed over the whole lifecycle, including the
# resubmissions that produce the terminal label. Notebook 10 Section 11 rebuilds
# the matrix at the **scheduled-episode** grain, where history is strictly prior
# (leakage guard PASS). With Phase B (notebook 10 Section 12) the full episode
# matrix `episode_features` carries Tier 2/3 too, so this section runs the whole
# prediction-point ablation at episode grain: submission, submission+history,
# at-scheduling, early-runtime, and all-reference. The leakage check compares
# `submission_plus_history` against the instance-grain value from 3.7
# (`pp_mcc["submission_plus_history"]`); the early-runtime point is the honest
# RQ1 baseline (where the >0.90 target is tested).
#
# Memory note: the episode base has ~90M rows. The per-instance negative cap,
# the group-aware (instance-keyed) train/test split, and the subsample are pushed
# into BigQuery so only a few-million-row extract reaches the Colab box. The cap
# and split mirror the helpers in notebook 10 Section 11.5. The validation side
# is left uncapped so MCC is read on the natural episode distribution.

# %%
from google.colab import auth

from utils.bq_client import get_client, table_ref

auth.authenticate_user()  # idempotent; ensures ADC for the BigQuery extract.
_bq = get_client()
# Full episode matrix (Tier 1 + 2 + 3) from notebook 10 Section 12 (Phase B).
# Run Section 12 first. If only Phase A has run, set this to
# 'episode_lifecycle_features_base' to score the submission points alone (the
# at-scheduling / early-runtime points then resolve to the submission set).
EPISODE_MATRIX_TABLE = 'episode_features'
EP_CAP_NEG = 5            # per-instance negative cap (notebook 10 Section 11.5 default)
EP_TRAIN_PERMILLE = 40    # ~4% subsample of the capped train side
EP_VAL_PERMILLE = 60      # ~6% subsample of the uncapped val side
EP_TEST_GRP = 0           # instance-key hash bucket (of 5) held out as validation


def _episode_extract_sql(*, train: bool) -> str:
    """Build the BigQuery extract for the episode leakage re-check.

    Group split: instances are bucketed 0..4 by a hash of the instance key;
    bucket `EP_TEST_GRP` is the validation side, so no instance straddles the
    split. Cap: train negatives are limited to `EP_CAP_NEG` per instance
    (positives never capped); validation is uncapped. Subsample: a hash bucket
    of the episode key thins each side to a few million rows.
    """
    key = "CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING))"
    epkey = ("CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING),"
             "'_',CAST(sched_seq AS STRING))")
    grp_pred = f"_grp != {EP_TEST_GRP}" if train else f"_grp = {EP_TEST_GRP}"
    permille = EP_TRAIN_PERMILLE if train else EP_VAL_PERMILLE
    cap_clause = (
        f"WHERE failure_label = 1 OR _rn <= {EP_CAP_NEG}" if train else ""
    )
    # Leading comma so the val side (no _rn) leaves a clean ``SELECT *``.
    rn_col = (
        ", ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index, failure_label "
        "ORDER BY _ephash) AS _rn"
    ) if train else ""
    return f"""
WITH split AS (
    SELECT
        b.*,
        MOD(ABS(FARM_FINGERPRINT({key})), 5) AS _grp,
        ABS(FARM_FINGERPRINT({epkey})) AS _ephash
    FROM {table_ref(EPISODE_MATRIX_TABLE)} b
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
SELECT * FROM capped
WHERE MOD(_ephash, 1000) < {permille}
"""


print("Pulling episode train/val extracts from BigQuery (capped + group-split + subsampled) ...")
ep_train_pd = _bq.query(_episode_extract_sql(train=True)).to_dataframe()
ep_val_pd = _bq.query(_episode_extract_sql(train=False)).to_dataframe()
ep_train = pl.from_pandas(ep_train_pd)
ep_val = pl.from_pandas(ep_val_pd)
del ep_train_pd, ep_val_pd
gc.collect()
print(f"Episode train extract: {ep_train.height:,} rows "
      f"({int(ep_train.filter(pl.col('failure_label') == 1).height):,} positive)")
print(f"Episode val extract:   {ep_val.height:,} rows "
      f"({int(ep_val.filter(pl.col('failure_label') == 1).height):,} positive)")

# %%
# Episode feature groups, derived by name from the extract (robust to one-hot
# suffixes). lifecycle_position is instance-grain only and absent here by design.
_ep_cols = set(ep_train.columns)
EP_SUBMIT_TEMPORAL = sorted(c for c in _ep_cols if c.startswith("submit_"))
EP_PRIORITY_ONEHOT = sorted(c for c in _ep_cols if c.startswith("priority_tier_"))
EP_REQUEST = [c for c in ("cpu_request", "memory_request", "request_ratio") if c in _ep_cols]
EP_HISTORY = [c for c in ("prior_fail_count", "has_prior_fail", "resubmission_count",
                          "prior_evict_count", "first_resubmission") if c in _ep_cols]

EP_PLATFORM_ONEHOT = sorted(c for c in _ep_cols if c.startswith("platform_"))
EP_TIER2 = [c for c in (
    "cpu_slope_5s", "cpu_slope_15s", "cpu_slope_30s",
    "memory_slope_5s", "memory_slope_15s", "memory_slope_30s",
    "initial_cpu_ramp", "initial_memory_ramp", "first_interval_util_ratio",
    "cpi_value", "mapi_value",
) if c in _ep_cols]

ep_submission_cols = (
    EP_REQUEST
    + EP_PRIORITY_ONEHOT
    + (["scheduling_class"] if "scheduling_class" in _ep_cols else [])
    + EP_SUBMIT_TEMPORAL
)
ep_submission_plus_history_cols = ep_submission_cols + EP_HISTORY
# At-scheduling adds queue time and the episode's assigned-machine platform.
ep_at_scheduling_cols = (
    ep_submission_cols
    + (["queue_time"] if "queue_time" in _ep_cols else [])
    + EP_PLATFORM_ONEHOT
)
# Early-runtime adds the Tier 2 slopes/ramps and the counter-availability flag.
# This is the RQ1-relevant point: where the >0.90 MCC target is tested.
ep_early_runtime_cols = (
    ep_at_scheduling_cols
    + EP_TIER2
    + (["has_hardware_counters"] if "has_hardware_counters" in _ep_cols else [])
)
# All-feature reference (every numeric feature present, including Tier 3).
_ep_nonfeature = {"collection_id", "instance_index", "sched_seq", LABEL_COL,
                  "outcome", "_grp", "_ephash", "_rn"}
ep_all_reference_cols = [c for c in ep_train.columns if c not in _ep_nonfeature]


def _ep_fit_eval(cols: list[str]) -> tuple[float, np.ndarray, np.ndarray]:
    """Train the baseline LightGBM on the episode extract; return
    (validation MCC, val labels, val predicted labels)."""
    Xtr = ep_train.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    ytr = ep_train.select(LABEL_COL).to_numpy().ravel().astype(np.int8)
    Xv = ep_val.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    yv = ep_val.select(LABEL_COL).to_numpy().ravel().astype(np.int8)
    dtrain = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
    bst = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_NUM_ROUNDS)
    pred = (bst.predict(Xv) >= 0.5).astype(np.int8)
    mcc = float(matthews_corrcoef(yv, pred))
    del Xtr, ytr, dtrain, bst
    gc.collect()
    return mcc, yv, pred


def _ep_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """Percentile 95% CI for MCC over resamples of the (paired) val arrays."""
    local = np.random.default_rng(SEED)
    n = y_true.size
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = local.integers(0, n, size=n)
        boots[b] = mcc_of(y_true[idx], y_pred[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


ep_points = {
    "episode_submission_conservative": ep_submission_cols,
    "episode_submission_plus_history": ep_submission_plus_history_cols,
    "episode_at_scheduling": ep_at_scheduling_cols,
    "episode_early_runtime": ep_early_runtime_cols,
    "episode_all_reference": ep_all_reference_cols,
}
ep_rows = []
print(f"Episode prediction-point re-check "
      f"(train {ep_train.height:,} rows, eval {ep_val.height:,} rows):")
for name, cols in ep_points.items():
    mcc_ep, yv, pred = _ep_fit_eval(cols)
    lo, hi = _ep_bootstrap_ci(yv, pred)
    ep_rows.append({"prediction_point": name, "n_features": len(cols),
                    "val_mcc": mcc_ep, "ci_low": lo, "ci_high": hi})
    print(f"  {name:32s} n_feat={len(cols):3d}  MCC={mcc_ep:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

ep_pp_df = pl.DataFrame(ep_rows)
ep_pp_df.write_csv(str(TABLES_DIR / "google_episode_prediction_point_recheck.csv"))
ep_mcc = {r["prediction_point"]: r["val_mcc"] for r in ep_rows}

# %%
# Compare the episode submission+history MCC to the instance-grain value from
# Section 3.7. The redesign succeeds when the episode value drops materially
# (the leaked lifecycle history no longer reaches past submission).
instance_sub_hist = pp_mcc.get("submission_plus_history")
episode_sub_hist = ep_mcc["episode_submission_plus_history"]
episode_sub_only = ep_mcc["episode_submission_conservative"]
drop_vs_instance = (instance_sub_hist - episode_sub_hist) if instance_sub_hist is not None else float("nan")
print(f"\nsubmission_plus_history MCC  instance-grain (3.7): {instance_sub_hist}")
print(f"                             episode-grain  (3.8): {episode_sub_hist:.4f}")
print(f"Drop from regrain: {drop_vs_instance:+.4f}")
print(f"Episode submission-only MCC: {episode_sub_only:.4f} "
      f"(strictly-prior history adds {episode_sub_hist - episode_sub_only:+.4f})")

# Runtime prediction points: the RQ1 >0.90 target is tested at early-runtime,
# not at submission. Present when the full episode matrix (Phase B) was used.
if "episode_at_scheduling" in ep_mcc:
    print(f"\nEpisode at_scheduling MCC:  {ep_mcc['episode_at_scheduling']:.4f}")
    print(f"Episode early_runtime MCC:  {ep_mcc['episode_early_runtime']:.4f}  (RQ1 target > 0.90)")
    print(f"Episode all_reference MCC:  {ep_mcc['episode_all_reference']:.4f}")
    record_check(
        "Section 3.8: early-runtime RQ1 point reported on the episode matrix",
        expected="reported (>0.90 is the RQ1 target; Tier 2/3 missingness is label-correlated)",
        observed=f"early_runtime={ep_mcc['episode_early_runtime']:.4f}, "
                 f"at_scheduling={ep_mcc['episode_at_scheduling']:.4f}",
        ok=True,
        notes="Honest episode-grain runtime baseline; interpret alongside Tier 2/3 null fractions.",
    )

record_check(
    "Section 3.8: episode-grain regrain removes the history leakage",
    expected=f"episode submission_plus_history << instance-grain "
             f"({instance_sub_hist if instance_sub_hist is not None else 'n/a'}); drop >= 0.10",
    observed=f"episode={episode_sub_hist:.4f}, instance={instance_sub_hist}, drop={drop_vs_instance:+.4f}",
    ok=(instance_sub_hist is not None and (instance_sub_hist - episode_sub_hist) >= 0.10),
    notes="Strictly-prior history at episode grain no longer encodes the terminal label.",
)

del ep_train, ep_val
gc.collect()

# %% [markdown]
# ### 3.9 Episode-grain learning curve (RQ1 adequacy base)
#
# Sections 3.1-3.7 fit the curve on the instance-grain matrix, which 3.7 and 3.8
# show is leakage-inflated near MCC 0.97. That curve is the leaked baseline
# (figure in 3.5b) and is not the RQ1 adequacy evidence. The honest adequacy
# curve is fit here on the per-attempt `episode_features` census at the
# early-runtime prediction point (the RQ1-reported configuration, V33), reusing
# `ep_early_runtime_cols` from 3.8.
#
# The base is pulled capped (per-instance negatives limited to `EP_CAP_NEG`,
# positives never capped) and instance-key group-split, so no instance straddles
# the train/validation boundary and an instance's earlier and later attempts
# never leak across it. It is then thinned to roughly `MAX_BASE_ROWS` episodes,
# exactly mirroring the instance-grain P05 design: if MCC has asymptoted at the
# cap, the full ~90M-episode census is more than adequate; if not, raise the
# permille (or use a High-RAM runtime) and re-run. The permilles below are sized
# for a base near the cap; the printed row counts confirm the realized sizes.

# %%
# Free the instance-grain arrays before pulling the episode base (peak memory).
del X_all, y_all, X_val, y_val
gc.collect()

EP_CURVE_TRAIN_PERMILLE = 250   # ~25% of the capped train census -> base near MAX_BASE_ROWS
EP_CURVE_VAL_PERMILLE = 80      # held-out instance-key bucket, thinned for eval speed


def _episode_curve_extract_sql(*, train: bool) -> str:
    """Capped + instance-key group-split episode extract for the learning curve.

    Mirrors the 3.8 re-check extract but with curve-sized permilles so the train
    side is a base large enough to take learning-curve fractions of. Train caps
    negatives per instance to ``EP_CAP_NEG`` (positives never capped); the
    held-out bucket ``EP_TEST_GRP`` is the uncapped validation side.
    """
    key = "CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING))"
    epkey = ("CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING),"
             "'_',CAST(sched_seq AS STRING))")
    grp_pred = f"_grp != {EP_TEST_GRP}" if train else f"_grp = {EP_TEST_GRP}"
    permille = EP_CURVE_TRAIN_PERMILLE if train else EP_CURVE_VAL_PERMILLE
    cap_clause = f"WHERE failure_label = 1 OR _rn <= {EP_CAP_NEG}" if train else ""
    rn_col = (
        ", ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index, failure_label "
        "ORDER BY _ephash) AS _rn"
    ) if train else ""
    return f"""
WITH split AS (
    SELECT
        b.*,
        MOD(ABS(FARM_FINGERPRINT({key})), 5) AS _grp,
        ABS(FARM_FINGERPRINT({epkey})) AS _ephash
    FROM {table_ref(EPISODE_MATRIX_TABLE)} b
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
SELECT * FROM capped
WHERE MOD(_ephash, 1000) < {permille}
"""


print("Pulling episode learning-curve base (capped + instance-key group-split) ...")
ep_pool_pd = _bq.query(_episode_curve_extract_sql(train=True)).to_dataframe()
ep_cval_pd = _bq.query(_episode_curve_extract_sql(train=False)).to_dataframe()
ep_pool = pl.from_pandas(ep_pool_pd)
ep_cval = pl.from_pandas(ep_cval_pd)
del ep_pool_pd, ep_cval_pd
gc.collect()

# RQ1 early-runtime feature set (from 3.8), intersected with the pulled columns.
EP_CURVE_FEATURES = [c for c in ep_early_runtime_cols if c in ep_pool.columns]
Xep = ep_pool.select([pl.col(c).cast(pl.Float32) for c in EP_CURVE_FEATURES]).to_numpy()
yep = ep_pool.select(LABEL_COL).to_numpy().ravel().astype(np.int8)
Xep_val = ep_cval.select([pl.col(c).cast(pl.Float32) for c in EP_CURVE_FEATURES]).to_numpy()
yep_val = ep_cval.select(LABEL_COL).to_numpy().ravel().astype(np.int8)
del ep_pool, ep_cval
gc.collect()
print(f"Episode curve base: train pool {Xep.shape[0]:,} x {len(EP_CURVE_FEATURES)} feats "
      f"({int(yep.sum()):,} positive); validation {Xep_val.shape[0]:,} "
      f"({int(yep_val.sum()):,} positive).")

# Stratified bootstrap subset of the validation fold (for CI speed).
if yep_val.size > BOOTSTRAP_VAL_CAP:
    ep_boot_idx, _ = train_test_split(
        np.arange(yep_val.size), train_size=BOOTSTRAP_VAL_CAP,
        random_state=SEED, stratify=yep_val,
    )
else:
    ep_boot_idx = np.arange(yep_val.size)
yep_boot = yep_val[ep_boot_idx]

# %%
ep_curve_rows = []
ep_boot_preds: dict[float, np.ndarray] = {}
_ep_pool_idx = np.arange(Xep.shape[0])
for frac in FRACTIONS:
    n_frac = max(2, int(round(frac * _ep_pool_idx.size)))
    if n_frac >= _ep_pool_idx.size:
        sub_idx = _ep_pool_idx
    else:
        sub_idx, _ = train_test_split(
            _ep_pool_idx, train_size=n_frac, random_state=SEED, stratify=yep
        )
    dtrain = lgb.Dataset(Xep[sub_idx], label=yep[sub_idx], free_raw_data=True)
    booster = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_NUM_ROUNDS)
    del dtrain
    gc.collect()
    val_pred = (booster.predict(Xep_val) >= 0.5).astype(np.int8)
    mcc_point = matthews_corrcoef(yep_val, val_pred)
    ep_boot_preds[frac] = (booster.predict(Xep_val[ep_boot_idx]) >= 0.5).astype(np.int8)
    ep_curve_rows.append({
        "fraction": frac, "n_train": int(sub_idx.size),
        "n_pos_train": int(yep[sub_idx].sum()), "mcc": float(mcc_point),
    })
    print(f"  frac={frac:>4.0%}  n_train={sub_idx.size:>10,}  MCC={mcc_point:.4f}")
    del booster, val_pred
    gc.collect()

ep_curve_df = pl.DataFrame(ep_curve_rows)
ep_curve_df.write_csv(str(TABLES_DIR / "google_episode_learning_curve.csv"))
print(ep_curve_df)

# %%
# Per-fraction CIs (ribbon) and consecutive-delta CIs (paired resamples), exactly
# as the instance curve in 3.5 but on the episode validation fold.
ep_boot_mcc = bootstrap_mcc(yep_boot, ep_boot_preds, N_BOOTSTRAP)
ep_ci_lo, ep_ci_hi = {}, {}
for frac in FRACTIONS:
    ep_ci_lo[frac], ep_ci_hi[frac] = np.percentile(ep_boot_mcc[frac], [2.5, 97.5])

ep_delta_rows = []
for i in range(len(FRACTIONS) - 1):
    f0, f1 = FRACTIONS[i], FRACTIONS[i + 1]
    mcc0 = ep_curve_df.filter(pl.col("fraction") == f0)["mcc"].item()
    mcc1 = ep_curve_df.filter(pl.col("fraction") == f1)["mcc"].item()
    delta_point = mcc1 - mcc0
    delta_boot = ep_boot_mcc[f1] - ep_boot_mcc[f0]
    d_lo, d_hi = np.percentile(delta_boot, [2.5, 97.5])
    straddles_zero = bool(d_lo <= 0.0 <= d_hi)
    converged = bool(abs(delta_point) < P05_DELTA_THRESHOLD and straddles_zero)
    ep_delta_rows.append({
        "from_fraction": f0, "to_fraction": f1,
        "delta_mcc": float(delta_point),
        "delta_ci_low": float(d_lo), "delta_ci_high": float(d_hi),
        "ci_straddles_zero": straddles_zero,
        "converged": converged,
    })

ep_delta_df = pl.DataFrame(ep_delta_rows)
print(ep_delta_df)
ep_n_train_by_frac = {r["fraction"]: r["n_train"] for r in ep_curve_rows}

# %% [markdown]
# ---
# ## 4. P05 adequacy decision (episode-grain RQ1 curve)
#
# P05: the modeling set is adequate when consecutive MCC deltas fall below
# `P05_DELTA_THRESHOLD` (0.005) in absolute value AND the 95% bootstrap CI on the
# delta straddles zero. The decision is read off the final transition
# (50% -> 100% of the base). This decision uses the **episode-grain** curve from
# Section 3.9 (the RQ1 modeling grain), not the instance-grain leaked baseline
# (3.5b). The leaked baseline's P05 is logged separately below for context only.

# %%
final = ep_delta_rows[-1]
p05_met = final["converged"]
print(f"Final transition {final['from_fraction']:.0%} -> {final['to_fraction']:.0%} "
      f"(episode-grain RQ1 curve):")
print(f"  delta MCC          = {final['delta_mcc']:+.4f} (threshold {P05_DELTA_THRESHOLD})")
print(f"  95% CI on delta    = [{final['delta_ci_low']:+.4f}, {final['delta_ci_high']:+.4f}]")
print(f"  CI straddles zero  = {final['ci_straddles_zero']}")
print(f"  P05 criterion met  = {p05_met}")

record_check(
    "Section 4: P05 episode-grain (RQ1) adequacy at the final transition",
    expected=f"|delta MCC| < {P05_DELTA_THRESHOLD} and 95% CI straddles 0",
    observed=(f"delta={final['delta_mcc']:+.4f}, "
              f"CI=[{final['delta_ci_low']:+.4f}, {final['delta_ci_high']:+.4f}]"),
    ok=p05_met,
    notes=("Adequate -> the full ~90M-episode census is more than sufficient for "
           "RQ1. Not met -> raise EP_CURVE_TRAIN_PERMILLE (or use High-RAM) and "
           "re-run Section 3.9."),
)

# Context only: the leaked instance-grain baseline's P05 (do not use for RQ1).
_leaked_final = delta_rows[-1]
record_check(
    "Section 4: leaked instance-grain baseline P05 (context only)",
    expected="informational; instance-grain curve is leakage-inflated (3.7, 3.8)",
    observed=(f"delta={_leaked_final['delta_mcc']:+.4f}, "
              f"CI=[{_leaked_final['delta_ci_low']:+.4f}, {_leaked_final['delta_ci_high']:+.4f}]"),
    ok=True,
    notes="Reported so the leaked-baseline curve stays auditable; not the RQ1 adequacy test.",
)

# Re-write the verification log so the P05 row is captured alongside Section 1.
verification_df = pl.DataFrame(verification_rows)
verification_df.write_csv(str(VERIFICATION_CSV))
print(f"\nVerification log updated: {VERIFICATION_CSV}")

# %% [markdown]
# ---
# ## 5. Learning-curve figure (episode-grain RQ1 adequacy)
#
# Episode-grain MCC vs base fraction with the 95% bootstrap CI ribbon, at the
# early-runtime prediction point (the RQ1-reported configuration). The x-axis is
# the absolute training-episode count (log scale) so the asymptote is legible.
# This is the RQ1 adequacy figure; the leaked instance-grain baseline is the
# separate figure written in Section 3.5b.

# %%
xs = np.array([ep_n_train_by_frac[f] for f in FRACTIONS], dtype=float)
ys = np.array([ep_curve_df.filter(pl.col("fraction") == f)["mcc"].item() for f in FRACTIONS])
lo = np.array([ep_ci_lo[f] for f in FRACTIONS])
hi = np.array([ep_ci_hi[f] for f in FRACTIONS])

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(xs, lo, hi, alpha=0.20, color="#1f77b4", label="95% bootstrap CI")
ax.plot(xs, ys, marker="o", color="#1f77b4", label="Validation MCC (early-runtime)")
for f, x, y in zip(FRACTIONS, xs, ys):
    ax.annotate(f"{f:.0%}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Training episodes (attempts, log scale)")
ax.set_ylabel("Matthews correlation coefficient (MCC)")
ax.set_title("Google Cluster Traces - episode-grain learning curve (RQ1, early-runtime LightGBM)")
status = "P05 met (converged)" if p05_met else "P05 not met (raise EP_CURVE_TRAIN_PERMILLE)"
ax.text(0.98, 0.04, status, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
ax.legend(loc="lower right")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(str(LEARNING_CURVE_PNG), dpi=150)
print(f"Episode-grain RQ1 adequacy figure saved: {LEARNING_CURVE_PNG}")
plt.show()

# %% [markdown]
# ---
# ## 6. Summary
#
# - Distribution verification: `outputs/tables/google_preprocessing_verification.csv`.
# - Tier 3 inversion guardrail: `outputs/tables/tier3_inversion_check.csv`.
# - Leakage diagnostics: `outputs/tables/google_leakage_ablation.csv`,
#   `outputs/tables/google_feature_importance.csv`, and
#   `outputs/tables/google_prediction_point_ablation.csv`.
# - Learning curve: `outputs/figures/learning_curve_google.png`.
#
# **Next step.** Resolve any Section 3.6 leakage screen before trusting the
# MCC. If P05 is met, lock the working set
# (`{OUTPUT_DIR}/working_sets/google/v1_*/`) with its `SamplingManifest`,
# this learning-curve plot, and the feature schema, and tag the git release.
# If not met, raise `MAX_BASE_ROWS` toward the 100M ceiling (or use a High-RAM
# runtime) and re-run before modeling.

# %%
n_failed = pl.DataFrame(verification_rows).filter(~pl.col("ok")).height
print(f"Total checks: {len(verification_rows)}; failed: {n_failed}")
print(f"P05 working-set adequacy met: {p05_met}")
