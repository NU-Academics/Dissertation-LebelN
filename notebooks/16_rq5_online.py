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
# # RQ5. Online learning and concept drift (Backblaze)
#
# **Question.** How can online learning maintain sustained predictive performance
# (MCC, F1) under concept drift?
#
# **Design.** A static batch model is frozen at a starting year and never updated.
# An adaptive online learner starts from the same warm-up window and keeps learning
# as time advances. Both are scored on the same natural-prevalence monthly windows.
# The comparison between them, not the absolute level, is the substantive result.
#
# **The absolute target will not be met, and that is reported as a finding.**
# Backblaze failure prediction caps at MCC about 0.20 at natural prevalence (the RQ1
# result), so no model on this dataset starts anywhere near the 0.85 sustained-MCC
# threshold. The threshold is tested honestly and reported unmet. It is not lowered,
# and the evaluation prevalence is not switched to a more forgiving one to
# manufacture a pass. The informative questions are how fast a frozen model decays,
# whether an adaptive learner holds its level, how quickly drift is detected against
# a known onset, and how much retraining recovers.
#
# **Evaluation prevalence.** Every reported metric is computed on natural-prevalence
# data, consistent with the evaluation protocol used for RQ1 and RQ3. The
# undersampled working set is a training-side stream only.
#
# **Evaluation window.** Natural-prevalence data exists for 2022 (used to fit the
# operating threshold and calibration, never scored) and for 2023 through 2025 (the
# scored window, identical to the RQ1 test set). The evaluation window is therefore
# fixed at the 36 months of 2023 to 2025 for every cell, and the starting year varies
# how stale the frozen model is when that window opens. All static baselines are
# scored on the identical months, so their trajectories are directly comparable and
# the difference between them is model age alone. The 2021 static baseline is the
# checkpointed RQ1 model, so its aggregate MCC over the window must reproduce the
# RQ1 figure; Section 3 asserts that.
#
# **Prequential ordering.** Within the evaluation window, the model predicts month m
# before it learns from any observation in month m. Predicting and learning are
# separate calls and are never fused, so no reported score is contaminated by the
# labels it is being scored against.
#
# **Prior shift is a confound, not a finding.** The natural failure-day rate declines
# across the evaluation years (0.0048% in 2023, 0.0044% in 2024, 0.0037% in 2025).
# MCC is prevalence-sensitive, so a falling base rate moves the metric with no
# covariate drift and no staleness. Every monthly MCC is therefore reported twice:
# raw, and recomputed at a fixed prevalence. Only the fixed-prevalence series
# supports a claim about drift.
#
# **Compute.** River processes one observation at a time and the working set holds
# 19.2M rows, so the stream is a drive-level stratified subsample: drives are
# sampled, never rows, so every sampled drive keeps its complete ordered history.
# The subsample is a stated limitation of the analysis.

# %% [markdown]
# ## 0. Session setup

# %%
# River is pinned below 0.23 deliberately: later releases raise the numpy floor
# above the ceiling the rest of the stack holds, and an unpinned install resolves a
# numpy upgrade that can break unpickling of the saved ensembles mid-session. It is
# installed before anything is loaded for the same reason.
# !pip install -q polars pandas pyarrow scikit-learn xgboost lightgbm matplotlib google-cloud-storage "river>=0.21,<0.23"

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
from datetime import date

import numpy as np
import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR, CHECKPOINT_DIR

from src.models.ensemble import build_wrapper
from src.models.online import (
    build_online_learner,
    balanced_class_weights,
    frame_to_dicts,
)
from src.evaluation.drift_detectors import (
    ADWINDetector,
    KSDriftDetector,
    PageHinkleyDetector,
    PSIDriftDetector,
)
from src.evaluation.metrics import (
    _mcc,
    drift_detection_latency,
    f1_with_ci,
    mcc_at_fixed_prevalence,
    mcc_with_ci,
    performance_degradation_rate,
    performance_sustainment_window,
    pr_auc_with_ci,
    retraining_effectiveness,
    sustainment_reference,
)
from src.evaluation.hypothesis import one_sample_threshold_test

setup_drive()

# Stale-module guard: assert on a symbol that only exists in the current modules, so
# a warm runtime holding an old copy fails here rather than three hours in.
assert hasattr(ADWINDetector, 'drift_indices'), "stale src.evaluation module; restart the runtime"
assert callable(sustainment_reference), "stale src.evaluation.metrics; restart the runtime"

TABLES_DIR = OUTPUT_DIR / 'tables'
FEATURES_DIR = OUTPUT_DIR / 'features'
FIG_DIR = OUTPUT_DIR / 'figures' / 'rq5'
RQ5_CKPT_DIR = CHECKPOINT_DIR / 'rq5'
for d in [TABLES_DIR, FIG_DIR, RQ5_CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Configuration and the feature contract
#
# The feature contract is rebuilt from the schema and asserted against the RQ1
# checkpoint sidecar. The online learner must consume exactly the columns the static
# baseline consumes, in the same order, or the comparison is between two different
# models and not between two strategies.

# %%
SEED = 42
np.random.seed(SEED)

HORIZON = 14                       # the checkpointed RQ1 horizon
RQ5_TARGET = 0.85                  # the locked sustained-MCC threshold
SUSTAINMENT_FRACTION = 0.80        # reachable reference: 80% of a model's own start

STARTING_YEARS = (2015, 2018, 2021)
STRATEGIES = ('static_baseline', 'incremental_online')

CALIBRATION_YEAR = 2022            # natural-prevalence, threshold and calibration only
EVAL_START, EVAL_END = date(2023, 1, 1), date(2025, 12, 31)

# Drive-level stream subsample. Drives are sampled, never rows, so each sampled
# drive keeps its full ordered history. Stated as a limitation in the write-up.
DRIVE_SAMPLE_PCT = 6               # percent of drives retained for the stream
WARMUP_YEARS = 2                   # warm-up window is [Y - WARMUP_YEARS, Y]

EVAL_ROWS_PER_MONTH = 100_000      # uniform, prevalence-preserving, per month
CAL_ROWS = 2_000_000               # uniform sample of the 2022 natural-prevalence set
NBOOT = 200                        # bootstrap resamples for the reported CIs

ARF_N_MODELS = 5                   # forest size; River scores row by row, so this is cost
SMOKE = True                       # run one cell end to end before the full grid
SMOKE_CELL = (2018, 'incremental_online')

# Known drift onsets (established by the schema-era census; not rediscovered here).
SCHEMA_ERA_ONSET = date(2021, 4, 1)      # SMART 187 and 188 fall below the availability bar
SCHEMA_ERA_ONSET_EARLY = date(2014, 4, 1)

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_WORKING_PREFIX = 'backblaze_features/working_set_20x'
GCS_NATURAL_PREFIX = 'backblaze_features/natural_test_2023_2025'
GCS_NATURAL_VAL_PREFIX = 'backblaze_features/natural_val_2022'
LOCAL_WORKING = Path('/content/working_set_20x')
LOCAL_NATURAL = Path('/content/natural_test_2023_2025')
LOCAL_NATURAL_VAL = Path('/content/natural_val_2022')
for d in [LOCAL_WORKING, LOCAL_NATURAL, LOCAL_NATURAL_VAL]:
    d.mkdir(parents=True, exist_ok=True)

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
      f"natural val: {len(natural_val_files)}")

# %%
# Load the RQ1 checkpoint. It is the static baseline for the 2021 cell and the
# source of the drive-model prior every stream must reproduce.
with (CHECKPOINT_DIR / f'rq1_backblaze_best_{HORIZON}d.pkl').open('rb') as fh:
    RQ1_CKPT = pickle.load(fh)
RQ1_SIDE = json.loads((CHECKPOINT_DIR / f'rq1_backblaze_best_{HORIZON}d.json').read_text())

SCHEMA = json.loads((FEATURES_DIR / 'backblaze_feature_schema.json').read_text())
HORIZON_TARGETS = {f'failure_within_{h}d' for h in (7, 14, 30)}
TARGET = f'failure_within_{HORIZON}d'
EXCLUDE_ALWAYS = {
    'date', 'serial_number', 'model', 'model_canonical', 'manufacturer', 'era',
    'failure', 'failure_observed', 'censored', 'is_last_obs',
    'year', 'fleet_age_days',
}
MODEL_PRIOR = 'model_prior'

BASE_FEATURES = [c for c in SCHEMA['columns'] if c not in (EXCLUDE_ALWAYS | HORIZON_TARGETS)]
FEATURES = BASE_FEATURES + [MODEL_PRIOR]

assert FEATURES == RQ1_SIDE['feature_columns'], "feature contract drifted from the RQ1 checkpoint"
assert len(FEATURES) == RQ1_SIDE['n_features']
assert not (set(FEATURES) & (HORIZON_TARGETS | EXCLUDE_ALWAYS)), "leak in the feature list"
print(f"Feature contract: {len(BASE_FEATURES)} base + prior = {len(FEATURES)}; "
      f"RQ1 baseline is {RQ1_SIDE['best_model']} at threshold {RQ1_SIDE['threshold']:.6f}")

# The prior is carried in the checkpoint payload and reused unchanged. It is fit on
# the training years only, and recomputing it here at a different prevalence would
# shift a feature under the model.
PRIOR_TABLE = pl.DataFrame(RQ1_CKPT['drive_model_prior'])
GLOBAL_PRIOR = float(RQ1_CKPT['global_prior'])


def attach_prior(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.join(PRIOR_TABLE, on='model_canonical', how='left')
        .with_columns(pl.col(MODEL_PRIOR).fill_null(GLOBAL_PRIOR).cast(pl.Float32))
    )


# %% [markdown]
# ## 2. Data
#
# Three sources, three roles, kept strictly apart.
#
# 1. **Training stream** (working set, 20:1, drive-level subsample). Streamed in date
#    order. Never scored.
# 2. **Calibration set** (2022, natural prevalence). Fits the operating threshold and
#    the probability calibrator for every strategy, once, before the evaluation window
#    opens. Never scored in the results.
# 3. **Evaluation windows** (2023 to 2025, natural prevalence, one per month). Scored,
#    never trained on.

# %% [markdown]
# ### 2.1 Training stream: drive-level subsample

# %%
STREAM_COLS = BASE_FEATURES + [TARGET, 'date', 'serial_number', 'model_canonical', 'year']

# Hash the drive identifier, not the row: every sampled drive keeps its entire
# ordered history, which is what makes a per-drive degradation signal learnable.
stream = (
    pl.scan_parquet(str(LOCAL_WORKING / 'bucket_*.parquet'))
    .select(STREAM_COLS)
    .filter((pl.col('serial_number').hash(seed=SEED) % 100) < DRIVE_SAMPLE_PCT)
    .with_columns([pl.col(c).cast(pl.Float32) for c in BASE_FEATURES])
    .collect(engine='streaming')
)
stream = attach_prior(stream).sort(['date', 'serial_number'])
n_drives = stream['serial_number'].n_unique()
print(f"stream: {stream.height:,} rows, {n_drives:,} drives, "
      f"{stream['date'].min()} to {stream['date'].max()}")
print(f"stream positive rate at {HORIZON}d: {100 * stream[TARGET].mean():.4f}% "
      f"(working-set prevalence, training side only)")
gc.collect()

# %% [markdown]
# ### 2.2 Calibration set (2022, natural prevalence)

# %%
cal = (
    pl.scan_parquet(str(LOCAL_NATURAL_VAL / 'bucket_*.parquet'))
    .select(BASE_FEATURES + [TARGET, 'model_canonical'])
    .collect(engine='streaming')
    .with_columns([pl.col(c).cast(pl.Float32) for c in BASE_FEATURES])
)
if cal.height > CAL_ROWS:
    cal = cal.sample(n=CAL_ROWS, seed=SEED)   # uniform: prevalence preserved
cal = attach_prior(cal)
y_cal = cal[TARGET].to_numpy().astype(np.int8)
X_cal = cal.select(FEATURES).to_numpy().astype(np.float32)
print(f"calibration (2022): {cal.height:,} rows, {100 * y_cal.mean():.4f}% positive")

# %% [markdown]
# ### 2.3 Evaluation windows (2023 to 2025, natural prevalence, monthly)
#
# Uniformly sampled per partition so prevalence is preserved exactly, then grouped
# into months. Built once and checkpointed, because a rebuild costs a full pass over
# 311M rows.

# %%
EVAL_CACHE = RQ5_CKPT_DIR / f'rq5_eval_natural_{HORIZON}d.parquet'

if EVAL_CACHE.exists():
    eval_set = pl.read_parquet(EVAL_CACHE)
    print(f"loaded cached evaluation set: {eval_set.height:,} rows")
else:
    total_natural = (
        pl.scan_parquet(str(LOCAL_NATURAL / 'bucket_*.parquet'))
        .select(pl.len()).collect(engine='streaming').item()
    )
    n_months = 36
    frac = min(1.0, (EVAL_ROWS_PER_MONTH * n_months) / total_natural)
    parts = []
    for f in sorted(LOCAL_NATURAL.glob('bucket_*.parquet')):
        part = (
            pl.read_parquet(f, columns=BASE_FEATURES + [TARGET, 'date', 'model_canonical'])
            .sample(fraction=frac, seed=SEED)
            .with_columns([pl.col(c).cast(pl.Float32) for c in BASE_FEATURES])
        )
        parts.append(part)
        del part
        gc.collect()
    eval_set = attach_prior(pl.concat(parts))
    del parts
    gc.collect()
    eval_set = eval_set.with_columns(
        pl.col('date').dt.truncate('1mo').alias('month')
    ).sort('month')
    eval_set.write_parquet(EVAL_CACHE)
    print(f"natural test total {total_natural:,}; sampled {eval_set.height:,} (frac {frac:.5f})")

EVAL_MONTHS = sorted(eval_set['month'].unique().to_list())
print(f"evaluation months: {len(EVAL_MONTHS)} ({EVAL_MONTHS[0]} to {EVAL_MONTHS[-1]})")

# Prevalence by month: this is the prior-shift confound, quantified before any model
# is scored, so it cannot later be mistaken for drift.
prevalence_by_month = (
    eval_set.group_by('month')
    .agg(pl.len().alias('n'), pl.col(TARGET).mean().alias('prevalence'))
    .sort('month')
)
print(prevalence_by_month.head(6).to_pandas().to_string(index=False))
REFERENCE_PREVALENCE = float(eval_set[TARGET].mean())
print(f"pooled evaluation prevalence (the fixed-prevalence reference): "
      f"{100 * REFERENCE_PREVALENCE:.4f}%")

# %% [markdown]
# ## 3. Static baselines, one per starting year
#
# Each static baseline is trained on the working-set rows of its warm-up window,
# calibrated and thresholded on the 2022 natural-prevalence set, and then frozen. It
# never updates again. That is the whole point of the comparison.
#
# The 2021 baseline is the checkpointed RQ1 model itself, so its aggregate MCC over
# the evaluation window must reproduce the RQ1 result. That is asserted, not assumed:
# a suspiciously clean number gets decomposed before it gets trusted.

# %%
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


def fit_calibrator(scores: np.ndarray, y: np.ndarray) -> tuple[str, object]:
    iso = IsotonicRegression(out_of_bounds='clip').fit(scores, y)
    platt = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), y)
    b_iso = float(np.mean((np.clip(iso.predict(scores), 0, 1) - y) ** 2))
    b_platt = float(np.mean((platt.predict_proba(scores.reshape(-1, 1))[:, 1] - y) ** 2))
    return ('isotonic', iso) if b_iso <= b_platt else ('platt', platt)


def apply_calibrator(kind: str, obj: object, scores: np.ndarray) -> np.ndarray:
    if kind == 'isotonic':
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


def score_rq1_checkpoint(X: np.ndarray) -> np.ndarray:
    """Calibrated P(failure) from the RQ1 checkpoint payload (single or stack)."""
    ck = RQ1_CKPT
    if ck['kind'] == 'single':
        Xm = ck['imputer'].transform(X).astype(np.float32) if ck['needs_impute'] else X
        return apply_calibrator(ck['calibrator_kind'], ck['calibrator'],
                                ck['wrapper'].predict_proba(Xm))
    cols = []
    for m in ck['members'].values():
        Xm = m['imputer'].transform(X).astype(np.float32) if m['needs_impute'] else X
        cols.append(apply_calibrator(m['calibrator_kind'], m['calibrator'],
                                     m['wrapper'].predict_proba(Xm)))
    return np.mean(np.vstack(cols), axis=0)


class StaticBaseline:
    """A frozen batch model plus its calibrator and operating threshold."""

    def __init__(self, name: str, scorer, threshold: float) -> None:
        self.name = name
        self._scorer = scorer
        self.threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._scorer(X)


def build_static_baseline(year: int) -> StaticBaseline:
    if year == 2021:
        # The RQ1 checkpoint: same model, same calibration, same threshold.
        return StaticBaseline('rq1_checkpoint_stack', score_rq1_checkpoint,
                              float(RQ1_SIDE['threshold']))

    warm = stream.filter(pl.col('year').is_between(year - WARMUP_YEARS, year))
    Xw = warm.select(FEATURES).to_numpy().astype(np.float32)
    yw = warm[TARGET].to_numpy().astype(np.int8)
    print(f"  static {year}: warm-up {warm.height:,} rows "
          f"({int(yw.sum()):,} positive) from {year - WARMUP_YEARS} to {year}")

    # LightGBM consumes nulls natively, matching how the RQ1 members treat them.
    model = build_wrapper('lightgbm', random_state=SEED).fit(Xw, yw)
    raw_cal = model.predict_proba(X_cal)
    kind, calib = fit_calibrator(raw_cal, y_cal)
    cal_scores = apply_calibrator(kind, calib, raw_cal)
    thr = best_threshold(y_cal, cal_scores)

    def scorer(X: np.ndarray, _m=model, _k=kind, _c=calib) -> np.ndarray:
        return apply_calibrator(_k, _c, _m.predict_proba(X))

    del Xw, yw, warm
    gc.collect()
    return StaticBaseline(f'lightgbm_frozen_{year}', scorer, thr)


static_baselines = {}
for y in STARTING_YEARS:
    print(f"Building static baseline for {y}")
    static_baselines[y] = build_static_baseline(y)
    print(f"  -> {static_baselines[y].name}, threshold {static_baselines[y].threshold:.6f}")

# %%
# Correctness check: the 2021 static baseline is the RQ1 model, so its pooled MCC
# over the whole evaluation window must reproduce the RQ1 figure (about 0.180).
X_eval_all = eval_set.select(FEATURES).to_numpy().astype(np.float32)
y_eval_all = eval_set[TARGET].to_numpy().astype(np.int8)
base21 = static_baselines[2021]
proba21 = base21.predict_proba(X_eval_all)
pooled21 = _mcc(y_eval_all.astype(np.int64), (proba21 >= base21.threshold).astype(np.int64))
print(f"2021 static baseline pooled MCC over 2023-2025: {pooled21:.4f} "
      f"(RQ1 reported {RQ1_SIDE['natural_mcc']:.4f})")
assert abs(pooled21 - RQ1_SIDE['natural_mcc']) < 0.05, (
    "the 2021 static baseline does not reproduce the RQ1 result; the handoff or the "
    "evaluation sample is not what it claims to be"
)
del X_eval_all, proba21
gc.collect()

# %% [markdown]
# ## 4. Online learners, one per starting year
#
# Each adaptive learner is warm-started on the same window as its static twin, then
# keeps learning through every year up to the evaluation window. Its operating
# threshold is fit on the 2022 calibration set, at the point it reaches 2022, and then
# held fixed, so the static and adaptive strategies are compared at a comparable
# operating point rather than one being re-tuned mid-flight.
#
# Cost-sensitive weighting matters here. The stream is undersampled but still skewed,
# and an unweighted incremental tree collapses to the majority class.

# %%
CLASS_WEIGHTS = balanced_class_weights(stream[TARGET].to_numpy())
print(f"class weights on the stream: {CLASS_WEIGHTS}")


def stream_rows(df: pl.DataFrame):
    """Yield (row_dict, label) in the frame's existing order."""
    rows = frame_to_dicts(df.select(FEATURES), FEATURES)
    labels = df[TARGET].to_numpy().astype(np.int8)
    return zip(rows, labels)


def build_online_learner_for(year: int):
    learner = build_online_learner(
        'adaptive_random_forest',
        n_models=ARF_N_MODELS,
        seed=SEED,
        feature_names=FEATURES,
        class_weights=CLASS_WEIGHTS,
    )
    # Learn everything from the warm-up window through the calibration year. All of
    # it precedes the evaluation window, so none of it is a peek at a scored label.
    pre = stream.filter(pl.col('year').is_between(year - WARMUP_YEARS, CALIBRATION_YEAR))
    print(f"  online {year}: streaming {pre.height:,} rows "
          f"({year - WARMUP_YEARS} to {CALIBRATION_YEAR})")
    for row, label in stream_rows(pre):
        learner.learn_one(row, int(label))
    del pre
    gc.collect()

    cal_scores = learner.predict_proba(cal.select(FEATURES))
    thr = best_threshold(y_cal, cal_scores)
    print(f"  online {year}: {learner.n_learned:,} observations learned, "
          f"threshold {thr:.6f}")
    return learner, thr


# %% [markdown]
# ## 5. Prequential evaluation over the monthly windows
#
# For each month: score the natural-prevalence window with the current model, record
# the metrics, then let the adaptive learner consume that month's training-stream
# rows. Predict, then learn. Never the other way round.
#
# Each month is reported twice: the raw MCC (what a deployed model would actually
# score, base-rate movement included) and the MCC recomputed at a fixed prevalence
# (which holds the base rate constant so any movement is attributable to drift). Only
# the second supports a drift claim.

# %%
def evaluate_month(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(np.int8)
    mcc, mcc_lo, mcc_hi = mcc_with_ci(y, pred, n_boot=NBOOT, seed=SEED)
    f1, f1_lo, f1_hi = f1_with_ci(y, pred, n_boot=NBOOT, seed=SEED)
    pr, pr_lo, pr_hi = pr_auc_with_ci(y, proba, n_boot=NBOOT, seed=SEED)
    return {
        'mcc': mcc, 'mcc_ci_low': mcc_lo, 'mcc_ci_high': mcc_hi,
        'mcc_fixed_prev': mcc_at_fixed_prevalence(y, pred, REFERENCE_PREVALENCE, seed=SEED),
        'f1': f1, 'f1_ci_low': f1_lo, 'f1_ci_high': f1_hi,
        'pr_auc': pr, 'pr_auc_ci_low': pr_lo, 'pr_auc_ci_high': pr_hi,
        'prevalence': float(y.mean()), 'n': int(y.size),
        'flag_rate': float(pred.mean()),
    }


# Pre-slice the evaluation set by month once (materializing per month inside the loop
# is the difference between minutes and hours).
eval_by_month = {
    m: eval_set.filter(pl.col('month') == m) for m in EVAL_MONTHS
}
eval_X_by_month = {
    m: df.select(FEATURES).to_numpy().astype(np.float32) for m, df in eval_by_month.items()
}
eval_y_by_month = {
    m: df[TARGET].to_numpy().astype(np.int8) for m, df in eval_by_month.items()
}
stream_by_month = {
    m: stream.filter(pl.col('date').dt.truncate('1mo') == m) for m in EVAL_MONTHS
}
print("months pre-sliced; "
      f"eval rows/month median {int(np.median([y.size for y in eval_y_by_month.values()])):,}; "
      f"stream rows/month median "
      f"{int(np.median([df.height for df in stream_by_month.values()])):,}")


def run_cell(year: int, strategy: str) -> tuple[list[dict], list[dict]]:
    """Run one (starting_year, strategy) cell. Returns (monthly rows, drift events)."""
    print(f"\n===== cell: {year} / {strategy} =====")
    monthly, events = [], []

    if strategy == 'static_baseline':
        model = static_baselines[year]
        threshold = model.threshold
        scorer = model.predict_proba
        learner = None
    else:
        learner, threshold = build_online_learner_for(year)
        scorer = lambda X: learner.predict_proba(X)  # noqa: E731

    # Detectors run on the prequential loss of this cell's model, over the evaluation
    # stream, so a detection has a timestamp and a measurable latency.
    adwin = ADWINDetector()
    ph = PageHinkleyDetector()

    for m in EVAL_MONTHS:
        y_m = eval_y_by_month[m]
        proba_m = scorer(eval_X_by_month[m])
        row = evaluate_month(y_m, proba_m, threshold)
        row.update({'starting_year': year, 'strategy': strategy, 'month': m})
        monthly.append(row)

        # Drift detection on the per-observation loss of the month just scored.
        pred_m = (proba_m >= threshold).astype(np.int8)
        for loss in (pred_m != y_m).astype(np.float64):
            adwin.update(loss)
            if adwin.drift_detected:
                events.append({'starting_year': year, 'strategy': strategy,
                               'detector': 'adwin', 'month': m, 'signal': 'performance'})
            ph.update(loss)
            if ph.drift_detected:
                events.append({'starting_year': year, 'strategy': strategy,
                               'detector': 'page_hinkley', 'month': m, 'signal': 'performance'})

        # Then, and only then, the adaptive learner consumes this month's stream.
        if learner is not None:
            for xrow, label in stream_rows(stream_by_month[m]):
                learner.learn_one(xrow, int(label))

        print(f"  {m}: MCC {row['mcc']:.4f} (fixed-prev {row['mcc_fixed_prev']:.4f}), "
              f"PR-AUC {row['pr_auc']:.4f}, prevalence {100 * row['prevalence']:.4f}%")

    if learner is not None:
        learner.save(RQ5_CKPT_DIR / f'online_{year}.pkl')
    return monthly, events


# %% [markdown]
# ### 5.1 Smoke run
#
# One cell end to end before committing the grid: confirms the learner streams, the
# monthly metrics compute in reasonable time, checkpointing works, and the detectors
# are not flagging every observation.

# %%
if SMOKE:
    smoke_monthly, smoke_events = run_cell(*SMOKE_CELL)
    smoke_df = pl.DataFrame(smoke_monthly)
    print(f"\nsmoke cell: {smoke_df.height} months, {len(smoke_events)} drift events")
    print(f"mean MCC {smoke_df['mcc'].mean():.4f}; "
          f"drift-event rate {len(smoke_events) / max(smoke_df['n'].sum(), 1):.2e} per observation")
    assert len(smoke_events) < 0.01 * smoke_df['n'].sum(), (
        "detectors are flagging on noise; retune before the grid"
    )

# %% [markdown]
# ### 5.2 The grid
#
# Three starting years by two strategies. Checkpoint after every cell, because the
# session boundary will land somewhere inside it.

# %%
RESULTS_CACHE = RQ5_CKPT_DIR / 'rq5_monthly.parquet'
EVENTS_CACHE = RQ5_CKPT_DIR / 'rq5_events.parquet'

all_monthly, all_events = [], []
done = set()
if RESULTS_CACHE.exists():
    cached = pl.read_parquet(RESULTS_CACHE)
    all_monthly = cached.to_dicts()
    done = {(r['starting_year'], r['strategy']) for r in all_monthly}
    if EVENTS_CACHE.exists():
        all_events = pl.read_parquet(EVENTS_CACHE).to_dicts()
    print(f"resuming; cells already done: {sorted(done)}")

for year in STARTING_YEARS:
    for strategy in STRATEGIES:
        if (year, strategy) in done:
            continue
        monthly, events = run_cell(year, strategy)
        all_monthly.extend(monthly)
        all_events.extend(events)
        pl.DataFrame(all_monthly).write_parquet(RESULTS_CACHE)
        if all_events:
            pl.DataFrame(all_events).write_parquet(EVENTS_CACHE)
        gc.collect()

monthly_df = pl.DataFrame(all_monthly).sort(['starting_year', 'strategy', 'month'])
monthly_df.write_csv(TABLES_DIR / 'rq5_monthly_trajectory.csv')
print(f"\nSaved {TABLES_DIR / 'rq5_monthly_trajectory.csv'} ({monthly_df.height} rows)")

# %% [markdown]
# ## 6. Covariate drift and the schema-era boundary
#
# The performance detectors above see drift only when it moves the loss. Covariate
# drift can move a feature distribution while the loss stays flat, so the window
# detectors run separately, per feature, over the training stream, which spans the
# schema-era boundary and therefore has a ground-truth onset to measure latency
# against.
#
# The reference window is drawn from the pre-boundary era; the sliding window advances
# through the stream. Window sizes are set well above the PSI noise floor: the
# detector refuses configurations where its own bands would be sampling noise.

# %%
# Features with the strongest documented discriminative power, plus the two whose
# availability collapses at the boundary. If the boundary is a real covariate shift,
# it should be loudest exactly here.
MONITORED = [c for c in (
    'smart_197_raw', 'smart_5_raw', 'smart_198_raw',
    'smart_187_raw', 'smart_188_raw', MODEL_PRIOR,
) if c in FEATURES]
print(f"monitored features: {MONITORED}")

REF_WINDOW, SLIDE_WINDOW, TEST_EVERY = 5_000, 5_000, 500

drift_rows = []
ref_period = stream.filter(pl.col('year').is_between(2019, 2020))   # pre-boundary
for feat in MONITORED:
    ref_values = ref_period[feat].drop_nulls().to_numpy()
    if ref_values.size < REF_WINDOW:
        print(f"  {feat}: insufficient reference data; skipped")
        continue

    ks = KSDriftDetector(reference_size=REF_WINDOW, window_size=SLIDE_WINDOW,
                         alpha=0.001, test_every=TEST_EVERY)
    psi = PSIDriftDetector(reference_size=REF_WINDOW, window_size=SLIDE_WINDOW,
                           n_bins=10, test_every=TEST_EVERY)
    ks.set_reference(ref_values[:REF_WINDOW])
    psi.set_reference(ref_values[:REF_WINDOW])

    post = stream.filter(pl.col('year') >= 2021).select(['date', feat]).sort('date')
    dates = post['date'].to_list()
    values = post[feat].to_numpy()
    first = {'ks': None, 'psi': None}
    for i, v in enumerate(values):
        ks.update(v)
        psi.update(v)
        if ks.drift_detected and first['ks'] is None:
            first['ks'] = dates[i]
        if psi.drift_detected and first['psi'] is None:
            first['psi'] = dates[i]
        if first['ks'] and first['psi']:
            break

    for det, when in first.items():
        drift_rows.append({
            'feature': feat, 'detector': det,
            'onset': SCHEMA_ERA_ONSET.isoformat(),
            'detected': when.isoformat() if when else None,
            'latency_days': drift_detection_latency(when, SCHEMA_ERA_ONSET) if when else None,
            'statistic': float(ks.statistic if det == 'ks' else psi.statistic),
        })
    print(f"  {feat}: KS {first['ks']}, PSI {first['psi']}")
    del post, values, dates
    gc.collect()

covariate_df = pl.DataFrame(drift_rows)
print(covariate_df.to_pandas().to_string(index=False))
print("\nNote: KS and PSI here test the distribution of the values that are present,"
      "\nwith nulls dropped by construction. The primary SMART raw values barely move"
      "\nacross the boundary (KS statistics near zero), so the value-distribution view"
      "\nunderstates the 2021Q2 schema drift, which is an availability shift. The"
      "\nmodel_prior row, which is fleet composition, carries the genuine covariate"
      "\nsignal and leads the schema boundary. The availability monitor below measures"
      "\nthe schema drift directly.")

# %% [markdown]
# ### 6.1 Availability monitor for the schema-era boundary
#
# The 2021Q2 boundary is a missingness shift, not a value shift: SMART 187 and 188
# fall below the 50% availability bar while the values that remain keep their
# distribution. A value-distribution detector that drops nulls cannot see that by
# construction, so the schema drift is measured directly as the monthly populated
# rate of each feature, and the onset is the first month the rate crosses below the
# 50% bar. This is the right instrument for the documented sudden drift, and it is
# what the detection-latency figure should be read from.

# %%
AVAILABILITY_BAR = 0.50
avail_rows = []
stream_months = (
    stream.with_columns(pl.col('date').dt.truncate('1mo').alias('month'))
    .sort('month')
)
for feat in MONITORED:
    monthly_avail = (
        stream_months.group_by('month')
        .agg((pl.col(feat).is_not_null().mean()).alias('populated_rate'))
        .sort('month')
    )
    crossed = monthly_avail.filter(pl.col('populated_rate') < AVAILABILITY_BAR)
    onset_month = crossed['month'][0] if crossed.height else None
    pre = monthly_avail.filter(pl.col('month') < SCHEMA_ERA_ONSET)['populated_rate']
    post = monthly_avail.filter(pl.col('month') >= SCHEMA_ERA_ONSET)['populated_rate']
    avail_rows.append({
        'feature': feat,
        'pre_boundary_availability': float(pre.mean()) if pre.len() else None,
        'post_boundary_availability': float(post.mean()) if post.len() else None,
        'first_month_below_50pct': onset_month.isoformat() if onset_month else None,
        'latency_days_vs_2021Q2': (
            drift_detection_latency(onset_month, SCHEMA_ERA_ONSET) if onset_month else None
        ),
    })
availability_df = pl.DataFrame(avail_rows)
print(availability_df.to_pandas().to_string(index=False))
availability_df.write_csv(TABLES_DIR / 'rq5_availability_drift.csv')

# %% [markdown]
# ## 7. Drift subtypes in this dataset
#
# Four canonical subtypes, characterized against the evidence rather than assumed.
# Several are already established by the era census and the exploratory analysis and
# are used here as anchors, not rediscovered.
#
# **Sudden.** The 2021Q2 schema-era boundary, where SMART 187 and 188 fall below the
# 50% availability bar (to about 42%). A schema and covariate shift with a known date,
# which is why it serves as the ground-truth onset for the detection-latency
# measurement in Section 6. The 2014Q2 boundary is the second instance and sits before
# the evaluation window.
#
# **Gradual.** Fleet turnover across the record (peak 87 drive models), replacing old
# models with new and shifting the SMART distributions progressively rather than
# abruptly. Quantified below as the share of evaluation-window drive-days belonging to
# models absent from the training years.
#
# **Incremental.** The 13-year covariate shift and fleet aging: many small accumulating
# changes, visible only over a long horizon, which is what the year-over-year
# trajectory is for.
#
# **Recurring.** Whether a previously observed distribution reappears. Tested below by
# distributional similarity between each evaluation month and the training years. If no
# month resembles an earlier regime more than it resembles its neighbors, the dataset
# does not exhibit recurring drift within the observation window, and the recurring
# mitigation cannot be empirically validated. That is a legitimate finding and is
# reported as one.
#
# **Prior shift (a confound, not a subtype).** The failure-day rate declines across the
# evaluation years. Because MCC is prevalence-sensitive, this alone moves the metric.
# Section 5 reports every month at both the raw and a fixed prevalence for exactly this
# reason, and no drift claim rests on the raw series.

# %%
# Gradual: how much of the evaluation window is drive models the training never saw.
train_models = set(
    stream.filter(pl.col('year') <= 2021)['model_canonical'].unique().to_list()
)
eval_models = eval_set.group_by('model_canonical').agg(pl.len().alias('n'))
unseen_share = (
    eval_models.filter(~pl.col('model_canonical').is_in(list(train_models)))['n'].sum()
    / eval_models['n'].sum()
)
print(f"fleet turnover: {len(train_models)} models in training; "
      f"{eval_models.height} in the evaluation window; "
      f"{100 * unseen_share:.2f}% of evaluation drive-days are unseen models")

# Recurring: distance of each month's SMART profile from the training profile.
#
# Each feature is standardized by its training mean and standard deviation before the
# profile is formed, so no single large-magnitude raw count dominates. A raw cosine on
# unstandardized means is uninformative here: the reallocated- and pending-sector raw
# counts run to millions, so the cosine is about 1 for every month regardless of the
# smaller features, which cannot detect anything. On standardized profiles the training
# profile is the origin, so the reported quantity is the Euclidean distance of each
# month from it: near zero means the month resembles the training regime, a later month
# returning toward zero after moving away would be recurrence.
profile_cols = [c for c in MONITORED if c != MODEL_PRIOR]
train_stats = stream.filter(pl.col('year') <= 2021).select(
    [pl.col(c).mean().alias(f'{c}__mean') for c in profile_cols]
    + [pl.col(c).std().alias(f'{c}__std') for c in profile_cols]
).row(0, named=True)
mu = np.array([train_stats[f'{c}__mean'] for c in profile_cols])
sigma = np.array([train_stats[f'{c}__std'] for c in profile_cols])
sigma[sigma == 0] = 1.0   # a constant feature contributes nothing rather than dividing by zero

recurring_rows = []
for m in EVAL_MONTHS:
    prof = eval_by_month[m].select(
        [pl.col(c).mean().alias(c) for c in profile_cols]
    ).to_numpy().ravel()
    z = (prof - mu) / sigma
    recurring_rows.append({
        'month': m,
        'distance_from_training': float(np.linalg.norm(z)),
    })
recurring_df = pl.DataFrame(recurring_rows)
_first = recurring_df['distance_from_training'][0]
_last = recurring_df['distance_from_training'][-1]
_max = recurring_df['distance_from_training'].max()
_argmax = recurring_df.filter(pl.col('distance_from_training') == _max)['month'][0]
print(f"standardized distance from the training profile: first month {_first:.3f}, "
      f"last month {_last:.3f}, peak {_max:.3f} at {_argmax}")
print("A distance that grows and does not return indicates incremental or gradual drift "
      "with no recurrence. A distance that rises and later falls back toward an earlier "
      "level would indicate a recurring regime.")

# %% [markdown]
# ## 8. Sustained MCC, the hypothesis test, and the custom metrics
#
# The headline number is the time-averaged MCC across the evaluation horizon, tested
# one-sided against 0.85. It will not clear it, on any strategy, at any starting year,
# because no model on this dataset starts near 0.85. That is reported as the answer to
# the hypothesis, and the substantive result is carried by the comparison below it.
#
# The sustainment window is reported twice: against 0.85 (which is 0 months for
# everything, and therefore cannot distinguish the strategies) and against a reachable
# reference at 80% of each model's own initial MCC (which can).

# %%
summary_rows = []
for (year, strategy), grp in monthly_df.group_by(['starting_year', 'strategy'], maintain_order=True):
    grp = grp.sort('month')
    months = grp['month'].to_list()
    mcc_series = grp['mcc'].to_list()
    fixed_series = grp['mcc_fixed_prev'].to_list()

    mean_mcc = float(np.mean(mcc_series))
    # The CI on the time-average comes from the monthly spread, which is what the
    # sustained-performance claim is about.
    boot = np.random.default_rng(SEED).choice(
        mcc_series, size=(1000, len(mcc_series)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    test = one_sample_threshold_test(
        mean_mcc, float(ci_low), float(ci_high), RQ5_TARGET,
        metric_name=f"sustained_MCC_{strategy}_{year}")

    initial = float(np.mean(mcc_series[:3]))   # first quarter as the starting level
    reference = sustainment_reference(initial, SUSTAINMENT_FRACTION)

    summary_rows.append({
        'starting_year': year, 'strategy': strategy,
        'mean_mcc': mean_mcc, 'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'meets_0.85': bool(test['reject']),
        'mean_mcc_fixed_prev': float(np.mean(fixed_series)),
        'initial_mcc': initial, 'final_mcc': float(np.mean(mcc_series[-3:])),
        'degradation_per_month': performance_degradation_rate(mcc_series, months),
        'degradation_per_month_fixed_prev': performance_degradation_rate(fixed_series, months),
        'sustainment_months_vs_0.85': performance_sustainment_window(mcc_series, RQ5_TARGET),
        'sustainment_reference': reference,
        'sustainment_months_vs_reference': performance_sustainment_window(mcc_series, reference),
        'mean_pr_auc': float(grp['pr_auc'].mean()),
        'mean_f1': float(grp['f1'].mean()),
    })

summary = pl.DataFrame(summary_rows).sort(['starting_year', 'strategy'])
summary.write_csv(TABLES_DIR / 'rq5_sustained_mcc.csv')
print(summary.to_pandas().to_string(index=False))
print(f"\nCells clearing the 0.85 sustained-MCC target: "
      f"{int(summary['meets_0.85'].sum())}/{summary.height}")

# %%
# Adaptive versus static, the substantive comparison. Retraining effectiveness is
# normalized against the static baseline's own initial level, which is the level the
# adaptation is recovering toward: a reachable reference, not the 0.85 target.
compare_rows = []
for year in STARTING_YEARS:
    s = summary.filter((pl.col('starting_year') == year) &
                       (pl.col('strategy') == 'static_baseline')).row(0, named=True)
    o = summary.filter((pl.col('starting_year') == year) &
                       (pl.col('strategy') == 'incremental_online')).row(0, named=True)
    compare_rows.append({
        'starting_year': year,
        'static_mean_mcc': s['mean_mcc'], 'online_mean_mcc': o['mean_mcc'],
        'mcc_gain': o['mean_mcc'] - s['mean_mcc'],
        'static_degradation_per_month': s['degradation_per_month_fixed_prev'],
        'online_degradation_per_month': o['degradation_per_month_fixed_prev'],
        'adaptation_effectiveness': retraining_effectiveness(
            s['final_mcc'], o['final_mcc'], s['initial_mcc']),
        'static_sustainment_vs_reference': s['sustainment_months_vs_reference'],
        'online_sustainment_vs_reference': o['sustainment_months_vs_reference'],
    })
compare = pl.DataFrame(compare_rows)
compare.write_csv(TABLES_DIR / 'rq5_adaptive_vs_static.csv')
print(compare.to_pandas().to_string(index=False))

# %%
# Drift event log, with latency measured against the schema-era onset where the
# onset precedes the detection.
event_rows = []
if all_events:
    ev = pl.DataFrame(all_events)
    event_rows = (
        ev.group_by(['starting_year', 'strategy', 'detector', 'month'])
        .agg(pl.len().alias('n_signals'))
        .sort(['starting_year', 'strategy', 'month'])
    )
    event_rows.write_csv(TABLES_DIR / 'rq5_drift_events.csv')
    print(event_rows.head(20).to_pandas().to_string(index=False))
covariate_df.write_csv(TABLES_DIR / 'rq5_covariate_drift.csv')
prevalence_by_month.write_csv(TABLES_DIR / 'rq5_prior_shift.csv')
recurring_df.write_csv(TABLES_DIR / 'rq5_recurring_similarity.csv')
print(f"\nSaved drift, prior-shift, and recurrence tables to {TABLES_DIR}")

# %% [markdown]
# ## 9. Figures

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MCC trajectories, one panel per starting year, raw and fixed-prevalence.
fig, axes = plt.subplots(1, len(STARTING_YEARS), figsize=(5 * len(STARTING_YEARS), 4.5),
                         sharey=True)
for ax, year in zip(np.atleast_1d(axes), STARTING_YEARS):
    for strategy, style in (('static_baseline', '-'), ('incremental_online', '--')):
        grp = monthly_df.filter((pl.col('starting_year') == year) &
                                (pl.col('strategy') == strategy)).sort('month')
        ax.plot(grp['month'], grp['mcc_fixed_prev'], style, label=strategy)
    ax.set_title(f'frozen at {year}')
    ax.set_xlabel('month')
    ax.tick_params(axis='x', rotation=45)
axes_list = np.atleast_1d(axes)
axes_list[0].set_ylabel(f'MCC at fixed prevalence ({HORIZON}-day horizon)')
axes_list[0].legend(fontsize=8)
fig.suptitle('RQ5. Adaptive versus static under drift (natural prevalence, prior shift removed)')
fig.tight_layout()
fig.savefig(FIG_DIR / 'mcc_trajectories.png', dpi=150)
plt.close(fig)

# Prior shift: the confound, shown next to the raw metric it moves.
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(prevalence_by_month['month'], 100 * prevalence_by_month['prevalence'],
         color='tab:gray', label='positive rate')
ax1.set_ylabel('monthly positive rate (%)')
ax1.set_xlabel('month')
ax2 = ax1.twinx()
ref = monthly_df.filter((pl.col('starting_year') == 2021) &
                        (pl.col('strategy') == 'static_baseline')).sort('month')
ax2.plot(ref['month'], ref['mcc'], color='tab:red', label='raw MCC')
ax2.plot(ref['month'], ref['mcc_fixed_prev'], color='tab:blue', label='MCC at fixed prevalence')
ax2.set_ylabel('MCC')
fig.legend(loc='upper right', fontsize=8)
ax1.set_title('Prior shift moves the raw metric without any covariate drift')
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / 'prior_shift.png', dpi=150)
plt.close(fig)

# Sustainment windows against the reachable reference (the 0.85 window is 0 for all).
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.35
x = np.arange(len(STARTING_YEARS))
for i, strategy in enumerate(STRATEGIES):
    vals = [summary.filter((pl.col('starting_year') == y) &
                           (pl.col('strategy') == strategy))['sustainment_months_vs_reference'][0]
            for y in STARTING_YEARS]
    ax.bar(x + i * width, vals, width, label=strategy)
ax.set_xticks(x + width / 2)
ax.set_xticklabels([str(y) for y in STARTING_YEARS])
ax.set_xlabel('starting year (model frozen)')
ax.set_ylabel('months sustained above 80% of initial MCC')
ax.set_title('Sustainment window against a reachable reference')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / 'sustainment_windows.png', dpi=150)
plt.close(fig)
print(f"Saved figures to {FIG_DIR}")

# %% [markdown]
# ## 10. Summary

# %%
print("RQ5 SUMMARY (natural prevalence, 14-day horizon, 2023-2025 evaluation window)")
print("=" * 78)
print(f"Sustained-MCC target 0.85: {int(summary['meets_0.85'].sum())} of {summary.height} "
      f"cells clear it. Best sustained MCC is {summary['mean_mcc'].max():.4f}.")
print("The target is not reachable on this dataset: failure prediction here caps near "
      "0.20 MCC at the deployment base rate, so no model starts within range of 0.85.")
print()
print("Adaptive versus static (the substantive result):")
print(compare.to_pandas().to_string(index=False))
print()
print(f"Stream: {stream.height:,} rows over {n_drives:,} drives "
      f"({DRIVE_SAMPLE_PCT}% drive-level subsample of the working set).")
print(f"Evaluation: {monthly_df.filter(pl.col('strategy') == 'static_baseline')['n'].sum():,} "
      f"natural-prevalence drive-days across {len(EVAL_MONTHS)} months.")
