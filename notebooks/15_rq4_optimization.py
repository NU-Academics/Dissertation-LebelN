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
# # 15. RQ4 - Resource Optimization (Google Cluster Traces)
#
# **Research question.** RQ4: to what extent can statistical pattern analysis enable
# a resource optimization improvement above the 25% dissertation target? The design
# compares a reactive baseline (do nothing) against three prediction-informed
# proactive strategies, measuring machine-level resource efficiency over rolling
# temporal windows in the held-out test period (schedule days 26-31).
#
# **Efficiency definition.** For a machine within a window,
# `efficiency = sum(actual CPU utilization) / sum(allocated CPU request)` over the
# instances active on that machine in that window. Allocation (CPU request) sits in
# the denominator because the cluster over-provisions: requests exceed realized use,
# so a doomed instance that holds an allocation but produces almost no useful work
# (V12: failing instances use far less CPU than successful ones) is wasted
# allocation. A proactive strategy raises efficiency by reclaiming that wasted
# allocation before it is committed.
#
# **Why at-submission, and why calibration first.** Borg failures are rapid-onset
# (median FAIL/LOST running duration ~22.6s, V09), so a proactive action has to fire
# at submission, not at runtime. The at-submission failure model
# (`rq1_google_best_atsubmission`, the tuned random forest) is a strong discriminator
# but over-forecasts probability under SMOTE plus class weighting (V37); RQ1 deferred
# probability calibration to here. Section 1 fits post-hoc calibration on the RQ1
# validation split and reports the Brier score before and after plus a reliability
# curve. Every probability-driven strategy reads the calibrated probability, because
# the operating point is a probability cutoff and raw over-forecasting would distort
# the migration and deferral counts.
#
# **Leakage discipline.** Scoring uses the calibrated at-submission model on the RQ1
# instance-keyed group split (test = FARM_FINGERPRINT instance-key buckets >= 17,
# matching notebook 12). The whole simulation is scoped to test-split instances so no
# instance the model trained on is scored. Timing and usage columns are observables,
# never model inputs.
#
# **Outputs.** `outputs/tables/rq4_google.csv` (baseline and per-strategy efficiency
# with paired deltas and CIs) and `outputs/tables/rq4_google_hypothesis_test.csv`
# (per-strategy Wilcoxon test against the 25% improvement target), plus a reliability
# figure and per-strategy efficiency figures under `outputs/figures/rq4_google/`.

# %% [markdown]
# ## 0. Colab session setup
#
# The `sys.modules` purge below drops any previously imported repo modules so that a
# `git pull` actually takes effect inside a running runtime (a stale cached module
# once produced a silent no-op result). Restart the runtime if anything still looks
# stale.

# %%
# !pip install -q polars lightgbm xgboost scikit-learn imbalanced-learn pandas pyarrow google-cloud-bigquery-storage matplotlib

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

# Drop cached repo modules so the freshly pulled source is what imports below.
for _m in [m for m in list(sys.modules)
           if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]

# %%
from utils.colab_setup import setup_drive, OUTPUT_DIR, CHECKPOINT_DIR

setup_drive()

# %%
import gc
import json
import pickle
import warnings

import numpy as np
import polars as pl

from src.evaluation.metrics import brier_score_with_ci, calibration_table, mcc_with_ci
from src.evaluation.hypothesis import paired_wilcoxon_cv

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

TABLES_DIR = OUTPUT_DIR / 'tables'
CACHE_DIR = OUTPUT_DIR / 'cache'
FIG_DIR = OUTPUT_DIR / 'figures' / 'rq4_google'
for directory in (TABLES_DIR, CACHE_DIR, FIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RQ4_TABLE_CSV = TABLES_DIR / 'rq4_google.csv'
RQ4_HYPOTHESIS_CSV = TABLES_DIR / 'rq4_google_hypothesis_test.csv'

LABEL_COL = "failure_label"
MICROS_PER_SEC = 1_000_000

# Efficiency target and trace-time constants.
RQ4_TARGET_IMPROVEMENT = 0.25        # dissertation resource-optimization target
DAY_SEC = 86_400
TRACE_OFFSET_SEC = 600               # trace time = us since 600s before trace start
# Held-out test period: schedule days 26-31 (matches the RQ1 test temporal block).
TEST_DAY_LO = 26
TEST_DAY_HI = 31
T_LO = ((TEST_DAY_LO - 1) * DAY_SEC + TRACE_OFFSET_SEC) * MICROS_PER_SEC
T_HI = (TEST_DAY_HI * DAY_SEC + TRACE_OFFSET_SEC) * MICROS_PER_SEC

# Rolling efficiency window length.
WINDOW_MIN = 60
WINDOW_US = WINDOW_MIN * 60 * MICROS_PER_SEC

# Instance-key group split (matches notebook 12): test = buckets >= 17 of 20.
N_BUCKETS = 20
TEST_BUCKET_MIN = 17

# %% [markdown]
# ## 1. Load and calibrate the at-submission model (Stage 1b)
#
# Load the checkpoint and its sidecar (the sidecar carries `feature_columns` in
# order and the tuned operating `threshold`). Calibration is fit on the RQ1
# validation split only (instance-key buckets [14, 17)); the Brier score before and
# after is reported on the test split (buckets >= 17). Isotonic and Platt (sigmoid)
# are both fit and the one with the lower validation Brier is kept. The calibrated
# model drives every probability-based strategy; the raw tuned threshold stays
# available for ranking-only uses.

# %%
CKPT_NAME = "rq1_google_best_atsubmission"
with open(CHECKPOINT_DIR / f"{CKPT_NAME}.pkl", "rb") as fh:
    at_submission_model = pickle.load(fh)
with open(CHECKPOINT_DIR / f"{CKPT_NAME}.json") as fh:
    sidecar = json.load(fh)

FEATURE_COLUMNS = sidecar["feature_columns"]
THRESHOLD = float(sidecar["threshold"])
print(f"Loaded {sidecar.get('model')} ({CKPT_NAME})")
print(f"  {len(FEATURE_COLUMNS)} feature columns | tuned threshold {THRESHOLD:.4f}")
print(f"  sidecar test metrics: {sidecar.get('test_metrics')}")


def score_raw(df: pl.DataFrame) -> np.ndarray:
    """Raw at-submission positive-class probability, features in sidecar order."""
    x = df.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLUMNS]).to_numpy()
    proba = at_submission_model.predict_proba(x)
    return np.asarray(proba)[:, 1] if np.ndim(proba) == 2 else np.asarray(proba)


# %%
from google.colab import auth

from utils.bq_client import get_client, table_ref

auth.authenticate_user()
_bq = get_client()

_inst_key = "CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING))"
CALIB_PERMILLE = 20                   # per-episode-key subsample for val / test pulls


def _calib_split_sql(bucket_lo: int, bucket_hi: int | None) -> str:
    """At-submission feature columns + label for one instance-key bucket band,
    thinned by an episode-key hash. ``bucket_hi`` None means the open upper band."""
    cols = ",\n        ".join(f"ef.{c}" for c in FEATURE_COLUMNS)
    epkey = f"CONCAT({_inst_key},'_',CAST(sched_seq AS STRING))"
    band = f"MOD(ABS(FARM_FINGERPRINT({_inst_key})), {N_BUCKETS}) >= {bucket_lo}"
    if bucket_hi is not None:
        band += f" AND MOD(ABS(FARM_FINGERPRINT({_inst_key})), {N_BUCKETS}) < {bucket_hi}"
    return f"""
SELECT
        {cols},
        ef.{LABEL_COL}
FROM {table_ref('episode_features')} ef
WHERE {band}
  AND MOD(ABS(FARM_FINGERPRINT({epkey})), 1000) < {CALIB_PERMILLE}
"""


def _pull(sql: str, name: str) -> pl.DataFrame:
    cache = CACHE_DIR / f"{name}.parquet"
    if cache.exists() and not REBUILD_CALIB:
        df = pl.read_parquet(cache)
        print(f"Loaded {df.height:,} rows <- {cache.name}")
        return df
    arrow = _bq.query(sql).to_arrow(create_bqstorage_client=True)
    df = pl.from_arrow(arrow)
    del arrow
    gc.collect()
    df.write_parquet(cache)
    print(f"Pulled {df.height:,} rows -> {cache.name}")
    return df


REBUILD_CALIB = True   # set False to reuse the cached val / test pulls
val_df = _pull(_calib_split_sql(14, 17), f"rq4_calib_val_p{CALIB_PERMILLE}")
test_df = _pull(_calib_split_sql(TEST_BUCKET_MIN, None), f"rq4_calib_test_p{CALIB_PERMILLE}")

# Fail-loud guard against a stale module / cache: the expected label and feature
# columns must be present before anything downstream runs.
assert LABEL_COL in val_df.columns, "label column missing; stale pull or schema drift."
assert all(c in val_df.columns for c in FEATURE_COLUMNS), "feature columns missing from val pull."

# %%
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

val_raw = score_raw(val_df)
val_y = val_df[LABEL_COL].to_numpy().astype(np.int64)
test_raw = score_raw(test_df)
test_y = test_df[LABEL_COL].to_numpy().astype(np.int64)

# Isotonic (non-parametric, monotone) calibrator.
_iso = IsotonicRegression(out_of_bounds="clip")
_iso.fit(val_raw, val_y)

# Platt / sigmoid calibrator: logistic regression on the raw score.
_platt = LogisticRegression(C=1e6, solver="lbfgs")
_platt.fit(val_raw.reshape(-1, 1), val_y)


def _apply_iso(scores: np.ndarray) -> np.ndarray:
    return _iso.predict(np.asarray(scores))


def _apply_platt(scores: np.ndarray) -> np.ndarray:
    return _platt.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


from sklearn.metrics import brier_score_loss

_brier_iso = brier_score_loss(val_y, _apply_iso(val_raw))
_brier_platt = brier_score_loss(val_y, _apply_platt(val_raw))
if _brier_iso <= _brier_platt:
    calibrate, CALIB_METHOD = _apply_iso, "isotonic"
else:
    calibrate, CALIB_METHOD = _apply_platt, "platt"
print(f"Validation Brier: isotonic {_brier_iso:.5f} | platt {_brier_platt:.5f} "
      f"-> using {CALIB_METHOD}")

# Brier before / after on the held-out test split, with CIs.
brier_raw = brier_score_with_ci(test_y, test_raw, seed=RANDOM_SEED)
brier_cal = brier_score_with_ci(test_y, calibrate(test_raw), seed=RANDOM_SEED)
print(f"Test Brier raw       {brier_raw[0]:.5f}  95% CI [{brier_raw[1]:.5f}, {brier_raw[2]:.5f}]")
print(f"Test Brier calibrated {brier_cal[0]:.5f}  95% CI [{brier_cal[1]:.5f}, {brier_cal[2]:.5f}]")

# Reliability curve (V37 expects the raw curve to sit below the diagonal).
reliability_raw = calibration_table(test_y, test_raw)
reliability_cal = calibration_table(test_y, calibrate(test_raw))
print("\nRaw reliability (mean_predicted should exceed observed_rate if over-forecasting):")
print(reliability_raw.select(["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"]))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
for tbl, lab in ((reliability_raw, "raw"), (reliability_cal, f"calibrated ({CALIB_METHOD})")):
    sub = tbl.filter(pl.col("mean_predicted").is_not_null())
    ax.plot(sub["mean_predicted"], sub["observed_rate"], marker="o", label=lab)
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed failure rate")
ax.set_title("At-submission reliability (test split)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "calibration_reliability.png", dpi=150)
plt.close(fig)
print(f"Saved {FIG_DIR / 'calibration_reliability.png'}")

# %% [markdown]
# ## 2. Reactive baseline efficiency (Stage 1)
#
# Build the per-machine, per-window simulation frame for the test period
# (days 26-31) at the scheduled-attempt (episode) grain, restricted to test-split
# instances. Each scheduled episode contributes one row, scored on its own
# at-submission features and judged by its own terminal label. This matches the
# episode-grain leakage discipline used throughout RQ1-RQ3 and avoids smearing one
# attempt's outcome across an instance's later attempts (the first attempt of a
# heavily-resubmitting failer would otherwise tag every later window of that
# instance). Per episode: `alloc_cpu` is its CPU request, `used_cpu` is its realized
# CPU over the schedule window (the Tier 3 `avg_cpu_60min` already materialized in
# `episode_features`, so the 7.5B-row usage table is never rescanned), `machine_id`
# is the machine assigned at its SCHEDULE event, and `window_idx` is derived from
# its schedule time. A machine-key hash subsample keeps whole machines (needed for
# the bin-packing strategy) while bounding memory.
#
# Baseline machine-window efficiency is `sum(used_cpu) / sum(alloc_cpu)`. EDA
# anticipates a low value (substantial over-provisioning).

# %%
SIM_MACHINE_PERMILLE = 50            # ~5% of machines, whole machines retained
SCHEDULE_EVENT_TYPE = 3              # SCHEDULE event (machine is assigned here)
USAGE_COL = "avg_cpu_60min"          # realized CPU over the schedule window (Tier 3)
_feat_cols = ",\n        ".join(f"ef.{c}" for c in FEATURE_COLUMNS)


def _sim_frame_sql() -> str:
    return f"""
WITH iv_test AS (
    -- Episodes scheduled in the test period, test-split instances only.
    SELECT collection_id, instance_index, sched_seq, schedule_time
    FROM {table_ref('episode_schedule_intervals')}
    WHERE schedule_time >= {T_LO} AND schedule_time < {T_HI}
      AND MOD(ABS(FARM_FINGERPRINT({_inst_key})), {N_BUCKETS}) >= {TEST_BUCKET_MIN}
),
sched AS (
    -- Machine assigned at the episode's SCHEDULE event, plus its efficiency window.
    SELECT
        iv.collection_id, iv.instance_index, iv.sched_seq,
        e.machine_id,
        CAST(FLOOR((iv.schedule_time - {T_LO}) / {WINDOW_US}) AS INT64) AS window_idx
    FROM iv_test iv
    JOIN {table_ref('instance_events_labeled')} e
      ON e.collection_id = iv.collection_id
     AND e.instance_index = iv.instance_index
     AND e.time = iv.schedule_time
     AND e.type = {SCHEDULE_EVENT_TYPE}
    WHERE e.machine_id IS NOT NULL
      AND e.time >= {T_LO} AND e.time < {T_HI}
      AND MOD(ABS(FARM_FINGERPRINT(CAST(e.machine_id AS STRING))), 1000) < {SIM_MACHINE_PERMILLE}
)
SELECT
    s.machine_id, s.window_idx,
    ef.collection_id, ef.instance_index, ef.sched_seq,
    ef.cpu_request AS alloc_cpu,
    COALESCE(ef.{USAGE_COL}, 0.0) AS used_cpu,
    ef.{LABEL_COL},
        {_feat_cols}
FROM sched s
JOIN {table_ref('episode_features')} ef
  USING (collection_id, instance_index, sched_seq)
WHERE ef.cpu_request > 0
"""


SIM_CACHE = CACHE_DIR / f"rq4_sim_ep_m{SIM_MACHINE_PERMILLE}_w{WINDOW_MIN}.parquet"
REBUILD_SIM = True   # set False to reload the simulation frame from Drive

if REBUILD_SIM or not SIM_CACHE.exists():
    print("Building per-machine x window episode frame (BigQuery Arrow stream) ...")
    arrow = _bq.query(_sim_frame_sql()).to_arrow(create_bqstorage_client=True)
    sim = pl.from_arrow(arrow)
    del arrow
    gc.collect()
    sim.write_parquet(SIM_CACHE)
    print(f"Built {sim.height:,} episode x window rows -> {SIM_CACHE.name}")
else:
    sim = pl.read_parquet(SIM_CACHE)
    print(f"Loaded {sim.height:,} episode x window rows <- {SIM_CACHE.name}")

assert {"machine_id", "window_idx", "used_cpu", "alloc_cpu", LABEL_COL}.issubset(sim.columns), \
    "simulation frame missing expected columns; stale pull or schema drift."

# Calibrated at-submission failure probability per scheduled episode.
sim = sim.with_columns(
    cal_prob=pl.Series(calibrate(score_raw(sim))).cast(pl.Float64)
)
print(f"  distinct machines {sim['machine_id'].n_unique():,} | "
      f"windows {sim['window_idx'].n_unique():,} | "
      f"episodes {sim.height:,} | "
      f"positive rate {float((sim[LABEL_COL] == 1).mean()):.4f}")


def machine_window_efficiency(frame: pl.DataFrame, used_col: str, alloc_col: str) -> pl.DataFrame:
    """Per machine x window efficiency = sum(used) / sum(alloc). Windows with zero
    surviving allocation are dropped (no allocation to be efficient about)."""
    agg = frame.group_by(["machine_id", "window_idx"]).agg(
        used=pl.col(used_col).sum(),
        alloc=pl.col(alloc_col).sum(),
    ).filter(pl.col("alloc") > 0)
    return agg.with_columns(efficiency=pl.col("used") / pl.col("alloc"))


baseline_eff = machine_window_efficiency(sim, "used_cpu", "alloc_cpu")
_b = baseline_eff["efficiency"].to_numpy()
print(f"\nBaseline efficiency over {baseline_eff.height:,} machine-windows: "
      f"mean {_b.mean():.4f} | median {np.median(_b):.4f}")

# %% [markdown]
# ## 3. Proactive strategies (Stage 2)
#
# Three strategies, each driven by the calibrated at-submission probability and
# firing at submission (rapid-onset failures rule out runtime action, V09). Each
# produces an adjusted per-machine-window allocation/usage frame whose efficiency is
# compared to the baseline window-for-window.
#
# 1. **Preemptive migration.** Instances whose calibrated failure probability clears
#    a tuned cutoff are rescheduled before failure: their wasted allocation is
#    reclaimed from the source machine-window (the doomed instance contributes ~0
#    useful work, V12, so removing it raises efficiency). The cutoff is tuned on a
#    Pareto frontier of (migrations triggered, true failures averted).
# 2. **Admission control.** New submissions whose calibrated probability clears a
#    cutoff are deferred for review and never land, removing their allocation. The
#    count of deferred "wasted" submissions is reported. Calibration matters most
#    here: a probability cutoff on over-forecast scores would over-defer.
# 3. **Capacity-aware bin packing.** A per-machine failure-risk score is derived from
#    `machine_events` history; a greedy rule steers new submissions on the riskiest
#    machines away (their allocation is reassigned off the high-risk machine-window).

# %%
def _pareto_threshold(prob: np.ndarray, y_true: np.ndarray, grid_n: int = 50) -> float:
    """Cutoff on the Pareto frontier trading migrations triggered against true
    failures averted. Picks the knee: the cutoff maximizing averted failures minus
    triggered migrations, both normalized to [0, 1]."""
    grid = np.quantile(prob, np.linspace(0.50, 0.999, grid_n))
    grid = np.unique(grid)
    n = prob.size
    pos = max(int((y_true == 1).sum()), 1)
    best_t, best_obj = float(grid[0]), -np.inf
    for t in grid:
        flagged = prob >= t
        triggered = flagged.sum() / n                       # cost, normalized
        averted = (flagged & (y_true == 1)).sum() / pos      # benefit, normalized
        obj = averted - triggered
        if obj > best_obj:
            best_obj, best_t = obj, float(t)
    return best_t


y_sim = sim[LABEL_COL].to_numpy().astype(np.int64)
prob_sim = sim["cal_prob"].to_numpy()

# Each strategy is a row-level "keep" mask over the episode frame: a removed
# episode (migrated, deferred, or steered off) drops out of the machine it would
# have occupied, so summing used / alloc over the kept episodes reclaims both its
# allocation and its (near-zero, V12) usage. Efficiency is aggregated per temporal
# window in Section 4, where removal-based strategies move the metric; pure
# reassignment would not, since it leaves total used / alloc unchanged.

# --- Strategy 1: preemptive migration -------------------------------------------
# Operating point: the Pareto knee trading migrations triggered against true
# failures averted. Flagged episodes are migrated before they run.
MIG_THRESHOLD = _pareto_threshold(prob_sim, y_sim)
sim = sim.with_columns(
    flag_migrate=(pl.col("cal_prob") >= MIG_THRESHOLD).cast(pl.Int8),
    keep_migrate=(pl.col("cal_prob") < MIG_THRESHOLD),
)
_n_mig = int((sim["flag_migrate"] == 1).sum())
_averted_mig = int(((sim["flag_migrate"] == 1) & (sim[LABEL_COL] == 1)).sum())
print(f"Migration: cutoff {MIG_THRESHOLD:.4f} | flagged {_n_mig:,} "
      f"| true failures averted {_averted_mig:,} "
      f"| precision {(_averted_mig / max(_n_mig, 1)):.3f}")

# --- Strategy 2: admission control ----------------------------------------------
# Operating point: a calibrated-probability cutoff (0.5). Calibration is what makes
# this honest: on the raw over-forecast scores a 0.5 cutoff would defer far too many
# submissions (V37). Flagged submissions are deferred for review.
ADMIT_THRESHOLD = 0.5
sim = sim.with_columns(
    flag_defer=(pl.col("cal_prob") >= ADMIT_THRESHOLD).cast(pl.Int8),
    keep_admit=(pl.col("cal_prob") < ADMIT_THRESHOLD),
)
_n_defer = int((sim["flag_defer"] == 1).sum())
_wasted_deferred = int(((sim["flag_defer"] == 1) & (sim[LABEL_COL] == 1)).sum())
print(f"Admission control: calibrated cutoff {ADMIT_THRESHOLD:.2f} | deferred {_n_defer:,} "
      f"| truly wasted (would fail) {_wasted_deferred:,} "
      f"| false-defer rate {(1 - _wasted_deferred / max(_n_defer, 1)):.3f}")

# %%
# --- Strategy 3: capacity-aware bin packing -------------------------------------
# Machine-targeted policy: a per-machine failure-risk score from event history
# strictly before the test period, then act only on the riskiest-quartile machines,
# reclaiming flagged-failure allocation there. This isolates whether concentrating
# intervention on high-risk machines suffices; the EDA failure model (intrinsic,
# resubmission-driven rather than machine-driven) predicts a limited lever.
MACHINE_RISK_SQL = f"""
SELECT
    e.machine_id,
    AVG(CASE WHEN e.type IN (5, 8) THEN 1.0 WHEN e.type = 6 THEN 0.0 END) AS machine_fail_rate,
    COUNTIF(e.type IN (5, 6, 8)) AS n_terminal
FROM {table_ref('instance_events_labeled')} e
WHERE e.machine_id IS NOT NULL
  AND e.time < {T_LO}
  AND e.type IN (5, 6, 8)
GROUP BY 1
HAVING n_terminal >= 20
"""
machine_risk = pl.from_arrow(_bq.query(MACHINE_RISK_SQL).to_arrow(create_bqstorage_client=True))
print(f"Machine risk scores for {machine_risk.height:,} machines "
      f"(median historical fail rate {machine_risk['machine_fail_rate'].median():.4f})")

_risk_cut = float(machine_risk["machine_fail_rate"].quantile(0.75))
sim = sim.join(machine_risk.select(["machine_id", "machine_fail_rate"]), on="machine_id", how="left")
sim = sim.with_columns(
    high_risk_machine=(pl.col("machine_fail_rate").fill_null(0.0) >= _risk_cut).cast(pl.Int8)
)
# Remove only flagged-failure episodes that sit on a high-risk machine.
sim = sim.with_columns(
    keep_pack=~((pl.col("cal_prob") >= MIG_THRESHOLD) & (pl.col("high_risk_machine") == 1))
)
_n_pack = int((~sim["keep_pack"]).sum())
print(f"Bin packing: high-risk fail-rate cutoff {_risk_cut:.4f} "
      f"| episodes steered off high-risk machines {_n_pack:,}")

# %% [markdown]
# ## 4. Efficiency improvement vs the 25% target (Stage 3)
#
# Efficiency is aggregated per temporal window (`window_idx`) as the global
# `sum(used) / sum(alloc)` over the episodes active in the window, for the baseline
# (all episodes) and for each strategy (the episodes it keeps). Pairing by temporal
# window avoids the bias a per-machine-window join introduces when a strategy empties
# a sparse machine-window. Per-window relative improvement is
# `(strategy_eff - baseline_eff) / baseline_eff`; the distribution over windows is
# tested against the 25% target with a one-sided Wilcoxon signed-rank test (per-window
# improvement against a constant 0.25 vector), with a 95% bootstrap CI on the median.
# Distrust a large gain: the printout shows windows improved and the allocation
# reclaimed so the number can be traced.

# %%
def _median_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    local = np.random.default_rng(RANDOM_SEED)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = np.median(local.choice(values, size=values.size, replace=True))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(values)), float(lo), float(hi)


def temporal_window_efficiency(frame: pl.DataFrame, keep_col: str | None) -> pl.DataFrame:
    """Global efficiency per temporal window = sum(used) / sum(alloc) over the
    episodes active in the window (all episodes if ``keep_col`` is None, else only
    the episodes the strategy keeps)."""
    f = frame if keep_col is None else frame.filter(pl.col(keep_col))
    agg = f.group_by("window_idx").agg(
        used=pl.col("used_cpu").sum(), alloc=pl.col("alloc_cpu").sum()
    ).filter(pl.col("alloc") > 0)
    return agg.with_columns(efficiency=pl.col("used") / pl.col("alloc"))


baseline_win = temporal_window_efficiency(sim, None).rename({"efficiency": "eff_base"})


def evaluate_strategy(name: str, keep_col: str) -> tuple[dict, dict]:
    """Per-temporal-window relative improvement vs baseline, the Wilcoxon test
    against the 25% target, and a bootstrap CI on the median improvement."""
    strat = temporal_window_efficiency(sim, keep_col).rename({"efficiency": "eff_strat"})
    joined = baseline_win.join(
        strat.select(["window_idx", "eff_strat"]), on="window_idx", how="inner"
    ).filter(pl.col("eff_base") > 0)
    improvement = ((joined["eff_strat"] - joined["eff_base"]) / joined["eff_base"]).to_numpy()
    med, lo, hi = _median_ci(improvement)
    wil = paired_wilcoxon_cv(
        list(improvement), [RQ4_TARGET_IMPROVEMENT] * improvement.size, alternative="greater")
    n_improved = int((improvement > 0).sum())
    reclaimed = float(sim.filter(~pl.col(keep_col))["alloc_cpu"].sum()
                      / max(float(sim["alloc_cpu"].sum()), 1e-12))
    print(f"\n[{name}] windows {improvement.size:,} | improved {n_improved:,} "
          f"({n_improved / max(improvement.size, 1):.3f}) | alloc reclaimed {reclaimed:.3f}")
    print(f"  median relative improvement {med:.4f} 95% CI [{lo:.4f}, {hi:.4f}] "
          f"(target {RQ4_TARGET_IMPROVEMENT:.2f})")
    print(f"  Wilcoxon vs 25%: p={wil['p_value']:.4g} -> {wil['decision']}")
    table_row = {
        "strategy": name, "n_windows": improvement.size,
        "median_improvement": round(med, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "frac_windows_improved": round(n_improved / max(improvement.size, 1), 4),
        "alloc_reclaimed_frac": round(reclaimed, 4),
    }
    hyp_row = {
        "strategy": name, "target_improvement": RQ4_TARGET_IMPROVEMENT,
        "median_improvement": round(med, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "wilcoxon_p": wil["p_value"], "reject_h0": bool(wil["p_value"] < 0.05 and med > RQ4_TARGET_IMPROVEMENT),
        "decision": wil["decision"],
    }
    return table_row, hyp_row


strategies = [
    ("preemptive_migration", "keep_migrate"),
    ("admission_control", "keep_admit"),
    ("capacity_aware_bin_packing", "keep_pack"),
]

rq4_rows: list[dict] = [{
    "strategy": "reactive_baseline", "n_windows": baseline_win.height,
    "median_improvement": 0.0, "ci_low": 0.0, "ci_high": 0.0,
    "frac_windows_improved": 0.0, "alloc_reclaimed_frac": 0.0,
}]
hyp_rows: list[dict] = []
for name, keep_col in strategies:
    t_row, h_row = evaluate_strategy(name, keep_col)
    rq4_rows.append(t_row)
    hyp_rows.append(h_row)

# %% [markdown]
# ## 5. Cost-benefit subsection (Stage 4)
#
# Prediction overhead is the per-submission inference time times the test-period
# submission rate. Resource savings are the efficiency gain times cluster capacity.
# The breakeven point, if any, is the submission rate at which inference overhead
# equals reclaimed capacity. The numbers are first-order estimates for the
# narrative, not a production cost model.

# %%
import time

# Measured per-row inference time of the calibrated at-submission model.
_probe = sim.head(min(20_000, sim.height))
_t0 = time.perf_counter()
_ = calibrate(score_raw(_probe))
_infer_sec_per_row = (time.perf_counter() - _t0) / max(_probe.height, 1)

n_submissions = sim.select(["collection_id", "instance_index", "sched_seq"]).n_unique()
test_period_sec = (T_HI - T_LO) / MICROS_PER_SEC
submission_rate = n_submissions / test_period_sec
overhead_sec = _infer_sec_per_row * n_submissions

# Reclaimed allocation (migration strategy) as the savings proxy, in CPU-units.
reclaimed_alloc = float(sim.filter(pl.col("flag_migrate") == 1)["alloc_cpu"].sum())
total_alloc = float(sim["alloc_cpu"].sum())
print(f"Inference {_infer_sec_per_row * 1e6:.2f} us/submission | "
      f"submissions {n_submissions:,} over {test_period_sec / DAY_SEC:.1f} days "
      f"({submission_rate:.2f}/s)")
print(f"Total inference overhead {overhead_sec:.2f} s of compute")
print(f"Reclaimed allocation (migration) {reclaimed_alloc:.2f} of {total_alloc:.2f} "
      f"CPU-units ({reclaimed_alloc / max(total_alloc, 1):.3f})")

cost_benefit = {
    "infer_us_per_submission": round(_infer_sec_per_row * 1e6, 3),
    "n_submissions": int(n_submissions),
    "submission_rate_per_sec": round(submission_rate, 3),
    "inference_overhead_sec": round(overhead_sec, 3),
    "reclaimed_alloc_cpu_units": round(reclaimed_alloc, 3),
    "reclaimed_alloc_fraction": round(reclaimed_alloc / max(total_alloc, 1), 4),
}
print(cost_benefit)

# %% [markdown]
# ## 6. Reporting
#
# Efficiency / improvement rows to `rq4_google.csv`; the per-strategy Wilcoxon tests
# against the 25% target plus the cost-benefit summary to
# `rq4_google_hypothesis_test.csv`. The baseline efficiency level and the calibration
# Brier are carried as context rows so the result table is self-contained.

# %%
rq4_df = pl.DataFrame(rq4_rows)
rq4_df = rq4_df.with_columns(
    baseline_mean_efficiency=pl.lit(round(float(_b.mean()), 4)),
    baseline_median_efficiency=pl.lit(round(float(np.median(_b)), 4)),
    window_minutes=pl.lit(WINDOW_MIN),
    calibration_method=pl.lit(CALIB_METHOD),
)
rq4_df.write_csv(str(RQ4_TABLE_CSV))
print(f"Wrote {rq4_df.height} rows -> {RQ4_TABLE_CSV}")
print(rq4_df)

# Hypothesis-test table, with calibration and cost-benefit context rows.
hyp_df = pl.DataFrame(hyp_rows)
hyp_df = hyp_df.with_columns(
    test_brier_raw=pl.lit(round(brier_raw[0], 5)),
    test_brier_calibrated=pl.lit(round(brier_cal[0], 5)),
    infer_us_per_submission=pl.lit(cost_benefit["infer_us_per_submission"]),
    reclaimed_alloc_fraction=pl.lit(cost_benefit["reclaimed_alloc_fraction"]),
)
hyp_df.write_csv(str(RQ4_HYPOTHESIS_CSV))
print(f"\nWrote {hyp_df.height} rows -> {RQ4_HYPOTHESIS_CSV}")
print(hyp_df)

# %%
# Per-strategy efficiency comparison figure.
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["baseline"] + [s[0] for s in strategies]
means = [float(baseline_win["eff_base"].mean())]
for _, keep_col in strategies:
    means.append(float(temporal_window_efficiency(sim, keep_col)["efficiency"].to_numpy().mean()))
ax.bar(range(len(labels)), means, color=["#888"] + ["#2a7"] * len(strategies))
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("mean machine-window efficiency")
ax.set_title(f"RQ4 efficiency by strategy (days {TEST_DAY_LO}-{TEST_DAY_HI})")
fig.tight_layout()
fig.savefig(FIG_DIR / "efficiency_by_strategy.png", dpi=150)
plt.close(fig)
print(f"Saved {FIG_DIR / 'efficiency_by_strategy.png'}")
