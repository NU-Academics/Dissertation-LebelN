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
# # 14. RQ3 - Failure Lead Time (Google Cluster Traces)
#
# **Research question.** RQ3: what time-series features can predict performance
# degradation with a useful lead time? For Google the prediction fires at
# submission, so the realizable lead time is the interval between a job attempt's
# submission and its terminal event (queue_time + running_duration). The
# dissertation target is 15 minutes.
#
# **Approach.** Reuse the RQ1 at-submission ensemble checkpoint
# (`rq1_google_best_atsubmission`) rather than fitting a new model. The checkpoint
# is a tuned impute-plus-classifier pipeline; its sidecar carries the exact
# `feature_columns` order and the tuned operating `threshold`. Because probability
# calibration is deferred to RQ4, the model's output is used for ranking and at its
# tuned threshold, not as a calibrated probability.
#
# **Two levels.**
# 1. *Per attempt (episode).* Honest but, by construction, hard: the median
#    FAIL/LOST running duration is ~22.6s (V09), so 15 minutes is infeasible at the
#    attempt level. The notebook reports the attempt-level lead-time distribution
#    and states this explicitly.
# 2. *Per collection.* Aggregate the at-submission scores of a collection's
#    attempts to a collection-level risk score, find the aggregation threshold that
#    yields collection-level MCC > 0.80, and measure the lead time between the
#    collection's elevated-risk flag and the subsequent member-instance failures.
#    This is where a 15-minute lead time can become realizable.
#
# **Leakage discipline.** Evaluation is on the RQ1 test split (the same
# instance-keyed FARM_FINGERPRINT group split as notebook 12), so no instance seen
# in training is scored here. The timing columns (submit / schedule / terminal
# times) are observables, never model inputs.
#
# **Output.** `outputs/tables/rq3_google.csv` and
# `outputs/tables/rq3_google_hypothesis_test.csv`.

# %% [markdown]
# ## 0. Colab session setup

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

# Colab caches imported modules in sys.modules, so a later `git pull` has no
# effect until the runtime restarts. Drop any previously imported repo modules
# here, so the freshly pulled source is what gets imported in the cells below.
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

from src.evaluation.metrics import mcc_with_ci
from src.evaluation.hypothesis import one_sample_threshold_test

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

TABLES_DIR = OUTPUT_DIR / 'tables'
CACHE_DIR = OUTPUT_DIR / 'cache'
for directory in (TABLES_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RQ3_TABLE_CSV = TABLES_DIR / 'rq3_google.csv'
RQ3_HYPOTHESIS_CSV = TABLES_DIR / 'rq3_google_hypothesis_test.csv'

LABEL_COL = "failure_label"
MICROS_PER_SEC = 1_000_000
SECONDS_PER_MIN = 60
RQ3_TARGET_MIN = 15.0   # dissertation lead-time target (minutes)

# %% [markdown]
# ## 1. Load the at-submission checkpoint and its sidecar
#
# The pickle is the fitted impute-plus-classifier pipeline; the JSON sidecar
# carries the exact `feature_columns` order and the validation-tuned `threshold`.
# Both are applied verbatim so the scores match the RQ1 at-submission model.

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


def score_episodes(df: pl.DataFrame) -> np.ndarray:
    """At-submission failure score per episode (positive-class probability),
    features taken in the sidecar's exact column order. Used for ranking and with
    the tuned threshold, not as a calibrated probability (calibration is RQ4)."""
    x = df.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLUMNS]).to_numpy()
    proba = at_submission_model.predict_proba(x)
    return np.asarray(proba)[:, 1] if np.ndim(proba) == 2 else np.asarray(proba)


# %% [markdown]
# ## 2. Pull the test-split episodes with at-submission features and timing
#
# The evaluation set is the RQ1 test split: instances bucketed by a
# FARM_FINGERPRINT of the instance key into 20 buckets, test = buckets >= 17
# (matching notebook 12's `VAL_BUCKET_MAX`). For each test episode the query joins:
#
# - `episode_features` for the at-submission features and `failure_label`;
# - `episode_segments_history` for `attempt_submit_time` and `schedule_time`;
# - a per-episode `terminal_time` computed as the first FAIL/LOST/FINISH event
#   inside the episode's schedule interval (`episode_schedule_intervals`).
#
# Lead time (submit to terminal) is `terminal_time - attempt_submit_time`, which
# equals `queue_time + running_duration`. A per-episode hash subsample bounds Colab
# memory.

# %%
from google.colab import auth

from utils.bq_client import get_client, table_ref

auth.authenticate_user()
_bq = get_client()

N_BUCKETS = 20
TEST_BUCKET_MIN = 17                 # buckets [17, 20) -> test (matches notebook 12)
EVAL_PERMILLE = 40                   # per-INSTANCE hash subsample (keeps all of a
                                     # sampled instance's episodes, so per-instance
                                     # and per-collection aggregations stay complete)

_inst_key = "CONCAT(CAST(collection_id AS STRING),'_',CAST(instance_index AS STRING))"
_FAILLOST_FINISH = "(5, 6, 8)"       # FAIL, FINISH, LOST


def _rq3_pull_sql() -> str:
    return f"""
WITH iv_test AS (
    SELECT *
    FROM {table_ref('episode_schedule_intervals')}
    WHERE MOD(ABS(FARM_FINGERPRINT({_inst_key})), {N_BUCKETS}) >= {TEST_BUCKET_MIN}
),
term AS (
    SELECT
        iv.collection_id, iv.instance_index, iv.sched_seq,
        MIN(e.time) AS terminal_time
    FROM iv_test iv
    JOIN {table_ref('instance_events_labeled')} e
      ON e.collection_id = iv.collection_id
     AND e.instance_index = iv.instance_index
     AND e.time >= iv.schedule_time
     AND (iv.next_schedule_time IS NULL OR e.time < iv.next_schedule_time)
     AND e.type IN {_FAILLOST_FINISH}
    GROUP BY iv.collection_id, iv.instance_index, iv.sched_seq
)
SELECT
    ef.*,
    h.attempt_submit_time,
    h.schedule_time,
    term.terminal_time
FROM {table_ref('episode_features')} ef
JOIN {table_ref('episode_segments_history')} h
  USING (collection_id, instance_index, sched_seq)
LEFT JOIN term
  USING (collection_id, instance_index, sched_seq)
WHERE MOD(ABS(FARM_FINGERPRINT({_inst_key})), {N_BUCKETS}) >= {TEST_BUCKET_MIN}
  AND MOD(ABS(FARM_FINGERPRINT({_inst_key})), 1000) < {EVAL_PERMILLE}
"""


RQ3_CACHE = CACHE_DIR / f"rq3_google_test_p{EVAL_PERMILLE}.parquet"
REBUILD_RQ3 = True   # set False to reload the test-episode pull from Drive

if REBUILD_RQ3 or not RQ3_CACHE.exists():
    print("Pulling RQ1 test-split episodes with timing (BigQuery Arrow stream) ...")
    arrow = _bq.query(_rq3_pull_sql()).to_arrow(create_bqstorage_client=True)
    episodes = pl.from_arrow(arrow)
    del arrow
    gc.collect()
    episodes.write_parquet(RQ3_CACHE)
    print(f"Pulled {episodes.height:,} test episodes -> {RQ3_CACHE.name}")
else:
    episodes = pl.read_parquet(RQ3_CACHE)
    print(f"Loaded {episodes.height:,} test episodes <- {RQ3_CACHE.name}")

# Derive lead time (submit -> terminal) in minutes; null when no terminal observed.
episodes = episodes.with_columns(
    lead_time_min=(
        (pl.col("terminal_time") - pl.col("attempt_submit_time")) / (MICROS_PER_SEC * SECONDS_PER_MIN)
    )
)
_p = int(episodes.filter(pl.col(LABEL_COL) == 1).height)
print(f"  positives {_p:,} ({_p / max(episodes.height, 1):.3f}) | "
      f"terminal observed {episodes['terminal_time'].is_not_null().sum():,}")

# %% [markdown]
# ## 3. Attempt-level lead time
#
# Score every test episode with the at-submission model, flag predicted-positive
# at the tuned threshold, and report the lead-time distribution for the
# true-positive attempts (predicted positive and actually FAIL/LOST), which is the
# warning the at-submission prediction would actually deliver before the failure.
# A bootstrap CI accompanies the median. The 15-minute target is reported but is
# expected to be infeasible at this grain (V09: ~22.6s median FAIL/LOST duration).

# %%
def _median_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the median of a 1-D array."""
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


episodes = episodes.with_columns(score=pl.Series(score_episodes(episodes)))
episodes = episodes.with_columns(pred_pos=(pl.col("score") >= THRESHOLD).cast(pl.Int8))

# True-positive attempts with an observed terminal time.
tp = episodes.filter(
    (pl.col("pred_pos") == 1) & (pl.col(LABEL_COL) == 1) & pl.col("lead_time_min").is_not_null()
)
lead = tp["lead_time_min"].to_numpy()
med, lo, hi = _median_ci(lead)
q25, q75 = (float(np.percentile(lead, 25)), float(np.percentile(lead, 75))) if lead.size else (float("nan"), float("nan"))
print(f"Attempt-level lead time (true positives, n={lead.size:,}):")
print(f"  median {med:.3f} min  IQR [{q25:.3f}, {q75:.3f}]  95% CI [{lo:.3f}, {hi:.3f}]")
print(f"  fraction with >= {RQ3_TARGET_MIN:.0f} min lead: "
      f"{float(np.mean(lead >= RQ3_TARGET_MIN)) if lead.size else float('nan'):.3f}")
print("  Note (V09): 15 min is infeasible at the attempt grain; the median "
      "FAIL/LOST attempt runs ~22.6s. Lead time is realized at the collection level (Section 4).")

attempt_test = one_sample_threshold_test(
    med, lo, hi, RQ3_TARGET_MIN, metric_name="attempt_lead_min")

# %% [markdown]
# ## 4. Collection-level lead time
#
# Aggregate the at-submission scores of each collection's test attempts to a
# collection risk score (configurable: max, mean, or top-k mean). The collection
# label is whether any member attempt fails. For each aggregation, search the
# collection-score threshold for the one maximizing collection-level MCC and check
# whether it clears 0.80. Then measure the lead time between the collection's
# elevated-risk flag (the earliest submit time among the member attempts whose own
# score clears the per-attempt threshold) and the first subsequent member failure.
#
# Caveat to carry into Chapter 4. The at-submission checkpoint was trained on an
# instance-grouped split, but a collection spans many instances, so a test
# collection generally has some instances that were in the model's training set;
# the collection-level scores are therefore mildly optimistic. A fully clean
# collection-level evaluation would retrain the at-submission model under a
# collection-grouped split. The MCC-optimal threshold is also selected on this same
# set (per the empirical-threshold instruction), so the collection MCC is an
# in-sample-threshold estimate, not a held-out one.

# %%
AGGREGATIONS = ("max", "mean", "top3_mean")


def _agg_expr(kind: str) -> pl.Expr:
    if kind == "max":
        return pl.col("score").max()
    if kind == "mean":
        return pl.col("score").mean()
    if kind == "top3_mean":
        return pl.col("score").sort(descending=True).head(3).mean()
    raise ValueError(kind)


def _best_mcc_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Score threshold maximizing collection-level MCC, with that MCC."""
    from sklearn.metrics import matthews_corrcoef

    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_m = float(np.median(scores)), -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, (scores >= t).astype(np.int8))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t, best_m


# Per-collection failure time (first member failure) and submit-flag time.
_coll_fail_time = (
    episodes.filter(pl.col(LABEL_COL) == 1)
    .group_by("collection_id")
    .agg(first_fail_time=pl.col("terminal_time").min())
)
_coll_flag_time = (
    episodes.filter(pl.col("pred_pos") == 1)
    .group_by("collection_id")
    .agg(flag_submit_time=pl.col("attempt_submit_time").min())
)

rq3_rows: list[dict] = []
coll_hyp_rows: list[dict] = []

for agg in AGGREGATIONS:
    coll = episodes.group_by("collection_id").agg(
        coll_score=_agg_expr(agg),
        coll_label=(pl.col(LABEL_COL).max()).cast(pl.Int8),   # any member fails
        n_attempts=pl.len(),
    )
    y_true = coll["coll_label"].to_numpy().astype(np.int8)
    scores = coll["coll_score"].to_numpy()
    thr, mcc_pt = _best_mcc_threshold(y_true, scores)
    y_pred = (scores >= thr).astype(np.int8)
    mcc_v, mcc_lo, mcc_hi = mcc_with_ci(y_true, y_pred, n_boot=1000, seed=RANDOM_SEED)

    # Lead time for flagged collections that do fail: first failure minus flag time.
    flagged = coll.filter(pl.col("coll_score") >= thr).join(_coll_fail_time, on="collection_id", how="inner")
    flagged = flagged.join(_coll_flag_time, on="collection_id", how="left").with_columns(
        coll_lead_min=((pl.col("first_fail_time") - pl.col("flag_submit_time"))
                       / (MICROS_PER_SEC * SECONDS_PER_MIN))
    ).filter(pl.col("coll_lead_min").is_not_null() & (pl.col("coll_lead_min") >= 0))
    lead_c = flagged["coll_lead_min"].to_numpy()
    cmed, clo, chi = _median_ci(lead_c)

    print(f"\nAggregation {agg}: collections={coll.height:,} | failing={int(y_true.sum()):,}")
    print(f"  MCC-optimal threshold {thr:.4f} | collection MCC {mcc_v:.4f} [{mcc_lo:.4f}, {mcc_hi:.4f}] "
          f"({'>' if mcc_v > 0.80 else '<='}0.80)")
    print(f"  collection lead time (flagged failers, n={lead_c.size:,}): "
          f"median {cmed:.2f} min  95% CI [{clo:.2f}, {chi:.2f}]")

    rq3_rows.append({"level": "collection", "aggregation": agg, "metric": "mcc",
                     "value": round(mcc_v, 4), "ci_low": round(mcc_lo, 4), "ci_high": round(mcc_hi, 4)})
    rq3_rows.append({"level": "collection", "aggregation": agg, "metric": "lead_time_min",
                     "value": round(cmed, 3), "ci_low": round(clo, 3), "ci_high": round(chi, 3)})

    # Lead-time hypothesis test vs 15 min, only where collection MCC clears 0.80
    # (a lead time is only meaningful if the collection-level predictor works).
    if mcc_v > 0.80:
        lt_test = one_sample_threshold_test(cmed, clo, chi, RQ3_TARGET_MIN,
                                            metric_name="collection_lead_min")
        coll_hyp_rows.append({
            "aggregation": agg, "collection_mcc": round(mcc_v, 4),
            "lead_median_min": round(cmed, 3), "ci_low": round(clo, 3), "ci_high": round(chi, 3),
            "threshold_target_min": RQ3_TARGET_MIN, "reject_h0": lt_test["reject"],
            "decision": lt_test["decision"], "margin": round(lt_test["margin"], 3),
        })

# %% [markdown]
# ## 5. Reporting
#
# Attempt-level and collection-level lead-time rows to `rq3_google.csv`, and the
# lead-time hypothesis tests (attempt level, plus each collection aggregation whose
# collection MCC clears 0.80) to `rq3_google_hypothesis_test.csv`.

# %%
rq3_rows.insert(0, {"level": "attempt", "aggregation": "none", "metric": "lead_time_min",
                    "value": round(med, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
rq3_df = pl.DataFrame(rq3_rows)
rq3_df.write_csv(str(RQ3_TABLE_CSV))
print(f"Wrote {rq3_df.height} rows -> {RQ3_TABLE_CSV}")
print(rq3_df)

hyp_rows = [{
    "level": "attempt", "aggregation": "none", "lead_median_min": round(med, 3),
    "ci_low": round(lo, 3), "ci_high": round(hi, 3), "threshold_target_min": RQ3_TARGET_MIN,
    "reject_h0": attempt_test["reject"], "decision": attempt_test["decision"],
    "margin": round(attempt_test["margin"], 3),
}]
for r in coll_hyp_rows:
    hyp_rows.append({"level": "collection", **r})
hyp_df = pl.DataFrame(hyp_rows)
hyp_df.write_csv(str(RQ3_HYPOTHESIS_CSV))
print(f"\nWrote {hyp_df.height} rows -> {RQ3_HYPOTHESIS_CSV}")
print(hyp_df)
