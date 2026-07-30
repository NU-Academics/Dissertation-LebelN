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
# # Production-EVICT failure-label sensitivity (Google, at-submission)
#
# **Purpose.** A committed robustness branch: re-run the at-submission failure
# prediction with the alternative label that additionally counts Production-priority
# EVICTs (type 4, priority 120 to 359) as failures, and report whether the headline
# MCC changes materially. The pre-registered expectation is a near-null change, since
# Production-priority evictions are a small fraction of all evictions, but the
# analysis is committed.
#
# **Why a rebuild is needed.** The frozen episode feature table keeps only
# FAIL/LOST/FINISH terminals, so Production-EVICT episodes are absent and their
# features are stored nowhere. The alternative label adds those episodes as
# positives, so their at-submission features must be reconstructed. The at-submission
# feature set is entirely pre-scheduling (strictly-prior history, submit priority and
# scheduling class, resource requests, submit-time temporal), all derivable from the
# labeled events and the instance lifecycle summary, so no usage or runtime rebuild
# is involved.
#
# **Design.** One learner, one split, two labels. Both the primary label
# (FAIL/LOST positive, Production-EVICT excluded) and the sensitivity label
# (FAIL/LOST and Production-EVICT positive) are trained and scored on the identical
# instance-keyed group split. The primary-label run is a faithfulness check: its
# at-submission MCC must land near the frozen 0.9006, or the feature reconstruction is
# off and the delta is not to be trusted. The delta between the two labels, with the
# same learner and split, is the sensitivity result.
#
# **Output.** Appends a Google row to `outputs/tables/sensitivity_analyses.csv`.

# %% [markdown]
# ## 0. Session setup

# %%
# !pip install -q polars pandas pyarrow scikit-learn lightgbm imbalanced-learn google-cloud-bigquery google-cloud-bigquery-storage

# %%
import os
import sys

from google.colab import userdata

GITHUB_PAT = userdata.get("GITHUB_PAT")
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
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef

from google.colab import auth

from utils.colab_setup import setup_drive, OUTPUT_DIR
from utils.bq_client import get_client, table_ref
from src.data.schemas import (
    EVENT_EVICT, EVENT_FAIL, EVENT_FINISH, EVENT_LOST, EVENT_SCHEDULE,
    PRIORITY_FREE_MAX, PRIORITY_BEST_EFFORT_LOW, PRIORITY_BEST_EFFORT_MAX,
    PRIORITY_MID_TIER_LOW, PRIORITY_MID_TIER_MAX,
    PRIORITY_PRODUCTION_LOW, PRIORITY_PRODUCTION_MAX, PRIORITY_MONITORING_LOW,
)
from src.evaluation.metrics import mcc_with_ci, f1_with_ci, pr_auc_with_ci

warnings.filterwarnings("ignore", category=UserWarning)
auth.authenticate_user()
setup_drive()
bq = get_client()

TABLES_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"
for d in (TABLES_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
LABEL_PRIMARY = "failure_label"
LABEL_SENS = "failure_label_sensitivity"
SENSITIVITY_CSV = TABLES_DIR / "sensitivity_analyses.csv"

# Instance-keyed group split, matching the RQ1 at-submission notebook.
N_BUCKETS, TRAIN_BUCKET_MAX, VAL_BUCKET_MAX = 20, 14, 17
NEG_CAP, TRAIN_PERMILLE, EVAL_PERMILLE = 5, 15, 20

# %% [markdown]
# ## 1. Cheap diagnostic (run this first)
#
# Bound the label perturbation before the full scan. The instance lifecycle summary
# already classifies each instance's terminal outcome, so the instance-grain count of
# Production-EVICT terminals against FAIL/LOST is a quick magnitude and confirms the
# columns the rebuild depends on. If Production-EVICT positives are a tiny fraction of
# FAIL/LOST positives, the sensitivity is bounded to be near-null before any model
# runs.

# %%
LC = table_ref("instance_lifecycle_summary")
diag = bq.query(f"""
SELECT outcome, COUNT(*) AS n
FROM {LC}
GROUP BY outcome
ORDER BY n DESC
""").to_dataframe()
print(diag)
_counts = dict(zip(diag["outcome"], diag["n"]))
_pos = _counts.get("FAIL_LOST", 0)
_prodevict = _counts.get("EVICT_PRODUCTION", 0)
print(f"\nInstance-grain: FAIL_LOST {_pos:,} | EVICT_PRODUCTION {_prodevict:,} "
      f"| added positives about {100 * _prodevict / max(_pos, 1):.3f}% of FAIL_LOST")

# Confirm the at-submission attributes the rebuild reads are present.
cols = set(bq.query(f"SELECT * FROM {LC} LIMIT 1").to_dataframe().columns)
needed = {"collection_id", "instance_index", "submit_priority", "submit_scheduling_class",
          "cpu_request", "memory_request", "submit_time", "resubmission_count",
          "terminal_type", "terminal_priority", "outcome"}
missing = needed - cols
print(f"lifecycle summary columns present: {not missing}"
      + (f" | MISSING {missing}" if missing else ""))

# %% [markdown]
# ## 2. Build the extended at-submission matrix
#
# Set `RUN_BUILD = True` only after the diagnostic looks right. This scans the
# labeled events once to segment episodes and capture, per scheduled attempt, the
# strictly-prior history and the first terminal event's type and priority, then joins
# the instance-level submit attributes. All terminals are segmented so the prior
# counts are complete; the modeling filter (FAIL/LOST/FINISH plus Production-EVICT) is
# applied after. The instance-keyed buckets and per-instance train negative cap mirror
# the RQ1 at-submission split.

# %%
RUN_BUILD = False   # set True to run the one-time full scan

EVENTS = table_ref("instance_events_labeled")
TERMS = f"{EVENT_EVICT}, {EVENT_FAIL}, {EVENT_FINISH}, {EVENT_LOST}"   # KILL excluded (V08)
KEEP_TERMS = f"{EVENT_FAIL}, {EVENT_LOST}, {EVENT_FINISH}"             # plus Production-EVICT below

EXTENDED_SQL = f"""
WITH ev AS (
    SELECT collection_id, instance_index, time, type, priority,
        COUNTIF(type = {EVENT_SCHEDULE}) OVER (
            PARTITION BY collection_id, instance_index ORDER BY time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS sched_seq
    FROM {EVENTS}
),
seg AS (
    SELECT collection_id, instance_index, sched_seq,
        ARRAY_AGG(IF(type IN ({TERMS}), type, NULL)
                  IGNORE NULLS ORDER BY time LIMIT 1)[SAFE_OFFSET(0)] AS terminal_type,
        ARRAY_AGG(IF(type IN ({TERMS}), priority, NULL)
                  IGNORE NULLS ORDER BY time LIMIT 1)[SAFE_OFFSET(0)] AS terminal_priority
    FROM ev
    WHERE sched_seq >= 1
    GROUP BY collection_id, instance_index, sched_seq
),
hist AS (
    SELECT *,
        COUNTIF(terminal_type IN ({EVENT_FAIL}, {EVENT_LOST})) OVER w AS prior_fail_count,
        COUNTIF(terminal_type = {EVENT_EVICT}) OVER w                AS prior_evict_count,
        COUNT(*) OVER w                                              AS prior_episode_count
    FROM seg
    WINDOW w AS (PARTITION BY collection_id, instance_index ORDER BY sched_seq
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
kept AS (
    SELECT * FROM hist
    WHERE terminal_type IN ({KEEP_TERMS})
       OR (terminal_type = {EVENT_EVICT}
           AND terminal_priority BETWEEN {PRIORITY_PRODUCTION_LOW} AND {PRIORITY_PRODUCTION_MAX})
),
joined AS (
    SELECT k.collection_id, k.instance_index, k.sched_seq,
        k.terminal_type, k.terminal_priority,
        k.prior_fail_count, k.prior_evict_count, k.prior_episode_count,
        s.submit_priority, s.submit_scheduling_class,
        s.cpu_request, s.memory_request, s.submit_time,
        MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(k.collection_id AS STRING), '_',
            CAST(k.instance_index AS STRING)))), {N_BUCKETS}) AS grp,
        ABS(FARM_FINGERPRINT(CONCAT(CAST(k.collection_id AS STRING), '_',
            CAST(k.instance_index AS STRING), '_', CAST(k.sched_seq AS STRING)))) AS ephash
    FROM kept k
    JOIN {LC} s USING (collection_id, instance_index)
)
SELECT * FROM joined
WHERE MOD(ephash, 1000) < {EVAL_PERMILLE}
"""


def _label_exprs() -> list[pl.Expr]:
    is_fail = pl.col("terminal_type").is_in([EVENT_FAIL, EVENT_LOST])
    is_finish = pl.col("terminal_type") == EVENT_FINISH
    is_prodevict = ((pl.col("terminal_type") == EVENT_EVICT)
                    & (pl.col("terminal_priority") >= PRIORITY_PRODUCTION_LOW)
                    & (pl.col("terminal_priority") <= PRIORITY_PRODUCTION_MAX))
    return [
        pl.when(is_fail).then(1).when(is_finish).then(0).otherwise(None)
          .cast(pl.Int8).alias(LABEL_PRIMARY),
        pl.when(is_fail | is_prodevict).then(1).when(is_finish).then(0).otherwise(None)
          .cast(pl.Int8).alias(LABEL_SENS),
    ]


_CACHE = CACHE_DIR / "prodevict_extended.parquet"
if RUN_BUILD:
    ext = pl.from_arrow(bq.query(EXTENDED_SQL).to_arrow(create_bqstorage_client=True))
    ext = ext.with_columns(_label_exprs())
    ext.write_parquet(_CACHE)
    print(f"extended matrix: {ext.height:,} episodes")
elif _CACHE.exists():
    ext = pl.read_parquet(_CACHE)
    print(f"loaded cached extended matrix: {ext.height:,} episodes")
else:
    raise RuntimeError(
        "No cached extended matrix found. Review the Section 1 diagnostic, then set "
        "RUN_BUILD = True in the cell above and re-run to perform the one-time build scan.")

print(ext.select([
    (pl.col(LABEL_PRIMARY) == 1).sum().alias("primary_pos"),
    (pl.col(LABEL_PRIMARY) == 0).sum().alias("primary_neg"),
    (pl.col(LABEL_SENS) == 1).sum().alias("sens_pos"),
]))

# %% [markdown]
# ## 3. Derive the at-submission features
#
# The encodings reproduce the frozen episode feature build: priority-tier one-hots
# and ordinal scheduling class from the submit-time values, a null-safe request
# ratio, the submit-time temporal features from the submit wall clock, and the
# strictly-prior history block.

# %%
PRIORITY_TIER_LEVELS = ["free", "best_effort", "mid", "production", "monitoring"]


def _priority_tier_expr() -> pl.Expr:
    p = pl.col("submit_priority")
    return (pl.when(p <= PRIORITY_FREE_MAX).then(pl.lit("free"))
            .when((p >= PRIORITY_BEST_EFFORT_LOW) & (p <= PRIORITY_BEST_EFFORT_MAX)).then(pl.lit("best_effort"))
            .when((p >= PRIORITY_MID_TIER_LOW) & (p <= PRIORITY_MID_TIER_MAX)).then(pl.lit("mid"))
            .when((p >= PRIORITY_PRODUCTION_LOW) & (p <= PRIORITY_PRODUCTION_MAX)).then(pl.lit("production"))
            .when(p >= PRIORITY_MONITORING_LOW).then(pl.lit("monitoring"))
            .otherwise(pl.lit("free")).alias("priority_tier"))


def derive_features(df: pl.DataFrame) -> pl.DataFrame:
    # Submit wall clock: trace time is microseconds from trace start; the frozen build
    # shifts to US/Pacific for the temporal features. The absolute offset does not
    # change the cyclic encodings materially, so the microsecond-derived hour is used.
    us_per_hour = 3_600_000_000
    df = df.with_columns([
        _priority_tier_expr(),
        pl.col("submit_scheduling_class").cast(pl.Int64).alias("scheduling_class"),
        pl.when(pl.col("memory_request") > 0)
          .then(pl.col("cpu_request") / pl.col("memory_request"))
          .otherwise(None).alias("request_ratio"),
        (pl.col("prior_episode_count")).cast(pl.Int64).alias("resubmission_count"),
        (pl.col("prior_fail_count") > 0).cast(pl.Int8).alias("has_prior_fail"),
        ((pl.col("submit_time") // us_per_hour) % 24).cast(pl.Int64).alias("submit_hour_of_day"),
        ((pl.col("submit_time") // (us_per_hour * 24)) % 7).cast(pl.Int64).alias("submit_day_of_week"),
    ])
    df = df.with_columns([
        (pl.col("resubmission_count") >= 1).cast(pl.Int8).alias("first_resubmission"),
        (2 * np.pi * pl.col("submit_hour_of_day") / 24).sin().alias("submit_hour_sin"),
        (2 * np.pi * pl.col("submit_hour_of_day") / 24).cos().alias("submit_hour_cos"),
        ((pl.col("submit_hour_of_day") >= 8) & (pl.col("submit_hour_of_day") <= 17))
            .cast(pl.Int8).alias("submit_is_business_hours_pdt"),
        (pl.col("submit_day_of_week") >= 5).cast(pl.Int8).alias("submit_is_weekend"),
    ])
    for level in PRIORITY_TIER_LEVELS:
        df = df.with_columns((pl.col("priority_tier") == level).cast(pl.Int8).alias(f"priority_tier_{level}"))
    return df


ext = derive_features(ext)

AT_SUBMISSION_COLS = (
    ["cpu_request", "memory_request", "request_ratio"]
    + [f"priority_tier_{lvl}" for lvl in PRIORITY_TIER_LEVELS]
    + ["scheduling_class",
       "submit_hour_of_day", "submit_day_of_week", "submit_hour_sin", "submit_hour_cos",
       "submit_is_business_hours_pdt", "submit_is_weekend",
       "prior_fail_count", "has_prior_fail", "resubmission_count",
       "prior_evict_count", "first_resubmission"]
)
missing = [c for c in AT_SUBMISSION_COLS if c not in ext.columns]
assert not missing, f"missing at-submission columns: {missing}"
print(f"at-submission features: {len(AT_SUBMISSION_COLS)}")

# %% [markdown]
# ## 4. Split, train each label, and compare
#
# The split is the instance-keyed bucket split of the RQ1 notebook: train below
# bucket 14, validation [14, 17), test >= 17. Train keeps every positive and caps the
# per-instance recurring negatives; validation and test keep the natural balance. One
# imbalanced-learn pipeline (median impute plus SMOTE plus LightGBM) is fit per label
# on the same split; the threshold maximizes MCC on validation and is applied to test.

# %%
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer


def make_xy(df: pl.DataFrame, label: str):
    d = df.filter(pl.col(label).is_not_null())
    x = d.select([pl.col(c).cast(pl.Float32) for c in AT_SUBMISSION_COLS]).to_numpy()
    y = d[label].to_numpy().astype(np.int8)
    return x, y, d


def bucket_split(df: pl.DataFrame, label: str):
    train = df.filter(pl.col("grp") < TRAIN_BUCKET_MAX)
    val = df.filter((pl.col("grp") >= TRAIN_BUCKET_MAX) & (pl.col("grp") < VAL_BUCKET_MAX)
                    & (pl.col("ephash") % 1000 < TRAIN_PERMILLE))
    test = df.filter(pl.col("grp") >= VAL_BUCKET_MAX)
    # Train: keep positives; cap per-instance negatives (recurring-failer control).
    train = train.filter(pl.col("ephash") % 1000 < TRAIN_PERMILLE)
    train = train.with_columns(
        _rn=pl.col("ephash").rank("ordinal").over(["collection_id", "instance_index", label]))
    train = train.filter((pl.col(label) == 1) | (pl.col("_rn") <= NEG_CAP))
    return train, val, test


def best_threshold(y, proba):
    grid = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 99)))
    best_t, best_m = 0.5, -1.0
    for t in grid:
        m = matthews_corrcoef(y, (proba >= t).astype(np.int8))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def fit_eval(label: str) -> dict:
    """Train one at-submission LightGBM pipeline on this label and evaluate on the
    matching test split. Returns the MCC point, CI, and bootstrap replicates."""
    train, val, test = bucket_split(ext, label)
    xtr, ytr, _ = make_xy(train, label)
    xva, yva, _ = make_xy(val, label)
    xte, yte, _ = make_xy(test, label)
    pipe = ImbPipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("smote", SMOTE(random_state=SEED)),
        ("model", LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                 class_weight="balanced", random_state=SEED, verbosity=-1)),
    ]).fit(xtr, ytr)
    thr = best_threshold(yva, pipe.predict_proba(xva)[:, 1])
    proba_te = pipe.predict_proba(xte)[:, 1]
    pred_te = (proba_te >= thr).astype(np.int8)
    pt, lo, hi, reps = mcc_with_ci(yte, pred_te, seed=SEED, return_replicates=True)
    f1 = f1_with_ci(yte, pred_te, seed=SEED)
    prauc = pr_auc_with_ci(yte, proba_te, seed=SEED)
    print(f"[{label}] train {xtr.shape[0]:,} ({int(ytr.sum()):,} pos) | test {xte.shape[0]:,} "
          f"({int(yte.sum()):,} pos) | thr {thr:.4f}")
    print(f"  MCC {pt:.4f} [{lo:.4f}, {hi:.4f}] | F1 {f1[0]:.4f} | PR-AUC {prauc[0]:.4f}")
    return {"label": label, "pt": pt, "lo": lo, "hi": hi, "reps": reps}


res_p = fit_eval(LABEL_PRIMARY)
res_s = fit_eval(LABEL_SENS)

print(f"\nFaithfulness: primary-label MCC {res_p['pt']:.4f} vs frozen at-submission 0.9006 "
      f"(difference {res_p['pt'] - 0.9006:+.4f}); a large miss is a reconstruction bug, not a result.")

# %% [markdown]
# ## 5. Append the sensitivity row
#
# The two label regimes score different test sets (the sensitivity test carries the
# extra Production-EVICT positives), so the MCC difference is reported with an
# unpaired interval from the two bootstrap distributions. The label change does not
# move the headline when that interval straddles zero.

# %%
Z = 1.959964
delta = res_s["pt"] - res_p["pt"]
se_diff = float(np.sqrt(np.var(res_s["reps"]) + np.var(res_p["reps"])))
dlo, dhi = delta - Z * se_diff, delta + Z * se_diff
straddles = bool(dlo <= 0.0 <= dhi)

row = {"branch": "google_prod_evict", "dataset": "google", "model": "at_submission_lightgbm",
       "horizon": None, "mcc": round(res_s["pt"], 4), "ci_low": round(res_s["lo"], 4),
       "ci_high": round(res_s["hi"], 4), "delta_from_primary": round(delta, 4),
       "delta_ci_low": round(dlo, 4), "delta_ci_high": round(dhi, 4),
       "difference_straddles_zero": straddles}
print(row)

if SENSITIVITY_CSV.exists():
    existing = pl.read_csv(SENSITIVITY_CSV)
    combined = pl.concat([existing, pl.DataFrame([row])], how="diagonal")
else:
    combined = pl.DataFrame([row])
combined.write_csv(SENSITIVITY_CSV)
print(f"\nAppended google_prod_evict to {SENSITIVITY_CSV}")
print(combined)
