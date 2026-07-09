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
# # 14. Backblaze RQ3: Lead Time as Maximum Horizon at MCC > 0.80
#
# **Purpose.** Reframe the multi-horizon failure-prediction results (notebook 12)
# as a lead-time question. Backblaze telemetry is daily and failures develop
# gradually, so the 15-minute lead-time target clears trivially at any horizon;
# the substantive question is the maximum days-ahead horizon at which an ensemble
# still sustains MCC > 0.80 at natural prevalence. This complements the Google
# RQ3 result, where discrimination is strong but rapid-onset crashes leave only
# seconds to minutes of warning.
#
# **Inputs.** `outputs/tables/rq1_backblaze.csv` (natural-prevalence MCC, PR-AUC,
# F1 by horizon and model) and the per-drive terminal table
# (`backblaze_preprocessed/backblaze_drive_terminal.parquet`) for the survival
# view over failing drives.
#
# **Output.** `outputs/tables/rq3_backblaze.csv` and
# `outputs/tables/rq3_backblaze_hypothesis_test.csv`.

# %% [markdown]
# ## 0. Colab session setup

# %%
# !pip install -q polars pandas pyarrow google-cloud-storage

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
import polars as pl

from utils.colab_setup import setup_drive, OUTPUT_DIR
from src.evaluation.hypothesis import one_sample_threshold_test

setup_drive()
TABLES_DIR = OUTPUT_DIR / 'tables'

GCS_BUCKET = f'{PROJECT_ID}-dissertation-data'
GCS_TERMINAL = 'backblaze_preprocessed/backblaze_drive_terminal.parquet'
LOCAL_TERMINAL = Path('/content/backblaze_drive_terminal.parquet')

MCC_TARGET = 0.80        # RQ3 discrimination bar for the lead-time claim
LEAD_TIME_TARGET_MIN = 15  # the 15-minute lead-time target
HORIZONS = (7, 14, 30)

# %% [markdown]
# ## 1. Load the RQ1 natural-prevalence results
#
# The lead-time question uses the natural-prevalence MCC per horizon and model.

# %%
rq1 = pl.read_csv(TABLES_DIR / 'rq1_backblaze.csv')
mcc = (
    rq1.filter((pl.col('metric') == 'mcc') & (pl.col('prevalence') == 'natural'))
    .select('horizon', 'model', 'value', 'ci_low', 'ci_high')
    .rename({'value': 'mcc', 'ci_low': 'mcc_ci_low', 'ci_high': 'mcc_ci_high'})
)
prauc = (
    rq1.filter((pl.col('metric') == 'pr_auc') & (pl.col('prevalence') == 'natural'))
    .select('horizon', 'model', 'value').rename({'value': 'pr_auc'})
)
cells = mcc.join(prauc, on=['horizon', 'model'], how='left').sort(['model', 'horizon'])
print(cells.to_pandas().to_string(index=False))

# %% [markdown]
# ## 2. Maximum horizon at MCC > 0.80
#
# For each model, the deepest horizon whose MCC lower bound clears 0.80. None is
# expected to clear it at natural prevalence, in which case the best achieved
# horizon and MCC are the reported finding (consistent with the RQ1 result).

# %%
summary_rows = []
for model in cells['model'].unique().to_list():
    sub = cells.filter(pl.col('model') == model).sort('horizon')
    clearing = sub.filter(pl.col('mcc_ci_low') > MCC_TARGET)
    max_h = int(clearing['horizon'].max()) if clearing.height else None
    best = sub.sort('mcc', descending=True).row(0, named=True)
    summary_rows.append({
        'model': model,
        'max_horizon_days_at_mcc_gt_080': max_h if max_h is not None else -1,
        'meets_mcc_080': max_h is not None,
        'best_horizon_days': int(best['horizon']),
        'best_mcc': best['mcc'], 'best_mcc_ci_low': best['mcc_ci_low'],
        'best_mcc_ci_high': best['mcc_ci_high'], 'best_pr_auc': best['pr_auc'],
        'meets_15min_lead': True,  # daily data yields days of warning, far above 15 minutes
    })
summary = pl.DataFrame(summary_rows).sort('best_mcc', descending=True)
summary.write_csv(TABLES_DIR / 'rq3_backblaze.csv')
print(summary.to_pandas().to_string(index=False))
print(f"\nSaved {TABLES_DIR / 'rq3_backblaze.csv'}")

any_clears = bool(summary['meets_mcc_080'].any())
overall_best = summary.row(0, named=True)
print(f"\nAny model-horizon clears MCC > {MCC_TARGET} at natural prevalence: {any_clears}")
print(f"Overall best: {overall_best['model']} at {overall_best['best_horizon_days']}d, "
      f"MCC {overall_best['best_mcc']:.4f} "
      f"CI [{overall_best['best_mcc_ci_low']:.4f}, {overall_best['best_mcc_ci_high']:.4f}]")

# %% [markdown]
# ## 3. Hypothesis tests
#
# Two targets: the 15-minute lead-time target (trivially cleared because daily
# telemetry provides days of warning at every horizon) and the RQ3 discrimination
# bar of MCC > 0.80 that qualifies the lead time as usable.

# %%
best = summary.row(0, named=True)
# Lead time in minutes at the best horizon: an N-day horizon is N * 1440 minutes.
lead_minutes = best['best_horizon_days'] * 24 * 60
lead_test = one_sample_threshold_test(
    float(lead_minutes), float(lead_minutes), float(lead_minutes),
    float(LEAD_TIME_TARGET_MIN), metric_name='lead_time_minutes')
mcc_test = one_sample_threshold_test(
    best['best_mcc'], best['best_mcc_ci_low'], best['best_mcc_ci_high'],
    MCC_TARGET, metric_name='best_mcc')
hyp = pl.DataFrame([
    {**lead_test, 'target': '15_minute_lead_time'},
    {**mcc_test, 'target': 'mcc_gt_080_at_best_horizon'},
])
hyp.write_csv(TABLES_DIR / 'rq3_backblaze_hypothesis_test.csv')
print(hyp.select('target', 'metric_value', 'threshold', 'reject', 'decision').to_pandas().to_string(index=False))
print(f"\nSaved {TABLES_DIR / 'rq3_backblaze_hypothesis_test.csv'}")

# %% [markdown]
# ## 4. Survival view over failing drives
#
# The per-drive terminal table classifies each drive's final observation as an
# observed failure or a right-censoring event. The observed-failure count is the
# survival denominator; the drive-day count is the classification denominator.

# %%
from google.cloud import storage

gcs_client = storage.Client(project=PROJECT_ID)
blob = gcs_client.bucket(GCS_BUCKET).blob(GCS_TERMINAL)
if not (LOCAL_TERMINAL.exists() and LOCAL_TERMINAL.stat().st_size == blob.size):
    blob.download_to_filename(str(LOCAL_TERMINAL))

term = pl.scan_parquet(str(LOCAL_TERMINAL))
term_cols = list(term.collect_schema().names())
print(f"terminal table columns: {term_cols}")

agg_exprs = []
if 'failure_observed' in term_cols:
    agg_exprs.append(pl.col('failure_observed').sum().alias('observed_failures'))
if 'censored' in term_cols:
    agg_exprs.append(pl.col('censored').sum().alias('censored_drives'))
agg_exprs.append(pl.len().alias('drives'))
term_counts = term.select(agg_exprs).collect()
print(term_counts.to_pandas().to_string(index=False))

# Days-to-failure distribution if a per-drive lifetime column is available.
lifetime_col = next((c for c in ('observed_span_days', 'drive_age_days', 'lifetime_days', 'age_days') if c in term_cols), None)
if lifetime_col and 'failure_observed' in term_cols:
    life = (
        term.filter(pl.col('failure_observed') == 1)
        .select(pl.col(lifetime_col).alias('days_to_failure'))
        .collect()['days_to_failure']
    )
    q = life.quantile
    print(f"\nDays-to-failure over observed failures (from {lifetime_col}):")
    print(f"  median {life.median():.0f}, p25 {q(0.25):.0f}, p75 {q(0.75):.0f}, "
          f"min {life.min():.0f}, max {life.max():.0f}")
else:
    print("\nPer-drive lifetime column not present in the terminal table; "
          "survival view reports observed-vs-censored counts only.")

# %% [markdown]
# ## 5. Summary
#
# Backblaze RQ3 reframed: the 15-minute lead-time target is cleared trivially by
# daily telemetry, but no horizon up to 30 days sustains MCC > 0.80 at natural
# prevalence, so the usable lead time is limited by discrimination rather than by
# horizon. This is the mirror image of Google RQ3 (V41), where discrimination is
# strong but rapid-onset crashes cap the lead time at seconds to minutes. The two
# datasets are complementary (V24): gradual degradation gives long horizons at
# modest discrimination; rapid-onset crashes give strong discrimination at short
# lead time.

# %%
print("BACKBLAZE RQ3 LEAD-TIME SUMMARY")
print("=" * 60)
print(f"  15-minute lead-time target: cleared (daily telemetry)")
print(f"  MCC > {MCC_TARGET} at any horizon up to 30d (natural): {any_clears}")
print(f"  best: {overall_best['best_horizon_days']}d horizon, "
      f"MCC {overall_best['best_mcc']:.4f}, PR-AUC {overall_best['best_pr_auc']:.4f}")
print("=" * 60)
