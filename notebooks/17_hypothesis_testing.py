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
# # Cross-cutting hypothesis testing
#
# **Purpose.** Assemble the per-question hypothesis tests into one family-wise
# decision table. Every per-question test was already computed and frozen by its
# own modeling notebook and lives in `outputs/tables/`. This notebook reads those
# frozen rows, reconciles each stored decision against the locked CI rule,
# designates one primary test per research question, applies the family-wise
# correction across the five primaries, and writes `outputs/tables/hypothesis_tests.csv`.
#
# **It aggregates; it does not recompute.** No model is refit and no metric is
# re-estimated from raw predictions here. Reconciliation re-runs only the decision
# rule on the stored point estimate and its stored bootstrap interval; a mismatch
# is treated as a finding to investigate, never silently overwritten.
#
# ## The decision rule and the p-value it needs
#
# The locked rule is a one-sided CI test: for a greater-is-better metric, reject the
# null only when the lower bound of the 95% bootstrap interval clears the target.
# That rule returns a decision but no p-value, and the family-wise correction
# (Holm and Benjamini-Hochberg) needs an ordering. The reusable primitive for that
# ordering is `bootstrap_threshold_pvalue` in `metrics.py`: the share of bootstrap
# resamples that fail to clear the target, read off the same resample distribution
# the interval is cut from. That primitive is the source-side companion to the CI
# and is unit-tested.
#
# Here, at the aggregation layer, the frozen per-question tables expose only the
# point estimate and its percentile interval, not the underlying resample arrays.
# So each family p-value is derived from the frozen interval by treating it as two
# known percentiles of an approximately normal sampling distribution, giving
# `SE = (ci_high - ci_low) / (2 * z_{0.975})` and a one-sided normal-tail p against
# the target. This reuses only frozen numbers and keeps every published point
# estimate untouched. Because all five primary outcomes are decisive (each raw
# p is either at the floor or at the ceiling except the one that clears its target
# with a comfortable margin), no p-value method could move a Holm or Benjamini-Hochberg
# decision; the approximation is a labeling convenience, not a load-bearing estimate.
# Where a question already carries a native p-value from its own notebook (the
# resource-optimization Wilcoxon test), that value is surfaced alongside.
#
# ## Primary versus secondary
#
# The family is the five research questions, so exactly one test per question enters
# the correction. Two questions produced both a Google and a Backblaze result, and
# the failure-prediction question was evaluated at three prediction points; those
# extra rows are reported as secondary and excluded from the family so the
# correction is not mis-specified. The failure-prediction primary is the Google
# result at the early-runtime prediction point, the designated operating point, not
# at-submission. The correction realistically bears only on the two marginal Google
# cells, and those are secondary.

# %% [markdown]
# ## 0. Session setup

# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IN_COLAB:
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

    # Purge cached repo modules so a git pull actually takes effect in a warm runtime.
    for _m in [m for m in list(sys.modules)
               if m == "src" or m.startswith("src.") or m == "utils" or m.startswith("utils.")]:
        del sys.modules[_m]
else:
    REPO_DIR = str(Path(__file__).resolve().parents[1]) if "__file__" in globals() else "."
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %%
import numpy as np
import polars as pl
from scipy.stats import norm

from src.evaluation.hypothesis import (
    benjamini_hochberg,
    holm_bonferroni,
    one_sample_threshold_test,
)
from src.evaluation.metrics import bootstrap_threshold_pvalue

# Stale-module guard: assert on a symbol that only exists in the current module, so
# a warm runtime holding an old copy fails here rather than at the write step.
assert callable(bootstrap_threshold_pvalue), "stale src.evaluation.metrics; restart the runtime"

# %%
# The per-question tables are gitignored generated artifacts, so in Colab they are
# read from the checkpoint output directory, not the git clone. Locally they are
# read from the repository copy.
if IN_COLAB:
    from utils.colab_setup import OUTPUT_DIR, setup_drive

    setup_drive()
    TABLES_DIR = OUTPUT_DIR / "tables"
else:
    TABLES_DIR = Path(REPO_DIR) / "outputs" / "tables"

print(f"Reading frozen per-question tables from: {TABLES_DIR}")

ALPHA = 0.05
CI_ALPHA = 0.05                       # the frozen intervals are 95% percentile bootstrap
Z = float(norm.ppf(1 - CI_ALPHA / 2))  # 1.959964, the two-sided 95% critical value

# %% [markdown]
# ## 1. Load the frozen per-question tests
#
# One file per question, plus the sustained-MCC table whose decision lives in its
# `meets_0.85` column. A missing input is an error, not a silent skip.

# %%
INPUTS = {
    "rq1_google": "rq1_google_hypothesis_test.csv",
    "rq1_backblaze": "rq1_backblaze_hypothesis_test.csv",
    "rq2": "rq2_hypothesis_test.csv",
    "rq3_google": "rq3_google_hypothesis_test.csv",
    "rq3_backblaze": "rq3_backblaze_hypothesis_test.csv",
    "rq4": "rq4_google_hypothesis_test.csv",
    "rq5": "rq5_sustained_mcc.csv",
}
missing = [f for f in INPUTS.values() if not (TABLES_DIR / f).exists()]
assert not missing, f"missing frozen inputs in {TABLES_DIR}: {missing}"

T = {k: pl.read_csv(TABLES_DIR / v) for k, v in INPUTS.items()}
for k, df in T.items():
    print(f"{k:14s} {df.height:3d} rows  cols={df.columns}")

# %% [markdown]
# ## 2. Select each test and reconcile it against the locked rule
#
# For every designated test, pull its single frozen row, then re-run the CI rule on
# the stored point estimate and stored interval. The recomputed decision must equal
# the decision the source notebook stored. A disagreement stops the notebook.

# %%
def one_row(df: pl.DataFrame, **eq) -> dict:
    """Return the single row matching every column==value pair as a dict."""
    sub = df
    for col, val in eq.items():
        sub = sub.filter(pl.col(col) == val)
    if sub.height != 1:
        raise ValueError(f"expected exactly one row for {eq}, found {sub.height}")
    return sub.row(0, named=True)


def ci_threshold_pvalue(value, ci_low, ci_high, threshold, greater_is_better=True) -> float:
    """One-sided normal-tail p against the target, from a frozen percentile interval.

    Treats the reported 95% bootstrap interval as two percentiles of an
    approximately normal sampling distribution: SE = (ci_high - ci_low) / (2 z).
    A degenerate interval (zero or non-finite width) returns 0.0 or 1.0 by the
    point estimate alone.
    """
    se = (float(ci_high) - float(ci_low)) / (2.0 * Z)
    if not np.isfinite(se) or se <= 0.0:
        clears = value > threshold if greater_is_better else value < threshold
        return 0.0 if clears else 1.0
    if greater_is_better:
        return float(norm.cdf((float(threshold) - float(value)) / se))
    return float(norm.cdf((float(value) - float(threshold)) / se))


def build_record(rq, dataset, point_label, model, metric, value, ci_low, ci_high,
                 threshold, greater_is_better, stored_reject, role, native_p=None):
    """Normalize one test, reconcile its decision, and attach the derived p-value."""
    check = one_sample_threshold_test(
        float(value), float(ci_low), float(ci_high), float(threshold),
        alpha=ALPHA, metric_name=metric, greater_is_better=greater_is_better,
    )
    if bool(check["reject"]) != bool(stored_reject):
        raise AssertionError(
            f"[{rq} {dataset} {point_label}] recomputed decision {check['reject']} "
            f"disagrees with the frozen decision {stored_reject}; investigate before aggregating"
        )
    raw_p = ci_threshold_pvalue(value, ci_low, ci_high, threshold, greater_is_better)
    se = (float(ci_high) - float(ci_low)) / (2.0 * Z)
    std_distance = ((float(value) - float(threshold)) / se
                    if np.isfinite(se) and se > 0 else float("nan"))
    return {
        "rq": rq,
        "dataset": dataset,
        "prediction_point": point_label,
        "model": model,
        "headline_metric": metric,
        "value": float(value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "threshold": float(threshold),
        "greater_is_better": bool(greater_is_better),
        "reject": bool(stored_reject),
        "decision": check["decision"],
        "margin": float(check["margin"]),
        "std_distance_from_threshold": float(std_distance),
        "raw_p": float(raw_p),
        "native_p": (float(native_p) if native_p is not None else None),
        "role": role,
    }


# %% [markdown]
# ### 2a. The five primaries (the correction family)

# %%
primaries = []

# Failure prediction (Google, early-runtime prediction point).
r = one_row(T["rq1_google"], prediction_point="early_runtime")
primaries.append(build_record(
    "RQ1", "Google", "early_runtime", r["model"], "mcc",
    r["value"], r["ci_low"], r["ci_high"], r["threshold_target"],
    greater_is_better=True, stored_reject=r["reject_h0"], role="primary"))

# Conflict resolution (best priority-inversion classifier).
r = one_row(T["rq2"], scope="priority_inversion")
primaries.append(build_record(
    "RQ2", "Google", "priority_inversion", r["model"], "mcc",
    r["value"], r["ci_low"], r["ci_high"], r["threshold_target"],
    greater_is_better=True, stored_reject=r["reject_h0"], role="primary"))

# Lead time (Google, collection level). The three collection aggregations agree;
# the max aggregation gives the longest lead and still fails, so it is the
# conservative choice for a fail-to-reject.
r = one_row(T["rq3_google"], level="collection", aggregation="max")
primaries.append(build_record(
    "RQ3", "Google", "collection_max", "collection_lead_time", "lead_time_minutes",
    r["lead_median_min"], r["ci_low"], r["ci_high"], r["threshold_target_min"],
    greater_is_better=True, stored_reject=r["reject_h0"], role="primary"))

# Resource optimization (best proactive strategy). Carries a native Wilcoxon p.
r = one_row(T["rq4"], strategy="preemptive_migration")
primaries.append(build_record(
    "RQ4", "Google", "preemptive_migration", "preemptive_migration",
    "efficiency_improvement",
    r["median_improvement"], r["ci_low"], r["ci_high"], r["target_improvement"],
    greater_is_better=True, stored_reject=r["reject_h0"], role="primary",
    native_p=r["wilcoxon_p"]))

# Online learning under drift (best strategy by time-averaged MCC). The stored
# decision is the meets_0.85 flag; reject means the flag is true.
rq5 = T["rq5"].sort("mean_mcc", descending=True)
r = rq5.row(0, named=True)
primaries.append(build_record(
    "RQ5", "Backblaze", f'{r["starting_year"]}_{r["strategy"]}', r["strategy"],
    "time_averaged_mcc",
    r["mean_mcc"], r["ci_low"], r["ci_high"], 0.85,
    greater_is_better=True, stored_reject=bool(r["meets_0.85"]), role="primary"))

primary_df = pl.DataFrame(primaries)
print(primary_df.select(
    ["rq", "dataset", "prediction_point", "value", "ci_low", "ci_high",
     "threshold", "decision", "raw_p"]))

# %% [markdown]
# ### 2b. Secondary tests (reported, excluded from the family)

# %%
secondaries = []

# Failure prediction, the other two Google prediction points. At-submission is the
# marginal fail-to-reject; at-scheduling clears at the raw level.
for pp in ("at_submission", "at_scheduling"):
    r = one_row(T["rq1_google"], prediction_point=pp)
    secondaries.append(build_record(
        "RQ1", "Google", pp, r["model"], "mcc",
        r["value"], r["ci_low"], r["ci_high"], r["threshold_target"],
        greater_is_better=True, stored_reject=r["reject_h0"], role="secondary"))

# Failure prediction, Backblaze, best model at the 14-day horizon.
r = one_row(T["rq1_backblaze"], metric_name="MCC_soft_voting_stack_14d")
secondaries.append(build_record(
    "RQ1", "Backblaze", "14d", r["model"], "mcc",
    r["metric_value"], r["ci_low"], r["ci_high"], r["threshold"],
    greater_is_better=bool(r["greater_is_better"]), stored_reject=r["reject"], role="secondary"))

# Lead time, Backblaze. The discrimination target at the best horizon is unmet; the
# 15-minute lead-time target clears trivially on daily telemetry.
r = one_row(T["rq3_backblaze"], target="mcc_gt_080_at_best_horizon")
secondaries.append(build_record(
    "RQ3", "Backblaze", "best_horizon_mcc", "best_horizon", r["metric_name"],
    r["metric_value"], r["ci_low"], r["ci_high"], r["threshold"],
    greater_is_better=bool(r["greater_is_better"]), stored_reject=r["reject"], role="secondary"))
r = one_row(T["rq3_backblaze"], target="15_minute_lead_time")
secondaries.append(build_record(
    "RQ3", "Backblaze", "15_minute_lead", "daily_telemetry", r["metric_name"],
    r["metric_value"], r["ci_low"], r["ci_high"], r["threshold"],
    greater_is_better=bool(r["greater_is_better"]), stored_reject=r["reject"], role="secondary"))

secondary_df = pl.DataFrame(secondaries)
print(secondary_df.select(
    ["rq", "dataset", "prediction_point", "value", "ci_low", "ci_high",
     "threshold", "decision", "raw_p"]))

# %% [markdown]
# ## 3. Family-wise error control across the five primaries
#
# Holm-Bonferroni is the primary correction (strong family-wise control);
# Benjamini-Hochberg is reported as the false-discovery-rate companion. Both take
# the five primary p-values in research-question order.

# %%
family_p = primary_df["raw_p"].to_list()
holm = holm_bonferroni(family_p, alpha=ALPHA)
bh = benjamini_hochberg(family_p, alpha=ALPHA)

primary_df = primary_df.with_columns(
    holm_adjusted_p=pl.Series(holm["adjusted"]),
    bh_adjusted_p=pl.Series(bh["adjusted"]),
    holm_reject=pl.Series(holm["reject"]),
    bh_reject=pl.Series(bh["reject"]),
)

print("Holm rejects:", holm["n_reject"], " Benjamini-Hochberg rejects:", bh["n_reject"])
print(primary_df.select(
    ["rq", "value", "threshold", "raw_p", "holm_adjusted_p", "bh_adjusted_p",
     "holm_reject", "bh_reject"]))

# The CI rule and the corrected decision must still agree for every primary; the
# correction only ever makes a rejection harder, so a primary that failed the CI
# rule cannot become a rejection here.
for row in primary_df.iter_rows(named=True):
    assert row["reject"] == row["holm_reject"], (
        f"{row['rq']}: Holm decision {row['holm_reject']} diverges from the CI decision "
        f"{row['reject']}; a marginal result did not survive correction, report it explicitly")

# %% [markdown]
# ## 4. Effect size and model separation
#
# For the one-sample threshold tests the effect measure is the standardized
# distance of the point estimate from its target (`std_distance_from_threshold`),
# already attached above. Paired fold-level comparisons (Wilcoxon on CV fold MCC,
# Cohen's d on the fold differences) require the per-fold score arrays produced in
# the individual modeling notebooks; those arrays are not carried in the frozen
# summary tables, so a paired fold test is not reconstructable at this layer. Where
# contenders were scored on a single natural-prevalence test rather than across
# folds, bootstrap-interval overlap is the available separation check. The
# resource-optimization question additionally carries its own Wilcoxon p, surfaced
# in `native_p`.

# %%
# Backblaze 14-day contenders: separation by interval overlap against the stack.
bb14 = (T["rq1_backblaze"]
        .filter(pl.col("horizon") == 14)
        .select(["model", "metric_value", "ci_low", "ci_high"])
        .sort("metric_value", descending=True))
stack = bb14.filter(pl.col("model") == "soft_voting_stack").row(0, named=True)
sep = bb14.with_columns(
    overlaps_stack=(pl.col("ci_high") >= stack["ci_low"]) & (pl.col("ci_low") <= stack["ci_high"])
)
print("Backblaze 14d contenders vs the stack (interval overlap):")
print(sep)

# %% [markdown]
# ## 5. Write the unified table
#
# One row per research question for the primary family, then the secondary rows,
# in one file. The `decision` column is the CI-rule decision; `holm_reject` and
# `bh_reject` carry the corrected outcomes for the primaries.

# %%
OUT_COLS = [
    "rq", "role", "dataset", "prediction_point", "model", "headline_metric",
    "value", "ci_low", "ci_high", "threshold", "greater_is_better",
    "margin", "std_distance_from_threshold",
    "raw_p", "holm_adjusted_p", "bh_adjusted_p",
    "holm_reject", "bh_reject", "native_p", "decision",
]

secondary_full = secondary_df.with_columns(
    holm_adjusted_p=pl.lit(None, dtype=pl.Float64),
    bh_adjusted_p=pl.lit(None, dtype=pl.Float64),
    holm_reject=pl.lit(None, dtype=pl.Boolean),
    bh_reject=pl.lit(None, dtype=pl.Boolean),
)

unified = pl.concat([primary_df.select(OUT_COLS), secondary_full.select(OUT_COLS)],
                    how="vertical")

out_path = TABLES_DIR / "hypothesis_tests.csv"
unified.write_csv(out_path)
print(f"Wrote {unified.height} rows to {out_path}")
with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60):
    print(unified.select(
        ["rq", "role", "dataset", "value", "threshold", "raw_p",
         "holm_adjusted_p", "holm_reject", "decision"]))

# %% [markdown]
# ## 6. Outcome summary and guardrail assertions
#
# The expected picture: two of the five primaries reject the null (failure
# prediction at early-runtime, and the priority-inversion classifier), and three
# fail to reject (lead time, resource-optimization efficiency, and sustained MCC
# under drift). Holm and Benjamini-Hochberg agree with the CI rule on all five, so
# the correction changes no primary decision. The correction's only real pressure
# is on the two marginal Google prediction points, which are secondary: at-scheduling
# clears at the raw level while at-submission does not.

# %%
n_primary_reject = int(primary_df["reject"].sum())
assert n_primary_reject == 2, f"expected 2 primary rejections, got {n_primary_reject}"
assert holm["n_reject"] == 2 and bh["n_reject"] == 2, "correction changed the family decision"

rej = primary_df.filter(pl.col("reject"))["rq"].to_list()
fail = primary_df.filter(~pl.col("reject"))["rq"].to_list()
print(f"Primary rejections (reject H0): {rej}")
print(f"Primary fail-to-reject:        {fail}")

sub = one_row(secondary_df.filter(pl.col("dataset") == "Google"),
              prediction_point="at_submission")
sch = one_row(secondary_df.filter(pl.col("dataset") == "Google"),
              prediction_point="at_scheduling")
print(f"Marginal secondary Google cells: "
      f"at_submission raw_p={sub['raw_p']:.4f} ({sub['decision']}); "
      f"at_scheduling raw_p={sch['raw_p']:.4f} ({sch['decision']})")
