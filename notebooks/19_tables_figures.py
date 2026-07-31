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
# # Chapter 4 tables and figures regeneration pipeline
#
# **Purpose.** Single source of truth for every Chapter 4 table and figure.
# This notebook reads only the frozen artifacts in `outputs/tables/` and
# `outputs/figures/` and regenerates the camera-ready Chapter 4 set into
# `outputs/tables/chapter4_final/` and `outputs/figures/chapter4_final/`,
# together with a regeneration manifest at `outputs/manifest_chapter4.json`.
#
# **It regenerates; it does not recompute.** No raw data is loaded, no model
# is refit, and no metric is re-estimated. Every number in every output is
# transcribed or derived arithmetically from a frozen input, and a
# reconciliation section asserts that cross-file copies of the same result
# agree before anything is written. A missing input is an error, not a
# silent skip: if an artifact cannot be produced from the frozen inputs,
# that is a defect in this notebook, never a reason to re-run a modeling
# notebook.
#
# **Two figure modes.** Figures whose full underlying data live in a frozen
# table are re-rendered here with one consistent style (`mode = render`).
# Figures whose underlying data are per-observation arrays that were never
# tabled (probability curves, calibration reliability, confusion panels,
# SHAP beeswarms) are carried forward from the frozen PNG byte for byte,
# with the source checksum recorded in the manifest (`mode = copy`). Both
# modes read only frozen artifacts.
#
# **Provenance and read order.** Result tables come from the per-question
# modeling notebooks (12 through 16), the cross-cutting evaluation
# notebooks (17 and 18 variants), and the decisions audit trail
# (`outputs/tables/eda_decisions.csv`). Tables are resolved from the
# committee-visible frozen snapshots first, in the tracked repository
# (`outputs/tables/google_block/`, `outputs/tables/backblaze_block/`,
# `outputs/tables/evaluation/`, tagged `google_block/v1`,
# `backblaze_block/v1`, and `evaluation/v1`), then from the tracked
# top-level tables (the decisions ledger), and only last from the untracked
# working-copy tables directory. A working copy that has drifted from its
# frozen snapshot is reported but never read. The SHAP run stamp is pinned
# from the frozen evaluation snapshot, so the carried-forward SHAP figures
# always match the frozen SHAP tables.

# %% [markdown]
# ## 1. Session setup and input registry

# %%
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
else:
    REPO_DIR = str(Path(__file__).resolve().parents[1]) if "__file__" in globals() else "."
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# %%
# The frozen tables and figures are generated artifacts. In Colab they live in
# the Drive output directory; locally they live under the repository outputs/.
if IN_COLAB:
    from utils.colab_setup import OUTPUT_DIR, setup_drive

    setup_drive()
    OUTPUTS_ROOT = OUTPUT_DIR
else:
    OUTPUTS_ROOT = Path(REPO_DIR) / "outputs"

TABLES_DIR = OUTPUTS_ROOT / "tables"
FIGURES_DIR = OUTPUTS_ROOT / "figures"
OUT_TABLES = TABLES_DIR / "chapter4_final"
OUT_FIGURES = FIGURES_DIR / "chapter4_final"
MANIFEST_PATH = OUTPUTS_ROOT / "manifest_chapter4.json"

for d in (OUT_TABLES, OUT_FIGURES):
    d.mkdir(parents=True, exist_ok=True)

print(f"Frozen inputs:  {TABLES_DIR}  |  {FIGURES_DIR}")
print(f"Chapter 4 out:  {OUT_TABLES}  |  {OUT_FIGURES}")

# %%
# Input registry. Every frozen artifact this notebook touches is resolved and
# checksummed through these helpers, so the manifest is a complete record of
# what the Chapter 4 set was built from. Resolution failures raise immediately.
#
# Tables resolve through an ordered search: the tracked frozen snapshots
# first (the committee-visible read surface), then the tracked top-level
# tables, then the untracked working copies. If a name resolves from a
# frozen snapshot while a same-named working copy exists with different
# bytes, the drift is reported; the snapshot is what gets read.

REPO_TABLES = Path(REPO_DIR) / "outputs" / "tables"
TABLE_SEARCH_ORDER = [
    ("google_block/v1", REPO_TABLES / "google_block"),
    ("backblaze_block/v1", REPO_TABLES / "backblaze_block"),
    ("evaluation/v1", REPO_TABLES / "evaluation"),
    ("repo_tracked", REPO_TABLES),
    ("working_copy", TABLES_DIR),
]

INPUTS_USED: dict[str, dict] = {}
OUTPUTS_MADE: dict[str, dict] = {}
DRIFT_WARNINGS: list[str] = []


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _register(path: Path, registry: dict, key: str, source: str = "") -> Path:
    if key not in registry:
        registry[key] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        if source:
            registry[key]["source"] = source
    return path


def in_table(name: str) -> Path:
    """Resolve a frozen table through the ordered search; register it."""
    for label, directory in TABLE_SEARCH_ORDER:
        path = directory / name
        if not path.exists():
            continue
        working = TABLES_DIR / name
        if label != "working_copy" and working.exists() and working != path:
            if _sha256(working) != _sha256(path):
                msg = (
                    f"working copy tables/{name} differs from the frozen "
                    f"{label} snapshot; reading the snapshot"
                )
                DRIFT_WARNINGS.append(msg)
                print(f"warning: {msg}")
        return _register(path, INPUTS_USED, f"tables/{name}", source=label)
    searched = ", ".join(str(d) for _, d in TABLE_SEARCH_ORDER)
    raise AssertionError(f"missing frozen input table {name}; searched {searched}")


def in_figure(rel: str) -> Path:
    """Resolve a frozen figure by path relative to outputs/figures/."""
    path = FIGURES_DIR / rel
    assert path.exists(), f"missing frozen input figure: {path}"
    return _register(path, INPUTS_USED, f"figures/{rel}", source="working_copy")


def read_table(name: str) -> pl.DataFrame:
    return pl.read_csv(in_table(name))


def write_out_table(df: pl.DataFrame, name: str) -> Path:
    path = OUT_TABLES / name
    df.write_csv(path)
    _register(path, OUTPUTS_MADE, f"tables/chapter4_final/{name}", source="render")
    print(f"wrote {path.name}  ({df.height} rows x {df.width} cols)")
    return path


def save_out_figure(fig, rel: str) -> Path:
    path = OUT_FIGURES / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _register(path, OUTPUTS_MADE, f"figures/chapter4_final/{rel}", source="render")
    print(f"wrote figures/chapter4_final/{rel}")
    return path


def copy_figure(src: Path, rel: str) -> Path:
    """Carry a frozen figure forward byte for byte under a stable name."""
    dst = OUT_FIGURES / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    _register(dst, OUTPUTS_MADE, f"figures/chapter4_final/{rel}", source="copy")
    print(f"copied figures/chapter4_final/{rel}  <-  {src.name}")
    return dst


# %%
# Formatting helpers shared by every camera-ready table.


def fmt(v: float, d: int = 4) -> str:
    return f"{float(v):.{d}f}"


def fmt_ci(v: float, lo: float, hi: float, d: int = 4) -> str:
    return f"{float(v):.{d}f} [{float(lo):.{d}f}, {float(hi):.{d}f}]"


def fmt_p(p) -> str:
    if p is None:
        return ""
    p = float(p)
    if p < 0.001:
        return "< .001"
    if p > 0.999:
        return "> .999"
    return f"{p:.3f}"


def fmt_pct(v: float, d: int = 1) -> str:
    return f"{100.0 * float(v):.{d}f}%"


# One consistent figure style for every rendered figure.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
    }
)
C_GOOGLE = "#1f77b4"
C_BACKBLAZE = "#d62728"
C_STATIC = "#7f7f7f"
C_ONLINE = "#2ca02c"
C_TARGET = "#000000"

# %% [markdown]
# ## 2. Population and Sample audit table
#
# Section 4.1 of Chapter 4. Dimensions are transcribed from the decisions
# audit trail and the frozen working-set artifacts; where a frozen artifact
# carries the same number, it is asserted here so a drifted copy cannot
# reach the draft. Learning-curve values are read from the frozen curves,
# not restated.

# %%
decisions = read_table("eda_decisions.csv")
_needed_ids = {"V30", "V32", "V46", "V47", "V48", "P01", "P05"}
_have_ids = set(decisions["id"].to_list())
assert _needed_ids <= _have_ids, f"decisions ledger missing rows: {_needed_ids - _have_ids}"

ws_manifest = json.loads(in_table("google_working_set_manifest.json").read_text())
assert ws_manifest["total_instances"] == 35_133_137, (
    "google working-set manifest no longer matches the audited eligible population"
)
assert ws_manifest["seed"] == 42, "working-set manifest seed drifted from 42"

g_lc = read_table("google_episode_learning_curve.csv").sort("fraction")
b_lc = read_table("backblaze_learning_curve.csv").sort("fraction")
assert g_lc["fraction"].to_list()[-1] == 1.0 and b_lc["fraction"].to_list()[-1] == 1.0

g_lc_first, g_lc_last = g_lc.row(0, named=True), g_lc.row(-1, named=True)
b_lc_first, b_lc_last = b_lc.row(0, named=True), b_lc.row(-1, named=True)

population_rows = [
    {
        "dimension": "Source population",
        "google_cluster_traces": (
            "9,315,375,176 rows across five tables "
            "(31 days, May 2019, cell a; 10,005 machines)"
        ),
        "backblaze_drive_data": (
            "681,749,417 daily drive observations, 31,062 failure events "
            "(2013 to 2025)"
        ),
        "sources": "notebook 02 Sec 2; notebook 05 Sec 2",
    },
    {
        "dimension": "Analytical population after cleaning",
        "google_cluster_traces": (
            "35,133,137 eligible instances (scheduled, FINISH or FAIL/LOST "
            "terminal); 89,683,949 scheduled-episode census rows "
            "(19.2M FAIL/LOST, 70.5M FINISH)"
        ),
        "backblaze_drive_data": (
            "676,445,899 HDD-only drive-day rows; 486,253 drives, "
            "30,801 failed drives"
        ),
        "sources": "eda_decisions V30, P01; notebook 09 Sec 6 (eda_decisions V45)",
    },
    {
        "dimension": "Modeling working set",
        "google_cluster_traces": (
            "Full episode census, no subsampling; per-instance negative cap "
            "and instance-keyed group split applied at training "
            f"({g_lc_last['n_train']:,} training rows at the full-data point)"
        ),
        "backblaze_drive_data": (
            "working_set_20x: 19,222,288 rows at 20:1 (training only); "
            "sensitivity branches 10:1 (10,073,758 rows) and 40:1 "
            "(37,538,538 rows); 915,504 positive rows in all branches"
        ),
        "sources": "eda_decisions V30, V32; eda_decisions V46",
    },
    {
        "dimension": "Evaluation prevalence",
        "google_cluster_traces": (
            "Natural episode distribution in validation and test "
            "(census-wide neg:pos 3.66:1)"
        ),
        "backblaze_drive_data": (
            "Natural prevalence throughout: 2023 to 2025 test set of "
            "311,209,261 drive-day rows (failure-day rate 0.0043%; "
            "horizon-positive rates 0.032/0.060/0.125% at 7/14/30 days); "
            "threshold and calibration fit on a natural-prevalence 2022 "
            "validation set"
        ),
        "sources": "eda_decisions V32; eda_decisions V48",
    },
    {
        "dimension": "Stratification variables",
        "google_cluster_traces": (
            "(priority_tier, scheduling_class); marginals exact because the "
            "full eligible population is retained"
        ),
        "backblaze_drive_data": (
            "Proportional by drive model and calendar year; all failures "
            "retained"
        ),
        "sources": "google_working_set_manifest.json; eda_decisions V46",
    },
    {
        "dimension": "Failure event preservation",
        "google_cluster_traces": "100% by design; positives are never subsampled",
        "backblaze_drive_data": "100% by design; positives are never subsampled",
        "sources": "eda_decisions P01, V46",
    },
    {
        "dimension": "Sample-size adequacy (learning curve)",
        "google_cluster_traces": (
            f"Flat: episode-grain MCC {fmt(g_lc_first['mcc'])} at 1% of "
            f"training data to {fmt(g_lc_last['mcc'])} at 100%"
        ),
        "backblaze_drive_data": (
            f"Flat by 1%: MCC {fmt(b_lc_first['mcc'])} at 1% to "
            f"{fmt(b_lc_last['mcc'])} at 100% (scored at the 20:1 "
            "working-set prevalence; natural-prevalence results are lower "
            "by construction)"
        ),
        "sources": "google_episode_learning_curve.csv; backblaze_learning_curve.csv (V47)",
    },
]
tbl_population = pl.DataFrame(population_rows)
write_out_table(tbl_population, "table_population_sample.csv")
with pl.Config(fmt_str_lengths=200, tbl_rows=-1, tbl_width_chars=200):
    print(tbl_population)

# %% [markdown]
# ## 3. Per-question result tables
#
# One camera-ready table per research question and dataset: model by metric
# with the 95% bootstrap interval, and the frozen decision against the
# locked target where a cell was formally tested.

# %%
# 3a. Failure prediction, Google, by prediction point.
rq1g = read_table("rq1_google.csv")
rq1g_hyp = read_table("rq1_google_hypothesis_test.csv")

POINT_ORDER = ["at_submission", "at_scheduling", "early_runtime"]
_decision_g = {
    (r["prediction_point"], r["model"]): r["decision"]
    for r in rq1g_hyp.iter_rows(named=True)
}

_rows = []
for point in POINT_ORDER:
    sub = rq1g.filter(pl.col("prediction_point") == point)
    assert sub.height > 0, f"rq1_google.csv has no rows for {point}"
    models = (
        sub.filter(pl.col("metric") == "mcc")
        .sort("value", descending=True)["model"]
        .to_list()
    )
    for model in models:
        m = {
            r["metric"]: r
            for r in sub.filter(pl.col("model") == model).iter_rows(named=True)
        }
        _rows.append(
            {
                "prediction_point": point,
                "model": model,
                "mcc_95ci": fmt_ci(m["mcc"]["value"], m["mcc"]["ci_low"], m["mcc"]["ci_high"]),
                "f1_95ci": fmt_ci(m["f1"]["value"], m["f1"]["ci_low"], m["f1"]["ci_high"]),
                "pr_auc_95ci": fmt_ci(
                    m["pr_auc"]["value"], m["pr_auc"]["ci_low"], m["pr_auc"]["ci_high"]
                ),
                "decision_vs_0.90_mcc": _decision_g.get((point, model), ""),
            }
        )
tbl_rq1_google = pl.DataFrame(_rows)
write_out_table(tbl_rq1_google, "table_rq1_google.csv")

# %%
# 3b. Failure prediction, Backblaze, by horizon, at natural prevalence.
rq1b = read_table("rq1_backblaze.csv")
rq1b_hyp = read_table("rq1_backblaze_hypothesis_test.csv")
assert set(rq1b["prevalence"].unique().to_list()) == {"natural"}, (
    "rq1_backblaze.csv contains non-natural-prevalence rows; the frozen "
    "protocol scores everything at natural prevalence"
)
_decision_b = {
    (int(r["horizon"]), r["model"]): r["decision"]
    for r in rq1b_hyp.iter_rows(named=True)
}

_rows = []
for horizon in (7, 14, 30):
    sub = rq1b.filter(pl.col("horizon") == horizon)
    assert sub.height > 0, f"rq1_backblaze.csv has no rows for horizon {horizon}"
    models = (
        sub.filter(pl.col("metric") == "mcc")
        .sort("value", descending=True)["model"]
        .to_list()
    )
    for model in models:
        m = {
            r["metric"]: r
            for r in sub.filter(pl.col("model") == model).iter_rows(named=True)
        }
        _rows.append(
            {
                "horizon_days": horizon,
                "model": model,
                "mcc_95ci": fmt_ci(m["mcc"]["value"], m["mcc"]["ci_low"], m["mcc"]["ci_high"]),
                "f1_95ci": fmt_ci(m["f1"]["value"], m["f1"]["ci_low"], m["f1"]["ci_high"]),
                "pr_auc_95ci": fmt_ci(
                    m["pr_auc"]["value"], m["pr_auc"]["ci_low"], m["pr_auc"]["ci_high"]
                ),
                "decision_vs_0.90_mcc": _decision_b.get((horizon, model), ""),
            }
        )
tbl_rq1_backblaze = pl.DataFrame(_rows)
write_out_table(tbl_rq1_backblaze, "table_rq1_backblaze.csv")

# %%
# 3c. Conflict resolution, Google, by conflict scope.
rq2 = read_table("rq2_results.csv")
rq2_hyp = read_table("rq2_hypothesis_test.csv")
_decision_rq2 = {
    (r["scope"], r["model"]): r["decision"] for r in rq2_hyp.iter_rows(named=True)
}

SCOPE_ORDER = ["pooled", "resource_contention", "priority_inversion", "scheduling_violation"]
_rows = []
for scope in SCOPE_ORDER:
    sub = rq2.filter(pl.col("conflict_scope") == scope)
    assert sub.height > 0, f"rq2_results.csv has no rows for scope {scope}"
    models = (
        sub.filter(pl.col("metric") == "mcc")
        .sort("value", descending=True)["model"]
        .to_list()
    )
    for model in models:
        m = {
            r["metric"]: r
            for r in sub.filter(pl.col("model") == model).iter_rows(named=True)
        }
        _rows.append(
            {
                "conflict_scope": scope,
                "model": model,
                "mcc_95ci": fmt_ci(m["mcc"]["value"], m["mcc"]["ci_low"], m["mcc"]["ci_high"]),
                "f1_95ci": fmt_ci(m["f1"]["value"], m["f1"]["ci_low"], m["f1"]["ci_high"]),
                "pr_auc_95ci": fmt_ci(
                    m["pr_auc"]["value"], m["pr_auc"]["ci_low"], m["pr_auc"]["ci_high"]
                ),
                "decision_vs_0.80_mcc": _decision_rq2.get((scope, model), ""),
            }
        )
tbl_rq2 = pl.DataFrame(_rows)
write_out_table(tbl_rq2, "table_rq2_google.csv")

# %%
# 3d. Lead time, Google. Attempt-level and collection-level views.
rq3g = read_table("rq3_google.csv")
rq3g_hyp = read_table("rq3_google_hypothesis_test.csv")

_rows = []
for r in rq3g_hyp.iter_rows(named=True):
    level, agg = r["level"], r["aggregation"]
    sub = rq3g.filter((pl.col("level") == level) & (pl.col("aggregation") == agg))
    mcc_row = sub.filter(pl.col("metric") == "mcc")
    _rows.append(
        {
            "level": level,
            "aggregation": agg,
            "median_lead_time_min_95ci": fmt_ci(
                r["lead_median_min"], r["ci_low"], r["ci_high"], d=3
            ),
            "collection_mcc_95ci": (
                fmt_ci(
                    mcc_row["value"][0], mcc_row["ci_low"][0], mcc_row["ci_high"][0]
                )
                if mcc_row.height
                else ""
            ),
            "target_min": fmt(r["threshold_target_min"], d=0),
            "decision": r["decision"],
        }
    )
tbl_rq3_google = pl.DataFrame(_rows)
write_out_table(tbl_rq3_google, "table_rq3_google.csv")

# %%
# 3e. Lead time, Backblaze. Deepest horizon sustaining MCC above 0.80.
rq3b = read_table("rq3_backblaze.csv")
rq3b_hyp = read_table("rq3_backblaze_hypothesis_test.csv")

tbl_rq3_backblaze = pl.DataFrame(
    [
        {
            "model": r["model"],
            "best_horizon_days": int(r["best_horizon_days"]),
            "best_mcc_95ci": fmt_ci(
                r["best_mcc"], r["best_mcc_ci_low"], r["best_mcc_ci_high"]
            ),
            "best_pr_auc": fmt(r["best_pr_auc"]),
            "max_horizon_days_at_mcc_gt_0.80": (
                "none" if int(r["max_horizon_days_at_mcc_gt_080"]) < 0
                else str(int(r["max_horizon_days_at_mcc_gt_080"]))
            ),
            "meets_15_min_lead": str(bool(r["meets_15min_lead"])).lower(),
        }
        for r in rq3b.sort("best_mcc", descending=True).iter_rows(named=True)
    ]
)
write_out_table(tbl_rq3_backblaze, "table_rq3_backblaze.csv")

# %%
# 3f. Resource optimization, Google, by strategy.
rq4 = read_table("rq4_google.csv")
rq4_hyp = read_table("rq4_google_hypothesis_test.csv")
_rq4_hyp = {r["strategy"]: r for r in rq4_hyp.iter_rows(named=True)}
_base = rq4.filter(pl.col("strategy") == "reactive_baseline").row(0, named=True)

_rows = []
for r in rq4.filter(pl.col("strategy") != "reactive_baseline").iter_rows(named=True):
    h = _rq4_hyp[r["strategy"]]
    _rows.append(
        {
            "strategy": r["strategy"],
            "median_efficiency_improvement_95ci": fmt_ci(
                r["median_improvement"], r["ci_low"], r["ci_high"]
            ),
            "windows_improved": fmt_pct(r["frac_windows_improved"]),
            "allocation_reclaimed": fmt_pct(r["alloc_reclaimed_frac"]),
            "wilcoxon_p_vs_25pct": fmt_p(h["wilcoxon_p"]),
            "decision_vs_25pct": h["decision"],
        }
    )
tbl_rq4 = pl.DataFrame(_rows)
write_out_table(tbl_rq4, "table_rq4_google.csv")
print(
    f"reactive baseline efficiency: mean {fmt(_base['baseline_mean_efficiency'])}, "
    f"median {fmt(_base['baseline_median_efficiency'])}; "
    f"{int(_base['n_windows'])} windows of {int(_base['window_minutes'])} min; "
    f"calibration {_base['calibration_method']} "
    f"(test Brier {fmt(_rq4_hyp['preemptive_migration']['test_brier_raw'], 3)} raw, "
    f"{fmt(_rq4_hyp['preemptive_migration']['test_brier_calibrated'], 3)} calibrated)"
)

# %%
# 3g. Online learning under drift, Backblaze. Sustained performance and the
# adaptive-versus-static contrast.
rq5 = read_table("rq5_sustained_mcc.csv")
assert rq5.height == 6, f"rq5_sustained_mcc.csv expected 6 cells, found {rq5.height}"
assert not rq5["meets_0.85"].any(), (
    "a drift cell now claims to meet 0.85; that contradicts the frozen verdict"
)

tbl_rq5 = pl.DataFrame(
    [
        {
            "training_frozen_at": int(r["starting_year"]),
            "strategy": r["strategy"],
            "time_averaged_mcc_95ci": fmt_ci(r["mean_mcc"], r["ci_low"], r["ci_high"]),
            "time_averaged_mcc_fixed_prevalence": fmt(r["mean_mcc_fixed_prev"]),
            "initial_to_final_mcc": f"{fmt(r['initial_mcc'])} to {fmt(r['final_mcc'])}",
            "degradation_per_month_fixed_prevalence": f"{float(r['degradation_per_month_fixed_prev']):+.5f}",
            "sustainment_months_vs_own_reference": int(r["sustainment_months_vs_reference"]),
            "meets_0.85": str(bool(r["meets_0.85"])).lower(),
        }
        for r in rq5.sort(["starting_year", "strategy"]).iter_rows(named=True)
    ]
)
write_out_table(tbl_rq5, "table_rq5_backblaze.csv")

rq5_avs = read_table("rq5_adaptive_vs_static.csv")
tbl_rq5_contrast = pl.DataFrame(
    [
        {
            "training_frozen_at": int(r["starting_year"]),
            "static_mean_mcc": fmt(r["static_mean_mcc"]),
            "online_mean_mcc": fmt(r["online_mean_mcc"]),
            "static_degradation_per_month": f"{float(r['static_degradation_per_month']):+.5f}",
            "online_degradation_per_month": f"{float(r['online_degradation_per_month']):+.5f}",
            "static_sustainment_months": int(r["static_sustainment_vs_reference"]),
            "online_sustainment_months": int(r["online_sustainment_vs_reference"]),
        }
        for r in rq5_avs.sort("starting_year").iter_rows(named=True)
    ]
)
write_out_table(tbl_rq5_contrast, "table_rq5_adaptive_vs_static.csv")

# %% [markdown]
# ## 4. Hypothesis-test summary table
#
# One row per primary test (the five-question correction family) followed
# by the reported secondary tests. Before formatting, every primary is
# reconciled against its per-question source file; a mismatch stops the
# notebook, because a summary that disagrees with its source is a
# reconciliation task, never a draft edit.

# %%
hyp = read_table("hypothesis_tests.csv")
primaries = hyp.filter(pl.col("role") == "primary")
secondaries = hyp.filter(pl.col("role") == "secondary")
assert primaries.height == 5, f"expected 5 primary rows, found {primaries.height}"
assert secondaries.height == 5, f"expected 5 secondary rows, found {secondaries.height}"


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def _one(df: pl.DataFrame, **eq) -> dict:
    sub = df
    for col, val in eq.items():
        sub = sub.filter(pl.col(col) == val)
    assert sub.height == 1, f"expected one row for {eq}, found {sub.height}"
    return sub.row(0, named=True)


# Reconcile each primary against the per-question frozen source.
_p = {r["rq"]: r for r in primaries.iter_rows(named=True)}
_src = _one(rq1g_hyp, prediction_point="early_runtime")
assert _close(_p["RQ1"]["value"], _src["value"]) and _p["RQ1"]["decision"] == _src["decision"]
_src = _one(rq2_hyp, scope="priority_inversion")
assert _close(_p["RQ2"]["value"], _src["value"]) and _p["RQ2"]["decision"] == _src["decision"]
_src = _one(rq3g_hyp, level="collection", aggregation="max")
assert _close(_p["RQ3"]["value"], _src["lead_median_min"])
_src = _one(rq4_hyp, strategy="preemptive_migration")
assert _close(_p["RQ4"]["value"], _src["median_improvement"])
_src = rq5.sort("mean_mcc", descending=True).row(0, named=True)
assert _close(_p["RQ5"]["value"], _src["mean_mcc"])

# The frozen decision picture: two rejections, three failures to reject, and
# neither correction changes any decision.
_rej = primaries.filter(pl.col("decision") == "reject H0")["rq"].to_list()
assert sorted(_rej) == ["RQ1", "RQ2"], f"primary rejections drifted: {_rej}"
for r in primaries.iter_rows(named=True):
    stored_reject = r["decision"] == "reject H0"
    assert bool(r["holm_reject"]) == stored_reject == bool(r["bh_reject"]), (
        f"{r['rq']}: family-wise correction disagrees with the frozen decision"
    )

tbl_hyp = pl.DataFrame(
    [
        {
            "rq": r["rq"],
            "role": r["role"],
            "dataset": r["dataset"],
            "test_point": r["prediction_point"],
            "model_or_strategy": r["model"],
            "metric": r["headline_metric"],
            "value_95ci": fmt_ci(r["value"], r["ci_low"], r["ci_high"]),
            "target": fmt(r["threshold"], d=2),
            "raw_p": fmt_p(r["raw_p"]),
            "holm_adjusted_p": fmt_p(r["holm_adjusted_p"]),
            "bh_adjusted_p": fmt_p(r["bh_adjusted_p"]),
            "decision": r["decision"],
        }
        for r in hyp.iter_rows(named=True)
    ]
)
write_out_table(tbl_hyp, "table_hypothesis_summary.csv")
with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60, tbl_width_chars=200):
    print(tbl_hyp)

# %% [markdown]
# ## 5. Sensitivity-analysis summary table
#
# The Backblaze working-set ratio branches carry re-scored MCCs and
# difference intervals. The Google Production-EVICT branch is a bounded
# analysis without a re-train, so it is rendered as a robustness statement;
# printing its empty numeric cells as blanks would misread as missing data.

# %%
sens = read_table("sensitivity_analyses.csv")

_rows = []
for r in sens.iter_rows(named=True):
    if r["method"] == "bounded_no_retrain":
        assert r["mcc"] is None and r["delta_from_primary"] is None, (
            "the bounded branch should carry no re-trained MCC or delta"
        )
        _rows.append(
            {
                "branch": r["branch"],
                "dataset": r["dataset"],
                "comparison": "at-submission relabeling bound (no re-train)",
                "result": (
                    "Bounded analysis: Production-priority EVICTs are "
                    f"{fmt_pct(r['prod_evict_share_of_evicts'], 2)} of evictions "
                    f"({int(r['prod_evict_events']):,} events, "
                    f"{fmt_pct(r['positives_added_share_of_faillost'], 2)} of FAIL/LOST); "
                    "relabeling adds under 1% hard positives, so the "
                    "at-submission decision is unaffected"
                ),
                "difference_straddles_zero": "",
            }
        )
    else:
        assert r["branch"].startswith("working_set_"), f"unexpected branch {r['branch']}"
        _rows.append(
            {
                "branch": r["branch"],
                "dataset": r["dataset"],
                "comparison": f"{r['model']}, {int(r['horizon'])}-day horizon, natural prevalence",
                "result": (
                    f"MCC {fmt_ci(r['mcc'], r['ci_low'], r['ci_high'])}; "
                    f"difference from primary {float(r['delta_from_primary']):+.4f} "
                    f"[{float(r['delta_ci_low']):+.4f}, {float(r['delta_ci_high']):+.4f}]"
                ),
                "difference_straddles_zero": str(bool(r["difference_straddles_zero"])).lower(),
            }
        )

_ws20 = _one(sens, branch="working_set_20x")
assert _close(_ws20["delta_from_primary"], 0.0), "the 20:1 branch is the primary; delta must be 0"

tbl_sens = pl.DataFrame(_rows)
write_out_table(tbl_sens, "table_sensitivity_analyses.csv")

# %% [markdown]
# ## 6. Feature ablation, SHAP tier alignment, and top importances

# %%
abl_g = read_table("google_feature_ablation.csv").with_columns(dataset=pl.lit("Google"))
abl_b = read_table("backblaze_feature_ablation.csv").with_columns(dataset=pl.lit("Backblaze"))

ABL_ORDER = {
    "Google": ["tier1_no_history", "tier1", "tier1_tier2", "all_tiers", "tier2_missingness_only"],
    "Backblaze": ["tier1", "tier1_tier2", "all_tiers", "all_tiers_plus_prior"],
}
_rows = []
for ds, abl in (("Google", abl_g), ("Backblaze", abl_b)):
    assert set(abl["feature_set"].unique().to_list()) == set(ABL_ORDER[ds]), (
        f"{ds} ablation arms drifted from the frozen set"
    )
    for fs in ABL_ORDER[ds]:
        m = {
            r["metric"]: r
            for r in abl.filter(pl.col("feature_set") == fs).iter_rows(named=True)
        }
        _rows.append(
            {
                "dataset": ds,
                "feature_set": fs,
                "n_features": int(m["mcc"]["n_features"]),
                "mcc_95ci": fmt_ci(m["mcc"]["value"], m["mcc"]["ci_low"], m["mcc"]["ci_high"]),
                "f1_95ci": fmt_ci(m["f1"]["value"], m["f1"]["ci_low"], m["f1"]["ci_high"]),
                "pr_auc_95ci": fmt_ci(
                    m["pr_auc"]["value"], m["pr_auc"]["ci_low"], m["pr_auc"]["ci_high"]
                ),
            }
        )
tbl_ablation = pl.DataFrame(_rows)
write_out_table(tbl_ablation, "table_feature_ablation.csv")

# %%
tier = read_table("tier_alignment.csv")
tbl_tier = pl.DataFrame(
    [
        {
            "model_block": r["dataset"],
            "feature_class": r["class"],
            "features_in_shap_top15": int(r["n_in_top15"]),
            "share_of_top15": fmt_pct(r["proportion"]),
        }
        for r in tier.iter_rows(named=True)
    ]
)
_g_t1 = _one(tier, dataset="rq1_google", **{"class": "tier1"})
assert int(_g_t1["n_in_top15"]) == 13, "the Google tier-1 SHAP alignment (13 of 15) drifted"
assert tier.filter(
    (pl.col("dataset") == "rq1_google") & (pl.col("class") == "tier3")
).height == 0, "a Google tier-3 feature entered the SHAP top 15; that contradicts V13"
write_out_table(tbl_tier, "table_shap_tier_alignment.csv")

# %%
imp = read_table("feature_importances.csv")
_rows = []
for ds in ("rq1_google", "rq1_backblaze"):
    sub = imp.filter(pl.col("dataset") == ds).sort("importance", descending=True).head(15)
    assert sub.height == 15, f"feature_importances.csv has fewer than 15 rows for {ds}"
    for rank, r in enumerate(sub.iter_rows(named=True), start=1):
        _rows.append(
            {
                "model_block": ds,
                "rank": rank,
                "feature": r["feature"],
                "mean_abs_shap": fmt(r["importance"], d=5),
            }
        )
tbl_imp = pl.DataFrame(_rows)
write_out_table(tbl_imp, "table_shap_top15_features.csv")

# %% [markdown]
# ## 7. Figures
#
# Rendered figures first (rebuilt from frozen tables with one shared
# style), then the carried-forward frozen figures.

# %%
# 7a. Learning curves, both datasets.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
ax.plot(g_lc["fraction"], g_lc["mcc"], marker="o", color=C_GOOGLE)
ax.set_xscale("log")
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Fraction of training data")
ax.set_ylabel("MCC")
ax.set_title("Google, scheduled-episode grain (early-runtime point)")
ax = axes[1]
ax.plot(b_lc["fraction"], b_lc["mcc"], marker="o", color=C_BACKBLAZE)
ax.fill_between(b_lc["fraction"], b_lc["mcc_lo"], b_lc["mcc_hi"], color=C_BACKBLAZE, alpha=0.2)
ax.set_xscale("log")
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Fraction of training data")
ax.set_ylabel("MCC (20:1 working-set prevalence)")
ax.set_title("Backblaze, 30-day horizon baseline")
fig.suptitle("Sample-size adequacy: learning curves are flat by 1% of training data", y=1.04)
save_out_figure(fig, "sample/learning_curves.png")

# %%
# 7b. Feature ablation, one panel per dataset.
for ds, fname in (("Google", "ablation/google_feature_ablation_mcc.png"),
                  ("Backblaze", "ablation/backblaze_feature_ablation_mcc.png")):
    sub = tbl_ablation.filter(pl.col("dataset") == ds)
    raw = (abl_g if ds == "Google" else abl_b).filter(pl.col("metric") == "mcc")
    order = ABL_ORDER[ds]
    vals = {r["feature_set"]: r for r in raw.iter_rows(named=True)}
    y = [vals[fs]["value"] for fs in order]
    lo = [vals[fs]["value"] - vals[fs]["ci_low"] for fs in order]
    hi = [vals[fs]["ci_high"] - vals[fs]["value"] for fs in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    color = C_GOOGLE if ds == "Google" else C_BACKBLAZE
    ax.bar(range(len(order)), y, yerr=[lo, hi], capsize=3, color=color, alpha=0.85)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([fs.replace("_", "\n") for fs in order], fontsize=8)
    ax.set_ylabel("MCC (95% bootstrap CI)")
    ax.set_title(f"{ds} feature ablation")
    save_out_figure(fig, fname)

# %%
# 7c. SHAP top-15 bars, both model blocks.
for ds, color, fname in (
    ("rq1_google", C_GOOGLE, "shap/rq1_google/top15_mean_abs_shap.png"),
    ("rq1_backblaze", C_BACKBLAZE, "shap/rq1_backblaze/top15_mean_abs_shap.png"),
):
    sub = imp.filter(pl.col("dataset") == ds).sort("importance", descending=True).head(15)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(range(sub.height - 1, -1, -1), sub["importance"], color=color, alpha=0.85)
    ax.set_yticks(range(sub.height - 1, -1, -1))
    ax.set_yticklabels(sub["feature"].to_list(), fontsize=8)
    ax.set_xlabel("Mean |SHAP| (normalized)")
    ax.set_title(f"{ds}: top 15 features by SHAP importance")
    save_out_figure(fig, fname)

# %%
# 7d. Resource-optimization efficiency improvements against the 25% target.
_strats = tbl_rq4["strategy"].to_list()
_raw4 = {r["strategy"]: r for r in rq4.iter_rows(named=True)}
y = [_raw4[s]["median_improvement"] for s in _strats]
lo = [_raw4[s]["median_improvement"] - _raw4[s]["ci_low"] for s in _strats]
hi = [_raw4[s]["ci_high"] - _raw4[s]["median_improvement"] for s in _strats]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(range(len(_strats)), y, yerr=[lo, hi], capsize=3, color=C_GOOGLE, alpha=0.85)
ax.axhline(0.25, color=C_TARGET, linestyle="--", linewidth=1, label="25% target")
ax.set_xticks(range(len(_strats)))
ax.set_xticklabels([s.replace("_", "\n") for s in _strats], fontsize=8)
ax.set_ylabel("Median per-window efficiency improvement")
ax.set_title("Prediction-informed strategies vs the 25% target")
ax.legend()
save_out_figure(fig, "rq4_google/efficiency_improvement.png")

# %%
# 7e. Drift-study figures from the frozen trajectory tables.
traj = read_table("rq5_monthly_trajectory.csv").with_columns(
    pl.col("month").str.to_date().alias("month_d")
)
years = sorted(traj["starting_year"].unique().to_list())
assert years == [2015, 2018, 2021], f"unexpected starting years {years}"

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
for ax, year in zip(axes, years):
    for strategy, color, label in (
        ("static_baseline", C_STATIC, "static (frozen)"),
        ("incremental_online", C_ONLINE, "incremental online"),
    ):
        sub = traj.filter(
            (pl.col("starting_year") == year) & (pl.col("strategy") == strategy)
        ).sort("month_d")
        assert sub.height > 0, f"trajectory missing for {year}/{strategy}"
        ax.plot(sub["month_d"], sub["mcc"], color=color, label=label, linewidth=1.4)
        ax.plot(
            sub["month_d"], sub["mcc_fixed_prev"], color=color,
            linestyle="--", linewidth=1.0, alpha=0.7,
        )
    ax.set_title(f"Training frozen at {year}")
    ax.set_xlabel("Evaluation month")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
axes[0].set_ylabel("Monthly MCC")
axes[0].legend(fontsize=8)
fig.suptitle(
    "Monthly MCC on the 2023 to 2025 window (solid: raw; dashed: fixed prevalence)",
    y=1.04,
)
save_out_figure(fig, "rq5/mcc_trajectories.png")

# %%
prior = read_table("rq5_prior_shift.csv").with_columns(
    pl.col("month").str.to_date().alias("month_d")
).sort("month_d")
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(prior["month_d"], prior["prevalence"], marker=".", color=C_BACKBLAZE)
ax.set_ylabel("Monthly failure-day prevalence")
ax.set_xlabel("Evaluation month")
ax.set_title("Prior shift across the evaluation window")
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
save_out_figure(fig, "rq5/prior_shift.png")

# %%
drift = read_table("rq5_drift_events.csv").with_columns(
    pl.col("month").str.to_date().alias("month_d")
)
d21 = drift.filter(pl.col("starting_year") == 2021)
assert d21.height > 0, "no drift events for the 2021 primary cell"
detectors = sorted(d21["detector"].unique().to_list())
fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.8), sharex=True)
for ax, strategy in zip(axes, ("static_baseline", "incremental_online")):
    sub = d21.filter(pl.col("strategy") == strategy)
    bottom = None
    months = sorted(sub["month_d"].unique().to_list())
    base = np.zeros(len(months))
    for det in detectors:
        counts = []
        dsub = {r["month_d"]: r["n_signals"] for r in
                sub.filter(pl.col("detector") == det).iter_rows(named=True)}
        counts = np.array([dsub.get(m, 0) for m in months])
        ax.bar(months, counts, bottom=base, width=20, label=det)
        base = base + counts
    ax.set_title(f"{strategy.replace('_', ' ')} (training frozen at 2021)", fontsize=9)
    ax.set_ylabel("Drift signals")
axes[-1].set_xlabel("Evaluation month")
axes[0].legend(fontsize=7, ncol=len(detectors))
fig.suptitle("Drift-detector events across the evaluation window", y=1.02)
save_out_figure(fig, "rq5/drift_events.png")

# %%
x = np.arange(len(years))
width = 0.35
fig, ax = plt.subplots(figsize=(6.0, 3.4))
_stat = [
    int(_one(rq5, starting_year=y, strategy="static_baseline")["sustainment_months_vs_reference"])
    for y in years
]
_onl = [
    int(_one(rq5, starting_year=y, strategy="incremental_online")["sustainment_months_vs_reference"])
    for y in years
]
ax.bar(x - width / 2, _stat, width, color=C_STATIC, label="static (frozen)")
ax.bar(x + width / 2, _onl, width, color=C_ONLINE, label="incremental online")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel("Training frozen at")
ax.set_ylabel("Months above own reference")
ax.set_title("Sustainment windows against each model's own initial-MCC reference")
ax.legend(fontsize=8)
save_out_figure(fig, "rq5/sustainment_windows.png")

# %%
# 7f. Carried-forward frozen figures (per-observation data never tabled).
COPY_PLAN = [
    ("rq1_google/pr_curves.png", "rq1_google/pr_curves.png"),
    ("rq1_google/roc_curves.png", "rq1_google/roc_curves.png"),
    ("rq1_google/calibration.png", "rq1_google/calibration.png"),
    ("rq1_google/confusion_matrices.png", "rq1_google/confusion_matrices.png"),
    ("rq1_backblaze/pr_curves_7d.png", "rq1_backblaze/pr_curves_7d.png"),
    ("rq1_backblaze/pr_curves_14d.png", "rq1_backblaze/pr_curves_14d.png"),
    ("rq1_backblaze/pr_curves_30d.png", "rq1_backblaze/pr_curves_30d.png"),
    ("rq4_google/calibration_reliability.png", "rq4_google/calibration_reliability.png"),
]
for scope in SCOPE_ORDER:
    for name in ("pr_curves", "roc_curves", "calibration", "confusion_matrix"):
        COPY_PLAN.append(
            (f"rq2_google/{scope}/{name}.png", f"rq2_google/{scope}/{name}.png")
        )
for src_rel, dst_rel in COPY_PLAN:
    copy_figure(in_figure(src_rel), dst_rel)

# SHAP figures carry a run stamp in the filename. The stamp is pinned from
# the frozen evaluation snapshot's run-metadata file, so the carried-forward
# figures always belong to the same SHAP run as the frozen SHAP tables. A
# stray later SHAP run on disk cannot be picked up silently.
_shap_meta = sorted((REPO_TABLES / "evaluation").glob("shap_run_metadata_*.json"))
assert len(_shap_meta) == 1, (
    f"expected exactly one shap_run_metadata_*.json in the evaluation "
    f"snapshot, found {[p.name for p in _shap_meta]}"
)
SHAP_STAMP = _shap_meta[0].stem.rsplit("_", 1)[-1]
_register(_shap_meta[0], INPUTS_USED, f"tables/{_shap_meta[0].name}", source="evaluation/v1")
print(f"SHAP run stamp pinned from the evaluation snapshot: {SHAP_STAMP}")

SHAP_FIGURES = [
    ("shap/rq1_google", "beeswarm"),
    ("shap/rq1_google", "top20_bar"),
    ("shap/rq1_google", "dependence_first_resubmission"),
    ("shap/rq1_google", "dependence_has_prior_fail"),
    ("shap/rq1_google", "dependence_prior_fail_count"),
    ("shap/rq1_google", "dependence_queue_time"),
    ("shap/rq1_google", "dependence_resubmission_count"),
    ("shap/rq1_backblaze", "beeswarm_lightgbm"),
    ("shap/rq1_backblaze", "beeswarm_random_forest"),
    ("shap/rq1_backblaze", "beeswarm_xgboost"),
    ("shap/rq1_backblaze", "top20_bar_stack"),
]
for subdir, stem in SHAP_FIGURES:
    copy_figure(
        in_figure(f"{subdir}/{stem}_{SHAP_STAMP}.png"), f"{subdir}/{stem}.png"
    )

# %% [markdown]
# ## 8. Regeneration manifest
#
# Records the exact frozen inputs (with checksums), every artifact
# produced, the notebook commit, and the library versions, so the Chapter 4
# set is reproducible and auditable from this one file.

# %%
def _git_commit(repo_dir: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"warning: git commit unavailable ({exc})")
        return "unavailable"


manifest = {
    "artifact": "chapter4_final",
    "generated_by": "notebooks/19_tables_figures.py",
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": _git_commit(REPO_DIR),
    "python": platform.python_version(),
    "library_versions": {
        "polars": pl.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "seed_dependence": "none; this notebook transcribes and re-renders frozen artifacts",
    "snapshot_tags": ["google_block/v1", "backblaze_block/v1", "evaluation/v1"],
    "shap_run_stamp": SHAP_STAMP,
    "working_copy_drift_warnings": DRIFT_WARNINGS,
    "inputs": dict(sorted(INPUTS_USED.items())),
    "outputs": dict(sorted(OUTPUTS_MADE.items())),
    "counts": {
        "inputs": len(INPUTS_USED),
        "output_tables": sum(1 for k in OUTPUTS_MADE if k.startswith("tables/")),
        "output_figures": sum(1 for k in OUTPUTS_MADE if k.startswith("figures/")),
    },
}
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
print(
    f"wrote {MANIFEST_PATH.name}: {manifest['counts']['inputs']} inputs, "
    f"{manifest['counts']['output_tables']} tables, "
    f"{manifest['counts']['output_figures']} figures, "
    f"commit {manifest['git_commit'][:12]}"
)

# %% [markdown]
# ## 9. Closing verification
#
# Every declared output must exist on disk, and the headline picture must
# match the frozen verdicts: the early-runtime failure-prediction and
# priority-inversion primaries reject their nulls, the other three
# primaries fail to reject, the family-wise corrections change nothing, no
# drift cell meets 0.85, and every Backblaze working-set difference
# interval straddles zero.

# %%
for rel in OUTPUTS_MADE:
    assert (OUTPUTS_ROOT / rel).exists(), f"declared output missing on disk: {rel}"

EXPECTED_TABLES = {
    "table_population_sample.csv",
    "table_rq1_google.csv",
    "table_rq1_backblaze.csv",
    "table_rq2_google.csv",
    "table_rq3_google.csv",
    "table_rq3_backblaze.csv",
    "table_rq4_google.csv",
    "table_rq5_backblaze.csv",
    "table_rq5_adaptive_vs_static.csv",
    "table_hypothesis_summary.csv",
    "table_sensitivity_analyses.csv",
    "table_feature_ablation.csv",
    "table_shap_tier_alignment.csv",
    "table_shap_top15_features.csv",
}
_actual_tables = {p.name for p in OUT_TABLES.glob("*.csv")}
assert _actual_tables == EXPECTED_TABLES, (
    f"unexpected table set: extra={_actual_tables - EXPECTED_TABLES}, "
    f"missing={EXPECTED_TABLES - _actual_tables}"
)

_ws_branches = sens.filter(pl.col("branch").str.starts_with("working_set_"))
assert _ws_branches.height == 3 and _ws_branches["difference_straddles_zero"].all()

print("Chapter 4 regeneration complete.")
print(f"  tables:  {len(_actual_tables)} in {OUT_TABLES}")
print(f"  figures: {manifest['counts']['output_figures']} in {OUT_FIGURES}")
print(f"  manifest: {MANIFEST_PATH}")
