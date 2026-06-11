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
# # 07. EDA Findings Summary & Phase 2 to Phase 3 Handoff
#
# **Purpose.** Consolidate every validated finding and analytical decision from
# notebooks 02 through 06 (Google Cluster Traces EDA, three Google follow-up
# queries, Backblaze ingest, Backblaze EDA, Data Quality Summary) into a single
# narrative document. This notebook is the bridge between Phase 2 (Data Understanding) and
# Phase 3 (Data Preparation).
#
# **Status.** An early deliverable of the Chapter 4 plan. Markdown-forward
# (Phase 2 executed every analytical query; this notebook does not re-derive).
#
# **Outputs.**
# 1. This notebook itself (the narrative consolidation).
# 2. `outputs/tables/eda_decisions.csv`: the decisions audit trail. One row per
#    analytical decision with evidence, source notebook, applicable RQ, and
#    resolution status. The CSV grows across Phases 3 through 7 as iteration
#    loops surface new decisions.
#
# **Reading order.** Skim the dataset-summary sections, then read the decisions
# log section in full. Open questions at the end queue work for Phases 3 and 4.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import polars as pl

# Resolve the repo root by walking up until we find the requirements.txt marker.
# This works in Colab (repo cloned to /content/Dissertation-LebelN), when
# running from the notebooks/ directory, and when running from the repo root.
_CANDIDATES = [
    Path("/content/Dissertation-LebelN"),  # Colab default
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
]
REPO_ROOT = next(
    (p for p in _CANDIDATES if (p / "requirements.txt").exists() and (p / "notebooks").exists()),
    Path.cwd(),
)
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DECISIONS_CSV = TABLES_DIR / "eda_decisions.csv"
print(f"Repo root:     {REPO_ROOT}")
print(f"Tables output: {TABLES_DIR}")
print(f"Decisions CSV: {DECISIONS_CSV}")

# %% [markdown]
# ## 1. Google Cluster Traces v3: validated EDA summary
#
# Source: BigQuery public dataset `google.com:google-cluster-data`, Cell `a`
# only. Cached to `dissertation_lebel.*_full` tables. Notebooks 02 (profiling),
# 03 (deep EDA), and three follow-up queries (pre-failure utilization, lifecycle
# reconstruction, CPI/MAPI missingness) supplied the findings below.
#
# ### Dataset dimensions (confirmed)
#
# | Table | Rows | Cols | Size (GB) | Duration |
# |-------|------|------|-----------|----------|
# | instance_events_full | 1,717,317,922 | 12 | 387.45 | 31 days |
# | instance_usage_full | 7,575,500,668 | 19 | 1,991.83 | 31 days |
# | collection_events_full | 20,807,441 | 14 | 3.12 | 31 days |
# | machine_events_full | 46,219 | 7 | 0.01 | 31 days |
# | machine_attributes_full | 1,702,926 | 5 | 0.01 | 31 days |
# | Total | ~9.3B | | ~2,382 | May 2019 |
#
# ### Failure definition (resolved)
#
# Primary: `failure_label = 1` where `type IN (5, 8)` (FAIL or LOST); `0` where
# `type = 6` (FINISH); NULL otherwise. EVICT (type 4) and KILL (type 7) are
# excluded from the primary target.
#
# Sensitivity: Production-priority EVICTs (`type = 4 AND priority >= 120`)
# additionally treated as failures in the sensitivity branch (Phase 5).
#
# Class ratio after labeling: 73,611,983 successes vs 21,709,490 failures
# (3.4:1, moderate imbalance, manageable with cost-sensitive learning + SMOTE).
#
# ### The unified failure model (three-query synthesis)
#
# Failures in Google Borg are predominantly rapid-onset crashes within seconds
# of scheduling, not gradual degradation. This fundamentally reshapes the
# prediction architecture.
#
# 1. **Pre-failure utilization is inverted.** FAIL_LOST median CPU utilization
#    ratio = 0.012 vs FINISH = 0.081. The discriminative signal is rate of
#    change: failing instances ramp CPU 3.6x faster and memory 2.3x faster
#    than successful ones (aggressive startup ramp followed by crash).
# 2. **Lifecycle is short.** 93.8% of FAIL_LOST instances crash within 10 s to
#    1 min. Median running duration: 22.6 s for FAIL_LOST vs 181.0 s for
#    FINISH. Runtime prediction windows >15 min are infeasible at the instance
#    level for the dominant failure mode.
# 3. **Resubmission history dominates.** 99.04% of FAIL_LOST instances have
#    been resubmitted at least once. First-resubmission failure rate = 10.12%
#    vs 0.14% single-pass (72x increase).
# 4. **Hardware-counter missingness is MNAR.** CPI/MAPI 20.5% overall null
#    rate is workload-type driven, not platform driven. FINISH = 87.2% null;
#    FAIL_LOST = 26.8% null. When present, FAIL_LOST CPI = 2.642 vs FINISH
#    = 1.900 (+39.1%). Requires indicator encoding, not MAR imputation.
#
# ### Feature engineering priority (EDA-informed tiers)
#
# - **Tier 1 (highest value).** Pre-scheduling and historical: prior_fail_count,
#   has_prior_fail, resubmission_count, has_hardware_counters, priority_tier,
#   scheduling_class, platform_id, cpu_request, memory_request, queue_time.
# - **Tier 2 (moderate value).** Early-runtime: cpu_slope, memory_slope,
#   initial_cpu_ramp, first_interval_util_ratio, cpi_value/mapi_value with
#   availability indicators, sequence_complexity, running_duration of prior
#   runs.
# - **Tier 3 (low value, ablation only).** Windowed utilization: avg_cpu,
#   avg_memory, max_cpu, max_memory. Included to empirically reconfirm the
#   confounding finding.
#
# ### Null and missingness strategy (resolved)
#
# | Pattern | Strategy | Reason |
# |---------|----------|--------|
# | machine_id 95-99% null pre-scheduling | Filter to types 3-8 for joins | Structural, no machine assigned yet |
# | cpu_request/memory_request 0.003% | Drop 47,933 rows | Negligible |
# | sample_memory 100% null | Drop column | Not collected in trace |
# | CPI/MAPI 20.5% null | Indicator + conditional value | MNAR, workload-type dependent |
# | max_memory 0.57% null | Drop or median-impute | Negligible |
# | collection max_per_machine and max_per_switch 99%+ | Drop columns | Near-empty |
#
# ### Prediction-point architecture (resolved)
#
# Three-level prediction trained in parallel:
# - **At-submission (primary).** Uses pre-scheduling features only. Tier 1 only.
# - **At-scheduling.** Adds machine assignment (platform_id, current load).
# - **Early-runtime (sensitivity).** Adds Tier 2 slope and ramp features for
#   the subset surviving past the first usage sampling interval.

# %% [markdown]
# ## 2. Backblaze Hard Drive Data: validated EDA summary
#
# Source: public CSV files from Backblaze, ingested to GCS and converted to
# Parquet via notebook 04. EDA in notebook 05.
#
# ### Dataset dimensions (confirmed)
#
# | Attribute | Value |
# |-----------|-------|
# | Total daily observations | 681,749,417 |
# | Failure events | 31,062 |
# | Grand daily failure rate | 0.0046% |
# | Date range | 2013 to 2025 (13 years) |
# | Peak fleet | 1,315,504 active drives (2025) |
# | Peak unique drive models | 87 (2024) |
# | Fleet expansion factor | ~45x |
# | Universal SMART IDs (present every file) | 40 |
# | Total unique SMART IDs across history | 93 |
# | Total dataset columns | 197 (5 metadata + 186 SMART + 6 derived) |
#
# ### Class imbalance and its implications
#
# Severity: roughly three orders of magnitude more severe than Google. Daily
# failure rate of 0.0046% places this in the extreme-imbalance regime
# (Li et al., 2021; Yu et al., 2023). Implications:
# - Working-set construction must undersample healthy observations aggressively
#   (100:1 ratio per Ch. 3 Sampling Strategy).
# - Evaluation metrics emphasize MCC, PR-AUC, F1 over accuracy and ROC-AUC.
# - SMOTE applied per training fold, never across folds.
#
# ### Top SMART predictors (EDA-validated)
#
# Primary discriminators: SMART 197 (Current Pending Sector Count), SMART 5
# (Reallocated Sectors Count), SMART 198 (Offline Uncorrectable). SMART 187
# and 188 (other literature-standard attributes) fall below 50% availability
# in the corpus and require availability indicators rather than blind use.
#
# Single-attribute thresholding achieves only ~10% detection, confirming HDD
# failure as a multi-signal phenomenon (Zhang et al., 2023) and motivating
# the ensemble approach.
#
# ### Zero-inflation pattern
#
# Top SMART attributes have median = 0 for both failed and healthy drives.
# Discrimination comes from the distribution tail. Feature engineering targets:
# - Binary indicators (`has_nonzero_smart_*`).
# - Upper-quantile rolling statistics (p95, max over 7, 14, 30 days).
# - Rate-of-change features.
# - Not mean/median aggregations.
#
# ### Model-specific variation
#
# Substantial variation in SMART discriminative power across top drive models.
# SMART 197 is most stable across models; SMART 5 most variable. Supports
# either per-manufacturer stratified modeling or drive-model-identity as a
# categorical feature. Working-set stratification uses drive model and year.
#
# ### Concept drift quantification (RQ5 grounding)
#
# Multiple validated drift sources across the 13-year span:
# - Monthly failure rates vary ~21-fold across the monitoring period.
# - Fleet composition changed dramatically (87 unique models; ~45x growth).
# - SMART distributions exhibit systematic covariate shifts driven by model
#   turnover and fleet aging.
# - Schema evolution: monitored attributes expanded over time.
#
# This constitutes moderate-to-severe concept drift, providing the empirical
# basis for RQ5.

# %% [markdown]
# ## 3. Cross-dataset complementarity
#
# The two corpora capture complementary failure regimes:
#
# | Dimension | Google Cluster Traces | Backblaze Hard Drive |
# |-----------|----------------------|---------------------|
# | Failure mode | Rapid-onset crash (seconds to minutes) | Gradual SMART degradation (days to weeks) |
# | Primary signal | Pre-scheduling + historical | Late-life SMART tail features |
# | Class imbalance | Moderate (3.4:1) | Extreme (~3 orders of magnitude) |
# | Temporal span | 31 days | 13 years |
# | Concept drift | Minimal within window | Documented and quantified |
# | Working-set construction | Collection-level stratified | 100:1 daily-observation stratified |
# | RQ coverage | RQ1, RQ2, RQ3, RQ4 | RQ1, RQ3, RQ5 |
#
# This complementarity is the empirical basis for the dissertation's
# methodology contribution: the same ensemble + online learning framework
# operates effectively across both failure regimes.

# %% [markdown]
# ## 4. Preliminary decisions log
#
# This is an audit trail with three status categories. Each entry traces back
# to a specific source (an EDA notebook section, a follow-up query, or a DP
# Chapter 3 commitment). Categories:
#
# - **Validated (Phase 2).** Decision is supported by EDA evidence that has
#   been executed and confirmed during the proposal phase. Sourced from the
#   "Preliminary Decisions Log" tables in notebook 03 Section 7.8, notebook 05
#   Section 7.8, notebook 06 Section 5, and the three Google follow-up queries.
#   The term "preliminary" matches how the source notebooks frame these: the
#   evidence is validated, but the resulting decision may still be revised
#   when Phase 3 or 4 modeling exposes a gap.
# - **Planned (DP).** Design decision committed in DP Chapter 3 that has not
#   yet been empirically validated. Examples include the 100:1 Backblaze
#   working-set ratio (needs learning-curve confirmation), the three-level
#   Google prediction architecture (motivated by EDA, not yet built), and the
#   four-subtype drift taxonomy (not yet executed). These move to a new
#   "Validated" row when the corresponding Phase 3/4/7 work confirms them.
# - **Open.** Question still under investigation. Sourced from notebook 03
#   Section 7.7 (4 unresolved Google open questions; 3 of the original 7 were
#   resolved by the follow-up queries) and notebook 05 Section 7.7 (all 7
#   Backblaze open questions are still open).
#
# As Phases 3 through 7 iterate, the table grows: rows move from Planned to
# Validated as evidence accumulates, new Validated rows appear when iteration
# surfaces a fresh decision, and Open rows are either closed (with the new
# answer logged) or refined as the investigation matures.

# %%
DECISIONS = [
    # --- VALIDATED (Phase 2): from notebook 03 Sec 7.8 Google preliminary decisions ---
    ("V01", "Failure Definition", "Google", "Primary failure label = FAIL (type 5) + LOST (type 8); FINISH (type 6) as success; EVICT and KILL excluded from primary target",
     "93% of EVICTs are Free/Best-effort tier (notebook 03 Sec 7.2); KILL is user-initiated", "Notebook 03 Sec 7.2.2 + 7.8",
     "RQ1, RQ2, RQ3, RQ4", "Validated (Phase 2)", "May be revisited if sensitivity branch (P04) reveals materially different patterns"),

    ("V02", "Imbalance Strategy", "Google", "Cost-sensitive learning + SMOTE; 3.4:1 ratio judged manageable without extreme oversampling",
     "73.6M FINISH vs 21.7M FAIL_LOST (notebook 03 Sec 7.2)", "Notebook 03 Sec 7.8; Li et al. (2021)",
     "RQ1, RQ2", "Validated (Phase 2)", "Approach confirmed Phase 4 modeling"),

    ("V03", "Null Handling: machine_id", "Google", "Do not impute machine_id; filter to post-scheduling event types when joining",
     "Structural nulls (95-99%) for pre-scheduling events (notebook 03 Sec 7.3)", "Notebook 03 Sec 7.3 + 7.8",
     "RQ1, RQ4", "Validated (Phase 2)", "Implemented Phase 3 preprocessing module"),

    ("V04", "Null Handling: Resources", "Google", "Drop or median-impute the 47,933 rows with missing cpu_request or memory_request",
     "0.003% null rate (notebook 03 Sec 7.3)", "Notebook 03 Sec 7.3 + 7.8",
     "RQ1, RQ4", "Validated (Phase 2)", "Implemented Phase 3"),

    ("V05", "Null Handling: Usage Columns", "Google", "Drop sample_memory column entirely; investigate CPI/MAPI further before deciding",
     "sample_memory is 100% null; CPI/MAPI 20.5% null pending follow-up (notebook 03 Sec 7.3)", "Notebook 03 Sec 7.3 + 7.8",
     "RQ1, RQ3", "Validated (Phase 2)", "CPI/MAPI follow-up resolved by V11"),

    ("V06", "Temporal Split", "Google", "Chronological 70/15/15 train/val/test with blocked temporal CV inside training",
     "31-day coverage (notebook 03 Sec 7.4)", "Notebook 03 Sec 7.8; Bergmeir & Benitez (2012); Raschka (2018)",
     "RQ1, RQ2, RQ3, RQ4", "Validated (Phase 2)", "Implemented Phase 4"),

    ("V07", "Machine Features", "Google", "Include platform, capacity, failure history, and churn metrics in the feature set",
     "Platform correlates with failure density (notebook 03 Sec 7.6)", "Notebook 03 Sec 7.6 + 7.8; Zhang et al. (2023)",
     "RQ1, RQ4", "Validated (Phase 2)", "Implemented Phase 3"),

    ("V08", "KILL Handling", "Google", "Exclude KILL events entirely from analysis",
     "Type 7 is user-initiated cancellation, not predictable system behavior (notebook 03 Sec 7.2.2)", "Notebook 03 Sec 7.2 + 7.8",
     "RQ1", "Validated (Phase 2)", "Implemented Phase 3"),

    # --- VALIDATED (Phase 2): from three Google follow-up queries (resolved Open Questions #3, #4, #6) ---
    ("V09", "Failure Mechanism", "Google", "Rapid-onset failure model: median FAIL_LOST running duration 22.6s vs FINISH 181.0s; 93.8% crash in 10s to 1min",
     "Lifecycle reconstruction query resolved Open Question #3", "sql/exploration/instance_lifecycle_reconstruction.sql",
     "RQ1, RQ3", "Validated (Phase 2)", "Drives the three-level architecture in P06"),

    ("V10", "Failure Predictor", "Google", "Resubmission history dominates: 99.04% of FAIL_LOST resubmitted at least once; first-resubmission failure rate 10.12% vs 0.14% single-pass (72x)",
     "Lifecycle reconstruction query resolved Open Question #3", "sql/exploration/instance_lifecycle_reconstruction.sql",
     "RQ1", "Validated (Phase 2)", "Anchors Tier 1 historical features in V13"),

    ("V11", "CPI/MAPI Encoding", "Google", "Indicator + conditional value encoding (MNAR), not MAR imputation",
     "Workload-type driven nulls: FINISH 87.2% null vs FAIL_LOST 26.8% null. Resolved Open Question #6", "sql/exploration/cpi_mapi_missingness_structure.sql",
     "RQ1, RQ3", "Validated (Phase 2)", "Implemented Phase 3"),

    ("V12", "Pre-Failure Utilization", "Google", "Utilization inversion: failing instances use LESS CPU/memory (FAIL_LOST median 0.012 vs FINISH 0.081) but ramp 3.6x faster CPU and 2.3x faster memory",
     "Pre-failure utilization query resolved Open Question #4", "sql/exploration/pre_failure_utilization_profiles.sql",
     "RQ1, RQ3", "Validated (Phase 2)", "Justifies Tier 2 slope features; relegates Tier 3 to ablation only"),

    ("V13", "Feature Tier Structure", "Google", "Tier 1 (pre-scheduling + historical) dominates; Tier 2 (early-runtime slopes) moderate; Tier 3 (windowed utilization) confounded by short crash window",
     "Synthesis of V09, V10, V11, V12", "Three follow-up queries (Phase 2 synthesis)",
     "RQ1, RQ3", "Validated (Phase 2)", "Phase 4 feature ablation will empirically retest"),

    # --- VALIDATED (Phase 2): from notebook 05 Sec 7.8 Backblaze preliminary decisions ---
    ("V14", "Primary SMART Features", "Backblaze", "SMART 197 (raw), 5 (raw), 198 (raw)",
     "AUC 0.7367, 0.7323, 0.6815 (top 3 per notebook 05)", "Notebook 05 Sec 7.8 D1; Cheng et al. (2022); Zhang et al. (2023)",
     "RQ1, RQ3, RQ5", "Validated (Phase 2)", "Built Phase 3"),

    ("V15", "Secondary SMART Features", "Backblaze", "SMART 4, 12, 193, 240, 1, 7, 9 as supplementary discriminators",
     "AUC > 0.56; cumulative wear indicators (notebook 05)", "Notebook 05 Sec 7.8 D2",
     "RQ1, RQ3, RQ5", "Validated (Phase 2)", "Built Phase 3"),

    ("V16", "SMART 187/188 Handling", "Backblaze", "Conditional inclusion with availability indicator encoding",
     "49.5% availability (notebook 05); literature priority but data gap", "Notebook 05 Sec 7.8 D3",
     "RQ1, RQ3", "Validated (Phase 2)", "Built Phase 3; O08 explores post-2015 recovery"),

    ("V17", "Imbalance Strategy", "Backblaze", "Cost-sensitive learning + severe undersampling; evaluate anomaly detection framing",
     "21,947:1 ratio at daily-observation level (notebook 05)", "Notebook 05 Sec 7.8 D4; Li et al. (2021)",
     "RQ1, RQ5", "Validated (Phase 2)", "Implemented Phase 3 + Phase 4"),

    ("V18", "Model Stratification", "Backblaze", "Include model identity as feature; evaluate per-manufacturer sub-models",
     "AUC spread up to 0.2744 across top-5 drive models (notebook 05)", "Notebook 05 Sec 7.8 D5; Zhang et al. (2023)",
     "RQ1, RQ5", "Validated (Phase 2)", "Depth of stratification still open (see O07)"),

    ("V19", "Feature Engineering Approach", "Backblaze", "Sliding-window temporal features: rolling mean, rolling std, rate-of-change, binary non-zero indicators",
     "Gradual degradation pattern + zero-inflation (notebook 05)", "Notebook 05 Sec 7.8 D6",
     "RQ1, RQ3, RQ5", "Validated (Phase 2)", "Built Phase 3; window length still open (see O06)"),

    ("V20", "Temporal Evaluation Strategy", "Backblaze", "Expanding-window or sliding-window cross-validation across years",
     "Non-stationary failure rates and fleet composition drift (notebook 05)", "Notebook 05 Sec 7.8 D7; Bergmeir & Benitez (2012); Campos et al. (2023)",
     "RQ1, RQ5", "Validated (Phase 2)", "Implemented Phase 4 + Phase 7"),

    ("V21", "Online Learning Justified", "Backblaze", "Online and incremental learning approach justified for RQ5",
     "21x monthly failure-rate range; schema evolution; 12-year fleet turnover (notebook 05)", "Notebook 05 Sec 7.8 D8; AlShafeey & Csaki (2024)",
     "RQ5", "Validated (Phase 2)", "Executed Phase 7"),

    ("V22", "Prediction Framing", "Backblaze", "Daily sliding-window prediction with multi-day lookahead",
     "Gradual degradation supports temporal features; contrasts with Google at-submission architecture", "Notebook 05 Sec 7.8 D9",
     "RQ1, RQ3", "Validated (Phase 2)", "Implemented Phase 4 (specific horizons in P07)"),

    ("V23", "Evaluation Metrics for Imbalance", "Both", "MCC, PR-AUC, F1 as primary metrics; not accuracy or standard ROC-AUC",
     "Extreme class imbalance makes accuracy uninformative", "Notebook 05 Sec 7.8 D10; Chicco & Jurman (2023); Saito & Rehmsmeier (2015)",
     "RQ1, RQ2, RQ5", "Validated (Phase 2)", "Reported with bootstrap CIs per P15"),

    # --- VALIDATED (Phase 2): cross-dataset comparison from notebook 06 ---
    ("V24", "Cross-Dataset Complementarity", "Both", "Two complementary failure regimes: rapid-onset software (Google) vs. gradual SMART degradation (Backblaze)",
     "Notebook 06 Section 5 cross-dataset comparison table", "Notebook 06 Sec 5",
     "RQ1, RQ3", "Validated (Phase 2)", "Anchors the methodology-generalizability claim in Ch. 5"),

    # --- PLANNED (DP): committed in DP Chapter 3 but not yet empirically validated ---
    ("P01", "Working Set Size", "Google", "Target 50-100 million instance events via collection-level stratified sampling",
     "Sample-size guidance from Rajput et al. (2023); compute envelope of Colab T4", "DP Ch. 3 Sampling Strategy",
     "RQ1, RQ2, RQ3, RQ4", "Planned (DP)", "Learning-curve validation in Phase 3"),

    ("P02", "Working Set Ratio", "Backblaze", "100:1 healthy-to-failure ratio (~3.13M observations) with full failure preservation",
     "Imbalance-handling recommendations from Saito & Rehmsmeier (2015), Chicco & Jurman (2023)", "DP Ch. 3 Sampling Strategy",
     "RQ1, RQ3, RQ5", "Planned (DP)", "Learning-curve validation in Phase 3"),

    ("P03", "Sensitivity Branches", "Backblaze", "Build 50:1 and 200:1 working sets alongside 100:1",
     "Tests robustness of the 100:1 choice to working-set construction", "DP Ch. 3 Sampling Strategy",
     "RQ1", "Planned (DP)", "Sensitivity analysis in Phase 5"),

    ("P04", "Sensitivity Branch", "Google", "Production-priority EVICTs (type 4, priority >= 120) added as failures in sensitivity analysis",
     "Tests whether production evictions exhibit distinct predictive patterns; only 0.13% of EVICTs", "Notebook 03 Sec 7.2.2 + DP Ch. 3",
     "RQ1", "Planned (DP)", "Sensitivity analysis in Phase 5"),

    ("P05", "Sample-Size Adequacy", "Both", "Learning curve at 1/5/10/25/50/100% of working set; convergence when MCC delta < 0.005 with 95% CI straddling zero",
     "Procedure committed in DP", "DP Ch. 3 Sampling Strategy",
     "RQ1, RQ2, RQ3, RQ4, RQ5", "Planned (DP)", "Executed Phase 3 Weeks 3 and 7"),

    ("P06", "Prediction Architecture", "Google", "Three-level prediction (at-submission primary, at-scheduling secondary, early-runtime sensitivity)",
     "Motivated by V09 rapid-onset model; runtime windows infeasible at the dominant failure mode", "DP Ch. 3 RQ1 specification",
     "RQ1, RQ3", "Planned (DP)", "Built Phase 4"),

    ("P07", "Multi-Horizon Prediction", "Backblaze", "Predict at 7-day, 14-day, and 30-day horizons",
     "Exploits the gradual SMART degradation pattern (V09 contrast)", "DP Ch. 3 RQ1 Backblaze + Notebook 05 D9",
     "RQ1, RQ3", "Planned (DP)", "Built Phase 4"),

    ("P08", "Hyperparameter Tuning Strategy", "Both", "Bayesian optimization for LightGBM and XGBoost (50 trials per family); grid search for Random Forest",
     "Efficient for high-dimensional parameter spaces", "DP Ch. 3 Hyperparameter Tuning Strategy; Snoek et al. (2012)",
     "RQ1, RQ4", "Planned (DP)", "Executed Phase 4 Weeks 4 and 8"),

    ("P09", "RQ2 Conflict Types", "Google", "Three conflict types labeled: resource contention, priority inversion, scheduling violations",
     "Operational definitions from DP", "DP Ch. 3 RQ2 specification",
     "RQ2", "Planned (DP)", "Built Phase 4"),

    ("P10", "RQ4 Proactive Strategies", "Google", "Three strategies compared against reactive baseline: preemptive migration, admission control, capacity-aware bin packing",
     "Operational definitions from DP", "DP Ch. 3 RQ4 specification",
     "RQ4", "Planned (DP)", "Built Phase 4 (depends on V09/P06 RQ1 outputs)"),

    ("P11", "RQ5 Drift Taxonomy", "Backblaze", "Address four drift subtypes (sudden, gradual, incremental, recurring) each with a specific detector-mitigation pair",
     "Castano et al. (2025); Lu et al. (2018) drift taxonomy", "DP Ch. 3 RQ5 specification",
     "RQ5", "Planned (DP)", "Executed Phase 7"),

    ("P12", "RQ5 Drift Detectors", "Backblaze", "ADWIN and Page-Hinkley (sudden); KS and PSI (gradual); long-horizon performance tracking (incremental); distributional similarity (recurring)",
     "Detector selection per subtype follows Lu et al. (2018)", "DP Ch. 3 RQ5 specification",
     "RQ5", "Planned (DP)", "Built Phase 7"),

    ("P13", "RQ5 Online Learners", "Backblaze", "Adaptive Random Forest, Hoeffding Adaptive Tree, Online Gradient Boosting (River implementations)",
     "Library validated by AlShafeey & Csaki (2024); Liu & Zhao (2023)", "DP Ch. 3 RQ5 specification",
     "RQ5", "Planned (DP)", "Built Phase 7"),

    ("P14", "Random Seed Convention", "Both", "Random seed = 42 across every stochastic operation",
     "Reproducibility per Allgaier & Pryss (2024) and Raschka (2018)", "DP Ch. 3 Reproducibility Plan",
     "RQ1, RQ2, RQ3, RQ4, RQ5", "Planned (DP)", "Enforced from Phase 3 onward"),

    ("P15", "Confidence Interval Method", "Both", "95% bootstrap CIs with 1,000 stratified resamples on every reported metric",
     "Stratified bootstrap preserves class balance per resample", "DP Ch. 3 Classification Metrics; Michelucci & Venturini (2021)",
     "RQ1, RQ2, RQ3, RQ4, RQ5", "Planned (DP)", "Reported Phase 4 onward"),

    ("P16", "Family-Wise Error Control", "Both", "Holm-Bonferroni primary, Benjamini-Hochberg supporting; both reported across the five RQs",
     "Strong control across primary hypothesis tests", "DP Ch. 3 Data Analysis",
     "RQ1, RQ2, RQ3, RQ4, RQ5", "Planned (DP)", "Applied Phase 5"),

    ("P17", "Interpretability Sampling", "Both", "Sample-based SHAP (target 10,000-20,000 stratified rows) rather than population",
     "Tractable compute; global rankings stabilize at this sample size", "DP Ch. 3 Model Interpretability",
     "RQ1", "Planned (DP)", "Executed Phase 5"),

    ("P18", "Reporting Standard", "Both", "TRIPOD+AI 27-item checklist with explicit cross-references to notebooks, tables, and figures",
     "Standard for ML prediction model reporting", "DP Ch. 3 Materials and Methods; Collins et al. (2024)",
     "RQ1, RQ2, RQ3, RQ4, RQ5", "Planned (DP)", "Completed Phase 6"),

    ("P19", "Regeneration Pipeline", "Both", "Single Notebook 19 regenerates every Chapter 4 table and figure from result Parquets",
     "Hermetic regeneration prevents drift between draft prose and source artifacts", "Phase 6",
     "RQ1, RQ2, RQ3, RQ4, RQ5", "Planned (DP)", "Built Phase 6"),

    # --- OPEN: questions still under investigation, from notebook 03 Sec 7.7 (4 remaining) ---
    ("O01", "Temporal Patterns", "Google", "Do event densities and failure rates vary by time of day or day of week?",
     "Determines whether temporal features (hour, weekday) are informative and whether temporal stratification is needed", "Notebook 03 Sec 7.7 #1",
     "RQ1, RQ3", "Open", "Investigate Phase 3 (post-preprocessing EDA)"),

    ("O02", "Monitoring-Tier Evictions", "Google", "Are the 7.8M monitoring-priority evictions (6.7% of all EVICTs) health-check/canary processes or genuine infrastructure issues?",
     "Could warrant a third sensitivity branch beyond P04 Production EVICTs", "Notebook 03 Sec 7.7 #2",
     "RQ1", "Open", "Investigate Phase 3; potential sensitivity branch in Phase 5"),

    ("O03", "Collection-Level Concentration", "Google", "Are failures concentrated in certain collections? collection_type 0 dominates (99.3% of events). Is collection_type informative or essentially constant?",
     "Material to RQ2 conflict labeling and to within-collection correlation structure", "Notebook 03 Sec 7.7 #5",
     "RQ2", "Open", "Investigate Phase 4 during RQ2 conflict labeling"),

    ("O04", "Sentinel Timestamp Handling", "Google", "How many instances have time=0 or time=MAX_INT64? Exclude as censored or handle as left/right-censored data?",
     "Affects lifecycle feature computation", "Notebook 03 Sec 7.7 #7",
     "RQ1, RQ3", "Open", "Resolve in Phase 3 preprocessing"),

    # --- OPEN: questions still under investigation, from notebook 05 Sec 7.7 (all 7 remain) ---
    ("O05", "Backblaze Non-Zero Indicators", "Backblaze", "Should binary indicators (has_nonzero_smart_*) be engineered alongside raw values and rolling stats? Are upper quantiles (p90, p95, p99) more discriminative than medians or means?",
     "Zero-inflation in SMART 197, 5, 198", "Notebook 05 Sec 7.7 #1",
     "RQ1, RQ3", "Open", "Investigate Phase 3 feature engineering"),

    ("O06", "Backblaze Window Length", "Backblaze", "What window lengths (7, 14, 30, 60 days) maximize predictive power for rate-of-change features?",
     "Degradation curves suggest 90-day detectability but zero-inflation complicates", "Notebook 05 Sec 7.7 #2",
     "RQ1, RQ3", "Open", "Investigate Phase 3 + Phase 4"),

    ("O07", "Backblaze Stratification Depth", "Backblaze", "Should stratification be at the model level (dozens of sub-models), the manufacturer level (3-4 groups: Seagate, HGST/WDC, Toshiba, Hitachi), or via model-identity features in a global classifier?",
     "Heatmap shows AUC variation up to 0.27 across drive models", "Notebook 05 Sec 7.7 #3",
     "RQ1", "Open", "Investigate Phase 4"),

    ("O08", "Backblaze SMART 187/188 Recovery", "Backblaze", "Can availability be improved by restricting to post-2015 files (where schema expanded)? Does discriminative power justify separate model branches?",
     "~49.5% availability; literature-standard but data gap", "Notebook 05 Sec 7.7 #4",
     "RQ1", "Open", "Investigate Phase 3 + Phase 4"),

    ("O09", "Backblaze Survival Modeling", "Backblaze", "For RQ3 (lead time analysis), should Backblaze support survival analysis (Cox PH or accelerated failure time) using days-from-first-observation-to-failure?",
     "EDA focused on binary failure prediction; RQ3 is about lead time", "Notebook 05 Sec 7.7 #5",
     "RQ3", "Open", "Investigate Phase 4 (RQ3 Backblaze)"),

    ("O10", "Backblaze Seasonal Features", "Backblaze", "Should month/quarter indicators be included, or is the signal too weak to justify the features?",
     "46% relative seasonal variation is modest but consistent", "Notebook 05 Sec 7.7 #6",
     "RQ1", "Open", "Investigate Phase 3"),

    ("O11", "Backblaze HGST Anomaly", "Backblaze", "The HGST HMS5C4040BLE640 shows consistently weak SMART discrimination. Genuine failure-mechanism difference or statistical artifact of 0.0011% daily failure rate? Exclude or treat as edge case?",
     "Affects training-set composition and per-model evaluation", "Notebook 05 Sec 7.7 #7",
     "RQ1", "Open", "Investigate Phase 4"),
]

COLUMNS = [
    "id",
    "category",
    "dataset",
    "item",
    "evidence_or_rationale",
    "source",
    "applies_to_rq",
    "status",
    "next_step",
]

decisions_df = pl.DataFrame(DECISIONS, schema=COLUMNS, orient="row")
print(f"Total entries logged: {decisions_df.height}")
print()
print("By status:")
print(decisions_df.group_by("status").len().sort("len", descending=True))
print()
print("By category:")
print(decisions_df.group_by("category").len().sort("len", descending=True))
print()
print("By dataset:")
print(decisions_df.group_by("dataset").len().sort("len", descending=True))

# %% [markdown]
# **How to read the status column.**
# - `Validated (Phase 2)`: supported by EDA evidence that has been executed and
#   confirmed. The decision may still be revised in later phases, but the
#   supporting evidence is in hand.
# - `Planned (DP)`: design decision committed in DP Chapter 3. Not yet
#   empirically validated. These rows are the ones that need follow-up Phase
#   3, 4, 5, or 7 work to confirm or revise.
# - `Open`: question still under investigation. Sourced from notebook 03
#   Section 7.7 and notebook 05 Section 7.7. The `next_step` column says
#   which week and notebook will address it.

# %%
decisions_df.write_csv(DECISIONS_CSV)
print(f"\nWrote {decisions_df.height} entries to {DECISIONS_CSV}")
print(f"File size: {DECISIONS_CSV.stat().st_size:,} bytes")

# %% [markdown]
# ## 5. Open questions queued for later phases
#
# This section restates the Open rows of the decisions log for narrative
# convenience. The source of truth is the CSV; this prose is a navigation aid.
#
# **Open in Google EDA (from notebook 03 Section 7.7; 3 of original 7 already
# resolved by follow-up queries):**
# 1. O01: Diurnal and weekly temporal patterns in event density (Phase 3).
# 2. O02: Monitoring-tier evictions (6.7% of all EVICTs); canary processes or
#    genuine infrastructure issues? (Phase 3 or Phase 5 sensitivity)
# 3. O03: Collection-level failure concentration; is collection_type informative?
#    (Phase 4 during RQ2 conflict labeling)
# 4. O04: Sentinel timestamp handling; censored or excluded? (Phase 3)
#
# **Open in Backblaze EDA (from notebook 05 Section 7.7; all 7 still open):**
# 5. O05: Non-zero indicator features vs upper-quantile rolling stats (Phase 3)
# 6. O06: Optimal sliding-window length 7/14/30/60 days (Phase 3 + Phase 4)
# 7. O07: Model stratification depth: model vs manufacturer vs global+identity (Phase 4)
# 8. O08: SMART 187/188 recovery via post-2015 restriction (Phase 3 + Phase 4)
# 9. O09: Survival modeling for RQ3 Backblaze (Phase 4)
# 10. O10: Seasonal features given 46% relative variation (Phase 3)
# 11. O11: HGST anomaly handling (Phase 4)
#
# When any of these gets resolved, append a new Validated entry to the CSV
# referencing the resolution; do not amend the Open entry (the audit trail
# preserves the original framing of the question).
