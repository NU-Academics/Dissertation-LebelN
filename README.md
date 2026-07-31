# Dissertation-LebelN

**Data-Driven Resilience Analytics for Distributed Data Mesh Systems**

PhD dissertation by Nathan Lebel, Doctor of Philosophy in Data Science, National University, San Diego, California. Dissertation Chair: Dr. Mohamed Nabeel. Subject Matter Expert: Dr. Gebreab Zewdie. Academic Reader: Dr. Seyedmohsen Hosseini.

## Overview

This repository contains the code and analysis for a quantitative predictive experimental study investigating failure prediction, conflict resolution, lead-time forecasting, resource optimization, and online learning under concept drift for distributed data mesh systems. The study uses two production-scale operational telemetry datasets:

- **Google Cluster Traces v3 (2019)**, Cell `a`: ~9.3B rows across 5 tables, 31 days, 10,005 machines. Supports RQ1, RQ2, RQ3, RQ4.
- **Backblaze Hard Drive Data (2013-2025)**: 681.7M daily observations, 31,062 failure events, 13 years, peak fleet of 1.3M drives. Supports RQ1, RQ3, RQ5.

The methodology follows the enhanced 7-phase CRISP-DM lifecycle (Bokrantz et al., 2024) and the TRIPOD+AI reporting guidelines (Collins et al., 2024).

## Current status

The proposal defended successfully in May 2026. All five research questions are modeled and frozen and the cross-cutting evaluation is complete; the work is now in the Chapter 4 reporting phase. Each phase below maps to specific notebooks in `notebooks/` and to modules extracted into `src/`.

- **Phase 1 (Business Understanding):** Complete (covered in Dissertation Proposal Chapter 1).
- **Phase 2 (Data Understanding):** Complete. Five cached BigQuery tables, three Google follow-up queries, full Backblaze EDA, data-quality summary, and `notebooks/07_eda_findings_summary.py` consolidating the preliminary decisions log.
- **Phase 3 (Data Preparation):** Complete (both dataset blocks). **Google preprocessing complete:** `notebooks/07b_phase3_front_loaded_eda.py` and `notebooks/08_google_preprocessing.py` validated, four `src/` modules extracted (`src/preprocessing/google_traces.py`, `src/preprocessing/lifecycle.py`, `src/data/schemas.py`, `src/data/validation.py`), Polars-native unit tests passing, and decisions log rows V25-V29 appended. **Google feature engineering complete:** the three-tier feature matrix (`notebooks/10_feature_engineering_google.py`), the post-feature distribution checks and learning-curve adequacy harness (`notebooks/11_preprocessed_eda_google.py`), and six `src/features/` modules extracted with tests. **Per-attempt (scheduled-episode) redesign complete:** a prediction-point ablation revealed that the original instance-grain setup leaked a job's lifecycle history into its own label; the Google failure-prediction problem was regrained to the individual scheduled attempt, with history restricted to earlier attempts only. This removed the leakage (a built-in guard confirms first attempts carry no prior history) and produced an honest RQ1 baseline that still meets the >0.90 MCC target once a job's early-runtime behavior is observable. Decisions log rows V30-V33 appended; logic extracted to `src/features/episodes.py` with `tests/test_episodes.py`. Per-directory READMEs are in place at `notebooks/`, `src/` (and every subpackage), `tests/`, `utils/`, and `sql/`. **Collection-level sampler complete:** `src/features/sampling.py` (`build_working_set_google` plus the BigQuery `build_working_set_sql`, with `tests/test_sampling.py`) constructs the Google working set with full failure retention. The eligible instance population (scheduled instances ending in FINISH or FAIL/LOST) is 35.1M, below the projected 50-100M band, so the full eligible population is used for the instance-grain matrices rather than subsampled. The episode-grain matrix the regrained RQ1 model fits on is a separate full census (~90M scheduled attempts), which sits inside the band; its sample-size adequacy is checked by an episode-grain learning curve, with the original instance-grain curve retained as a labeled leakage baseline. **Backblaze block complete:** the SMART schema-evolution era census (`notebooks/07c_backblaze_era_census.py`; three eras materialized as `BACKBLAZE_ERAS` in `src/data/schemas.py`), the HDD-only preprocessing (`notebooks/09_backblaze_preprocessing.py` with `src/preprocessing/backblaze.py` and four Backblaze validation asserts) producing 676.4M cleaned drive-day rows across 486,253 drives plus a per-drive terminal table, the tiered SMART feature matrix and 7/14/30-day multi-horizon targets (`notebooks/10_feature_engineering_backblaze.py` with `src/features/backblaze_smart.py`), the horizon-positive working-set sampler (`src/features/sampling.py::build_working_set_backblaze`, 20:1 primary with 10:1 and 40:1 sensitivity branches), and the learning-curve adequacy check (`notebooks/11_preprocessed_eda_backblaze.py`; the curve is flat by 1% of the training data, so the working set is adequate). Decisions log rows V44-V47 record the era census, the preprocessing outcome, the working-set definition, and the adequacy finding. With both dataset blocks done, Phase 3 (Data Preparation) is complete.
- **Phase 4 (Modeling):** Complete (both dataset blocks). **RQ1 (failure prediction) complete:** the episode-grain ensemble (`notebooks/12_rq1_ensemble_google.py`) meets the >0.90 MCC target at the early-runtime prediction point (tuned random forest, MCC ~0.95), with an honest at-submission curve that is hard for a novel job and stronger once strictly-prior resubmission history is available. The reusable learners, metrics, and statistical tests were extracted to `src/models/ensemble.py`, `src/evaluation/metrics.py` (each metric with a stratified-bootstrap CI), and `src/evaluation/hypothesis.py` (CI-based threshold tests, paired Wilcoxon, effect size, family-wise correction). **RQ2 (conflict resolution) complete:** `notebooks/13_rq2_conflict.py` with `src/features/conflict_labels.py` derives three conflict types (resource contention, priority inversion, scheduling violations) with detection-time features and an entity-grouped split; on honest within-type evaluation, priority inversion meets the >0.80 MCC target (driven by strictly-prior resubmission history) while contention and scheduling violations do not. The three non-ensemble learners were extracted to `src/models/classifier.py` with `tests/test_classifier.py`. **RQ3 (lead time, Google) complete:** `notebooks/14_rq3_leadtime_google.py` reuses the RQ1 at-submission predictor; failure discrimination is strong but the realizable lead time is seconds to a few minutes, short of the 15-minute target, because Google failures are rapid-onset crashes (the target is met instead on Backblaze's gradual degradation). **RQ4 (resource optimization, Google) complete:** `notebooks/15_rq4_optimization.py` compares a reactive baseline against three prediction-informed proactive strategies (preemptive migration, admission control, capacity-aware bin packing), driven by the at-submission ensemble after post-hoc isotonic calibration (test Brier 0.107 to 0.036, closing the probability-calibration item deferred from RQ1). All three fall short of the 25% efficiency-improvement target because only about 18 to 20% of cluster allocation is failure-bound and failing jobs already consume almost no CPU; the result is a finding rather than a met target, consistent with the rapid-onset failure model. **Feature ablation (Google) complete:** `notebooks/18_feature_ablation_google.py` confirms the tier hierarchy (scheduling-plus-temporal floor MCC 0.813, plus strictly-prior history 0.915, plus early-runtime 0.950, plus windowed utilization 0.969; a null-indicators-only model scores 0.268). The Google modeling block is frozen at tag `google_block/v1`. Decisions log rows V34-V43 record the RQ1-RQ4 results, the leakage and calibration corrections, and the block freeze. **Backblaze RQ1 (failure prediction) and RQ3 (lead time) complete:** the multi-horizon ensemble (`notebooks/12_rq1_ensemble_backblaze.py`) trains the shared cost-sensitive learners and a soft-voting stack at the 7, 14, and 30-day horizons on 2021-and-earlier data, adding a leakage-safe drive-model failure-rate prior (target encoding fit on prior years only) and fitting the operating threshold and probability calibration on a natural-prevalence 2022 validation slice. Evaluation is at the true class prevalence on a purpose-built 2023-2025 test set (`notebooks/10b_backblaze_natural_test.py`, 311.2M drive-day rows at a 0.0043% failure-day rate; the 2022 validation is `notebooks/10c_backblaze_natural_val_2022.py`), because MCC and PR-AUC are prevalence-sensitive and the undersampled working set is training-only. At natural prevalence the >0.90 MCC target is not met (best soft-voting-stack MCC 0.20 at the 30-day horizon, PR-AUC 0.11, roughly 90 to 260 times the random baseline), reported as a dataset-level moderate-discriminability finding analogous to the unmet Google RQ3 and RQ4 targets. A single-model bridge analysis (`notebooks/12b_backblaze_bridge_singlemodel.py`) decomposes the gap to prior work that reports much higher numbers: on one high-volume drive model the same pipeline reaches MCC 0.72 under an in-era, balanced-test protocol, so test-set balancing and in-era splitting rather than modeling account for the difference. RQ3 (`notebooks/14_rq3_leadtime_backblaze.py`) reframes lead time as the deepest horizon sustaining MCC above 0.80: none clears it at natural prevalence, while the 15-minute target clears trivially on daily telemetry, the mirror image of the Google RQ3 result and the intended cross-dataset complementarity. Decisions log rows V48-V51 record the natural-prevalence evaluation protocol, the RQ1 result, the evaluation-design decomposition, and the RQ3 result. With RQ5 complete (Phase 7 below), modeling is finished for all five research questions.
- **Phase 5 (Evaluation):** Complete. **Cross-cutting hypothesis testing** (`notebooks/17_hypothesis_testing.py`) aggregates the frozen per-question tests into one decision table with one designated primary test per research question, applies Holm-Bonferroni and Benjamini-Hochberg family-wise corrections, and reconciles each stored decision against the locked CI rule. Because that CI-based rule returns a decision but not a p-value, a one-sided bootstrap threshold p-value (the share of resamples that fail to clear the target) was added to `src/evaluation/metrics.py` and unit-tested; all five primary outcomes are decisive, so both corrections agree with the CI rule and change no decision. Two targets are met (Google RQ1 failure prediction at early-runtime and RQ2 priority inversion) and three are not (RQ3 lead time, RQ4 optimization, RQ5 sustained MCC). **SHAP interpretability** (`notebooks/17_shap_analysis.py`) explains both RQ1 ensembles: the Google model's leading features are pre-scheduling history rather than live utilization (Tier 1 fills thirteen of the top fifteen, Tier 3 utilization none), confirming the rapid-onset premise from the model itself, while the Backblaze stack's single most influential feature is the drive-model prior, so part of its discrimination is drive-model identity rather than SMART degradation, connecting to the V50 bridge finding; a tier-alignment table records both. **Sensitivity analyses** (`notebooks/18_feature_ablation_backblaze.py`, `notebooks/18_sensitivity_prodevict_google.py`) confirm the Backblaze result is stable across the 10:1, 20:1, and 40:1 training working sets (paired-bootstrap MCC differences straddle zero, and the 20:1 re-run reproduces the frozen 0.180), and resolve the Google Production-EVICT relabeling as a bounded analysis (Production-priority evictions are 0.13% of evictions and under 1% of failures, and such instances are almost never evicted, so the label choice cannot move the headline). **Feature ablation** on Backblaze shows the Tier 2 rolling dynamics provide the main lift while the drive-model prior raises the operating point without improving ranking, matching the interpretability finding. Decisions log rows V56-V58 record the aggregation and p-value route, the Backblaze ablation and working-set sensitivity, and the Google Production-EVICT sensitivity.
- **Phase 6 (Deployment):** Tables/figures regeneration pipeline, TRIPOD+AI checklist, Chapter 4 reporting.
- **Phase 7 (Operation and Maintenance):** Complete. **RQ5 (online learning under concept drift):** `notebooks/16_rq5_online.py` compares a static batch baseline frozen at each of three starting years against an incremental Adaptive Random Forest, both scored month by month at natural prevalence on the fixed 2023-2025 window, with the frozen 2021 baseline reproducing the checkpointed RQ1 result as a gating check on the handoff. The 0.85 sustained-MCC target is not met in any of the six cells (best time-averaged MCC 0.184), which follows from the RQ1 discriminability ceiling rather than from the drift machinery and is reported as such. The substantive result is the relative comparison: every frozen baseline degrades across the evaluation window (fixed-prevalence slope -0.0017 to -0.0023 MCC per month) while the incremental learner holds flat, so continuous adaptation removes model staleness rather than raising the performance ceiling. Drift is dominated by prior shift (the failure-day rate falls roughly threefold across the window) and fleet turnover (16.93% of evaluation drive-days are unseen drive models) rather than by shifts in the SMART value distributions; recurring drift is absent within the observation window; and the 2021Q2 schema boundary proves to be a gradual availability decline in SMART 187 and 188 rather than a sudden event, so an availability-rate monitor rather than a value-distribution test is the correct instrument. Because MCC is prevalence-sensitive, every monthly figure is reported both raw and at a fixed prevalence, and only the fixed-prevalence series supports a drift claim. Reusable logic was extracted to `src/models/online.py` (River wrappers with checkpoint-and-resume and structurally enforced prequential ordering) and `src/evaluation/drift_detectors.py` (ADWIN, Page-Hinkley, KS, PSI behind one interface), with the custom drift metrics added to `src/evaluation/metrics.py`. Decisions log rows V52-V55 record the machinery and its data-derived detector defaults, the online-boosting substitution, the RQ5 result, and the drift characterization.

## Following along (for committee members)

Commit messages follow a CRISP-DM-aligned prefix convention:

| Prefix | CRISP-DM phase | Example |
|--------|----------------|---------|
| `infra:` | Phase 1 / setup | `infra(deps): add Phase 3-4 ML dependencies` |
| `eda:` | Phase 2 | `eda(google): three-query failure synthesis` |
| `prep:` | Phase 3 (preprocessing) | `prep(google): lifecycle reconstruction validated against EDA` |
| `feat:` | Phase 3 (features + sampling) | `feat(backblaze): Tier 2 rolling SMART statistics` |
| `model:` | Phase 4 | `model(rq1): three-level architecture results` |
| `eval:` | Phase 5 | `eval(hypothesis): per-RQ threshold tests + Holm-adjusted p-values` |
| `operate:` | Phase 7 | `operate(rq5): year-over-year sustained MCC vs. 0.85` |
| `deploy:` | Phase 6 | `deploy(report): TRIPOD+AI 27-item checklist` |
| `test:` | cross-cutting | `test(prep): Google preprocessing tests + Parquet snapshot manifest` |
| `docs:` | cross-cutting | `docs(scaffold): Chapter 4 scaffolding complete` |

One commit per work day. Scope tags in parentheses identify the affected component.

## Repository structure

```
Dissertation-LebelN/
├── README.md                  This file
├── requirements.txt           Python dependencies (core stack; River and SHAP installed in-notebook)
├── .gitignore                 Excludes data files, .ipynb, scratch/, secrets
│
├── sql/                       BigQuery cache + exploration queries (Phase 2)
│   ├── cache_*.sql            Five population-caching queries (run once each)
│   └── exploration/           Follow-up queries supporting the rapid-onset failure model
│
├── notebooks/                 Jupytext .py files (primary work product)
│   ├── 00_setup_environment.py
│   ├── 01_bigquery_caching.py
│   ├── 02_initial_profiling.py
│   ├── 03_google_eda.py
│   ├── 04_backblaze_ingest.py
│   ├── 05_backblaze_eda.py
│   ├── 06_data_quality.py
│   ├── 07_eda_findings_summary.py    Phase 2 to Phase 3 handoff + decisions log
│   ├── 08-11                         Phase 3 (Data Preparation); 10b/10c natural-prevalence eval sets
│   ├── 12-16                         Phase 4 and 7 (12 RQ1 Google + Backblaze, 12b Backblaze bridge,
│   │                                 13 RQ2, 14 RQ3 Google + Backblaze, 15 RQ4, 16 RQ5 drift)
│   ├── 17                            Phase 5 (17_hypothesis_testing aggregation + 17_shap_analysis)
│   └── 18-19                         Phase 5 and 6 (18 feature ablation + sensitivity branches;
│                                     19 tables/figures reporting planned)
│
├── utils/                     Minimal shared utilities (BQ client, Drive setup, checkpoints)
├── src/                       Extracted modules (preprocessing, features, models, evaluation)
├── configs/                   Hyperparameter configs (populated during Phase 4)
├── tests/                     pytest suite (test_smoke + per-module tests)
└── outputs/                   Generated tables and figures (gitignored except .gitkeep
                               and outputs/tables/eda_decisions.csv audit trail)
```

The `outputs/tables/eda_decisions.csv` audit trail captures every analytical decision with one of several statuses: **Validated (Phase 2)** for EDA-confirmed decisions, **Validated (Phase 3 ...)** for decisions confirmed during data preparation (including the per-attempt regrain, rows V30-V33), **Validated (Phase 4 Modeling)** for results and corrections confirmed during modeling (rows V34-V43 cover the RQ1 tuned-ensemble result and curve reconciliation, the at-submission feature-source leak fix, the RQ2 conflict-labeling design, history features, entity-grouped split and result, the RQ3 Google lead-time result, the RQ4 resource-optimization result with at-submission calibration, and the Google block freeze; rows V44-V47 cover the Backblaze schema-evolution era census, the preprocessing outcome, the horizon-positive working-set definition, and the learning-curve adequacy finding; rows V48-V51 cover the natural-prevalence evaluation protocol, the Backblaze RQ1 multi-horizon result, the single-model evaluation-design decomposition, and the Backblaze RQ3 lead-time result), **Validated (Phase 7 Operation and Maintenance)** for the drift work (rows V52-V55 cover the online-learning and drift-detection machinery with its data-derived detector defaults, the online-boosting substitution, the RQ5 adaptive-versus-static result, and the drift characterization), **Validated (Phase 5 Evaluation)** for the cross-cutting evaluation (rows V56-V58 cover the cross-RQ hypothesis-test aggregation and its bootstrap p-value route, the Backblaze feature ablation and working-set sensitivity, and the Google Production-EVICT label sensitivity), **Planned (DP)** for design commitments awaiting empirical validation, and **Open** or **Resolved** for questions under or past investigation. Each row traces back to a specific notebook section, follow-up query, or Dissertation Proposal Chapter 3 reference.

## Methodology and conventions

- **Methodology:** Enhanced 7-phase CRISP-DM (Bokrantz et al., 2024). Iterative; later phases can loop back to earlier ones when modeling exposes a feature gap.
- **Reporting:** TRIPOD+AI (Collins et al., 2024). Checklist filled in Phase 6.
- **Primary metric:** Matthews correlation coefficient (MCC). 95% bootstrap CIs (1,000 stratified resamples) on every reported value. F1 and PR-AUC co-reported. ROC-AUC reported as complementary baseline only.
- **Splits:** Group-aware holdout so no entity straddles train and test. The episode-grain failure models bucket instances by a hash of the instance key; the RQ2 conflict models split by the conflict's entity (machine, instance, or collection). Walk-forward (expanding-window) cross-validation runs inside the training period for temporal robustness. Random seed = 42 everywhere.
- **Imbalance handling:** Cost-sensitive learning (class weights = inverse prior) plus SMOTE applied only inside training folds.
- **Sampling:** Working-set construction with full failure preservation. Learning-curve adequacy evidence per dataset.
- **Code organization:** Notebooks are the primary work product; modules in `src/` are extracted only after notebooks validate the logic.
- **Data scale:** Polars (not Pandas) is the primary DataFrame library for memory efficiency in Colab.

## Environment

- **Local development:** IntelliJ IDEA Ultimate with Ruff linting and pytest. Python 3.12.
- **Execution:** Google Colab with T4 GPU runtime, ~12.7 GB RAM. Notebooks are stored as Jupytext `.py` files; `.ipynb` is generated on demand in Colab and is gitignored.
- **Data storage:** BigQuery (Google Cluster Traces cached tables), Google Cloud Storage (Backblaze Parquet), Google Drive (checkpoints, intermediate outputs).
- **Credentials:** Loaded from Colab Secrets (`GCP_PROJECT_ID`, `GITHUB_PAT`). Never hardcoded.

## Quick start (for new contributors)

```bash
# Clone (replace with the actual repo URL)
git clone https://github.com/YOUR-USERNAME/Dissertation-LebelN.git
cd Dissertation-LebelN

# Install dependencies
pip install -r requirements.txt

# Run the test suite (use the module form, not a bare pytest; see tests/README.md)
python -m pytest

# Open notebooks/00_setup_environment.py in Colab to verify connectivity to
# BigQuery and Google Drive. Subsequent notebooks build on this setup.
```

For a deeper orientation, run `notebooks/07_eda_findings_summary.py` to see the Phase 2 findings consolidation and the decisions log. The EDA narratives in notebooks 03 (Section 7.8) and 05 (Section 7.8) are the source of truth for validated preliminary decisions.

## References

The full reference library is managed in Zotero. Key methodological sources cited across the repository include Bokrantz et al. (2024) for the enhanced CRISP-DM lifecycle, Chicco & Jurman (2023) for MCC as primary classification metric, Saito & Rehmsmeier (2015) for PR-AUC under imbalance, Bergmeir & Benitez (2012) for time-series cross-validation, Castano et al. (2025) and Lu et al. (2018) for concept drift taxonomy, and Collins et al. (2024) for TRIPOD+AI reporting.

## License

Code is published under the MIT License (see LICENSE file when added). Datasets used in this study retain their own licenses: Google Cluster Traces v3 under CC-BY 4.0 (Google, 2019); Backblaze Hard Drive Data under Backblaze's custom open-source license (Backblaze, 2024).
