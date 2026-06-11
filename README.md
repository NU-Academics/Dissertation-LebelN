# Dissertation-LebelN

**Data-Driven Resilience Analytics for Distributed Data Mesh Systems**

PhD dissertation by Nathan Lebel, Doctor of Philosophy in Data Science, National University, San Diego, California. Dissertation Chair: Dr. Mohamed Nabeel. Subject Matter Expert: Dr. Gebreab Zewdie. Academic Reader: Dr. Seyedmohsen Hosseini.

## Overview

This repository contains the code and analysis for a quantitative predictive experimental study investigating failure prediction, conflict resolution, lead-time forecasting, resource optimization, and online learning under concept drift for distributed data mesh systems. The study uses two production-scale operational telemetry datasets:

- **Google Cluster Traces v3 (2019)**, Cell `a`: ~9.3B rows across 5 tables, 31 days, 10,005 machines. Supports RQ1, RQ2, RQ3, RQ4.
- **Backblaze Hard Drive Data (2013-2025)**: 681.7M daily observations, 31,062 failure events, 13 years, peak fleet of 1.3M drives. Supports RQ1, RQ3, RQ5.

The methodology follows the enhanced 7-phase CRISP-DM lifecycle (Bokrantz et al., 2024) and the TRIPOD+AI reporting guidelines (Collins et al., 2024).

## Current status

The proposal defended successfully in May 2026. The analysis is now in the Chapter 4 execution window. Each phase below maps to specific notebooks in `notebooks/` and to modules extracted into `src/`.

- **Phase 1 (Business Understanding):** Complete (covered in Dissertation Proposal Chapter 1).
- **Phase 2 (Data Understanding):** Complete. Five cached BigQuery tables, three Google follow-up queries, full Backblaze EDA, data-quality summary, and `notebooks/07_eda_findings_summary.py` consolidating the preliminary decisions log.
- **Phase 3 (Data Preparation):** Active; the Google block is substantially complete. **Google preprocessing complete:** `notebooks/07b_phase3_front_loaded_eda.py` and `notebooks/08_google_preprocessing.py` validated, four `src/` modules extracted (`src/preprocessing/google_traces.py`, `src/preprocessing/lifecycle.py`, `src/data/schemas.py`, `src/data/validation.py`), Polars-native unit tests passing, and decisions log rows V25-V29 appended. **Google feature engineering complete:** the three-tier feature matrix (`notebooks/10_feature_engineering_google.py`), the post-feature distribution checks and learning-curve adequacy harness (`notebooks/11_preprocessed_eda_google.py`), and six `src/features/` modules extracted with tests. **Per-attempt (scheduled-episode) redesign complete:** a prediction-point ablation revealed that the original instance-grain setup leaked a job's lifecycle history into its own label; the Google failure-prediction problem was regrained to the individual scheduled attempt, with history restricted to earlier attempts only. This removed the leakage (a built-in guard confirms first attempts carry no prior history) and produced an honest RQ1 baseline that still meets the >0.90 MCC target once a job's early-runtime behavior is observable. Decisions log rows V30-V33 appended; logic extracted to `src/features/episodes.py` with `tests/test_episodes.py`. Per-directory READMEs are in place at `notebooks/`, `src/` (and every subpackage), `tests/`, `utils/`, and `sql/`. **Collection-level sampler complete:** `src/features/sampling.py` (`build_working_set_google` plus the BigQuery `build_working_set_sql`, with `tests/test_sampling.py`) constructs the Google working set with full failure retention. The eligible instance population (scheduled instances ending in FINISH or FAIL/LOST) is 35.1M, below the projected 50-100M band, so the full eligible population is used for the instance-grain matrices rather than subsampled. The episode-grain matrix the regrained RQ1 model fits on is a separate full census (~90M scheduled attempts), which sits inside the band; its sample-size adequacy is checked by an episode-grain learning curve, with the original instance-grain curve retained as a labeled leakage baseline. **Remaining Phase 3 work:** the Backblaze block.
- **Phase 4 (Modeling):** Per-RQ ensemble and classifier training with temporal cross-validation and Bayesian hyperparameter tuning. The Google RQ1 ensemble notebook (`notebooks/12_rq1_ensemble_google.py`) is scaffolded against the episode-grain matrix.
- **Phase 5 (Evaluation):** Cross-cutting hypothesis testing with family-wise error control, SHAP interpretability, sensitivity analyses, feature ablation.
- **Phase 6 (Deployment):** Tables/figures regeneration pipeline, TRIPOD+AI checklist, Chapter 4 reporting.
- **Phase 7 (Operation and Maintenance):** RQ5 drift simulation with River-based online learning and four drift-subtype detector/mitigation pairs.

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
├── requirements.txt           Python dependencies (Phase 2-4 active; River/SHAP deferred)
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
│   ├── 08-11                         Phase 3 (Data Preparation)
│   ├── 12-16                         Phase 4 and 7 (Modeling and Operation)
│   └── 17-19                         Phase 5 and 6 (Evaluation and Deployment)
│
├── utils/                     Minimal shared utilities (BQ client, Drive setup, checkpoints)
├── src/                       Extracted modules (preprocessing, features, models, evaluation)
├── configs/                   Hyperparameter configs (populated during Phase 4)
├── tests/                     pytest suite (test_smoke + per-module tests)
└── outputs/                   Generated tables and figures (gitignored except .gitkeep
                               and outputs/tables/eda_decisions.csv audit trail)
```

The `outputs/tables/eda_decisions.csv` audit trail captures every analytical decision with one of four statuses: **Validated (Phase 2)** for EDA-confirmed decisions, **Validated (Phase 3 ...)** for decisions confirmed during data preparation (including the per-attempt regrain, rows V30-V33), **Planned (DP)** for design commitments awaiting empirical validation, and **Open** for questions still under investigation. Each row traces back to a specific notebook section, follow-up query, or Dissertation Proposal Chapter 3 reference.

## Methodology and conventions

- **Methodology:** Enhanced 7-phase CRISP-DM (Bokrantz et al., 2024). Iterative; later phases can loop back to earlier ones when modeling exposes a feature gap.
- **Reporting:** TRIPOD+AI (Collins et al., 2024). Checklist filled in Phase 6.
- **Primary metric:** Matthews correlation coefficient (MCC). 95% bootstrap CIs (1,000 stratified resamples) on every reported value. F1 and PR-AUC co-reported. ROC-AUC reported as complementary baseline only.
- **Splits:** Temporal only. Walk-forward (expanding-window) cross-validation inside training periods. Random seed = 42 everywhere.
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

# Install Phase 2-4 dependencies
pip install -r requirements.txt

# Run the test smoke suite (verifies src/ skeleton is intact)
pytest tests/ -v

# Open notebooks/00_setup_environment.py in Colab to verify connectivity to
# BigQuery and Google Drive. Subsequent notebooks build on this setup.
```

For a deeper orientation, run `notebooks/07_eda_findings_summary.py` to see the Phase 2 findings consolidation and the decisions log. The EDA narratives in notebooks 03 (Section 7.8) and 05 (Section 7.8) are the source of truth for validated preliminary decisions.

## References

The full reference library is managed in Zotero. Key methodological sources cited across the repository include Bokrantz et al. (2024) for the enhanced CRISP-DM lifecycle, Chicco & Jurman (2023) for MCC as primary classification metric, Saito & Rehmsmeier (2015) for PR-AUC under imbalance, Bergmeir & Benitez (2012) for time-series cross-validation, Castano et al. (2025) and Lu et al. (2018) for concept drift taxonomy, and Collins et al. (2024) for TRIPOD+AI reporting.

## License

Code is published under the MIT License (see LICENSE file when added). Datasets used in this study retain their own licenses: Google Cluster Traces v3 under CC-BY 4.0 (Google, 2019); Backblaze Hard Drive Data under Backblaze's custom open-source license (Backblaze, 2024).
