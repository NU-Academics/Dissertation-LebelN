# Notebooks

Jupytext percent-format `.py` files are the primary work product. Each notebook captures the analytical narrative for one CRISP-DM step and produces the artifacts that downstream notebooks and `src/` modules depend on. `.ipynb` files are generated on demand in Colab and are gitignored.

The decisions audit trail at `outputs/tables/eda_decisions.csv` is the canonical cross-reference for every validated analytical choice (rows `V01` through `V41` at the time of writing, plus the `P` planned and `O` open / resolved rows). When a notebook section references a decision by ID, the CSV is where to read the evidence and the next step.

## Phase 2: Data Understanding (complete)

### `00_setup_environment.py`

Bridge cell that verifies the Colab session, mounts Google Drive, pulls Colab Secrets (`GCP_PROJECT_ID`, `GITHUB_PAT`), clones the repo, and confirms BigQuery connectivity. Run first in every Colab session.

### `01_bigquery_caching.py`

Caches the five Google Cluster Traces v3 tables from the public BigQuery dataset (`google.com:google-cluster-data`, cell `a`) into `{project_id}.dissertation_lebel.*_full`. Caching is a one-time cost. Every subsequent notebook reads from the cached tables to keep BigQuery spend predictable.

### `02_initial_profiling.py`

Fast reconnaissance sweep over the five cached tables. Outputs `schema_summary.csv`, `null_profile.csv`, `numeric_stats.csv`, and `categorical_cardinality.csv`. Drives the column-type decisions encoded in `src/data/schemas.py`.

### `03_google_eda.py`

Deep exploratory analysis of the Google trace. Section 7.8 lists the preliminary decisions log that seeds `eda_decisions.csv` (`V01` through `V08`). Three follow-up queries (`sql/exploration/`) close Phase 2 by resolving Open Questions #3, #4, and #6, producing the rapid-onset failure model (`V09`), resubmission dominance (`V10`), MNAR hardware-counter encoding (`V11`), and the utilization inversion (`V12`).

### `04_backblaze_ingest.py`

Downloads the public Backblaze quarterly CSV bundles, converts to Parquet, and stages them in GCS at `gs://{project_id}-dissertation-data/backblaze_parquet/`. SSD rows are excluded to preserve the 13-year temporal consistency.

### `05_backblaze_eda.py`

Deep EDA on the Backblaze corpus. Produces the SMART feature ranking (`V14`, `V15`), schema-evolution observations for SMART 187 and 188 (`V16`), the extreme-imbalance characterization (`V17`), the model-stratification finding (`V18`), the zero-inflation pattern (`V19`), the temporal split strategy (`V20`), the online-learning justification (`V21`), the multi-horizon prediction framing (`V22`), and the cross-dataset metric strategy (`V23`).

### `06_data_quality.py`

Consolidates the per-column quality assessments from both datasets into the Chapter 3 Data Quality Summary Table. Emits `data_quality_summary.csv`, `google_column_profile.csv`, `backblaze_column_profile.csv`, and `class_imbalance.csv`.

### `07_eda_findings_summary.py`

Markdown-forward consolidation of every Phase 2 finding. Section 4 materializes `outputs/tables/eda_decisions.csv` with three status categories: **Validated (Phase 2)** for EDA-confirmed decisions, **Planned (DP)** for design commitments awaiting empirical validation, and **Open** for questions still under investigation. Cross-dataset complementarity (`V24`) is the closing finding.

## Phase 3: Data Preparation (active focus on Google)

### `07b_phase3_front_loaded_eda.py`

Four front-loaded EDA checks executed early in Phase 3 (Data Preparation) to resolve open items that shape preprocessing:

- **F1** (sentinel timestamp inventory) closes `O04` and produces `V25`. Sentinel-bearing rows are dropped outright in preprocessing.
- **F2** (diurnal and weekly density patterns) closes `O01` and produces `V26`. Hour-of-day and day-of-week bucketing operates in PDT wall-clock; FAIL_LOST rate varies up to ~8x across hour buckets, motivating stratified chronological splits and temporal features.
- **F3** (monitoring-tier eviction triage) closes `O02` and produces `V27`. The F3.2 repeats distribution shows monitoring-priority EVICTs are canary / health-check preemptions, not failures; they are excluded from every failure label.
- **F4** (CPI/MAPI within-instance variance) refines `V11` and produces `V28`. 39.84% of instances flip the indicator within their lifetime, so preprocessing aggregates `has_hardware_counters` to a per-instance majority vote.

Emits `outputs/tables/sentinel_inventory.csv`, `temporal_density.csv`, `monitoring_evict_profile.csv`, `cpi_mapi_within_instance_variance.csv`, and `outputs/figures/diurnal_density.png`.

### `08_google_preprocessing.py`

The Phase 3 Google preprocessing notebook. Seven analytical sections plus the export:

1. **Sentinel timestamp filtering** (`V25`): drops `time = 0` and `time = 2^63 - 1` into `instance_events_clean`.
2. **Structural null handling** (`V03`, `V04`, `V05`): drops the ~48K rows with missing `cpu_request` or `memory_request`; documents the column-level drops on `instance_usage` and `collection_events`.
3. **Failure label construction** (`V01`, `V08`, `V27`, `P04`): primary `failure_label` and the optional `failure_label_sensitivity_prod_evict` branch, with explicit V27 exclusion of monitoring-priority EVICTs.
4. **MNAR indicator encoding** (`V11`, `V28`): per-observation `has_cpi_value` and `has_mapi_value`, plus the per-instance `has_hardware_counters_majority` table. Verifies the V11 record-level null rates against the EDA targets (87.2% FINISH, 26.8% FAIL_LOST) using the V11-exact triple-key join.
5. **Lifecycle reconstruction** (`V09`, `V10`, `V29`): the per-instance summary materialized as `instance_lifecycle_summary`. Reproduces the V10 99.04% FAIL_LOST resubmission rate using the 3-day window plus Part B schedule filter, and captures the full-trace contrast (~76%) as an informational row. The V29 entry in the decisions log documents both reproduction modes.
6. **Post-preprocessing assertion suite**: verifies every section's invariants against the EDA-confirmed statistics. A failed assertion blocks the export.
7. **Export to GCS Parquet plus Drive manifest**: writes the preprocessed events table to `gs://{project_id}-dissertation-data/google_preprocessed/instance_events_preprocessed/` as partitioned Parquet (Drive cannot host a 1.7B-row Parquet) and records the export in `manifest.json`.

### `10_feature_engineering_google.py`

The Google feature-engineering notebook. Section 1.1 locks the working set via the collection-level sampler (`src/features/sampling.py::build_working_set_sql`): it reads the per-instance `instance_lifecycle_summary` (never the 1.7B-row events table), retains every failure-containing collection in full, and stratified-samples successful collections. The eligible instance population is 35.1M, below the projected 50-100M band, so Section 1.1a asserts full-population retention rather than subsampling and writes the `SamplingManifest`. Sections 0-10 build the instance-grain three-tier matrix (`instance_features`) per the `V13` priority hierarchy: Tier 1 historical/scheduling/temporal, Tier 2 early-runtime slopes via a two-stage BigQuery filter, Tier 3 windowed utilization (ablation only). Sections 11-12 add the **per-attempt (scheduled-episode) redesign** that removes the instance-grain lifecycle-history leakage (`V30`-`V33`): Section 11 segments episodes and builds `episode_lifecycle_features_base` with strictly-prior history (leakage guard PASS; submission+history MCC 0.949 → 0.681), and Section 11.3 verifies the episode census (~89.68M rows) lands in the 50-100M `P01` band; Section 12 adds episode Tier 2/3 by interval assignment and assembles the full `episode_features` matrix. The episode matrix is built over all instances from `instance_events_labeled`, independent of the working set. Durable outputs are the BigQuery tables and GCS exports (no multi-GB Drive sink for the episode matrix). Logic extracted to `src/features/` (the five tier modules, `episodes.py`, and `sampling.py`).

### `11_preprocessed_eda_google.py`

Validates the engineered matrix and decides working-set adequacy (`P05`). Sections 1-2 re-verify the distribution invariants and the `V12` Tier 3 inversion guard; Section 3 runs the LightGBM learning curve, the leakage diagnostics, and the prediction-point ablation (which surfaced the instance-grain leakage). Section 3.8 is the **episode-grain re-check**: a BigQuery-side capped, instance-keyed group-split extract from `episode_features` runs the honest prediction-point curve (submission 0.666 → early-runtime 0.909, meeting the RQ1 >0.90 target). Section 3.9 is the **episode-grain learning curve** at the early-runtime prediction point: it is the RQ1 sample-size-adequacy evidence (`P05`), and Sections 4 (P05 decision) and 5 (figure) read from it. The instance-grain curve from Sections 3.1-3.7 is retained as a labeled leakage baseline (figure in Section 3.5b), because its ~0.97 MCC is leakage-inflated.

### `11b_attempt_structure_google.py`

Characterizes the attempt/episode structure of the event stream before the redesign: event-sequence dumps, episode segmentation reconciliation (99.1% of failures post-schedule, 93.4% well-formed, ~1.2% open, ~5.4% multi-terminal), and the FINISH-doubling analysis (a recurring tail of genuine separate scheduled runs, not duplicate records). Source for the `V30`-`V32` segmentation rules.

### Planned

`09_backblaze_preprocessing.py` and `10_feature_engineering_backblaze.py` are the next preprocessing-and-feature notebooks.

## Phase 4: Modeling (RQ1 to RQ3 Google complete)

### `12_rq1_ensemble_google.py`

RQ1 ensemble failure prediction at the **episode grain** (the leakage-free grain from the per-attempt redesign). Loads the durable `episode_features` matrix, attaches `schedule_time` and a 1-based `sched_day`, applies the per-instance negative cap (`src/features/episodes.py::cap_negative_episodes`), and runs per-prediction-point (`at_submission`, `at_scheduling`, `early_runtime`) training across a model set (logistic regression, decision tree, random forest, balanced random forest, XGBoost, LightGBM, gradient boosting, plus a top-3 soft-voting stack) with walk-forward CV, a validation-tuned threshold, and percentile bootstrap CIs. Two leakage corrections were resolved during the build (the terminal-vs-submit feature-source leak `V35`, and a split realignment to the instance-keyed group split); the tuned model zoo then met the >0.90 target at early-runtime (random forest MCC ~0.95, `V36`), with the formal one-sided CI test and the model-agnostic tuning ceiling recorded in `V37`. Sections 8 to 10 add Optuna / walk-forward-grid tuning to `configs/models/`, checkpointing of the best ensemble per prediction point (with a metadata sidecar), and the artifact and hypothesis-test outputs; the wrappers, metrics, and tests are extracted to `src/models/ensemble.py`, `src/evaluation/metrics.py`, and `src/evaluation/hypothesis.py`. Writes `outputs/tables/rq1_google.csv`.

### `13_rq2_conflict.py`

RQ2 conflict resolution. Builds labeled conflict episodes via `src/features/conflict_labels.py` over three conflict types (resource contention, priority inversion, scheduling violations) from two working-set scopes (machine-scoped for contention and inversion, preserving co-residency; collection-scoped for violations), with detection-time features only and a failure-based `resolution_outcome`. Trains a model zoo (logistic regression, decision tree, linear SVM, the random forest reused from `ensemble.py`, a one-hidden-layer Keras NN, and a most-frequent baseline) per conflict type plus pooled, splitting by the conflict entity (`group_key`) so no entity straddles train and test, with imputation and SMOTE training-only and a validation-tuned threshold. Scores with the `metrics.py` CI helpers and tests the >0.80 MCC target with `hypothesis.py`. On honest within-type evaluation, priority inversion meets the target (driven by strictly-prior resubmission history) while resource contention and scheduling violations do not; the pooled number is a between-type-prevalence artifact. Writes `outputs/tables/rq2_results.csv` and `outputs/tables/rq2_hypothesis_test.csv` (`V38` to `V40`). The Decision Tree / SVM / Keras wrappers are extracted to `src/models/classifier.py`.

### `14_rq3_leadtime_google.py`

RQ3 lead time (Google). Reuses the RQ1 at-submission checkpoint (`rq1_google_best_atsubmission`) on the RQ1 instance-grouped test split, scoring at its tuned threshold for ranking rather than as a calibrated probability (calibration is deferred to RQ4). Computes per-attempt lead time (submission to terminal) and aggregates member scores to the collection level (max / mean / top-3 mean) with an MCC-optimal threshold search, testing both against the 15-minute target. Failure discrimination is strong (collection MCC up to 0.87) but realizable lead time is only seconds to a few minutes, short of 15 minutes, because Google failures are rapid-onset crashes; the target is met instead on Backblaze's gradual degradation (`V41`). Writes `outputs/tables/rq3_google.csv` and `outputs/tables/rq3_google_hypothesis_test.csv`.

### Remaining Phase 4 onward (planned)

Notebooks 15 through 19 cover RQ4 resource optimization, the RQ5 online-learning drift simulation, hypothesis testing, the feature ablation, and the Chapter 4 tables/figures regeneration pipeline. Each will get its own narrative-first treatment, with logic moving into `src/` only after the notebook validates it.

## Conventions

- **Jupytext only.** Edit the `.py` source; `.ipynb` is generated on demand in Colab.
- **Colab as the execution environment.** Local IntelliJ runs are for editing, linting, and unit tests. The full pipelines run on Colab with the T4 GPU runtime.
- **One commit per work day.** Commit messages prefixed by CRISP-DM phase per the top-level `README.md` convention table.
- **Polars, not Pandas.** Polars LazyFrames are the default. Pandas is reserved for ML-library interop where unavoidable.
- **Random seed = 42** across every stochastic operation in every notebook. Locked in `tests/test_smoke.py::test_random_seed_constant_is_42`.
