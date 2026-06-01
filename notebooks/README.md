# Notebooks

Jupytext percent-format `.py` files are the primary work product. Each notebook captures the analytical narrative for one CRISP-DM step and produces the artifacts that downstream notebooks and `src/` modules depend on. `.ipynb` files are generated on demand in Colab and are gitignored.

The decisions audit trail at `outputs/tables/eda_decisions.csv` is the canonical cross-reference for every validated analytical choice (rows `V01` through `V29` at the time of writing). When a notebook section references a decision by ID, the CSV is where to read the evidence and the next step.

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

Four front-loaded EDA checks executed at the start of Week 2 to resolve open items that shape preprocessing:

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

### Planned

`09_backblaze_preprocessing.py`, `10_feature_engineering_google.py`, `11_preprocessed_eda_google.py` are the next preprocessing-and-feature notebooks.

## Phase 4 onward (planned)

Notebooks 12 through 19 cover modeling, online-learning drift simulation, hypothesis testing, ablation, and the Chapter 4 tables/figures regeneration pipeline. Each will get its own narrative-first treatment, with logic moving into `src/` only after the notebook validates it.

## Conventions

- **Jupytext only.** Edit the `.py` source; `.ipynb` is generated on demand in Colab.
- **Colab as the execution environment.** Local IntelliJ runs are for editing, linting, and unit tests. The full pipelines run on Colab with the T4 GPU runtime.
- **One commit per work day.** Commit messages prefixed by CRISP-DM phase per the top-level `README.md` convention table.
- **Polars, not Pandas.** Polars LazyFrames are the default. Pandas is reserved for ML-library interop where unavoidable.
- **Random seed = 42** across every stochastic operation in every notebook. Locked in `tests/test_smoke.py::test_random_seed_constant_is_42`.
