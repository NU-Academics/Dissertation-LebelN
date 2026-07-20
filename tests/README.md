# `tests/` - Test Suite

Pytest test suite for the extracted `src/` modules. Synthetic Polars LazyFrame fixtures hand-crafted to hit each branch. No BigQuery, no Drive, no GCS. Fast: the full suite runs in a few seconds on a laptop. The TensorFlow-dependent neural-network tests skip automatically when TensorFlow is absent (they run in Colab), as do the XGBoost / LightGBM wrapper tests when those libraries are not installed locally.

## Running

```bash
# From the repo root
python -m pytest -v
```

Use `python -m pytest`, not a bare `pytest`. The module form binds the run to the active interpreter; a bare `pytest` can resolve to a different environment's console script, which silently skips every test whose optional dependency is installed only in the active one. `pytest.ini` scopes collection to `tests/` because some Jupytext notebook sources match pytest's default `test_*.py` / `*_test.py` patterns and would otherwise be imported and fail on their Colab-only imports.

The suite is 146 tests across twelve files (locally 143 pass and 3 skip where TensorFlow is absent; all run in Colab):

```
tests/test_smoke.py              2 tests
tests/test_preprocessing.py     23 tests
tests/test_lifecycle.py          7 tests
tests/test_episodes.py           6 tests
tests/test_sampling.py           8 tests
tests/test_backblaze_smart.py    5 tests
tests/test_metrics.py           19 tests
tests/test_hypothesis.py        17 tests
tests/test_ensemble.py          15 tests
tests/test_classifier.py        11 tests
tests/test_drift_detectors.py   19 tests
tests/test_online.py            23 tests
```

To run a single file: `pytest tests/test_preprocessing.py -v`. To run a single test: `pytest tests/test_preprocessing.py::test_failure_label_primary -v`.

## Test inventory

### `test_smoke.py`

Repo-skeleton sanity checks that must remain green throughout Chapter 4 execution.

- `test_src_subpackages_importable` confirms every `src/` subpackage imports cleanly.
- `test_random_seed_constant_is_42` locks the reproducibility contract. If this assertion needs to change, the entire reproducibility plan in Chapter 3 is being modified and must be updated first.

### `test_preprocessing.py` (Phase 3)

Unit tests for the four pure functions in `src/preprocessing/google_traces.py` and the six in `src/preprocessing/backblaze.py`, plus the four Backblaze validation asserts. Each fixture is hand-crafted to hit the branches the function must handle.

- `test_filter_sentinel_timestamps_drops_zero_and_max` and `..._accepts_alternate_time_column` cover `V25`.
- `test_failure_label_primary` covers `V01` and `V08` (KILL exclusion).
- `test_failure_label_sensitivity_prod_evict` covers `P04` (Production-priority EVICTs labeled in the sensitivity column).
- `test_failure_label_excludes_monitoring_evict` is the regression guard for `V27`. Monitoring-priority EVICTs must stay NULL under every label, while a production-boundary EVICT confirms the sensitivity branch still activates.
- `test_failure_label_invalid_sensitivity_branch_raises` confirms the ValueError contract on bad inputs.
- `test_mnar_indicators_per_observation` builds a 1000-row fixture engineered to match V11's 87.2% / 26.8% null targets, verifies the per-row indicator flips with the underlying null pattern, and confirms the aggregated null rates land within +/- 1 percentage point of the V11 targets.
- `test_mnar_indicators_per_instance_majority` covers `V28` with five hand-crafted instances (all-present, all-absent, 60% majority, 40% minority, exact 50% boundary).
- `test_mnar_indicators_opt_in_skips_majority_column` confirms the opt-in flag.

The Backblaze section uses a three-era synthetic fixture (early, standard, recent drives plus an SSD to exclude):

- `test_filter_hdds_only_removes_ssd_models` and `..._noop_without_models` cover SSD exclusion.
- `test_assign_era_bins_by_date` and `..._marks_out_of_range_unknown` cover era assignment from `BACKBLAZE_ERAS` (`V44`).
- `test_reconcile_smart_schema_adds_missing_columns` confirms absent SMART columns are added all-null.
- `test_encode_smart_availability_indicators_track_nullness` covers the `has_smart_{id}` indicator (`V16`).
- `test_canonicalize_drive_model_derives_manufacturer` and `..._applies_aliases` cover `V18`.
- `test_mark_censoring_flags_terminal_observations` covers the observed-failure versus censored terminal marking.
- `test_assert_failure_event_count_pass_and_fail`, `test_assert_one_row_per_drive_day_pass_and_fail`, `test_assert_era_assignment_complete_pass_and_fail`, and `test_assert_fleet_expansion_pass_and_fail` cover the Backblaze validation asserts, and `test_backblaze_eras_constant_is_well_formed` checks the 187/188 placement (`V44`).

### `test_lifecycle.py` (Phase 3)

Unit tests for the lifecycle reconstruction semantics. A private Polars-native helper `_reconstruct_lifecycle_polars` mirrors the BigQuery DDL in `src/preprocessing/lifecycle.py::_build_lifecycle_ddl`. Tests build small synthetic event tables, run the helper, and assert on the resulting per-instance summary. The BigQuery path itself is validated end-to-end against the EDA-confirmed statistics during `notebooks/08_google_preprocessing.py` Section 5.

- `test_lifecycle_resubmission_count` covers the SUBMIT-count arithmetic.
- `test_lifecycle_prior_fail_flag` distinguishes single-pass success, single-pass failure, and resubmitted instances.
- `test_lifecycle_queue_time` and `..._with_resubmission_uses_first_schedule` cover the queue-time and running-duration arithmetic in the presence of multiple SUBMIT and SCHEDULE events.
- `test_lifecycle_running_duration_sentinel_aware` is the regression guard for the EDA-discovered 2.3-billion-second mean. Without `filter_sentinel_timestamps`, a sentinel-bearing FAIL terminal pushes `running_duration_sec` past 1 billion; with the filter applied first, the realistic three-second duration is recovered.
- `test_lifecycle_schema_matches_bigquery_ddl` is the parity guard between the Polars reference and the BQ DDL.
- `test_lifecycle_counts_lost_with_fail` covers LOST events being aggregated with FAIL.

### `test_episodes.py` (Phase 3, per-attempt regrain)

Six unit tests for `src/features/episodes.py`, covering the scheduled-episode regrain (`V30`-`V32`). A Polars-native mirror of the segmentation and strictly-prior history is exercised on hand-built event sequences.

- Two-episode segmentation with strictly-prior history, and the first-episode zero-prior-history guard (the leakage check that anchors the whole redesign).
- The multi-terminal first-by-time rule.
- The per-instance negative cap (positives kept in full, negatives capped per instance).
- The instance-keyed group split (no instance straddles train and test).
- The SQL builders rendering balanced and at the episode grain.

### `test_sampling.py` (Phase 3, working-set sampler)

Eight unit tests for `src/features/sampling.py`. The four Google tests (`P01`) use synthetic instance-events frames with a known stratum structure to verify the contract properties plus the degenerate and SQL-render paths:

- Full failure retention: every failure-bearing collection is kept in full and every FAIL instance survives, with subsampling of successful collections actually occurring.
- Priority-tier and scheduling-class marginals preserved within 2%.
- Target-exceeds-population path: when the target dwarfs the data, the full population is kept (the behavior the 35.1M eligible Google population actually triggers).
- `build_working_set_sql` renders with balanced parentheses, the unconditional failure-retention branch, and the stratified prefix selection.

The four Backblaze tests (`V17`, `P07`) use a feature-shaped frame with horizon-positive and healthy rows across two models and two years:

- Every horizon-positive row is retained and the achieved healthy-to-positive ratio matches the target.
- The healthy sample preserves the drive-model composition of the healthy population (proportional stratified).
- No-op when the healthy population already sits below the target ratio.
- Determinism under a fixed seed.

### `test_backblaze_smart.py` (Phase 3, Backblaze SMART features)

Five unit tests for `src/features/backblaze_smart.py`, using a single date-sorted drive fixture with zero-inflated SMART columns:

- Tier 1 non-zero indicators and the leakage-safe days-since-first-non-zero timer (null before the first non-zero), plus manufacturer one-hot and capacity.
- Tier 2 rolling mean / p95 / std, 1-day and 7-day deltas, and the reduced secondary set.
- Tier 3 cohort ages, calendar parts, and the era-availability flags (0 for SMART 187/188 in the recent era, 1 in the standard era).
- Multi-horizon targets flagging the correct pre-failure window and zero for a never-failing drive.

### `test_metrics.py` (Phase 4 and Phase 7)

Nineteen tests for `src/evaluation/metrics.py`. The Phase 4 set covers the closed-form MCC / F1 against scikit-learn, the stratified bootstrap returning `(point, ci_low, ci_high)` with the point matching the direct computation, the single-class degenerate path returning NaN bounds, PR-AUC / ROC-AUC / Brier on synthetic predictions, and the `calibration_table` bin structure and observed-versus-predicted columns.

The Phase 7 set covers the custom drift metrics: a known-slope recovery for `performance_degradation_rate` (with date and numeric month indices), the signed-day and negative-latency cases of `drift_detection_latency`, the no-recovery / full-recovery / overshoot / undefined cases of `retraining_effectiveness`, and the sustainment window. Two tests encode findings rather than mechanics. `test_sustainment_reference_makes_the_window_informative` builds a holding series and a decaying series that are indistinguishable under the 0.85 bar (both 0 months) and separate cleanly under a reachable reference (4 versus 1), which is why `sustainment_reference` exists. `test_fixed_prevalence_mcc_separates_prior_shift_from_covariate_drift` constructs two windows of identical discriminative power at different base rates, confirms the raw MCC moves by more than 0.02 anyway, and confirms the fixed-prevalence MCC agrees within 0.02, which is the guard against reading a declining base rate as model staleness (`V55`).

### `test_hypothesis.py` (Phase 4)

Seventeen tests for `src/evaluation/hypothesis.py`: the reject / fail / boundary / less-is-better cases of `one_sample_threshold_test`, paired Wilcoxon on identical and consistently-different folds plus the length-mismatch guard, analytic Cohen's d, hand-computed Holm-Bonferroni and Benjamini-Hochberg adjustments with input-order preservation, and a parity check against statsmodels' `multipletests`.

### `test_ensemble.py` (Phase 4)

Fifteen tests for `src/models/ensemble.py`: per-wrapper fit / predict / importance and save-load round-trips (XGBoost / LightGBM / Balanced Random Forest skipped if the library is absent), the Polars / NumPy boundary equivalence, the soft-voting stack (equal and weighted), XGBoost's auto `scale_pos_weight`, and the `build_wrapper` error path.

### `test_classifier.py` (Phase 4)

Eleven tests for `src/models/classifier.py`: fit / predict / importances and save-load round-trips for the Decision Tree and linear SVM, the Polars / NumPy / pandas boundary equivalence, a check that the SVM score is the unbounded decision function (not a [0, 1] probability), the Keras NN fit / predict / save-load and its `NotImplementedError` on feature importances (the three Keras tests skipped when TensorFlow is absent), and the factory error path plus registry completeness.

### `test_drift_detectors.py` (Phase 7)

Nineteen tests for `src/evaluation/drift_detectors.py`. Each detector is driven with a synthetic stream whose change point is known by construction and asserted on three properties: silence before the change, detection within a bounded latency after it, and silence on a matched stationary stream. That third property is the one that matters: both false-alarm guards caught real defects. `test_page_hinkley_is_quiet_on_a_stationary_stream` failed at the library-typical settings (a false drift event at observation 624 on a stream containing no drift), which produced the tuned defaults, and `test_page_hinkley_defaults_keep_the_noise_scale_below_the_threshold` pins the derivation so a future default change trips a test rather than quietly refilling the drift log with noise. `test_psi_refuses_windows_where_its_own_bands_are_noise` locks the constructor guard. Remaining tests cover the shared interface and reset semantics, explicit versus auto-built reference windows, non-finite skipping, and PSI analytic cases including the degenerate all-zero reference that the zero-inflated SMART attributes produce.

### `test_online.py` (Phase 7)

Twenty-three tests for `src/models/online.py`, driven on a small separable synthetic stream. Every learner in the registry is asserted to satisfy the protocol and to beat chance (positives scoring above negatives), which is the guard against a River signature change silently producing a learner that trains but never learns. `test_checkpoint_resumes_exactly_where_it_left_off` is the load-bearing one: a learner checkpointed halfway and resumed must produce predictions identical to one that streamed straight through, because the RQ5 grid cannot fit in a single 12-hour Colab session. Others cover the row-dict boundary (including the null-drop encoding), balanced class weights, prequential ordering (`learn_one` returns self; predicting does not consume the observation), reset, and the soft-voting ensemble in both static and dynamically reweighted modes, including that a deliberately sabotaged member loses weight relative to a healthy one.

### Planned

- `test_features.py` (Phase 3): the Google Tier 1 / 2 / 3 builders (`historical`, `scheduling`, `temporal`, `runtime`, `utilization`) against synthetic LazyFrames. The Backblaze feature builders are already covered by `test_backblaze_smart.py`.
- `test_conflict_labels.py` (Phase 4): the RQ2 labelers and the `prior_counts` strict-prior helper (currently validated by hand and in-notebook).

## Fixture conventions

- Tests build their own LazyFrames inline rather than depending on shared `conftest.py` fixtures. Each test reads top-to-bottom as a self-contained story; the cost is some repetition.
- Constants and event-type codes come from `src/data/schemas.py` so the test inputs use the same vocabulary as the production functions. No magic numbers in tests.
- Polars arithmetic and comparison checks use exact equality where possible. For aggregate rate checks (e.g. V11 null rates) tests use a tolerance band keyed to the test fixture's construction precision.
- Tests do not perform I/O. Modules in `src/` that require I/O (notably `src/preprocessing/lifecycle.py::reconstruct_instance_lifecycle`) are tested via a Polars-native reference helper rather than by mocking the BigQuery client.
