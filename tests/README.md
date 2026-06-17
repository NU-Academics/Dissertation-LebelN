# `tests/` - Test Suite

Pytest test suite for the extracted `src/` modules. Synthetic Polars LazyFrame fixtures hand-crafted to hit each branch. No BigQuery, no Drive, no GCS. Fast: the full suite runs in a few seconds on a laptop. The TensorFlow-dependent neural-network tests skip automatically when TensorFlow is absent (they run in Colab), as do the XGBoost / LightGBM wrapper tests when those libraries are not installed locally.

## Running

```bash
# From the repo root
pytest tests/ -v
```

The suite is 81 tests across nine files (locally 72 pass and 9 skip where TensorFlow / XGBoost / LightGBM are absent; all run in Colab):

```
tests/test_smoke.py              2 tests
tests/test_preprocessing.py      9 tests
tests/test_lifecycle.py          7 tests
tests/test_episodes.py           6 tests
tests/test_sampling.py           4 tests
tests/test_metrics.py           10 tests
tests/test_hypothesis.py        17 tests
tests/test_ensemble.py          15 tests
tests/test_classifier.py        11 tests
```

To run a single file: `pytest tests/test_preprocessing.py -v`. To run a single test: `pytest tests/test_preprocessing.py::test_failure_label_primary -v`.

## Test inventory

### `test_smoke.py`

Repo-skeleton sanity checks that must remain green throughout Chapter 4 execution.

- `test_src_subpackages_importable` confirms every `src/` subpackage imports cleanly.
- `test_random_seed_constant_is_42` locks the reproducibility contract. If this assertion needs to change, the entire reproducibility plan in Chapter 3 is being modified and must be updated first.

### `test_preprocessing.py` (Phase 3)

Unit tests for the four pure functions in `src/preprocessing/google_traces.py`. Each fixture is hand-crafted to hit the branches the function must handle.

- `test_filter_sentinel_timestamps_drops_zero_and_max` and `..._accepts_alternate_time_column` cover `V25`.
- `test_failure_label_primary` covers `V01` and `V08` (KILL exclusion).
- `test_failure_label_sensitivity_prod_evict` covers `P04` (Production-priority EVICTs labeled in the sensitivity column).
- `test_failure_label_excludes_monitoring_evict` is the regression guard for `V27`. Monitoring-priority EVICTs must stay NULL under every label, while a production-boundary EVICT confirms the sensitivity branch still activates.
- `test_failure_label_invalid_sensitivity_branch_raises` confirms the ValueError contract on bad inputs.
- `test_mnar_indicators_per_observation` builds a 1000-row fixture engineered to match V11's 87.2% / 26.8% null targets, verifies the per-row indicator flips with the underlying null pattern, and confirms the aggregated null rates land within +/- 1 percentage point of the V11 targets.
- `test_mnar_indicators_per_instance_majority` covers `V28` with five hand-crafted instances (all-present, all-absent, 60% majority, 40% minority, exact 50% boundary).
- `test_mnar_indicators_opt_in_skips_majority_column` confirms the opt-in flag.

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

Four unit tests for `src/features/sampling.py` (`P01`). Synthetic instance-events frames with a known stratum structure verify the two contract properties plus the degenerate and SQL-render paths.

- Full failure retention: every failure-bearing collection is kept in full and every FAIL instance survives, with subsampling of successful collections actually occurring.
- Priority-tier and scheduling-class marginals preserved within 2%.
- Target-exceeds-population path: when the target dwarfs the data, the full population is kept (the behavior the 35.1M eligible Google population actually triggers).
- `build_working_set_sql` renders with balanced parentheses, the unconditional failure-retention branch, and the stratified prefix selection.

### `test_metrics.py` (Phase 4)

Ten tests for `src/evaluation/metrics.py`: the closed-form MCC / F1 against scikit-learn, the stratified bootstrap returning `(point, ci_low, ci_high)` with the point matching the direct computation, the single-class degenerate path returning NaN bounds, PR-AUC / ROC-AUC / Brier on synthetic predictions, and the `calibration_table` bin structure and observed-versus-predicted columns.

### `test_hypothesis.py` (Phase 4)

Seventeen tests for `src/evaluation/hypothesis.py`: the reject / fail / boundary / less-is-better cases of `one_sample_threshold_test`, paired Wilcoxon on identical and consistently-different folds plus the length-mismatch guard, analytic Cohen's d, hand-computed Holm-Bonferroni and Benjamini-Hochberg adjustments with input-order preservation, and a parity check against statsmodels' `multipletests`.

### `test_ensemble.py` (Phase 4)

Fifteen tests for `src/models/ensemble.py`: per-wrapper fit / predict / importance and save-load round-trips (XGBoost / LightGBM / Balanced Random Forest skipped if the library is absent), the Polars / NumPy boundary equivalence, the soft-voting stack (equal and weighted), XGBoost's auto `scale_pos_weight`, and the `build_wrapper` error path.

### `test_classifier.py` (Phase 4)

Eleven tests for `src/models/classifier.py`: fit / predict / importances and save-load round-trips for the Decision Tree and linear SVM, the Polars / NumPy / pandas boundary equivalence, a check that the SVM score is the unbounded decision function (not a [0, 1] probability), the Keras NN fit / predict / save-load and its `NotImplementedError` on feature importances (the three Keras tests skipped when TensorFlow is absent), and the factory error path plus registry completeness.

### Planned

- `test_features.py` (Phase 3): Tier 1 / 2 / 3 feature derivations against synthetic LazyFrames.
- `test_conflict_labels.py` (Phase 4): the RQ2 labelers and the `prior_counts` strict-prior helper (currently validated by hand and in-notebook).
- `test_drift_detectors.py` (Phase 7): ADWIN, Page-Hinkley, KS, PSI behavior under synthetic drift.

## Fixture conventions

- Tests build their own LazyFrames inline rather than depending on shared `conftest.py` fixtures. Each test reads top-to-bottom as a self-contained story; the cost is some repetition.
- Constants and event-type codes come from `src/data/schemas.py` so the test inputs use the same vocabulary as the production functions. No magic numbers in tests.
- Polars arithmetic and comparison checks use exact equality where possible. For aggregate rate checks (e.g. V11 null rates) tests use a tolerance band keyed to the test fixture's construction precision.
- Tests do not perform I/O. Modules in `src/` that require I/O (notably `src/preprocessing/lifecycle.py::reconstruct_instance_lifecycle`) are tested via a Polars-native reference helper rather than by mocking the BigQuery client.
