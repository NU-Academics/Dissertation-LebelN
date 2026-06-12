# `src/evaluation/` — Metrics, Confidence Intervals, Hypothesis Tests

**Status:** `metrics.py` and `hypothesis.py` populated and unit-tested; `interpretability.py` remains planned. Reusable metric helpers landed early so the per-RQ notebooks share one canonical implementation of each metric. Modules are extracted from the per-RQ training notebooks (`12` through `16`) and the cross-cutting evaluation notebook (`17_hypothesis_testing.py`).

## Modules

- `metrics.py`, populated — Canonical metrics, each returned as `(point, ci_low, ci_high)`: `mcc_with_ci` (the primary metric per Chicco & Jurman, 2023), `f1_with_ci`, `pr_auc_with_ci` (Saito & Rehmsmeier, 2015), `roc_auc_with_ci`, `brier_score_with_ci`, plus `calibration_table` (reliability bins and observed-vs-predicted, a TRIPOD+AI calibration item, `P18`). The 95% confidence interval is a stratified percentile bootstrap (1,000 resamples by default; positives and negatives resampled independently to preserve class balance and keep both classes present for the AUC metrics, `P15`). Inputs accept Polars Series, NumPy arrays, or lists; the label metrics take `y_pred` already at the chosen operating point and the score metrics take `y_score`. Tested in `tests/test_metrics.py`. The bootstrap engine lives inside this module, so there is no separate `bootstrap.py`.
- `hypothesis.py`, populated — `one_sample_threshold_test` (the CI-based one-sided decision committed in Chapter 3: reject only when the (1 - alpha) CI lower bound clears the target), `paired_wilcoxon_cv` (signed-rank comparison of two models across the same CV folds, with the small-fold power caveat stated), `cohens_d` (pooled effect size), and the family-wise corrections `holm_bonferroni` (primary) and `benjamini_hochberg` (supporting) for control across the five RQs (`P16`). Each returns a plain dict that serializes into `outputs/tables/`. Tested in `tests/test_hypothesis.py`.
- Calibration and Brier-score helpers are part of `metrics.py` (`calibration_table` and `brier_score_with_ci`), not a separate `calibration.py`.
- `interpretability.py` — Sample-based SHAP (10,000 to 20,000 stratified rows per `P17`) and LIME on high-stakes cases. Produces global feature-importance rankings and per-prediction explanations.

## Cross-cutting conventions

- The label metrics (`mcc_with_ci`, `f1_with_ci`) take `y_pred` already thresholded at the chosen operating point; the score metrics (`pr_auc_with_ci`, `roc_auc_with_ci`, `brier_score_with_ci`) and `calibration_table` take `y_score`. Functions never read a threshold from a global; the caller selects the operating point upstream.
- Every reported metric in `outputs/tables/` carries its bootstrap CI; the bootstrap helper is the single source of truth.
- Hypothesis-test results write to `outputs/tables/hypothesis_tests.csv` with one row per RQ, including raw p, Holm-adjusted p, BH-adjusted p, effect size, and decision.
- Reporting follows TRIPOD+AI item 23 (calibration), item 24 (discrimination), and item 25 (uncertainty quantification).

Modules land here once the corresponding RQ has at least one validated result and the metric / CI pattern stabilizes.
