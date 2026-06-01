# `src/evaluation/` — Metrics, Confidence Intervals, Hypothesis Tests

**Status:** Planned for Phase 5 with reusable metric helpers landing earlier so the per-RQ notebooks share one canonical implementation of each metric. Modules are extracted from the per-RQ training notebooks (`12` through `16`) and the cross-cutting evaluation notebook (`17_hypothesis_testing.py`).

## Planned modules

- `metrics.py` — Canonical implementations of the primary classification metrics: Matthews correlation coefficient (the primary hypothesis-testing metric per Chicco & Jurman, 2023), F1, PR-AUC (Saito & Rehmsmeier, 2015), confusion-matrix components. Each metric is computed at the operating point selected by PR-curve tuning rather than at the default 0.5 threshold.
- `bootstrap.py` — 95% bootstrap confidence intervals with 1,000 stratified resamples (`P15`). Stratification preserves class balance per resample. Used to report every metric in `outputs/tables/`.
- `hypothesis.py` — Per-RQ threshold tests, paired Wilcoxon model comparisons, Cohen's d effect sizes, and family-wise error control across the five RQs using Holm-Bonferroni as the primary method and Benjamini-Hochberg as supporting (`P16`).
- `calibration.py` — Reliability diagrams and Brier scores. Calibration is a TRIPOD+AI checklist item (`P18`).
- `interpretability.py` — Sample-based SHAP (10,000 to 20,000 stratified rows per `P17`) and LIME on high-stakes cases. Produces global feature-importance rankings and per-prediction explanations.

## Cross-cutting conventions

- All metrics returned by `metrics.py` accept ground-truth and predicted probability arrays plus the operating-point threshold. Functions never read the threshold from a global.
- Every reported metric in `outputs/tables/` carries its bootstrap CI; the bootstrap helper is the single source of truth.
- Hypothesis-test results write to `outputs/tables/hypothesis_tests.csv` with one row per RQ, including raw p, Holm-adjusted p, BH-adjusted p, effect size, and decision.
- Reporting follows TRIPOD+AI item 23 (calibration), item 24 (discrimination), and item 25 (uncertainty quantification).

Modules land here once the corresponding RQ has at least one validated result and the metric / CI pattern stabilizes.
