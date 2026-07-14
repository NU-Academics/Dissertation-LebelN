"""Evaluation modules.

Phase 4 (Modeling) and Phase 5 (Evaluation) deliverables.
Metric computation with bootstrap confidence intervals, hypothesis testing
against locked thresholds, and drift detection support for RQ5.

Submodules:
- metrics (populated): MCC (primary), F1, PR-AUC, ROC-AUC, Brier, each as a
  (point, ci_low, ci_high) triple with a stratified bootstrap 95% CI, plus a
  calibration table. Custom RQ5 metrics (degradation rate, detection latency,
  retraining effectiveness, sustainment window) will be added for RQ5.
- hypothesis (populated): one-sample tests against locked thresholds, paired
  Wilcoxon for model-versus-model comparisons, Cohen's d effect sizes,
  family-wise error control (Holm and Benjamini-Hochberg).
- drift_detectors (populated): ADWIN and Page-Hinkley over a per-observation loss
  stream (sudden drift), KS and PSI over a reference-versus-sliding feature window
  (gradual drift and covariate shift), behind one update / drift_detected /
  warning_detected / reset interface, plus a standalone PSI function.
"""
