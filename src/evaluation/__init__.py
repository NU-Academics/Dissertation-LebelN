"""Evaluation modules.

Phase 4 (Modeling) and Phase 5 (Evaluation) deliverables.
Metric computation with bootstrap confidence intervals, hypothesis testing
against locked thresholds, and drift detection support for RQ5.

Planned submodules (created during Weeks 4-10):
- metrics: MCC (primary), F1, PR-AUC, ROC-AUC, Brier, calibration tables, all
  with stratified bootstrap 95% CIs. Custom RQ5 metrics (degradation rate,
  detection latency, retraining effectiveness, sustainment window).
- hypothesis: one-sample tests against locked thresholds, paired Wilcoxon for
  model-versus-model comparisons, Cohen's d effect sizes, family-wise error
  control (Holm and Benjamini-Hochberg).
- drift_detectors: ADWIN, Page-Hinkley, KS, PSI detector wrappers for RQ5
  drift simulation.
"""
