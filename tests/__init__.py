"""Test suite for src/ modules.

Tests are added as modules are extracted from notebooks during Phases 3, 4,
and 7. Coverage targets:
- tests/test_preprocessing.py: preprocessing modules (Phase 3)
- tests/test_features.py: feature engineering modules (Phase 3)
- tests/test_sampling.py: working-set samplers (Phase 3)
- tests/test_metrics.py: bootstrap CI helpers, MCC/F1/PR-AUC (Phase 4)
- tests/test_hypothesis.py: one-sample threshold tests + paired Wilcoxon (Phase 4)
- tests/test_drift_detectors.py: ADWIN, Page-Hinkley, KS, PSI wrappers (Phase 7)
"""
