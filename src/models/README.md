# `src/models/` — Ensemble Training and Online Learners

**Status:** `ensemble.py` populated (extracted from the RQ1 modeling notebook and unit-tested). `classifier.py` and `online.py` remain planned for the RQ2 and RQ5 notebooks. Modules are extracted after the per-RQ notebooks validate the training logic against the methodology committed in Dissertation Proposal Chapter 3.

## Modules

- `ensemble.py`, populated — Reusable wrappers for the RQ1 tree-ensemble learners: `RandomForestWrapper`, `BalancedRandomForestWrapper`, `XGBoostWrapper`, `LightGBMWrapper`, `GradientBoostingWrapper`, and a `SoftVotingStack`, all satisfying one `EnsembleWrapper` protocol (`fit` / `predict_proba` / `feature_importances` / `save` / `load`). They accept Polars DataFrames at the boundary (converting to a dense `float32` array once per call), set cost-sensitive class weighting to the inverse class prior where the learner supports it (XGBoost via `scale_pos_weight` computed at fit; Gradient Boosting via balanced `sample_weight`), and leave SMOTE and the per-instance negative cap to the caller as training-only fold steps. `predict_proba` returns the 1-D positive-class probability. `build_wrapper(name, **params)` is the factory used by the modeling notebook and the hyperparameter search. Extracted from `notebooks/12_rq1_ensemble_google.py` (the three-level prediction architecture, `P06`); tested in `tests/test_ensemble.py`. Walk-forward cross-validation and Bayesian tuning (50 trials per family) are orchestrated by the notebook around these wrappers.
- `classifier.py` — Single-model classifiers for RQ2 (conflict resolution). Decision tree, SVM, random forest, and shallow neural network variants. Targets the >80% conflict-resolution success rate (`P09`).
- `online.py` — Online learners for RQ5 (concept drift). River-based Adaptive Random Forest, Hoeffding Adaptive Tree, and Online Gradient Boosting (`P13`). Paired with the drift detectors specified in `P12` (ADWIN, Page-Hinkley, KS, PSI).

## Cross-cutting conventions

- Every stochastic operation uses `RANDOM_SEED = 42`, enforced repository-wide and locked by `tests/test_smoke.py::test_random_seed_constant_is_42`.
- Hyperparameter configurations are stored under `configs/` and loaded at runtime so model recipes are inspectable and version-controlled.
- Class-imbalance handling combines cost-sensitive learning (class weights = inverse prior) with SMOTE applied only inside training folds (`V02` for Google, `V17` for Backblaze).
- Walk-forward cross-validation is the only acceptable split strategy. Random k-fold is disallowed because both datasets carry temporal structure.

Modules land here once the corresponding notebook (12 through 16) validates the training logic and the per-RQ result tables land in `outputs/tables/`.
