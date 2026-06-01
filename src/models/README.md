# `src/models/` — Ensemble Training and Online Learners

**Status:** Planned for Phase 4 and Phase 7. Modules are extracted after the per-RQ notebooks validate the training logic against the methodology committed in Dissertation Proposal Chapter 3.

## Planned modules

- `ensemble.py` — Ensemble training and prediction harness for RQ1 (failure prediction). Wraps scikit-learn, XGBoost, and LightGBM through a single interface with walk-forward cross-validation (Cerqueira et al., 2020), Bayesian hyperparameter tuning (50 trials per family), and cost-sensitive learning. Extracts from `notebooks/12_rq1_ensemble_google.py` once the three-level prediction architecture (`P06`) is validated.
- `classifier.py` — Single-model classifiers for RQ2 (conflict resolution). Decision tree, SVM, random forest, and shallow neural network variants. Targets the >80% conflict-resolution success rate (`P09`).
- `online.py` — Online learners for RQ5 (concept drift). River-based Adaptive Random Forest, Hoeffding Adaptive Tree, and Online Gradient Boosting (`P13`). Paired with the drift detectors specified in `P12` (ADWIN, Page-Hinkley, KS, PSI).

## Cross-cutting conventions

- Every stochastic operation uses `RANDOM_SEED = 42`, enforced repository-wide and locked by `tests/test_smoke.py::test_random_seed_constant_is_42`.
- Hyperparameter configurations are stored under `configs/` and loaded at runtime so model recipes are inspectable and version-controlled.
- Class-imbalance handling combines cost-sensitive learning (class weights = inverse prior) with SMOTE applied only inside training folds (`V02` for Google, `V17` for Backblaze).
- Walk-forward cross-validation is the only acceptable split strategy. Random k-fold is disallowed because both datasets carry temporal structure.

Modules land here once the corresponding notebook (12 through 16) validates the training logic and the per-RQ result tables land in `outputs/tables/`.
