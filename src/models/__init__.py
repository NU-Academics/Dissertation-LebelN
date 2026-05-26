"""Model wrappers.

Phase 4 (Modeling) and Phase 7 (Operation and Maintenance) deliverables.
Thin wrappers around scikit-learn, XGBoost, LightGBM, and River that expose a
common interface for fitting, predicting, feature-importance extraction, and
checkpointing.

Planned submodules (created during Weeks 4-9):
- ensemble: Random Forest, Balanced Random Forest, XGBoost, LightGBM, Gradient
  Boosting, soft-voting stack. Used by RQ1 (Google and Backblaze) and as the
  static baseline for RQ5.
- classifier: Decision Tree, SVM, Random Forest, simple Keras NN for RQ2
  conflict resolution.
- online: River incremental learners (Adaptive Random Forest, Hoeffding
  Adaptive Tree, Online Gradient Boosting) and online soft-voting ensemble for
  RQ5 drift adaptation.
"""
