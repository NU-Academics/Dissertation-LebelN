"""Model wrappers.

Phase 4 (Modeling) and Phase 7 (Operation and Maintenance) deliverables.
Thin wrappers around scikit-learn, XGBoost, LightGBM, and River that expose a
common interface for fitting, predicting, feature-importance extraction, and
checkpointing.

Submodules (extracted from the per-RQ modeling notebooks):
- ensemble (populated): Random Forest, Balanced Random Forest, XGBoost,
  LightGBM, Gradient Boosting, and a soft-voting stack behind one
  EnsembleWrapper interface (fit / predict_proba / feature_importances /
  save / load), accepting Polars at the boundary. Used by RQ1 (Google and
  Backblaze) and as the static baseline for RQ5.
- classifier (planned): Decision Tree, SVM, Random Forest, simple Keras NN for
  RQ2 conflict resolution.
- online (planned): River incremental learners (Adaptive Random Forest,
  Hoeffding Adaptive Tree, Online Gradient Boosting) and online soft-voting
  ensemble for RQ5 drift adaptation.
"""
