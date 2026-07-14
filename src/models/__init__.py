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
- classifier (populated): Decision Tree, linear SVM, and a one-hidden-layer
  Keras NN for RQ2 conflict resolution, reusing the EnsembleWrapper protocol and
  the boundary / save-load from ensemble. Random Forest is reused from ensemble
  (RandomForestWrapper), not duplicated. Extracted from notebook 13.
- online (populated): River incremental learners (Adaptive Random Forest,
  Hoeffding Adaptive Tree, online AdaBoost over Hoeffding trees, incremental
  logistic regression) plus an online soft-voting ensemble with optional
  performance-based reweighting, behind one OnlineLearner interface
  (learn_one / predict_proba_one / predict_proba / save / load) with
  checkpoint-and-resume. River has no online gradient boosting, so the boosting
  arm is incremental AdaBoost. Used by RQ5; the static baseline it is compared
  against is the calibrated ensemble checkpoint from RQ1.
"""
