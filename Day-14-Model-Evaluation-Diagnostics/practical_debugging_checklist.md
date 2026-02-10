# Practical Debugging Checklist — Model Evaluation & Diagnostics

Data and Splits
- Confirm stratified splits for classification; preserve group/time structure when applicable (GroupKFold/TimeSeriesSplit).
- Check class/target distribution in train vs test; watch for covariate shift.
- Validate no leakage: preprocessing, encoding, imputation, feature construction fit on training only.

Metric Choice and Baselines
- Choose metrics aligned with the task: ROC-AUC/PR-AUC/F1 for imbalanced classification; MAE/R2 for regression; NDCG/AP for ranking.
- Always compare against simple baselines (e.g., logistic regression, single tree, dummy classifier/regressor).
- Report multiple metrics (accuracy + F1 + ROC-AUC); inspect confusion matrix and per-class performance.

Learning and Validation Curves
- Plot learning curves: large gap → high variance (consider regularization or more data); both curves low → high bias (more capacity/features).
- Plot validation curves across key hyperparameters (e.g., C for SVC, max_depth for trees) to find under/over-regularization regimes.

Error Analysis
- Confusion matrix: identify dominant error types (e.g., false positives vs false negatives).
- Slice-based analysis: compute metrics across subgroups or feature ranges; prioritize fixes where performance is worst.
- Examine misclassified examples: look for label noise, feature quality issues, or systematic patterns.

Calibration and Uncertainty
- Calibrate probabilities (CalibratedClassifierCV) when thresholding decisions or communicating risk scores.
- Bootstrap confidence intervals for your chosen metric to quantify uncertainty and avoid overinterpreting small differences.

Regularization and Hyperparameters
- Start conservative: smaller learning rates (boosting), regularization on; increase capacity only if underfitting.
- Tune 2–3 impactful hyperparameters first (e.g., depth/leaves, learning_rate, C); use RandomizedSearch/Optuna to explore interactions.

Reproducibility and Logging
- Set random_state consistently (model, CV, data generation).
- Log seeds, splits, best params, metrics, and environment (library versions).
- Use Pipelines to freeze preprocessing; serialize final models and configs.

Operational Concerns
- Check runtime and memory; simplify model if needed.
- Monitor drift over time; consider periodic re-evaluation and recalibration.
- Build simple visual dashboards (learning curves, reliability diagrams, slice metrics) for ongoing diagnostics.
