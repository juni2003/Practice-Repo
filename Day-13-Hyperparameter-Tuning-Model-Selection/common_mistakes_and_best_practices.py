COMMON MISTAKES & BEST PRACTICES — Hyperparameter Tuning & Model Selection

Mistakes:
1) Data Leakage
   - Scaling/encoding on the full dataset before CV.
   - Fix: Use Pipeline with transformers fit on training folds only.

2) Wrong CV Strategy
   - Using standard KFold for classification (class imbalance) or ignoring groups/time.
   - Fix: StratifiedKFold for classification, GroupKFold for grouped data, TimeSeriesSplit for temporal data.

3) Overly Narrow Search Spaces
   - Missing good regions due to tight grids.
   - Fix: Start with RandomizedSearchCV over broad ranges; refine with GridSearchCV.

4) Ignoring Reproducibility
   - Random results across runs.
   - Fix: Set random_state everywhere, including CV shuffling and search random_state.

5) Optimizing Wrong Metric
   - Using accuracy for imbalanced classes.
   - Fix: Prefer ROC-AUC/F1/PR-AUC depending on task.

6) Too Many Folds / Too Large CV
   - Excessive runtime with minimal benefit.
   - Fix: 5–10 folds typical; match dataset size and computational budget.

7) Nested Early Stopping with CV
   - Leakage from using internal validation during CV.
   - Fix: Disable internal early stopping in CV; or use an outer train/val split for early stopping.

8) Ignoring Interaction of Hyperparameters
   - Tuning one param while others fixed suboptimally.
   - Fix: RandomizedSearch or Bayesian optimization to explore interactions.

Best Practices:
- Use Pipelines and ColumnTransformer to avoid leakage.
- Begin with RandomizedSearchCV; follow with a focused GridSearchCV.
- For boosting: small learning_rate with early stopping.
- Log-scale for multiplicative params (e.g., C, n_estimators); sensible bounds for depth/leaves.
- Document seeds, CV splits, and final best params; ensure repeatable experiments.
