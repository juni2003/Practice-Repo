# Day 13: Hyperparameter Tuning & Model Selection

Overview
- Goal: Systematically find good hyperparameters while avoiding data leakage and preserving reproducibility.
- Tools: GridSearchCV, RandomizedSearchCV, Bayesian optimization (Optuna), cross-validation strategies, reproducibility controls.
- Outcomes: Faster iteration, better generalization, and reliable comparisons.

What You’ll Learn
- Grid vs Random search: when each is appropriate and how to measure effectiveness.
- Bayesian optimization (Optuna): smarter search using priors and acquisition functions.
- Cross-validation strategies: StratifiedKFold, GroupKFold, TimeSeriesSplit.
- Reproducibility: consistent seeding for NumPy, scikit-learn, Optuna; deterministic pipelines.
- Search spaces: picking ranges and distributions that make sense.

Module Structure
Day-13-Hyperparameter-Tuning-Model-Selection/
- README.md
- grid_vs_random_search.py
- optuna_bayes_optimization.py
- cross_validation_strategies.py
- reproducibility_and_search_spaces.py
- sklearn_pipeline_tuning_examples.py
- tuning_with_early_stopping_templates.py
- common_mistakes_and_best_practices.txt

Quick Start
- Grid vs Random search
```python
# Run: python grid_vs_random_search.py
# Compares time, best params, and ROC-AUC for LogisticRegression.
```

- Bayesian optimization (Optuna)
```python
# Run: python optuna_bayes_optimization.py
# Tunes HistGradientBoostingClassifier via CV AUC and prints best trial.
```

- Cross-validation strategies
```python
# Run: python cross_validation_strategies.py
# Shows StratifiedKFold vs GroupKFold vs TimeSeriesSplit usage and caveats.
```

Practical Tips
- Always split data before tuning; fit transformers on training only.
- Start with RandomizedSearchCV for broad exploration; refine with GridSearchCV.
- Use small learning rates and early stopping for boosting models.
- Keep a global random_state for reproducibility; control shuffling in CV.
- Balance metrics: use ROC-AUC/F1 for imbalanced classification.

Common Pitfalls (see file)
- Data leakage via scaling/encoding on full dataset.
- Overly narrow search spaces; or too wide with unrealistic ranges.
- Ignoring CV strategy for grouped/time-dependent data.
- Not fixing random_state, leading to unstable results.

Enjoy robust tuning!
