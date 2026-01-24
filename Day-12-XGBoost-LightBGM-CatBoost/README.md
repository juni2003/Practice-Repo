# Day 12: XGBoost, LightGBM, CatBoost — Gradient-Boosted Trees

## Overview
Gradient-boosted trees win often on tabular data because they:
- Fit powerful nonlinear relationships and interactions without heavy feature engineering
- Use regularization, shrinkage (learning rate), subsampling, and early stopping to control overfitting
- Handle missing values natively (XGBoost, LightGBM, CatBoost)
- Are fast and scalable (hist-based algorithms, sparse-aware)

This module covers:
- Why gradient-boosted trees excel on tabular data
- Core training loop (sequential weak learners on gradients/residuals)
- Early stopping and regularization
- Feature importance and SHAP
- Handling categorical features (CatBoost)
- Hyperparameter tuning templates

Libraries:
- XGBoost: optimized gradient boosting with rich features
- LightGBM: fast, memory-efficient, leaf-wise growth, great for large datasets
- CatBoost: native categorical handling, strong defaults

## Module Structure
```
Day-12-XGBoost-LightGBM-CatBoost/
├── README.md
├── xgboost_basics.py
├── lightgbm_basics.py
├── catboost_categorical_handling.py
├── early_stopping_and_regularization.py
├── feature_importance_and_shap.py
├── hyperparameter_tuning_templates.py
└── common_mistakes_and_best_practices.txt
```

## Install
```bash
pip install xgboost lightgbm catboost shap scikit-learn matplotlib pandas seaborn
```

## Quick Starts

XGBoost (classification with early stopping)
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

model = XGBClassifier(
    n_estimators=1000, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method="hist", eval_metric="logloss", random_state=42
)
model.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
print("Best iteration:", model.best_iteration_)
print("Test accuracy:", (model.predict(Xte) == yte).mean())
```

LightGBM (classification with early stopping)
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

model = LGBMClassifier(
    n_estimators=2000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42
)
model.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="logloss",
          early_stopping_rounds=50, verbose=False)
print("Best iteration:", model.best_iteration_)
print("Test accuracy:", (model.predict(Xte) == yte).mean())
```

CatBoost (categorical features)
```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "income": rng.normal(50_000, 15_000, n),
    "city": rng.choice(["A", "B", "C", "D"], n),
    "device": rng.choice(["mobile", "desktop", "tablet"], n)
})
y = ((df["age"] > 40).astype(int) + (df["city"].isin(["A", "B"])).astype(int) + (df["income"] > 60000).astype(int) + (df["device"] == "desktop").astype(int) + rng.integers(0, 2, n)) > 2
y = y.astype(int)

cat_idx = [2, 3]
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, stratify=y, random_state=42)
Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
val_pool = Pool(Xval, yval, cat_features=cat_idx)
test_pool = Pool(Xte, yte, cat_features=cat_idx)

model = CatBoostClassifier(
    iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
    loss_function="Logloss", eval_metric="AUC", random_state=42, verbose=False
)
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)
print("Best iteration:", model.get_best_iteration())
print("Test accuracy:", (model.predict(test_pool).astype(int).reshape(-1) == yte).mean())
```

## Why Gradient-Boosted Trees Win Often
- Strong base learners (trees) capture nonlinearity and interactions
- Shrinkage (learning_rate) + many estimators approximate complex functions smoothly
- Regularization: depth limits, l1/l2, subsampling, column sampling
- Early stopping picks optimal complexity via validation
- Native handling of missing values and monotonic constraints (varies by library)

## Early Stopping & Regularization
- Early stopping: monitor validation metric, stop when it stops improving
- Regularization knobs:
  - XGBoost: max_depth, min_child_weight, gamma, subsample, colsample_bytree, reg_lambda, reg_alpha
  - LightGBM: num_leaves, max_depth, min_child_samples, feature_fraction, bagging_fraction, lambda_l1/l2
  - CatBoost: depth, l2_leaf_reg, random_strength, subsample

## Feature Importance
- Gain/weight importance built-in
- SHAP values for consistent local/global explanations
- Permutation importance for model-agnostic validation

## Handling Categorical Features
- CatBoost natively handles categorical features via target statistics + ordered boosting
- XGBoost/LightGBM: one-hot or categorical support (LightGBM has native categorical if data provided in specific format; scikit-learn wrapper often uses integers + parameter setting; CatBoost is simpler)

## Hyperparameter Tuning Templates
- Start with sensible defaults
- Use RandomizedSearchCV, small learning_rate, and early stopping via fit params
- Tune tree depth/num_leaves, subsampling, regularization; monitor validation loss/metric

## Practice Flow
1) Run xgboost_basics.py and lightgbm_basics.py to see training/early stopping.
2) Explore catboost_categorical_handling.py for categorical columns.
3) Use early_stopping_and_regularization.py to compare regularization effects.
4) Inspect feature_importance_and_shap.py to understand model explanations.
5) Try hyperparameter_tuning_templates.py on your data to find good settings.
6) Read common_mistakes_and_best_practices.txt to avoid pitfalls.

Enjoy boosting!
