# Day 11: Bagging & Boosting (Foundations)

## Overview
Ensembles combine many “weak” learners to form a stronger model. Two foundational paradigms:

- Bagging (Bootstrap Aggregating): trains many models independently on bootstrapped samples and averages their predictions. Reduces variance.
- Boosting: trains models sequentially; each new model focuses on correcting previous errors. Reduces bias (while controlling variance).

This module covers:
- Bagging vs Pasting, bias–variance tradeoff, out-of-bag (OOB) evaluation
- AdaBoost intuition and exponential loss
- Gradient Boosting: weak learners on residuals
- Model comparison vs single trees
- Practical guidance and pitfalls

## Why Ensembles?
- Robust baselines that often outperform single models
- Handle nonlinearity and interactions (via trees)
- Flexible (classification, regression), widely applicable

Limitations:
- Less interpretable than single models
- Training time/memory increases with number of estimators
- Boosting can overfit if not tuned (learning rate, depth)

## Module Structure
```
Day-11-Bagging-Boosting/
├── README.md                                     # This file
├── bagging_vs_pasting.py                         # Compare bagging vs pasting + OOB
├── bias_variance_tradeoff_visualization.py       # Curves vs n_estimators/base depth
├── adaboost_from_scratch.py                      # Educational AdaBoost (binary, decision stumps)
├── gradient_boosting_regression_from_scratch.py  # Educational GB for regression (residual fitting)
├── sklearn_ensembles_examples.py                 # Bagging, RF, AdaBoost, GBT comparisons
└── common_mistakes_and_best_practices.txt        # Pitfalls and tips
```

## Quick Starts

Bagging (with OOB score)
```python
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None, random_state=42),
    n_estimators=200, max_samples=0.8, bootstrap=True, oob_score=True, random_state=42
)
bag.fit(Xtr, ytr)
print("OOB score:", bag.oob_score_)
print("Test accuracy:", accuracy_score(yte, bag.predict(Xte)))
```

AdaBoost (stumps)
```python
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=200, learning_rate=0.5, random_state=42
)
ada.fit(Xtr, ytr)
print("Test accuracy:", accuracy_score(yte, ada.predict(Xte)))
```

Gradient Boosting (classification)
```python
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(Xtr, ytr)
print("Test accuracy:", accuracy_score(yte, gb.predict(Xte)))
```

## Concepts in a Nutshell
- Bagging vs Pasting: both resample data for each estimator; bagging uses bootstrap (sampling with replacement), pasting uses without replacement.
- Bias–Variance Tradeoff:
  - Bagging reduces variance by averaging many high-variance models (e.g., deep trees).
  - Boosting reduces bias by focusing sequentially on hard examples.
- AdaBoost:
  - Uses weighted errors to update sample weights: harder examples get higher weights.
  - Learner weight α_t ≈ 0.5 ln((1 − err_t)/err_t).
- Gradient Boosting:
  - Fits new learners to current pseudo-residuals (gradients of the loss).
  - For squared error (regression): residuals = y − prediction.
  - For log-loss (classification): residuals are negative gradients in probability space.

## When to Use
- Bagging/Random Forests: strong out-of-the-box classifiers/regressors; robust, less tuning.
- AdaBoost: works well on clean, low-noise data; sensitive to outliers.
- Gradient Boosting: powerful and flexible; tune learning_rate, n_estimators, depth.

## Practice Flow
1) Run bagging_vs_pasting.py to see resampling and OOB.
2) Explore bias_variance_tradeoff_visualization.py for curves vs estimators/depth.
3) Inspect adaboost_from_scratch.py to understand boosting weights and α.
4) Run gradient_boosting_regression_from_scratch.py to see residual fitting.
5) Compare scikit-learn ensembles in sklearn_ensembles_examples.py.
6) Review common_mistakes_and_best_practices.txt.

Happy ensembling!
