"""
Early Stopping Templates in Tuning

- HistGradientBoosting (sklearn): early_stopping and validation_fraction
- Optional XGBoost/LightGBM templates (if installed) using eval_set and early_stopping_rounds

Note:
- For cross-validation, prefer disabling model-internal early stopping to avoid nested validation leakage.
- For a single train/val split, early stopping is very effective.
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier


def sklearn_histgb_template():
    X, y = make_classification(n_samples=5000, n_features=30, n_informative=12, n_redundant=6, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Use internal early stopping with a validation fraction of the training split
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    )
    model.fit(Xtr, ytr)
    y_proba = model.predict_proba(Xval)[:, 1]
    print("HistGB (sklearn) AUC:", round(roc_auc_score(yval, y_proba), 4))


def optional_xgb_lgb_templates():
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print("XGBoost/LightGBM not available:", e)
        return

    X, y = make_classification(n_samples=5000, n_features=30, n_informative=12, n_redundant=6, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    xgb = XGBClassifier(
        n_estimators=5000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist",
        eval_metric="logloss", random_state=42
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=100, verbose=False)
    print("XGB best_iteration:", xgb.best_iteration_)

    lgb = LGBMClassifier(
        n_estimators=5000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42
    )
    lgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="logloss", early_stopping_rounds=100, verbose=False)
    print("LGB best_iteration:", lgb.best_iteration_)


def main():
    sklearn_histgb_template()
    optional_xgb_lgb_templates()


if __name__ == "__main__":
    main()
