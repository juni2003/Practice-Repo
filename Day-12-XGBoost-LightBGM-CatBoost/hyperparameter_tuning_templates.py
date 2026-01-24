"""
Hyperparameter Tuning Templates for XGBoost, LightGBM, CatBoost

- RandomizedSearchCV templates
- Early stopping via fit params with a validation set
- Parameter ranges based on common practices

Note:
- Early stopping inside CV is tricky; here we split train/valid and pass eval_set to fit().
- For larger datasets or more robust searches, consider libraries like Optuna.
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def xgb_search(Xtr, ytr, Xval, yval):
    xgb = XGBClassifier(
        tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42
    )
    params = {
        "n_estimators": [800, 1200, 2000],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.5, 1.0],
        "gamma": [0.0, 0.1, 0.2],
    }
    rs = RandomizedSearchCV(
        xgb, params, n_iter=20, scoring="roc_auc", cv=3, verbose=0, n_jobs=-1, random_state=42
    )
    # Pass eval_set for early stopping to fit()
    rs.fit(Xtr, ytr, **{"eval_set": [(Xval, yval)], "early_stopping_rounds": 50, "verbose": False})
    print("XGB best params:", rs.best_params_)
    return rs.best_estimator_


def lgb_search(Xtr, ytr, Xval, yval):
    lgb = LGBMClassifier(random_state=42, n_jobs=-1)
    params = {
        "n_estimators": [1000, 2000, 4000],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 6, 8],
        "min_child_samples": [10, 20, 40],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.5, 1.0],
    }
    rs = RandomizedSearchCV(
        lgb, params, n_iter=20, scoring="roc_auc", cv=3, verbose=0, n_jobs=-1, random_state=42
    )
    rs.fit(Xtr, ytr, **{"eval_set": [(Xval, yval)], "eval_metric": "logloss", "early_stopping_rounds": 50, "verbose": False})
    print("LGB best params:", rs.best_params_)
    return rs.best_estimator_


def cat_search(Xtr, ytr, Xval, yval):
    # CatBoost's sklearn wrapper can be tuned via RandomizedSearchCV; early stopping passed via fit params.
    cat = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", verbose=False, random_state=42
    )
    params = {
        "iterations": [1000, 2000, 4000],
        "learning_rate": [0.02, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1.0, 3.0, 5.0],
        "bootstrap_type": ["Bayesian", "Bernoulli"],
        "subsample": [0.7, 0.8, 0.9],
    }
    rs = RandomizedSearchCV(
        cat, params, n_iter=20, scoring="roc_auc", cv=3, verbose=0, n_jobs=-1, random_state=42
    )
    rs.fit(Xtr, ytr, **{"eval_set": (Xval, yval), "early_stopping_rounds": 50, "verbose": False})
    print("CAT best params:", rs.best_params_)
    return rs.best_estimator_


def demo():
    X, y = make_classification(n_samples=5000, n_features=30, n_informative=12, n_redundant=6, random_state=42)
    Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(Xtr_all, ytr_all, test_size=0.25, stratify=ytr_all, random_state=42)

    print("Searching XGBoost...")
    xgb_best = xgb_search(Xtr, ytr, Xval, yval)
    auc_xgb = roc_auc_score(yte, xgb_best.predict_proba(Xte)[:, 1])
    print("XGB AUC:", auc_xgb)

    print("\nSearching LightGBM...")
    lgb_best = lgb_search(Xtr, ytr, Xval, yval)
    auc_lgb = roc_auc_score(yte, lgb_best.predict_proba(Xte)[:, 1])
    print("LGB AUC:", auc_lgb)

    print("\nSearching CatBoost...")
    cat_best = cat_search(Xtr, ytr, Xval, yval)
    auc_cat = roc_auc_score(yte, cat_best.predict_proba(Xte)[:, 1])
    print("CAT AUC:", auc_cat)


if __name__ == "__main__":
    demo()
