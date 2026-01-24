"""
Early Stopping & Regularization — Side-by-Side

- Compares XGBoost, LightGBM, CatBoost on same synthetic dataset
- Uses early stopping on validation set
- Demonstrates common regularization knobs

Requires: xgboost, lightgbm, catboost, scikit-learn
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def build_data(seed=42):
    X, y = make_classification(
        n_samples=4000, n_features=30, n_informative=12, n_redundant=6,
        class_sep=1.0, random_state=seed
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    return (Xtr, ytr), (Xval, yval), (Xte, yte)


def run():
    (Xtr, ytr), (Xval, yval), (Xte, yte) = build_data()

    xgb = XGBClassifier(
        n_estimators=5000, learning_rate=0.05, max_depth=4,
        min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
        tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=100, verbose=False)
    y_proba_xgb = xgb.predict_proba(Xte)[:, 1]
    print(f"XGB: best_iter={xgb.best_iteration_}, AUC={roc_auc_score(yte, y_proba_xgb):.4f}, Acc={accuracy_score(yte, (y_proba_xgb>=0.5).astype(int)):.4f}")

    lgb = LGBMClassifier(
        n_estimators=8000, learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0, random_state=42, n_jobs=-1
    )
    lgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="logloss", early_stopping_rounds=100, verbose=False)
    y_proba_lgb = lgb.predict_proba(Xte)[:, 1]
    print(f"LGB: best_iter={lgb.best_iteration_}, AUC={roc_auc_score(yte, y_proba_lgb):.4f}, Acc={accuracy_score(yte, (y_proba_lgb>=0.5).astype(int)):.4f}")

    cat = CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, loss_function="Logloss", eval_metric="AUC",
        random_state=42, verbose=False
    )
    cat.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=100, verbose=False)
    y_proba_cat = cat.predict_proba(Xte)[:, 1]
    print(f"CAT: best_iter={cat.get_best_iteration()}, AUC={roc_auc_score(yte, y_proba_cat):.4f}, Acc={accuracy_score(yte, (y_proba_cat>=0.5).astype(int)):.4f}")


if __name__ == "__main__":
    run()
