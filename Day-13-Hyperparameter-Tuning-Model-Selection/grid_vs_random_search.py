"""
GridSearchCV vs RandomizedSearchCV — Comparison

- Dataset: synthetic binary classification
- Model: Pipeline(StandardScaler + LogisticRegression)
- Metrics: ROC-AUC with StratifiedKFold CV
- Outputs: time taken, best params, best CV score, test ROC-AUC

Note:
- Grid search is exhaustive on given grid (can be slow).
- Random search samples candidates (faster, good for large spaces).
"""

from __future__ import annotations
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def build_data(seed=42):
    X, y = make_classification(
        n_samples=3000, n_features=20, n_informative=8, n_redundant=4,
        class_sep=1.0, weights=[0.6, 0.4], random_state=seed
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    return Xtr, Xte, ytr, yte


def grid_search(Xtr, ytr, cv):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
    ])
    # Fairly small grid — robust but not huge
    params = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l2"],  # lbfgs supports only l2
        "clf__class_weight": [None, "balanced"]
    }
    t0 = time.time()
    gs = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(Xtr, ytr)
    dt = time.time() - t0
    return gs, dt


def random_search(Xtr, ytr, cv):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
    ])
    # Sample candidate values; RandomizedSearchCV will pick n_iter samples
    C_candidates = np.logspace(-3, 2, 50)  # 0.001 to 100
    params = {
        "clf__C": C_candidates,
        "clf__penalty": ["l2"],
        "clf__class_weight": [None, "balanced"]
    }
    t0 = time.time()
    rs = RandomizedSearchCV(pipe, params, n_iter=20, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42, verbose=0)
    rs.fit(Xtr, ytr)
    dt = time.time() - t0
    return rs, dt


def main():
    Xtr, Xte, ytr, yte = build_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs, t_grid = grid_search(Xtr, ytr, cv)
    rs, t_rand = random_search(Xtr, ytr, cv)

    # Evaluate on test
    y_proba_gs = gs.predict_proba(Xte)[:, 1]
    y_proba_rs = rs.predict_proba(Xte)[:, 1]
    auc_gs = roc_auc_score(yte, y_proba_gs)
    auc_rs = roc_auc_score(yte, y_proba_rs)

    print("GridSearchCV:")
    print("  Time (s):", round(t_grid, 2))
    print("  Best params:", gs.best_params_)
    print("  Best CV ROC-AUC:", round(gs.best_score_, 4))
    print("  Test ROC-AUC:", round(auc_gs, 4))

    print("\nRandomizedSearchCV:")
    print("  Time (s):", round(t_rand, 2))
    print("  Best params:", rs.best_params_)
    print("  Best CV ROC-AUC:", round(rs.best_score_, 4))
    print("  Test ROC-AUC:", round(auc_rs, 4))


if __name__ == "__main__":
    main()
