"""
Bayesian Optimization with Optuna — HistGradientBoostingClassifier

- Objective: maximize CV ROC-AUC via StratifiedKFold
- Search space: learning_rate, max_depth, max_leaf_nodes, min_samples_leaf, l2_regularization
- Prints best trial params and score

Install:
    pip install optuna scikit-learn

Note:
- We disable estimator's early stopping inside CV to avoid nested validation leakage.
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import optuna
except ImportError:
    raise SystemExit("Please install optuna: pip install optuna")


def build_data(seed=42):
    X, y = make_classification(
        n_samples=5000, n_features=30, n_informative=12, n_redundant=6,
        class_sep=1.0, random_state=seed
    )
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def objective(trial: "optuna.trial.Trial", Xtr: np.ndarray, ytr: np.ndarray) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
    }
    clf = HistGradientBoostingClassifier(
        **params,
        early_stopping=False,  # avoid nested validation during CV
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def main(n_trials: int = 30, seed: int = 42):
    Xtr, Xte, ytr, yte = build_data(seed)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, Xtr, ytr), n_trials=n_trials)

    print("Best trial:")
    print("  Value (CV ROC-AUC):", round(study.best_value, 4))
    print("  Params:", study.best_params)

    # Train final model on full training set with best params
    best = HistGradientBoostingClassifier(
        **study.best_params, early_stopping=False, random_state=42
    ).fit(Xtr, ytr)
    y_proba = best.predict_proba(Xte)[:, 1]
    print("Test ROC-AUC:", round(roc_auc_score(yte, y_proba), 4))


if __name__ == "__main__":
    main(n_trials=30)
