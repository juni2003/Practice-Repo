"""
LightGBM Basics — Classification with Early Stopping, Regularization, and Importance

- Synthetic dataset (binary classification)
- Early stopping on validation set
- Prints accuracy, ROC-AUC, best_iteration
- Plots feature importance

Requires: lightgbm, scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier


def main(plot_importance: bool = True):
    X, y = make_classification(
        n_samples=3000, n_features=25, n_informative=10, n_redundant=5,
        class_sep=1.0, random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

    model = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,              # -1 means no limit
        min_child_samples=20,
        subsample=0.8,             # bagging_fraction in core API
        colsample_bytree=0.8,      # feature_fraction in core API
        reg_lambda=1.0,            # L2
        reg_alpha=0.0,             # L1
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        eval_metric="logloss",
        early_stopping_rounds=100,
        verbose=False
    )

    print("Best iteration:", model.best_iteration_)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    print("Accuracy:", accuracy_score(yte, y_pred))
    print("ROC-AUC:", roc_auc_score(yte, y_proba))

    if plot_importance:
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=imp[order], y=[f"f{idx}" for idx in order], orient="h", color="purple")
        plt.title("LightGBM Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(plot_importance=True)
