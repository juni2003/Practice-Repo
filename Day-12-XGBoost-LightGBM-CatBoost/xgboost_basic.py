"""
XGBoost Basics — Classification with Early Stopping, Regularization, and Importance

- Synthetic dataset (binary classification)
- Early stopping on validation set
- Prints accuracy, ROC-AUC, best_iteration
- Plots feature importance (gain)

Requires: xgboost, scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def main(plot_importance: bool = True):
    X, y = make_classification(
        n_samples=3000, n_features=25, n_informative=10, n_redundant=5,
        class_sep=1.0, random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

    model = XGBClassifier(
        n_estimators=3000,
        max_depth=4,
        min_child_weight=1.0,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        early_stopping_rounds=100,
        verbose=False
    )

    print("Best iteration:", model.best_iteration_)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    print("Accuracy:", accuracy_score(yte, y_pred))
    print("ROC-AUC:", roc_auc_score(yte, y_proba))

    if plot_importance:
        booster = model.get_booster()
        importance = booster.get_score(importance_type="gain")
        # Map feature indices to importance
        imp = np.zeros(X.shape[1])
        for k, v in importance.items():
            # k is 'f{idx}'
            idx = int(k[1:])
            imp[idx] = v
        order = np.argsort(imp)[::-1][:20]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=imp[order], y=[f"f{idx}" for idx in order], orient="h", color="teal")
        plt.title("XGBoost Feature Importance (gain)")
        plt.xlabel("Gain")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(plot_importance=True)
