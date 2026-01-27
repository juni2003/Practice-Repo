"""
Feature Importance & SHAP

- Trains XGBoost on breast cancer dataset
- Shows gain-based importance
- Computes SHAP values (if shap installed); falls back to permutation importance if not

Requires: xgboost, shap (optional), scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

    model = XGBClassifier(
        n_estimators=4000, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        tree_method="hist", eval_metric="logloss", random_state=42
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=100, verbose=False)
    y_proba = model.predict_proba(Xte)[:, 1]
    print("ROC-AUC:", roc_auc_score(yte, y_proba))

    # Gain importance
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = np.array([gain.get(f"f{i}", 0.0) for i in range(X.shape[1])])
    order = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp[order], y=[feature_names[i] for i in order], orient="h", color="steelblue")
    plt.title("XGBoost Gain Importance (Top 15)")
    plt.xlabel("Gain")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # SHAP if available
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xte)
        shap.summary_plot(shap_vals, Xte, feature_names=feature_names, show=True)
    except Exception as e:
        print("SHAP unavailable or failed:", e)
        print("Falling back to permutation importance...")
        pi = permutation_importance(model, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1][:15]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=pi.importances_mean[order], y=[feature_names[i] for i in order], orient="h", color="darkgreen")
        plt.title("Permutation Importance (Top 15)")
        plt.xlabel("Mean importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
