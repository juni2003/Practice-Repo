"""
SHAP interpretation (global vs local)

This script trains a tree model and uses SHAP to explain:
- Global: summary plot (feature importance + direction)
- Local: explanation for a single example

SHAP is optional. If it's not installed, the script exits with instructions.

Install:
  pip install shap

Notes:
- For tree models, TreeExplainer is fast.
- For linear models, LinearExplainer may be used.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
except ImportError:
    raise SystemExit("SHAP not installed. Run: pip install shap")


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    print("Test ROC-AUC:", round(float(roc_auc_score(yte, proba)), 4))

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model)
    # SHAP values for class 1
    shap_values = explainer.shap_values(Xte)
    # Depending on shap version, shap_values can be list[class] or array
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ---- Global explanation: summary plot ----
    # Shows which features have largest absolute SHAP values (importance),
    # and the direction of effect (color encodes feature value).
    shap.summary_plot(sv, Xte, feature_names=feature_names, show=False)
    plt.title("Global SHAP Summary (class=1)")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    # ---- Local explanation: one row ----
    i = 0
    print("\nLocal explanation for sample index", i)
    # Waterfall plot is good for single predictions (why this row)
    shap.plots.waterfall(shap.Explanation(values=sv[i], base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                                         data=Xte[i], feature_names=feature_names), show=False)
    plt.title("Local SHAP Waterfall")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    print("\nHow to read SHAP:")
    print("- Positive SHAP value pushes prediction toward class 1.")
    print("- Negative SHAP value pushes prediction away from class 1.")
    print("- Global summary: features with biggest absolute SHAP are most influential overall.")


if __name__ == "__main__":
    main()
