"""
Calibration & Confidence Intervals

- Train a classifier prone to miscalibration (GaussianNB) and calibrate it using CalibratedClassifierCV
- Plot reliability diagram (calibration curve)
- Bootstrap confidence intervals for ROC-AUC

Requirements: scikit-learn, numpy, matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    X, y = make_classification(
        n_samples=3000, n_features=20, n_informative=8, n_redundant=4,
        class_sep=1.0, weights=[0.6, 0.4], random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Baseline (uncalibrated)
    base = GaussianNB(var_smoothing=1e-9)
    base.fit(Xtr, ytr)
    proba_base = base.predict_proba(Xte)[:, 1]
    auc_base = roc_auc_score(yte, proba_base)

    # Calibrated (sigmoid)
    calib = CalibratedClassifierCV(base_estimator=GaussianNB(var_smoothing=1e-9), method="sigmoid", cv=5)
    calib.fit(Xtr, ytr)
    proba_cal = calib.predict_proba(Xte)[:, 1]
    auc_cal = roc_auc_score(yte, proba_cal)

    print(f"ROC-AUC (uncalibrated): {auc_base:.4f}")
    print(f"ROC-AUC (calibrated)  : {auc_cal:.4f}")

    # Bootstrap CI for AUC
    ci_base = bootstrap_ci(yte, proba_base, roc_auc_score)
    ci_cal = bootstrap_ci(yte, proba_cal, roc_auc_score)
    print(f"95% CI AUC (uncalibrated): [{ci_base[0]:.4f}, {ci_base[1]:.4f}]")
    print(f"95% CI AUC (calibrated)  : [{ci_cal[0]:.4f}, {ci_cal[1]:.4f}]")

    # Reliability diagram
    frac_pos_base, mean_pred_base = calibration_curve(yte, proba_base, n_bins=10, strategy="uniform")
    frac_pos_cal, mean_pred_cal = calibration_curve(yte, proba_cal, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred_base, frac_pos_base, "o-", label="Uncalibrated")
    plt.plot(mean_pred_cal, frac_pos_cal, "o-", label="Calibrated (sigmoid)")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print("Plot display skipped:", e)


if __name__ == "__main__":
    main()
