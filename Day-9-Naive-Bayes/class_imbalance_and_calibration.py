"""
Class Imbalance & Probability Calibration with Naive Bayes

Covers:
- Creating an imbalanced dataset
- Using sample_weight to reflect class weights in NB fit
- Calibrating probabilities with CalibratedClassifierCV
- Evaluating with precision, recall, F1, Brier score
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    brier_score_loss,
    accuracy_score,
)
import matplotlib.pyplot as plt


def build_imbalanced(n_samples: int = 1000, weights=(0.9, 0.1)):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        weights=list(weights),
        flip_y=0.01,
        random_state=42,
    )
    return X, y


def sample_weights_for_classes(y: np.ndarray, class_weight: dict[int, float]):
    return np.array([class_weight[label] for label in y], dtype=float)


def run():
    X, y = build_imbalanced(n_samples=1200, weights=(0.92, 0.08))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Baseline GaussianNB (no weights)
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
    print("Baseline Report:\n", classification_report(y_test, y_pred))

    # Weighted fit: upweight minority class
    weights = sample_weights_for_classes(y_train, class_weight={0: 1.0, 1: 8.0})
    gnb_w = GaussianNB(var_smoothing=1e-9)
    gnb_w.fit(X_train, y_train, sample_weight=weights)
    y_pred_w = gnb_w.predict(X_test)
    print("\nWeighted Fit Accuracy:", accuracy_score(y_test, y_pred_w))
    print("Weighted Report:\n", classification_report(y_test, y_pred_w))

    # Probability calibration (sigmoid)
    # Note: GaussianNB provides predict_proba, but calibration can reduce over/under-confidence
    calib = CalibratedClassifierCV(base_estimator=GaussianNB(var_smoothing=1e-9), cv=5, method="sigmoid")
    calib.fit(X_train, y_train)
    y_proba = calib.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_proba >= 0.5).astype(int)

    print("\nCalibrated Accuracy:", accuracy_score(y_test, y_pred_cal))
    print("Calibrated Report:\n", classification_report(y_test, y_pred_cal))
    print("Brier score (lower is better):", brier_score_loss(y_test, y_proba))

    # Reliability (calibration) curve
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, "o-", label="Calibrated NB")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Sigmoid Calibrated GaussianNB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
