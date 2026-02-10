"""
Error Analysis Workflow

- Dataset: make_classification with a 'feature' that can be used for slicing
- Model: Pipeline(StandardScaler + LogisticRegression)
- Outputs:
  - Confusion matrix and classification report
  - Slice-based metrics: split by threshold on a chosen feature
  - Visual confusion matrix (seaborn heatmap)

Requirements: scikit-learn, numpy, seaborn, matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


def slice_metrics(Xte: np.ndarray, yte: np.ndarray, ypred: np.ndarray, feature_index: int = 0, threshold: float = 0.0):
    mask_lo = Xte[:, feature_index] <= threshold
    mask_hi = ~mask_lo
    for name, mask in [("slice <= threshold", mask_lo), ("slice > threshold", mask_hi)]:
        if mask.sum() == 0:
            continue
        acc = accuracy_score(yte[mask], ypred[mask])
        pr, rc, f1, _ = precision_recall_fscore_support(yte[mask], ypred[mask], average="binary", zero_division=0)
        print(f"  {name}: n={mask.sum()}, acc={acc:.3f}, precision={pr:.3f}, recall={rc:.3f}, f1={f1:.3f}")


def main():
    # Create data with a feature that influences class to enable slicing
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=5, n_redundant=2,
        class_sep=1.0, weights=[0.6, 0.4], random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
    ])
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)

    # Confusion matrix and report
    cm = confusion_matrix(yte, ypred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(yte, ypred, digits=3))

    # Slice-based analysis on feature 0
    thr = np.median(Xte[:, 0])
    print(f"Slice-based metrics using feature[0] threshold={thr:.3f}:")
    slice_metrics(Xte, yte, ypred, feature_index=0, threshold=thr)

    # Visualize confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print("Plot display skipped:", e)


if __name__ == "__main__":
    main()
