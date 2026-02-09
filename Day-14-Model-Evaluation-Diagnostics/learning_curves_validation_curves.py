"""
Learning Curves & Validation Curves

- Dataset: digits (classification)
- Model: Pipeline(StandardScaler + SVC)
- Learning curve: score vs training set size
- Validation curve: score vs hyperparameter C
- Guidance printed to console

Notes:
- Learning curves indicate if more data would help (gap between train/validation curves).
- Validation curves indicate under/over-regularization trends.

Requirements: scikit-learn, numpy, matplotlib, seaborn
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve


def plot_learning_curve(pipe, X, y, cv):
    train_sizes = np.linspace(0.1, 1.0, 7)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=cv, scoring="accuracy", train_sizes=train_sizes, n_jobs=-1, shuffle=True, random_state=42
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Train", color="tab:blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="tab:blue")

    plt.plot(train_sizes, val_mean, "o-", label="Validation", color="tab:orange")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="tab:orange")

    plt.title("Learning Curve (SVC + StandardScaler)")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print("Plot display skipped:", e)


def plot_validation_curve(pipe, X, y, cv):
    C_values = np.logspace(-3, 2, 8)
    val_scores = validation_curve(
        pipe, X, y, param_name="svc__C", param_range=C_values,
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    mean = val_scores.mean(axis=1)
    std = val_scores.std(axis=1)

    plt.figure(figsize=(7, 5))
    plt.semilogx(C_values, mean, "o-", color="tab:green", label="Validation")
    plt.fill_between(C_values, mean - std, mean + std, alpha=0.2, color="tab:green")
    plt.title("Validation Curve (C for SVC)")
    plt.xlabel("C (log scale)")
    plt.ylabel("Accuracy")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print("Plot display skipped:", e)

    # Guidance
    best_idx = np.argmax(mean)
    print(f"Best C by validation curve: {C_values[best_idx]:.4f} (acc={mean[best_idx]:.4f})")
    print("Trend: low C (strong regularization) → higher bias; high C (weak regularization) → potential overfitting.")


def main():
    data = load_digits()
    X, y = data.data, data.target

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma="scale", C=1.0, probability=False, random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plot_learning_curve(pipe, X, y, cv)
    plot_validation_curve(pipe, X, y, cv)


if __name__ == "__main__":
    main()
