"""
Bias–Variance Demo with Polynomial Regression

- Dataset: synthetic regression with noise
- Model: Pipeline(PolynomialFeatures + StandardScaler + LinearRegression)
- Sweep polynomial degree and plot train/test MSE

Interpretation:
- Low degree → high bias (underfitting): both errors high, small gap.
- Very high degree → high variance (overfitting): train error low, test error high.

Requirements: scikit-learn, numpy, matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def build_data(n=600, noise=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    # True function: sin(x) + 0.3x
    y = np.sin(X[:, 0]) + 0.3 * X[:, 0] + rng.normal(0, noise, size=n)
    return X, y


def main():
    X, y = build_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = list(range(1, 21))
    train_mse, test_mse = [], []

    for d in degrees:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ])
        model.fit(Xtr, ytr)
        train_mse.append(mean_squared_error(ytr, model.predict(Xtr)))
        test_mse.append(mean_squared_error(yte, model.predict(Xte)))

    plt.figure(figsize=(7, 5))
    plt.plot(degrees, train_mse, "o--", label="Train MSE", color="tab:blue")
    plt.plot(degrees, test_mse, "o-", label="Test MSE", color="tab:orange")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("Bias–Variance via Polynomial Degree")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print("Plot display skipped:", e)

    # Print guidance
    best_d = degrees[int(np.argmin(test_mse))]
    print(f"Lowest test MSE at degree={best_d}. If train MSE << test MSE at high degrees, that indicates high variance (overfitting).")


if __name__ == "__main__":
    main()
