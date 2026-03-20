"""
AR, MA, ARIMA basics (minimal implementation + intuition)

Goal:
- Show what AR and MA mean in practice.
- Implement a simple AR(p) fit via linear regression on lag features.
- Show how differencing (d) turns AR into an ARIMA-like workflow conceptually.

This is NOT a full professional ARIMA implementation.
For full ARIMA estimation, typically use statsmodels.
But this script teaches the mechanics:
- AR(p) = regression on past values
- differencing makes non-stationary series closer to stationary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def simulate_ar1(n: int = 600, phi: float = 0.75, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=float)
    eps = rng.normal(0, 1.0, size=n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return pd.Series(y, name="y")


def make_supervised(y: pd.Series, lags: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({"y": y})
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)
    df = df.dropna()
    X = df[[f"lag_{L}" for L in lags]]
    target = df["y"]
    return X, target


def fit_ar(y: pd.Series, p: int, train_frac: float = 0.75) -> None:
    lags = list(range(1, p + 1))
    X, target = make_supervised(y, lags)

    split = int(len(X) * train_frac)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = target.iloc[:split], target.iloc[split:]

    model = LinearRegression()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print(f"AR({p}) | MAE:", round(float(mean_absolute_error(yte, pred)), 4))
    print("Estimated coefficients (phi_1..phi_p):", [round(float(c), 4) for c in model.coef_])


def main() -> None:
    print("=== AR example: simulate AR(1) and fit AR(p) ===")
    y_ar = simulate_ar1()
    fit_ar(y_ar, p=1)
    fit_ar(y_ar, p=3)

    print("\n=== ARIMA concept: differencing + AR fit ===")
    # Create a trending series (non-stationary)
    t = np.arange(700)
    y_trend = pd.Series(0.05 * t + np.sin(2 * np.pi * t / 30) + np.random.default_rng(42).normal(0, 0.8, size=len(t)))

    # AR on original (often poor because trend violates stationarity assumptions)
    print("AR on raw trending series:")
    fit_ar(y_trend, p=5)

    # Differencing corresponds to the 'I' in ARIMA
    y_diff = y_trend.diff(1).dropna()
    print("AR on differenced series (ARIMA-like idea):")
    fit_ar(y_diff, p=5)

    print("\nMA intuition (concept only):")
    print("- MA(q) models y[t] as depending on current and past error terms ε[t-1..t-q].")
    print("- Fitting MA properly needs iterative estimation (not just simple regression on y-lags).")
    print("- In practice, use statsmodels ARIMA/SARIMAX if you need full ARIMA.")


if __name__ == "__main__":
    main()
