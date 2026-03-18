"""
Time split & leakage demo

This script shows why random train/test split is WRONG for time series.

We generate a synthetic time series:
- trend + weekly seasonality + noise

Then we compare:
1) Random split (leaky) -> unrealistically good results
2) Time-based split -> realistic results

Model: LinearRegression on lag features

Key learning:
- The ONLY safe way is chronological split (or walk-forward backtesting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def make_series(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    trend = 0.03 * t
    weekly = 1.5 * np.sin(2 * np.pi * t / 7.0)
    noise = rng.normal(0, 0.6, size=n)

    y = 10 + trend + weekly + noise
    return pd.DataFrame({"t": t, "y": y})


def add_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out["y"].shift(L)
    return out.dropna().reset_index(drop=True)


def fit_eval(X_train, y_train, X_test, y_test) -> float:
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(mean_absolute_error(y_test, pred))


def main() -> None:
    df = make_series()
    df_feat = add_lags(df, lags=[1, 2, 7, 14])

    X = df_feat[[c for c in df_feat.columns if c.startswith("lag_")]]
    y = df_feat["y"]

    # 1) WRONG: random split (leaks future patterns into training)
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
    mae_random = fit_eval(Xtr_r, ytr_r, Xte_r, yte_r)

    # 2) RIGHT: time split (train early, test later)
    split = int(len(df_feat) * 0.75)
    Xtr_t, Xte_t = X.iloc[:split], X.iloc[split:]
    ytr_t, yte_t = y.iloc[:split], y.iloc[split:]
    mae_time = fit_eval(Xtr_t, ytr_t, Xte_t, yte_t)

    print("MAE with RANDOM split (leaky):", round(mae_random, 4))
    print("MAE with TIME split (correct):", round(mae_time, 4))
    print("\nInterpretation:")
    print("- Random split usually looks better because it mixes time periods.")
    print("- Time split tests on the future, which is what forecasting requires.")


if __name__ == "__main__":
    main()
