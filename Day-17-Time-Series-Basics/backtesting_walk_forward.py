"""
Walk-forward backtesting (rolling-origin evaluation)

Instead of one train/test split, we simulate repeated forecasting:

Fold i:
- train = [0 .. train_end]
- test  = (train_end .. train_end + horizon]
Then slide forward by 'step'.

This script:
- builds lag+rolling features safely (past only)
- runs backtesting for multiple folds
- reports MAE per fold + mean/std

Model: Ridge regression (fast baseline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


def make_series(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = (
        50
        + 0.015 * t
        + 2.2 * np.sin(2 * np.pi * t / 7)
        + 1.0 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 0.9, size=n)
    )
    return pd.DataFrame({"ds": dates, "y": y})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["ds"].dt.dayofweek

    # Lags
    for L in [1, 7, 14, 28]:
        out[f"lag_{L}"] = out["y"].shift(L)

    # rolling on past only
    y_past = out["y"].shift(1)
    out["roll_mean_14"] = y_past.rolling(14).mean()
    out["roll_std_14"] = y_past.rolling(14).std()
    out["roll_mean_56"] = y_past.rolling(56).mean()
    out["roll_std_56"] = y_past.rolling(56).std()

    return out.dropna().reset_index(drop=True)


def walk_forward_splits(n: int, initial_train: int, horizon: int, step: int):
    """
    yields (train_end, test_end) indices over a feature dataframe of length n.
    train = [0:train_end)
    test  = [train_end:test_end)
    """
    train_end = initial_train
    while train_end + horizon <= n:
        test_end = train_end + horizon
        yield train_end, test_end
        train_end += step


def main() -> None:
    df = make_series()
    data = make_features(df)

    feature_cols = [c for c in data.columns if c not in ("ds", "y")]
    X_all = data[feature_cols]
    y_all = data["y"]

    initial_train = 500
    horizon = 60
    step = 60

    maes = []
    for fold, (train_end, test_end) in enumerate(walk_forward_splits(len(data), initial_train, horizon, step), start=1):
        Xtr, ytr = X_all.iloc[:train_end], y_all.iloc[:train_end]
        Xte, yte = X_all.iloc[train_end:test_end], y_all.iloc[train_end:test_end]

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = float(mean_absolute_error(yte, pred))
        maes.append(mae)

        print(f"Fold {fold:02d} | train_end={train_end} test_end={test_end} | MAE={mae:.4f}")

    maes = np.array(maes)
    print("\nBacktesting summary:")
    print("  folds:", len(maes))
    print("  MAE mean:", round(float(maes.mean()), 4))
    print("  MAE std :", round(float(maes.std(ddof=1)) if len(maes) > 1 else 0.0, 4))

    print("\nWhy this matters:")
    print("- Single split can be lucky/unlucky.")
    print("- Walk-forward shows performance stability over time and regime changes.")


if __name__ == "__main__":
    main()
