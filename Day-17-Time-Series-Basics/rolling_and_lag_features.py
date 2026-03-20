"""
Rolling features & lagged variables (turn time series into supervised ML)

We build a synthetic series and create:
- lag features: y[t-1], y[t-7], y[t-14]
- rolling mean/std: mean(y[t-7..t-1]), std(y[t-7..t-1])
- calendar-like features: day-of-week from an artificial date index

Then we train a model to predict y[t] using past data only.

Key leakage guard:
- Rolling features must be computed using ONLY PAST values.
  Use shift(1) before rolling to avoid including y[t] in its own features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def make_daily_series(n_days: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    weekly = 2.0 * np.sin(2 * np.pi * t / 7)
    yearly = 1.2 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.01 * t
    noise = rng.normal(0, 0.7, size=n_days)

    y = 20 + trend + weekly + yearly + noise
    return pd.DataFrame({"ds": dates, "y": y})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["ds"].dt.dayofweek  # 0=Mon..6=Sun

    # Lags
    for L in [1, 7, 14, 28]:
        out[f"lag_{L}"] = out["y"].shift(L)

    # Rolling stats: IMPORTANT to shift(1) so the window ends at t-1 (past only)
    y_past = out["y"].shift(1)
    out["roll_mean_7"] = y_past.rolling(7).mean()
    out["roll_std_7"] = y_past.rolling(7).std()
    out["roll_mean_28"] = y_past.rolling(28).mean()
    out["roll_std_28"] = y_past.rolling(28).std()

    return out.dropna().reset_index(drop=True)


def main() -> None:
    df = make_daily_series()
    data = make_features(df)

    feature_cols = [c for c in data.columns if c not in ("ds", "y")]
    X = data[feature_cols]
    y = data["y"]

    # Time split (no shuffle)
    split = int(len(data) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)

    print("Features used:", feature_cols)
    print("Test MAE:", round(float(mae), 4))

    # Show top importances
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(imp.head(10).round(4))


if __name__ == "__main__":
    main()
