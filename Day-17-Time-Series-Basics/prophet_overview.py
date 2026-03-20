"""
Prophet overview (optional)

This file is intentionally lightweight and safe:
- If Prophet isn't installed, it prints how to install and exits.
- Uses a simple synthetic daily series with weekly + yearly seasonality.

Prophet is an additive model:
  y(t) = trend(t) + seasonality(t) + holidays(t) + error

It is often useful for business time series where:
- multiple seasonalities exist
- missing dates happen
- you want a fast strong baseline

Install:
  pip install prophet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    raise SystemExit("Prophet not installed. Run: pip install prophet (optional script).")


def make_data(n: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)

    y = (
        100
        + 0.03 * t
        + 4.0 * np.sin(2 * np.pi * t / 7)
        + 2.0 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 1.2, size=n)
    )

    return pd.DataFrame({"ds": ds, "y": y})


def main() -> None:
    df = make_data()

    # time split: last 90 days as test
    train = df.iloc[:-90].copy()
    test = df.iloc[-90:].copy()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=90, freq="D")
    forecast = m.predict(future)

    # Evaluate on the last 90 days (simple holdout)
    pred = forecast.iloc[-90:]["yhat"].values
    mae = float(np.mean(np.abs(test["y"].values - pred)))
    print("Prophet holdout MAE (last 90 days):", round(mae, 4))

    print("\nTip:")
    print("- For real projects, do walk-forward backtesting instead of one holdout.")


if __name__ == "__main__":
    main()
