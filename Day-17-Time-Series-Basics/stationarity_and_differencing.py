"""
Stationarity & differencing (practical intuition)

We generate a non-stationary time series (trend + seasonality),
then apply:
- first differencing
- seasonal differencing

We do not rely on external stationarity tests to keep dependencies minimal.
Instead we show:
- rolling mean / rolling std changes over time

Key learning:
- Differencing can remove trend/seasonality and make series more stationary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_series(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    season = 2.0 * np.sin(2 * np.pi * t / 12.0)  # period 12
    noise = rng.normal(0, 0.8, size=n)
    y = 20 + trend + season + noise
    return pd.Series(y, name="y")


def rolling_stats(y: pd.Series, win: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": y,
            "roll_mean": y.rolling(win).mean(),
            "roll_std": y.rolling(win).std(),
        }
    )


def plot_series(df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(11, 4))
    plt.plot(df["y"], label="series", linewidth=1.2)
    plt.plot(df["roll_mean"], label="rolling mean", linewidth=2.0)
    plt.plot(df["roll_std"], label="rolling std", linewidth=2.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    y = make_series()

    base = rolling_stats(y)
    plot_series(base, "Original series (non-stationary: trend + seasonality)")

    # First difference removes trend (often)
    diff1 = y.diff(1).dropna()
    d1 = rolling_stats(diff1)
    plot_series(d1, "1st difference: y[t] - y[t-1]")

    # Seasonal difference removes seasonality (period=12)
    diff_season = y.diff(12).dropna()
    ds = rolling_stats(diff_season)
    plot_series(ds, "Seasonal difference (period=12): y[t] - y[t-12]")

    # Combine (often used in seasonal ARIMA style)
    diff_both = y.diff(12).diff(1).dropna()
    db = rolling_stats(diff_both)
    plot_series(db, "Seasonal + 1st difference")

    print("Notes:")
    print("- If rolling mean/std changes a lot over time, the series is likely non-stationary.")
    print("- Differencing often stabilizes mean; seasonal differencing targets repeating cycles.")


if __name__ == "__main__":
    main()
