# Day 17 — Time Series Basics

This day teaches the **core workflow for time series modeling**:

- How to split data for time (**no shuffling**)
- **Stationarity**, **differencing**, and why they matter
- AR, MA, ARIMA (practical understanding)
- Seasonality & a quick **Prophet overview** (conceptual + optional code)
- Feature engineering: **lags** and **rolling features**
- **Backtesting** (walk-forward validation) and evaluation

> Time series is different from “normal ML” because **future information must never leak into the past**.
> This affects splitting, cross-validation, feature engineering, and evaluation.

---

## Folder Structure

```
Day-17-Time-Series-Basics/
├── README.md
├── time_split_and_leakage_demo.py
├── stationarity_and_differencing.py
├── ar_ma_arima_basics.py
├── rolling_and_lag_features.py
├── backtesting_walk_forward.py
├── prophet_overview_optional.py
└── time_series_checklist.md
```

---

## Install

Minimum (everything except Prophet):
```bash
pip install numpy pandas scikit-learn matplotlib
```

Optional (only for Prophet file):
```bash
pip install prophet
```

> If Prophet install fails on your system, skip that script — it’s optional.

---

## How to Run

```bash
python Day-17-Time-Series-Basics/time_split_and_leakage_demo.py
python Day-17-Time-Series-Basics/stationarity_and_differencing.py
python Day-17-Time-Series-Basics/ar_ma_arima_basics.py
python Day-17-Time-Series-Basics/rolling_and_lag_features.py
python Day-17-Time-Series-Basics/backtesting_walk_forward.py
python Day-17-Time-Series-Basics/prophet_overview_optional.py
```

---

## 1) Train/Test splits for time (the most important rule)

### Wrong (leakage)
- Random train/test split shuffles time:
  - The model trains on “future” points and predicts the “past”.
  - Your metrics look amazing but fail in production.

### Right
- Split by time:
  - Train = early period
  - Test  = later period

### Walk-forward validation (backtesting)
Instead of one split, do multiple:
- Train on `[0..t]`, test on `(t..t+h]`
- Move forward and repeat

This simulates reality: at each point, you only know the past.

---

## 2) Stationarity & differencing

Many classical models (especially ARIMA) assume **stationarity**:
- mean roughly constant over time
- variance roughly constant
- autocorrelation structure stable

If a series has trend/seasonality, it’s often non-stationary.

**Differencing** is the simplest fix:
- 1st difference: `y[t] - y[t-1]` removes trend
- seasonal difference: `y[t] - y[t-s]` removes seasonality (period `s`)

---

## 3) AR, MA, ARIMA (intuition first)

### AR(p)
“Today depends on past values”:
- `y[t] = c + φ1 y[t-1] + ... + φp y[t-p] + noise`

### MA(q)
“Today depends on past shocks/errors”:
- `y[t] = c + ε[t] + θ1 ε[t-1] + ... + θq ε[t-q]`

### ARIMA(p,d,q)
- AR(p) + differencing `d` + MA(q)
- `d` helps stationarity

> In this module we implement AR/MA style fitting and show ARIMA as a concept.
> (Full ARIMA fitting is typically done with `statsmodels`—not required here.)

---

## 4) Rolling features & lagged variables (ML-style time series)

Instead of specialized time-series models, you can transform into a supervised dataset:

Features at time `t`:
- `lag_1 = y[t-1]`, `lag_7 = y[t-7]`
- rolling mean: `mean(y[t-7..t-1])`
- rolling std / min / max, etc.

Target:
- `y[t]` (or `y[t+h]` for forecasting horizon `h`)

Then use any ML regressor (linear, random forest, XGBoost, etc.)

---

## 5) Evaluation & Backtesting

Common regression metrics:
- MAE (robust and easy to interpret)
- RMSE (punishes large errors)
- MAPE (careful when values near 0)

**The correct evaluation is walk-forward backtesting**:
- compute metrics on each fold
- report mean + std over folds

---

## Prophet overview (high level)
Prophet is an additive model:
- trend + seasonality + holidays + noise
It is often strong for:
- business series with multiple seasonalities
- missing dates
- holiday effects

But it’s not magic—still requires correct backtesting.

---

## What you should be able to explain after Day 17
- Why time series splits must be chronological
- What stationarity is and how differencing helps
- Difference between AR vs MA
- How lag/rolling features convert time series → supervised learning
- How walk-forward backtesting simulates real forecasting

---
If you paste your real dataset schema (date column name + target column), I can tailor these scripts to it.
