# Time Series Basics — Checklist

## Splitting & evaluation (most important)
- [ ] Never shuffle time series for train/test split.
- [ ] Use chronological split.
- [ ] Prefer walk-forward backtesting for evaluation.

## Leakage guards
- [ ] Any rolling feature must use only past data:
  - use `shift(1)` before `rolling(...)`
- [ ] Any target-related feature must be computable at prediction time.
- [ ] If you use scaling, fit scaler only on train, apply to test.

## Stationarity basics
- [ ] If trend exists, try first differencing.
- [ ] If seasonality exists, try seasonal differencing (lag = period).
- [ ] Don’t difference blindly: you can destroy meaningful signal.

## Model choices (quick guidance)
- AR/ARIMA family: good when series has strong autocorrelation and relatively stable dynamics.
- ML with lag/rolling features: flexible, can include exogenous variables (promo, holidays, weather).
- Prophet: strong baseline for business seasonality, holidays, missing dates.

## Feature ideas
- Lags: 1, 7, 14, 28 (depends on cadence)
- Rolling mean/std over 7/14/28
- Calendar: day-of-week, month, holidays
- External regressors: price, campaign, inventory, temperature, etc.

## Common failure patterns
- Too-good CV score → leakage (random splits or rolling features using current y)
- Model works on some periods but fails later → regime change; use backtesting + re-train strategy
- Underfitting seasonality → add seasonal lags or model seasonality explicitly
