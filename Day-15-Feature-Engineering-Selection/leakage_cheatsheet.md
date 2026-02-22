# Leakage Cheatsheet (Feature Engineering & Selection)

Leakage = using information at training time that would not be available at prediction time.

## Most common leakage sources
1. **Scaling/Imputation/Encoding outside a Pipeline**
   - Wrong: `X_scaled = scaler.fit_transform(X)` then CV
   - Right: `Pipeline([("scaler", ...), ("model", ...)])`

2. **Target Encoding / Mean Encoding**
   - Wrong: compute category mean target on full dataset (or on the full train before CV)
   - Right: compute encoding **inside CV folds** (out-of-fold encoding)

3. **Feature Selection outside CV**
   - Wrong: Select features using all data, then cross-validate
   - Right: Put selection in the Pipeline so each fold selects using only its training data

4. **Time-based data**
   - Wrong: random shuffling splits for time series
   - Right: TimeSeriesSplit / forward chaining

5. **Group leakage**
   - Wrong: same patient/user appears in both train and validation folds
   - Right: GroupKFold or GroupShuffleSplit

## Quick sanity checks
- If your CV score is very high but test score collapses → leakage is likely.
- If a single engineered feature makes performance “too good to be true” → inspect how it was built.
- Always re-check your split strategy: stratified vs grouped vs time-based.

## Golden rule
**Every step that learns from data must live inside the training fold.**
That includes preprocessing, encoding, feature selection, and even oversampling.
