COMMON MISTAKES & BEST PRACTICES — Naive Bayes

Mistakes:
1) Using MultinomialNB with non-count features
   - Multinomial NB assumes non-negative counts/frequencies.
   - Fix: Use CountVectorizer or transform to counts; consider TF (term frequency).

2) No smoothing (alpha=0)
   - Zero counts yield zero probabilities → log-likelihood collapses.
   - Fix: Use Laplace/Lidstone smoothing (alpha > 0). Default in sklearn is good.

3) Ignoring feature scaling for GaussianNB
   - GaussianNB models per-class variance; extreme scales can destabilize.
   - Fix: Consider standardization if features vary wildly; or use models suited for continuous data.

4) Treating correlated features as independent
   - NB assumes conditional independence. Highly correlated features can mislead.
   - Fix: Feature selection or dimensionality reduction; consider other models.

5) Misusing BernoulliNB with count features
   - Bernoulli NB expects binary presence/absence.
   - Fix: Use CountVectorizer(binary=True) or Binarizer.

6) Assuming probability outputs are perfectly calibrated
   - NB can be overconfident.
   - Fix: Calibrate with CalibratedClassifierCV (sigmoid/isotonic) for thresholded decisions.

7) Ignoring class imbalance
   - Priors heavily affect posterior; majority class dominates.
   - Fix: Adjust priors via sample_weight; evaluate Precision/Recall/F1.

8) Over-reliance on TF-IDF with MultinomialNB
   - NB derivation assumes counts; TF-IDF can still work but may underperform depending on task.
   - Fix: Benchmark Count vs TF-IDF; consider combining with feature selection.

9) Not validating with stratified splits
   - Class imbalance can produce misleading metrics.
   - Fix: Use StratifiedKFold or stratified train/test splits.

10) Using accuracy alone
   - Accuracy hides performance on minority class.
   - Fix: Report precision/recall/F1 and examine confusion matrix.

Best Practices:
- Start with CountVectorizer + MultinomialNB (alpha=1.0) for text.
- Use BernoulliNB for binary presence/absence features.
- For continuous data, GaussianNB with var_smoothing to avoid zero variance.
- Consider feature selection and dimensionality reduction for high-dimensional settings.
- Always evaluate with precision/recall/F1; calibrate probabilities when needed.
