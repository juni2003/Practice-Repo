# Naive Bayes: Theory and Concepts

## Bayes' Theorem
For classes y ∈ {0, 1, ..., K-1} and feature vector x:
```
P(y | x) = [ P(x | y) P(y) ] / P(x)
```
For classification, we compare unnormalized posteriors:
```
P(y | x) ∝ P(x | y) P(y)
```
We pick:
```
y* = argmax_y [ P(x | y) P(y) ]
```

## Naive Assumption
The “naive” part assumes conditional independence of features given y:
```
P(x | y) = Π_j P(x_j | y)
```
This simplifies computation and works surprisingly well in practice, particularly for text.

## Log-Space
To avoid underflow and improve numerical stability:
```
log P(y | x) ∝ log P(y) + Σ_j log P(x_j | y)
```

## Gaussian Naive Bayes (Continuous Features)
Assumes each feature is Gaussian within each class y:
```
P(x_j | y=k) = N(x_j; μ_{k,j}, σ_{k,j}^2)
log P(x | y=k) = Σ_j [ - 0.5 log(2πσ_{k,j}^2) - (x_j - μ_{k,j})^2 / (2σ_{k,j}^2) ]
log Posterior ∝ log P(y=k) + (above sum)
```
Parameters:
- μ_{k,j}: mean of feature j in class k
- σ_{k,j}^2: variance of feature j in class k
- P(y=k): prior (class frequency or custom)

## Multinomial Naive Bayes (Counts/Frequencies)
Ideal for bag-of-words counts:
```
P(x | y=k) ∝ Π_j θ_{k,j}^{x_j}
θ_{k,j} = (N_{k,j} + α) / (Σ_j N_{k,j} + α * V)
```
- N_{k,j}: count of feature j in class k (e.g., word count)
- V: vocabulary size
- α: smoothing (Laplace α=1 or Lidstone α∈(0,1])

Log form:
```
log P(x | y=k) = Σ_j x_j log θ_{k,j} + constant
```

## Bernoulli Naive Bayes (Binary Features)
For feature presence/absence:
```
P(x_j | y=k) = p_{k,j}^{x_j} (1 - p_{k,j})^{(1 - x_j)}
p_{k,j} = (n_{present_in_class_k} + α) / (n_k + 2α)
```
- Useful when features are binary indicators (e.g., presence of a word)
- Complement NB (variant) can help when class imbalance is strong

## Priors and Class Imbalance
- P(y=k) typically the class frequency
- You can set priors manually or emulate via sample_weight
- Evaluate using precision/recall/F1 rather than accuracy

## Probability Calibration
- NB probabilities can be overconfident
- Use CalibratedClassifierCV with “sigmoid” or “isotonic” to calibrate

## When to Use Which Variant
- Gaussian NB: numeric continuous features, roughly normal per class
- Multinomial NB: text counts/frequencies (bag-of-words/TF)
- Bernoulli NB: binary indicators (presence/absence)

## Practical Notes
- Always apply smoothing (α>0) for Multinomial/Bernoulli
- For text, CountVectorizer often pairs very well with MultinomialNB
- TF-IDF can work too, but classic NB derivation assumes counts
- Consider feature selection to reduce noise in high-dimensional text
