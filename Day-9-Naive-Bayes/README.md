# Day 9: Naive Bayes & Bayesian Thinking

## Overview
Naive Bayes (NB) is a family of probabilistic classifiers based on Bayes’ theorem with a strong independence assumption between features. Despite the “naive” assumption, NB performs remarkably well on:
- Text classification (spam detection, sentiment analysis)
- Simple tabular problems with relatively independent features
- Baseline models that are fast and easy to train

This module covers:
- Bayesian foundations: priors, likelihoods, posteriors
- Variants: Gaussian, Multinomial, Bernoulli
- Text classification pipeline (Count/TF-IDF + NB)
- Handling class imbalance and probability calibration
- Visualizing posterior probabilities and decision logic

## Why Learn Naive Bayes?
- Fast training and inference
- Works well with high-dimensional sparse text data
- Solid baseline for many classification tasks
- Clear probabilistic interpretation

Limitations:
- Strong independence assumption
- Continuous data often handled simply (Gaussian), may underperform if features are correlated
- Raw probability outputs may need calibration for decision-making thresholds

## Module Structure
```
Day-9-Naive-Bayes/
├── README.md                                   # This file
├── nb_theory_and_concepts.md                   # Theory, formulas, assumptions
├── gaussian_nb_from_scratch.py                 # Educational implementation for continuous data
├── multinomial_nb_text_pipeline.py             # Text classification pipeline with Count/TF-IDF
├── bernoulli_nb_binary_features.py             # Bernoulli NB on binary features
├── class_imbalance_and_calibration.py          # Sample weighting & probability calibration
├── bayes_posterior_visualization.py            # Visualize likelihood/prior/posterior
└── common_mistakes_and_best_practices.txt      # Pitfalls and practical tips
```

## Quick Start: Multinomial NB (Text)
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

texts = [
    "Win cash now click here",
    "Meeting scheduled at 10am",
    "Limited time offer for free prize",
    "Please review the attached document",
    "Urgent account verification required",
    "Thanks for your purchase, order shipped",
]
y = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.33, random_state=42, stratify=y)

pipe = Pipeline([
    ("vectorizer", CountVectorizer(stop_words="english")),
    ("nb", MultinomialNB(alpha=1.0))
])

pipe.fit(X_train, y_train)
print("Accuracy:", pipe.score(X_test, y_test))
```

## Quick Start: Gaussian NB (Continuous)
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=600, n_features=5, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
print("Accuracy:", gnb.score(X_test, y_test))
```

## Variants at a Glance
- Gaussian NB: continuous features (assumes per-class Gaussian per feature)
- Multinomial NB: counts/frequencies (bag-of-words)
- Bernoulli NB: binary features (presence/absence)

## Key Concepts
- Bayes' theorem: P(y|x) ∝ P(x|y) P(y)
- Priors (P(y)): class probabilities
- Likelihood (P(x|y)): how likely features are given the class
- Naive assumption: features are conditionally independent given y
- Log-space: sum of log-likelihoods and log-priors to avoid underflow

## Handling Class Imbalance
- Adjust class priors or use sample_weight in fit (sklearn NB supports sample_weight)
- Evaluate with precision/recall/F1 rather than accuracy

## Probability Calibration
- NB probabilities can be poorly calibrated
- Use CalibratedClassifierCV (sigmoid/isotonic) when thresholding probabilities is critical

## Practice Flow
1) Read nb_theory_and_concepts.md to get the math intuition  
2) Run gaussian_nb_from_scratch.py to see the inner workings  
3) Build a text classifier in multinomial_nb_text_pipeline.py  
4) Try Bernoulli NB for binary features in bernoulli_nb_binary_features.py  
5) Explore class imbalance and calibration in class_imbalance_and_calibration.py  
6) Visualize Bayesian components in bayes_posterior_visualization.py  
7) Review practical pitfalls in common_mistakes_and_best_practices.txt

Enjoy the Bayesian journey!
