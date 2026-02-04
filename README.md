# 📘 Practice Repository — ML | DS | Python | NN

Welcome! This repo is my personal learning journal for Machine Learning, Data Science, Python, and Neural Networks. The purpose is simple:
- Learn something new whenever I get time
- Keep clean, runnable code with comments
- Write short READMEs to explain concepts and logic
- Track progress day-by-day in well-organized folders

If you're browsing this as a visitor, feel free to explore each day’s folder — most topics include multiple scripts, visualizations, and a README explaining the approach.

---

## 📂 What’s Inside (Completed Days)

- Day-1-Feature-Scaling
  - MinMax, Standard, Robust, MaxAbs; visual comparisons; when to use; common pitfalls.
- Day-2-Reinforcement-Learning
  - RL basics, MDPs, Q-Learning, DQN, Policy Gradients, REINFORCE.
- Day-3-Linear-Regression
  - Simple & Multiple LR, Gradient Descent, cost functions, metrics, house-price pipeline.
- Day-4-Logistic-Regression
  - Sigmoid, log loss, decision boundary, confusion matrix, spam detection example.
- Day-5-Data-Preprocessing
  - Missing values, encodings, feature engineering, pipelines, unit tests, common mistakes.
- Day-6-Decision-Tree
  - Entropy, information gain, from-scratch tree, pruning, random forests, visualization.
- Day-7-Support-vector-machines-SVM
  - Linear vs kernel SVM, margin, hyperparameters (C, gamma), SVR, best practices.
- Day-8-KNN-KMeans
  - KNN classification/regression; K-Means clustering; elbow & silhouette methods; scaling and distance metrics.
- Day 9 — Naive Bayes & Bayesian Thinking
  - conditional independence, likelihoods, priors, posteriors; Gaussian, Multinomial, Bernoulli; Text classification pipeline (TF-IDF + NB); Handling class imbalance and calibration

- Day 10 — Dimensionality Reduction (PCA and Friends)
  - variance, eigenvectors, scree plot, Whitening, explained variance, reconstruction error; Incremental PCA; Visualization

- Day 11 — Bagging & Boosting (Foundations)
  - Bagging vs pasting; bias-variance tradeoff; AdaBoost intuition and exponential loss; Gradient Boosting; Model comparison against single trees

- Day 12 — XGBoost, LightGBM, CatBoost
- Why gradient-boosted trees win often; Handling categorical features (CatBoost), Early stopping, regularization, feature importance, Hyperparameter tuning templates

Tip: Each day typically contains 5–8 files with commented code, visualizations where helpful, and a README for quick understanding.

---

## 🧭 How to Use This Repo

- Clone and create a fresh environment:
```bash
git clone https://github.com/juni2003/Practice-Repo.git
cd Practice-Repo
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt  # if present, otherwise install per-folder needs
```

- Run any script from a day folder (most are standalone):
```bash
cd Day-8-KNN-KMeans
python knn_sklearn_examples.py
```

- Typical folder layout for new topics:
```
Day-N-Topic/
├── README.md
├── core_concepts.md or notes.md
├── algo_from_scratch.py
├── sklearn_examples.py
├── visualizations.py
├── real_world_applications.py
└── common_mistakes_and_best_practices.txt
```

---

## 🚀 Next 6 Days (Planned Roadmap)

These are the next topics I’ll study, each with subtopics and a mini-project or visualization.


### Day 13 — Hyperparameter Tuning & Model Selection
- GridSearchCV, RandomizedSearchCV
- Bayesian optimization with Optuna
- Cross-validation strategies (StratifiedKFold, GroupKFold)
- Reproducibility and search spaces

### Day 14 — Model Evaluation & Diagnostics
- Bias-variance, learning curves, validation curves
- Metrics by task (classification, regression, ranking)
- Error analysis workflow; calibration; confidence intervals
- Practical debugging checklist

### Day 15 — Feature Engineering & Selection
- ColumnTransformer pipelines
- Univariate selection, RFE, model-based selection
- Interaction features, target encoding (with leakage guards)
- SHAP for interpretation (global vs local)

### Day 16 — Advanced Clustering
- DBSCAN, HDBSCAN for density-based clustering
- Gaussian Mixture Models (soft assignments)
- Spectral Clustering for non-convex shapes
- Cluster validation beyond silhouette (e.g., stability)

### Day 17 — Time Series Basics
- Train/test splits for time; stationarity; differencing
- AR, MA, ARIMA; seasonality; Prophet overview
- Rolling features; lagged variables
- Backtesting and evaluation

### Day 18 — Neural Networks Fundamentals
- Perceptron and MLP; activations; backprop
- Initialization, normalization, regularization
- PyTorch/TensorFlow starter; simple MLP
- Overfitting control (dropout, weight decay)

---

## 🔮 Future Topics (Names Only)

1) Natural Language Processing (NLP) Basics  
2) Word Embeddings & Vector Semantics (Word2Vec/GloVe)  
3) Convolutional Neural Networks (CNNs)  
4) Recurrent Networks (RNN/LSTM/GRU)  
5) Transformers & Attention  
6) Recommender Systems (Collaborative/Content-based)  
7) MLOps & Deployment (FastAPI, Docker)  
8) Experiment Tracking & Data Versioning (MLflow, DVC)  
9) Causal Inference (Intro)  
10) Fairness, Bias, and Explainability  

---

## 🛠️ Conventions & Notes

- Code is written to be readable first; comments > micro-optimizations.
- Visualizations aim to show intuition (decision boundaries, distributions).
- Feature scaling is applied where required (distance/gradient-based models).
- Each new topic will include:
  - A simple “from scratch” implementation (if reasonable)
  - A scikit-learn pipeline for practical use
  - A “common mistakes” text file with do’s and don’ts
  - A small real-world(ish) example

---

## 🤝 Contributions / My Future Self

- If I revisit older days, I’ll add:
  - Better visualizations
  - More robust evaluation
  - Extra datasets and edge cases
- If you’re reading this and have ideas, feel free to open an issue with suggestions.

Enjoy the journey!
