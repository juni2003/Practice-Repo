# Day 5 — Data Preprocessing & Feature Engineering

This folder contains learning materials and runnable examples for core preprocessing and feature engineering steps including feature scaling:

Learning goals
- Understand handling missing values and best practices.
- Learn categorical encoding: label encoding vs one-hot encoding and when to use each.
- Apply basic feature selection (filter & embedded methods).
- Apply feature extraction (PCA basics).
- Practice train-test split and cross-validation for robust evaluation.
- Build a reusable scikit-learn pipeline that combines preprocessing and a model.

Contents (files)
- preprocessing_pipeline.py — A complete, commented pipeline example with synthetic data, missing value handling, encoding, scaling, train-test split, and cross-validation.
- feature_engineering_examples.py — (planned) feature selection & PCA examples and visualizations.
- experiments/ — (planned) Jupyter notebooks with step-by-step interactive experiments.
- data/ — (planned) small example CSVs (if needed) or scripts to create toy datasets.
- common_mistakes.txt — (planned) common pitfalls when preprocessing and feature engineering.
- README.md — (this file): conceptual guide and file map.

Quick notes / Concepts (short)
- Missing values: Impute where appropriate. Use SimpleImputer with median for skewed numeric data, mean for roughly Gaussian, and constant (or most_frequent) for categorical.
- Encoding: One-hot encoding is good for nominal features with few categories. For high-cardinality categorical features, consider target encoding or embedding (not covered here).
- Feature selection: Use filter methods (variance threshold, correlation, univariate tests) for quick pruning. Use embedded methods (L1-regularized models, tree-based feature importances) for stronger selection.
- PCA: Use after scaling. PCA is for dimensionality reduction/visualization — interpretability of components is lower than original features.
- Pipelines: Use scikit-learn Pipelines & ColumnTransformer so preprocessing is safe and reproducible (no leakage).
- Cross-validation: Use cross_val_score or cross_validate with a pipeline — ensures all preprocessing steps run inside CV folds to avoid leakage.

How to use the example
1. Run the Python file:
   python Day-5-Data-Preprocessing/preprocessing_pipeline.py
2. It will:
   - Create a toy dataset
   - Build preprocessing pipeline
   - Fit pipeline and show transformed shape/features
   - Run a simple cross-validation run and print scores

If you want me to commit these first two files to a new branch and open a PR, say "Commit Day-5 files" and I'll proceed (I will need permission to push or you can run the commands locally).
