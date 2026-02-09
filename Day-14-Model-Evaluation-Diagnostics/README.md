# Day 14: Model Evaluation & Diagnostics

## Overview
Evaluating models is more than a single metric. This module provides runnable, self-contained scripts to help you:
- Understand bias–variance via controlled demos
- Plot learning curves and validation curves
- Select appropriate metrics by task (classification, regression, ranking)
- Run an error analysis workflow (confusion matrix, slice analysis)
- Calibrate probabilities and estimate confidence intervals via bootstrapping
- Follow a practical debugging checklist

Dependencies: scikit-learn, numpy, matplotlib, seaborn, pandas (optional)

## Module Structure
```
Day-14-Model-Evaluation-Diagnostics/
├── README.md
├── learning_curves_validation_curves.py
├── bias_variance_demo.py
├── metrics_by_task.py
├── error_analysis_workflow.py
├── calibration_and_confidence_intervals.py
├── model_comparison_report.py
└── practical_debugging_checklist.md
```

## Quick Starts

- Learning & Validation Curves
```bash
python Day-14-Model-Evaluation-Diagnostics/learning_curves_validation_curves.py
```
Outputs learning curves for a pipeline (StandardScaler + SVC) and a validation curve over C, with guidance in console.

- Bias–Variance Demo (Polynomial Regression)
```bash
python Day-14-Model-Evaluation-Diagnostics/bias_variance_demo.py
```
Plots train/test MSE vs polynomial degree, highlighting under/over-fitting.

- Metrics by Task
```bash
python Day-14-Model-Evaluation-Diagnostics/metrics_by_task.py
```
Shows classification (accuracy, precision, recall, f1, ROC-AUC), regression (MSE, MAE, R2), and ranking (AP, NDCG) metrics.

- Error Analysis Workflow
```bash
python Day-14-Model-Evaluation-Diagnostics/error_analysis_workflow.py
```
Prints confusion matrix, per-class report, and slice-based analysis; renders a heatmap.

- Calibration & Confidence Intervals
```bash
python Day-14-Model-Evaluation-Diagnostics/calibration_and_confidence_intervals.py
```
Trains an uncalibrated classifier, calibrates it, draws a reliability diagram, and bootstraps CI for AUC.

- Model Comparison Report
```bash
python Day-14-Model-Evaluation-Diagnostics/model_comparison_report.py
```
Compares LogisticRegression, RandomForest, GradientBoosting on the same dataset with multiple metrics.

## Best Practices
- Always use stratified splits for classification; avoid leakage by fitting preprocessors on training data only (use Pipelines).
- Prefer task-appropriate metrics: ROC-AUC/F1/PR-AUC for imbalanced classification, MAE/R2 for regression, NDCG/AP for ranking.
- Inspect learning and validation curves to decide if you need more data or more/less regularization.
- Calibrate probabilities when thresholding or communicating risk; use bootstrap to estimate metric uncertainty.
- Perform slice-based error analysis to uncover hidden failure modes (e.g., specific cohorts or ranges).

Enjoy robust evaluation!
