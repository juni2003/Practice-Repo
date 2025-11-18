"""
Model Evaluation Metrics for Linear Regression

After training a linear regression model, we need to evaluate how well it
performs. This module covers the most important evaluation metrics.

Key Metrics:
    1. R² Score (Coefficient of Determination)
    2. RMSE (Root Mean Squared Error)
    3. MAE (Mean Absolute Error)
    4. Adjusted R²
    
Each metric provides different insights into model performance.

Author: Practice Repository
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² Score (Coefficient of Determination).
    
    R² = 1 - (SS_res / SS_tot)
    
    Where:
    - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
    - SS_tot = Σ(y_true - ȳ)² (total sum of squares)
    
    Interpretation:
    - R² = 1.0: Perfect predictions (all points on the line)
    - R² = 0.0: Model no better than predicting the mean
    - R² < 0.0: Model worse than predicting the mean
    - R² = 0.7: Model explains 70% of variance in the data
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: R² score
    
    Example:
        >>> y_true = np.array([3, 5, 7, 9])
        >>> y_pred = np.array([3.1, 4.9, 7.2, 8.8])
        >>> r2 = r2_score(y_true, y_pred)
        >>> print(f"R² = {r2:.4f}")
    """
    # Calculate residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_res / ss_tot)
    
    return r2


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calculate Adjusted R² Score.
    
    Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    
    Where:
    - n: number of samples
    - p: number of features
    
    Unlike R², Adjusted R² penalizes adding features that don't improve
    the model. It's better for comparing models with different numbers
    of features.
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
        n_features (int): Number of features in the model
    
    Returns:
        float: Adjusted R² score
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate adjusted R²
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    
    return adjusted_r2


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    RMSE = √[(1/n) * Σ(y_true - y_pred)²]
    
    RMSE is in the same units as the target variable, making it
    easier to interpret than MSE. Lower is better.
    
    Advantages:
    - Same units as target variable
    - Penalizes large errors (due to squaring)
    - Most common metric for regression
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: RMSE value
    
    Example:
        >>> y_true = np.array([100, 200, 300])
        >>> y_pred = np.array([110, 190, 310])
        >>> error = rmse(y_true, y_pred)
        >>> print(f"RMSE = {error:.2f}")
    """
    n = len(y_true)
    squared_errors = (y_true - y_pred) ** 2
    mse = (1 / n) * np.sum(squared_errors)
    rmse_value = np.sqrt(mse)
    
    return rmse_value


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    MAE is less sensitive to outliers than RMSE because it doesn't
    square the errors. Also in the same units as the target variable.
    
    Advantages:
    - More robust to outliers than RMSE
    - Easy to interpret
    - Same units as target variable
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: MAE value
    """
    n = len(y_true)
    absolute_errors = np.abs(y_true - y_pred)
    mae_value = (1 / n) * np.sum(absolute_errors)
    
    return mae_value


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    
    MAPE expresses error as a percentage, making it scale-independent.
    However, it's undefined when y_true contains zeros.
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        raise ValueError("MAPE is undefined when all true values are zero")
    
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    mape_value = (100 / len(y_true[mask])) * np.sum(percentage_errors)
    
    return mape_value


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a model.
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
        n_features (int): Number of features used (for adjusted R²)
    
    Returns:
        Dict[str, float]: Dictionary of all metrics
    """
    metrics = {
        'R²': r2_score(y_true, y_pred),
        'Adjusted R²': adjusted_r2_score(y_true, y_pred, n_features),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
    }
    
    # Add MAPE if possible
    try:
        metrics['MAPE'] = mape(y_true, y_pred)
    except ValueError:
        metrics['MAPE'] = None
    
    return metrics


def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Model Predictions"):
    """
    Create comprehensive visualization of model predictions.
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
        title (str): Title for the plots
    """
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    axes[0, 0].set_xlabel('Actual Values', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Values', fontsize=11)
    axes[0, 0].set_title(f'Actual vs Predicted\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values', fontsize=11)
    axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residual Distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title(f'Residual Distribution\nMean = {np.mean(residuals):.4f}', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error Metrics Summary
    axes[1, 1].axis('off')
    
    # Create text summary
    summary_text = f"""
    {title}
    {'=' * 50}
    
    Performance Metrics:
    
    R² Score:            {r2:.4f}
    {'(Explains ' + f'{r2*100:.2f}' + '% of variance)'}
    
    RMSE:                {rmse_val:.4f}
    (Average error magnitude)
    
    MAE:                 {mae_val:.4f}
    (Average absolute error)
    
    Residuals:
    Mean:                {np.mean(residuals):.4f}
    Std Dev:             {np.std(residuals):.4f}
    Min:                 {np.min(residuals):.4f}
    Max:                 {np.max(residuals):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.show()


def compare_models_example():
    """
    Compare multiple models using different metrics.
    Demonstrates how to interpret and use evaluation metrics.
    """
    np.random.seed(42)
    
    # Generate true values
    X = np.linspace(0, 10, 100)
    y_true = 2 * X + 3 + np.random.randn(100) * 2
    
    # Create three different prediction scenarios
    # Model 1: Good fit
    y_pred_good = 2 * X + 3 + np.random.randn(100) * 2.5
    
    # Model 2: Biased (systematic error)
    y_pred_biased = 2 * X + 5 + np.random.randn(100) * 2
    
    # Model 3: High variance (random errors)
    y_pred_variance = 2 * X + 3 + np.random.randn(100) * 5
    
    models = [
        ("Good Model", y_pred_good),
        ("Biased Model", y_pred_biased),
        ("High Variance Model", y_pred_variance),
    ]
    
    print("\nModel Comparison:")
    print("=" * 70)
    
    # Compare metrics
    for name, y_pred in models:
        metrics = evaluate_model(y_true, y_pred)
        
        print(f"\n{name}:")
        print("-" * 70)
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name:<15}: {value:10.4f}")
            else:
                print(f"  {metric_name:<15}: {'N/A':>10}")
    
    print("\n" + "=" * 70)
    
    # Visualize each model
    for name, y_pred in models:
        visualize_predictions(y_true, y_pred, title=name)


def demonstrate_metric_interpretation():
    """
    Show what different R² values look like visually.
    """
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 * X + 3
    
    # Create predictions with different R² scores
    scenarios = [
        (0.95, "Excellent Fit (R² ≈ 0.95)"),
        (0.70, "Good Fit (R² ≈ 0.70)"),
        (0.40, "Poor Fit (R² ≈ 0.40)"),
        (0.10, "Very Poor Fit (R² ≈ 0.10)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (target_r2, label) in enumerate(scenarios):
        # Generate noise to achieve approximately the target R²
        # More noise = lower R²
        noise_std = np.sqrt((1 - target_r2) / target_r2) * np.std(y_true) * 1.5
        y_pred = y_true + np.random.randn(100) * noise_std
        
        actual_r2 = r2_score(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        
        # Plot
        axes[idx].scatter(y_true, y_pred, alpha=0.6, s=30)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[idx].set_xlabel('Actual', fontsize=11)
        axes[idx].set_ylabel('Predicted', fontsize=11)
        axes[idx].set_title(f'{label}\nActual R² = {actual_r2:.3f}, RMSE = {rmse_val:.2f}',
                          fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating model evaluation metrics.
    """
    print("=" * 70)
    print("MODEL EVALUATION METRICS DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Example 1: Calculate metrics manually
    print("EXAMPLE 1: Basic Metric Calculation")
    print("-" * 70)
    
    y_true = np.array([3, 5, 7, 9, 11])
    y_pred = np.array([3.1, 4.9, 7.2, 8.8, 11.1])
    
    print(f"True values:      {y_true}")
    print(f"Predicted values: {y_pred}")
    print()
    
    r2 = r2_score(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    
    print(f"R² Score:  {r2:.4f}  (explains {r2*100:.2f}% of variance)")
    print(f"RMSE:      {rmse_val:.4f}  (average error magnitude)")
    print(f"MAE:       {mae_val:.4f}  (average absolute error)")
    print()
    
    # Example 2: Understanding R²
    print("EXAMPLE 2: Understanding R²")
    print("-" * 70)
    print("R² = 1.0  → Perfect predictions")
    print("R² = 0.9  → Excellent model")
    print("R² = 0.7  → Good model")
    print("R² = 0.5  → Moderate model")
    print("R² = 0.0  → No better than predicting the mean")
    print("R² < 0.0  → Worse than predicting the mean")
    print()
    
    # Example 3: RMSE vs MAE
    print("EXAMPLE 3: RMSE vs MAE with Outliers")
    print("-" * 70)
    
    y_true_normal = np.array([10, 20, 30, 40, 50])
    y_pred_normal = np.array([11, 19, 31, 39, 51])
    
    y_true_outlier = np.array([10, 20, 30, 40, 50])
    y_pred_outlier = np.array([11, 19, 31, 39, 80])  # One large error
    
    print("Without outlier:")
    print(f"  RMSE: {rmse(y_true_normal, y_pred_normal):.4f}")
    print(f"  MAE:  {mae(y_true_normal, y_pred_normal):.4f}")
    print()
    
    print("With outlier (last prediction is 80 instead of 51):")
    print(f"  RMSE: {rmse(y_true_outlier, y_pred_outlier):.4f}")
    print(f"  MAE:  {mae(y_true_outlier, y_pred_outlier):.4f}")
    print()
    print("Notice: RMSE is much more affected by the outlier!")
    print()
    
    print("=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)
    print("Generating plots... (close the plot windows to continue)")
    print()
    
    # Generate sample data for visualization
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true_vis = 2 * X + 3 + np.random.randn(100) * 2
    y_pred_vis = 2.1 * X + 2.8 + np.random.randn(100) * 2.5
    
    print("1. Comprehensive prediction visualization...")
    visualize_predictions(y_true_vis, y_pred_vis, "Sample Model")
    
    print("2. Comparing different models...")
    compare_models_example()
    
    print("3. Visual interpretation of R² values...")
    demonstrate_metric_interpretation()
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. R² measures proportion of variance explained (0 to 1, higher is better)")
    print("2. Adjusted R² accounts for number of features")
    print("3. RMSE is in the same units as target, penalizes large errors")
    print("4. MAE is more robust to outliers than RMSE")
    print("5. Always use multiple metrics for complete evaluation")
    print("6. Residual plots help identify patterns and issues")
    print("7. Perfect metrics on training data may indicate overfitting")
    print("=" * 70)


if __name__ == "__main__":
    main()
