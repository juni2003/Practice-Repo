"""
Cost Function Implementation and Visualization

The cost function (also called loss function) measures how well our model
fits the training data. For linear regression, we typically use Mean Squared
Error (MSE) as our cost function.

Mean Squared Error (MSE):
    J(θ) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    Where:
    - J(θ): cost as a function of parameters θ
    - m: number of training examples
    - h(x⁽ⁱ⁾): predicted value for example i
    - y⁽ⁱ⁾: actual value for example i
    - 1/2: mathematical convenience for derivatives

Why MSE?
    1. Differentiable: Allows us to use gradient descent
    2. Convex: For linear regression, has a single global minimum
    3. Penalizes larger errors more: Squared term emphasizes outliers
    4. Mathematical properties: Makes calculus easier

Alternative Cost Functions:
    - Mean Absolute Error (MAE): |h(x) - y| - More robust to outliers
    - Huber Loss: Combination of MSE and MAE
    - Log Loss: For classification problems

Author: Practice Repository
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    MSE = (1/m) * Σ(y_true - y_pred)²
    
    Note: Some implementations use 1/2m for mathematical convenience
    in gradient computation, but 1/m is more standard for reporting.
    
    Args:
        y_true (np.ndarray): Actual target values, shape (m,)
        y_pred (np.ndarray): Predicted values, shape (m,)
    
    Returns:
        float: Mean squared error
    
    Example:
        >>> y_true = np.array([3, 5, 7, 9])
        >>> y_pred = np.array([2.5, 5.5, 7.0, 8.5])
        >>> mse = mean_squared_error(y_true, y_pred)
        >>> print(f"MSE: {mse:.2f}")
    """
    m = len(y_true)
    squared_errors = (y_true - y_pred) ** 2
    mse = (1 / m) * np.sum(squared_errors)
    return mse


def cost_function_linear_regression(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Calculate the cost for linear regression using MSE.
    
    J(θ) = (1/2m) * Σ(Xθ - y)²
    
    The 1/2 factor is for mathematical convenience: when we take the
    derivative, the 2 from the squared term cancels with 1/2.
    
    Args:
        X (np.ndarray): Feature matrix with bias term, shape (m, n+1)
        y (np.ndarray): Target values, shape (m,)
        theta (np.ndarray): Parameters, shape (n+1,)
    
    Returns:
        float: Cost value
    """
    m = len(y)
    predictions = X @ theta
    squared_errors = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/m) * Σ|y_true - y_pred|
    
    MAE is more robust to outliers than MSE because it doesn't
    square the errors.
    
    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        float: Mean absolute error
    """
    m = len(y_true)
    absolute_errors = np.abs(y_true - y_pred)
    mae = (1 / m) * np.sum(absolute_errors)
    return mae


def visualize_cost_surface_1d():
    """
    Visualize how cost changes with different parameter values (1D case).
    For simple linear regression: y = mx + b
    """
    # Create simple dataset
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])  # approximately y = x + 1
    
    # Add bias term to X
    X_with_bias = np.c_[np.ones(len(X)), X]
    
    # Calculate optimal parameters using Normal Equation
    theta_optimal = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
    
    # Create range of parameter values to test
    slopes = np.linspace(-2, 4, 100)
    intercepts = np.linspace(-2, 4, 100)
    
    # Calculate cost for different slopes (keeping optimal intercept)
    costs_slope = []
    for slope in slopes:
        theta = np.array([theta_optimal[0], slope])
        cost = cost_function_linear_regression(X_with_bias, y, theta)
        costs_slope.append(cost)
    
    # Calculate cost for different intercepts (keeping optimal slope)
    costs_intercept = []
    for intercept in intercepts:
        theta = np.array([intercept, theta_optimal[1]])
        cost = cost_function_linear_regression(X_with_bias, y, theta)
        costs_intercept.append(cost)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cost vs Slope
    axes[0].plot(slopes, costs_slope, 'b-', linewidth=2)
    axes[0].axvline(x=theta_optimal[1], color='r', linestyle='--', linewidth=2, 
                    label=f'Optimal Slope = {theta_optimal[1]:.2f}')
    axes[0].set_xlabel('Slope (m)', fontsize=12)
    axes[0].set_ylabel('Cost J(θ)', fontsize=12)
    axes[0].set_title('Cost Function vs Slope\n(with optimal intercept)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cost vs Intercept
    axes[1].plot(intercepts, costs_intercept, 'g-', linewidth=2)
    axes[1].axvline(x=theta_optimal[0], color='r', linestyle='--', linewidth=2, 
                    label=f'Optimal Intercept = {theta_optimal[0]:.2f}')
    axes[1].set_xlabel('Intercept (b)', fontsize=12)
    axes[1].set_ylabel('Cost J(θ)', fontsize=12)
    axes[1].set_title('Cost Function vs Intercept\n(with optimal slope)', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_cost_surface_2d():
    """
    Visualize the cost function as a 2D surface (both slope and intercept).
    This shows the convex "bowl" shape of the MSE cost function.
    """
    # Create simple dataset
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    
    # Add bias term
    X_with_bias = np.c_[np.ones(len(X)), X]
    
    # Calculate optimal parameters
    theta_optimal = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
    
    # Create grid of parameter values
    intercepts = np.linspace(-2, 6, 100)
    slopes = np.linspace(-1, 3, 100)
    I, S = np.meshgrid(intercepts, slopes)
    
    # Calculate cost for each combination
    costs = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            theta = np.array([I[i, j], S[i, j]])
            costs[i, j] = cost_function_linear_regression(X_with_bias, y, theta)
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(I, S, costs, cmap='viridis', alpha=0.8)
    ax1.scatter([theta_optimal[0]], [theta_optimal[1]], 
                [cost_function_linear_regression(X_with_bias, y, theta_optimal)],
                color='red', s=100, marker='*', label='Optimal')
    ax1.set_xlabel('Intercept (b)', fontsize=10)
    ax1.set_ylabel('Slope (m)', fontsize=10)
    ax1.set_zlabel('Cost J(θ)', fontsize=10)
    ax1.set_title('Cost Function Surface\n(3D View)', fontsize=11, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Plot 2: Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(I, S, costs, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(theta_optimal[0], theta_optimal[1], 'r*', markersize=15, 
             label=f'Optimal: ({theta_optimal[0]:.2f}, {theta_optimal[1]:.2f})')
    ax2.set_xlabel('Intercept (b)', fontsize=11)
    ax2.set_ylabel('Slope (m)', fontsize=11)
    ax2.set_title('Cost Function Contours\n(Top View)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heat map
    ax3 = fig.add_subplot(133)
    im = ax3.imshow(costs, extent=[intercepts.min(), intercepts.max(), 
                                   slopes.min(), slopes.max()],
                    origin='lower', cmap='viridis', aspect='auto')
    ax3.plot(theta_optimal[0], theta_optimal[1], 'r*', markersize=15, 
             label='Optimal')
    ax3.set_xlabel('Intercept (b)', fontsize=11)
    ax3.set_ylabel('Slope (m)', fontsize=11)
    ax3.set_title('Cost Function Heat Map', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    fig.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.show()


def compare_cost_functions():
    """
    Compare different cost functions (MSE vs MAE) and their behavior.
    Demonstrates how MSE is more sensitive to outliers.
    """
    # Create data with an outlier
    y_true = np.array([2, 4, 6, 8, 10, 12])
    y_pred_good = np.array([2.1, 4.2, 5.9, 7.8, 10.1, 12.2])  # Good predictions
    y_pred_outlier = np.array([2.1, 4.2, 5.9, 7.8, 10.1, 20.0])  # One bad prediction
    
    # Calculate costs
    mse_good = mean_squared_error(y_true, y_pred_good)
    mae_good = mean_absolute_error(y_true, y_pred_good)
    
    mse_outlier = mean_squared_error(y_true, y_pred_outlier)
    mae_outlier = mean_absolute_error(y_true, y_pred_outlier)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Good predictions
    axes[0].plot(y_true, label='True values', marker='o', linewidth=2)
    axes[0].plot(y_pred_good, label='Good predictions', marker='s', linewidth=2)
    axes[0].set_xlabel('Sample Index', fontsize=11)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title(f'Good Predictions\nMSE = {mse_good:.2f}, MAE = {mae_good:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: With outlier
    axes[1].plot(y_true, label='True values', marker='o', linewidth=2)
    axes[1].plot(y_pred_outlier, label='Predictions with outlier', 
                marker='s', linewidth=2)
    axes[1].scatter([5], [20.0], color='red', s=200, marker='X', 
                   label='Outlier', zorder=5)
    axes[1].set_xlabel('Sample Index', fontsize=11)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title(f'Predictions with Outlier\nMSE = {mse_outlier:.2f}, MAE = {mae_outlier:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nCost Function Comparison:")
    print("-" * 70)
    print("Good Predictions:")
    print(f"  MSE: {mse_good:.4f}")
    print(f"  MAE: {mae_good:.4f}")
    print()
    print("With One Outlier:")
    print(f"  MSE: {mse_outlier:.4f}  (increase: {((mse_outlier/mse_good - 1) * 100):.1f}%)")
    print(f"  MAE: {mae_outlier:.4f}  (increase: {((mae_outlier/mae_good - 1) * 100):.1f}%)")
    print()
    print("Observation: MSE is much more sensitive to outliers due to squaring!")


def demonstrate_cost_minimization():
    """
    Show how minimizing cost improves predictions.
    """
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2 * X + 3 + np.random.randn(50) * 2
    
    X_with_bias = np.c_[np.ones(len(X)), X]
    
    # Test different parameter values
    test_params = [
        (0, 0, "Random start"),
        (1, 1, "Better guess"),
        (3, 2, "Even better"),
    ]
    
    # Calculate optimal
    theta_optimal = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
    test_params.append((theta_optimal[0], theta_optimal[1], "Optimal"))
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (intercept, slope, label) in enumerate(test_params):
        theta = np.array([intercept, slope])
        cost = cost_function_linear_regression(X_with_bias, y, theta)
        
        # Make predictions
        y_pred = X_with_bias @ theta
        
        # Plot
        axes[idx].scatter(X, y, alpha=0.6, s=30, label='Data')
        axes[idx].plot(X, y_pred, 'r-', linewidth=2, 
                      label=f'y = {slope:.2f}x + {intercept:.2f}')
        axes[idx].set_xlabel('X', fontsize=11)
        axes[idx].set_ylabel('y', fontsize=11)
        axes[idx].set_title(f'{label}\nCost = {cost:.4f}', 
                          fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating cost functions.
    """
    print("=" * 70)
    print("COST FUNCTION DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Example 1: Basic MSE calculation
    print("EXAMPLE 1: Basic Cost Calculation")
    print("-" * 70)
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2.5, 5.5, 7.0, 8.5])
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"True values:     {y_true}")
    print(f"Predicted values: {y_pred}")
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print()
    
    # Example 2: Understanding individual errors
    print("EXAMPLE 2: Breaking Down the Error")
    print("-" * 70)
    errors = y_true - y_pred
    squared_errors = errors ** 2
    
    print(f"{'Index':<8} {'True':<8} {'Pred':<8} {'Error':<10} {'Squared Error':<15}")
    print("-" * 70)
    for i, (yt, yp, err, sq_err) in enumerate(zip(y_true, y_pred, errors, squared_errors)):
        print(f"{i:<8} {yt:<8.1f} {yp:<8.1f} {err:<10.2f} {sq_err:<15.4f}")
    print(f"\nSum of squared errors: {np.sum(squared_errors):.4f}")
    print(f"MSE (divide by n={len(y_true)}): {mse:.4f}")
    print()
    
    print("=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)
    print("Generating plots... (close the plot windows to continue)")
    print()
    
    # Visualizations
    print("1. Cost vs individual parameters...")
    visualize_cost_surface_1d()
    
    print("2. Cost surface (2D)...")
    visualize_cost_surface_2d()
    
    print("3. Comparing MSE vs MAE...")
    compare_cost_functions()
    
    print("4. Cost minimization in action...")
    demonstrate_cost_minimization()
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Cost function measures how well the model fits the data")
    print("2. MSE is the most common cost function for regression")
    print("3. MSE creates a convex surface - single global minimum")
    print("4. Lower cost = better predictions")
    print("5. MSE heavily penalizes large errors (due to squaring)")
    print("6. MAE is more robust to outliers than MSE")
    print("7. Gradient descent minimizes cost iteratively")
    print("=" * 70)


if __name__ == "__main__":
    main()
