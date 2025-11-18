"""
Multiple Linear Regression Implementation from Scratch

This module demonstrates multiple linear regression with multiple features.
Unlike simple linear regression (one feature), this handles multiple input
variables to predict a single output.

Mathematical Formula:
    y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
    
    Or in matrix form:
    y = Xθ
    
    Where:
    - y: predicted value (n_samples,)
    - X: feature matrix (n_samples, n_features)
    - θ: coefficients vector (n_features + 1,) including bias
    - b₀: bias/intercept term
    - b₁, b₂, ..., bₙ: coefficients for each feature

Normal Equation (Closed-Form Solution):
    θ = (XᵀX)⁻¹Xᵀy
    
    This directly computes the optimal parameters without iteration.
    However, it can be computationally expensive for very large datasets
    (inverting an n×n matrix has O(n³) complexity).

Author: Practice Repository
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D


class MultipleLinearRegression:
    """
    Multiple Linear Regression model using the Normal Equation.
    
    This model can handle multiple input features to predict a single
    continuous output value.
    
    Attributes:
        coefficients (np.ndarray): Model coefficients including bias term
        is_fitted (bool): Whether the model has been trained
        n_features (int): Number of features in the training data
    """
    
    def __init__(self):
        """Initialize the model with no parameters."""
        self.coefficients = None
        self.is_fitted = False
        self.n_features = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultipleLinearRegression':
        """
        Fit the multiple linear regression model using the Normal Equation.
        
        The Normal Equation: θ = (XᵀX)⁻¹Xᵀy
        
        Steps:
        1. Add bias column (all ones) to X
        2. Compute XᵀX
        3. Compute the inverse of XᵀX
        4. Multiply by Xᵀy to get optimal coefficients
        
        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training targets, shape (n_samples,)
        
        Returns:
            self: The fitted model
        
        Example:
            >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
            >>> y = np.array([3, 5, 7, 9])
            >>> model = MultipleLinearRegression()
            >>> model.fit(X, y)
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, self.n_features = X.shape
        
        # Add bias term (column of ones) to X
        # This allows us to include the intercept in our coefficient vector
        X_with_bias = np.c_[np.ones(n_samples), X]
        
        # Apply Normal Equation: θ = (XᵀX)⁻¹Xᵀy
        # Step 1: Compute XᵀX
        X_transpose_X = X_with_bias.T @ X_with_bias
        
        # Step 2: Compute XᵀY
        X_transpose_y = X_with_bias.T @ y
        
        # Step 3: Solve for θ
        # We use np.linalg.solve instead of computing inverse directly
        # This is more numerically stable and efficient
        try:
            self.coefficients = np.linalg.solve(X_transpose_X, X_transpose_y)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse as fallback
            print("Warning: Using pseudo-inverse due to singular matrix")
            self.coefficients = np.linalg.pinv(X_transpose_X) @ X_transpose_y
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Formula: ŷ = Xθ
        
        Args:
            X (np.ndarray): Features to predict, shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted values, shape (n_samples,)
        
        Raises:
            ValueError: If model hasn't been fitted yet or feature count mismatch
        
        Example:
            >>> X_test = np.array([[5, 6], [6, 7]])
            >>> predictions = model.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check feature count
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Add bias term
        n_samples = X.shape[0]
        X_with_bias = np.c_[np.ones(n_samples), X]
        
        # Compute predictions: ŷ = Xθ
        return X_with_bias @ self.coefficients
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the R² (coefficient of determination) score.
        
        R² = 1 - (SS_res / SS_tot)
        
        Args:
            X (np.ndarray): Features, shape (n_samples, n_features)
            y (np.ndarray): True target values, shape (n_samples,)
        
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        
        # Residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # R² score
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def get_coefficients(self) -> Tuple[float, np.ndarray]:
        """
        Get the model coefficients.
        
        Returns:
            Tuple[float, np.ndarray]: (intercept, feature_coefficients)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        intercept = self.coefficients[0]
        feature_coeffs = self.coefficients[1:]
        
        return intercept, feature_coeffs


def create_sample_data_2d() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data with 2 features.
    
    Generate synthetic data following:
    y = 3 + 2*x₁ + 5*x₂ + noise
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) training data
    """
    np.random.seed(42)
    
    # Generate 100 samples with 2 features
    n_samples = 100
    X1 = np.random.uniform(0, 10, n_samples)
    X2 = np.random.uniform(0, 10, n_samples)
    X = np.column_stack([X1, X2])
    
    # True relationship: y = 3 + 2*x₁ + 5*x₂ + noise
    y = 3 + 2 * X1 + 5 * X2 + np.random.randn(n_samples) * 3
    
    return X, y


def create_sample_data_multi() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data with multiple features (5 features).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) training data
    """
    np.random.seed(42)
    
    # Generate 200 samples with 5 features
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients: [2, -3, 1.5, 0.5, -1]
    true_coeffs = np.array([2, -3, 1.5, 0.5, -1])
    
    # Generate y with these coefficients plus noise
    y = 10 + X @ true_coeffs + np.random.randn(n_samples) * 2
    
    return X, y, true_coeffs


def visualize_2d_regression(model: MultipleLinearRegression, X: np.ndarray, y: np.ndarray):
    """
    Visualize regression with 2 features in 3D.
    
    Args:
        model (MultipleLinearRegression): Fitted model
        X (np.ndarray): Feature matrix with 2 features
        y (np.ndarray): Target values
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: 3D scatter with regression plane
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot actual data points
    ax1.scatter(X[:, 0], X[:, 1], y, color='blue', alpha=0.6, s=30, label='Actual Data')
    
    # Create mesh for regression plane
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    # Predict on mesh
    X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
    y_mesh = model.predict(X_mesh).reshape(x1_mesh.shape)
    
    # Plot regression plane
    ax1.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.3, color='red')
    
    ax1.set_xlabel('X₁ (Feature 1)', fontsize=10)
    ax1.set_ylabel('X₂ (Feature 2)', fontsize=10)
    ax1.set_zlabel('y (Target)', fontsize=10)
    ax1.set_title('Multiple Linear Regression (2 Features)\n3D Visualization', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Plot 2: Feature importance
    ax2 = fig.add_subplot(122)
    intercept, coeffs = model.get_coefficients()
    
    feature_names = [f'Feature {i+1}' for i in range(len(coeffs))]
    colors = ['green' if c > 0 else 'red' for c in coeffs]
    
    bars = ax2.barh(feature_names, coeffs, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Coefficient Value', fontsize=11)
    ax2.set_ylabel('Features', fontsize=11)
    ax2.set_title(f'Feature Importance\nIntercept = {intercept:.2f}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, coeff in zip(bars, coeffs):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{coeff:.2f}', 
                ha='left' if width > 0 else 'right',
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def compare_simple_vs_multiple():
    """
    Compare simple linear regression (1 feature) with multiple (2 features).
    Shows how adding relevant features improves predictions.
    """
    np.random.seed(42)
    
    # Generate data where y depends on both x1 and x2
    n_samples = 100
    X1 = np.random.uniform(0, 10, n_samples)
    X2 = np.random.uniform(0, 10, n_samples)
    X_full = np.column_stack([X1, X2])
    
    # True relationship: y = 2*x1 + 5*x2 + noise
    y = 2 * X1 + 5 * X2 + np.random.randn(n_samples) * 3
    
    # Model 1: Use only first feature
    model_simple = MultipleLinearRegression()
    model_simple.fit(X1.reshape(-1, 1), y)
    r2_simple = model_simple.score(X1.reshape(-1, 1), y)
    
    # Model 2: Use both features
    model_multiple = MultipleLinearRegression()
    model_multiple.fit(X_full, y)
    r2_multiple = model_multiple.score(X_full, y)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Simple regression (only X1)
    axes[0].scatter(X1, y, alpha=0.6, s=30, color='blue')
    X1_line = np.linspace(X1.min(), X1.max(), 100).reshape(-1, 1)
    y_pred_simple = model_simple.predict(X1_line)
    axes[0].plot(X1_line, y_pred_simple, 'r-', linewidth=2, label='Fit line')
    axes[0].set_xlabel('X₁ (Feature 1)', fontsize=11)
    axes[0].set_ylabel('y (Target)', fontsize=11)
    axes[0].set_title(f'Simple Linear Regression\nR² = {r2_simple:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Multiple regression results
    y_pred_multiple = model_multiple.predict(X_full)
    residuals = y - y_pred_multiple
    
    axes[1].scatter(y_pred_multiple, y, alpha=0.6, s=30, color='blue', label='Predictions')
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect fit')
    axes[1].set_xlabel('Predicted y', fontsize=11)
    axes[1].set_ylabel('Actual y', fontsize=11)
    axes[1].set_title(f'Multiple Linear Regression (2 Features)\nR² = {r2_multiple:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nComparison Results:")
    print(f"  Simple Regression (1 feature):  R² = {r2_simple:.4f}")
    print(f"  Multiple Regression (2 features): R² = {r2_multiple:.4f}")
    print(f"  Improvement: {(r2_multiple - r2_simple) * 100:.2f}%")


def main():
    """
    Main function demonstrating multiple linear regression.
    """
    print("=" * 70)
    print("MULTIPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Example 1: Two features
    print("EXAMPLE 1: Two Features")
    print("-" * 70)
    print("Creating sample data with 2 features...")
    X_2d, y_2d = create_sample_data_2d()
    print(f"Generated {len(X_2d)} samples with {X_2d.shape[1]} features")
    print(f"True relationship: y = 3 + 2*x₁ + 5*x₂ + noise")
    print()
    
    # Train model
    print("Training the model...")
    model_2d = MultipleLinearRegression()
    model_2d.fit(X_2d, y_2d)
    
    intercept, coeffs = model_2d.get_coefficients()
    print(f"✓ Model trained successfully!")
    print(f"  Learned equation: y = {intercept:.2f} + {coeffs[0]:.2f}*x₁ + {coeffs[1]:.2f}*x₂")
    print(f"  Expected: y = 3.00 + 2.00*x₁ + 5.00*x₂")
    print()
    
    # Evaluate
    r2_2d = model_2d.score(X_2d, y_2d)
    print(f"R² Score: {r2_2d:.4f}")
    print()
    
    # Example 2: Multiple features
    print("EXAMPLE 2: Five Features")
    print("-" * 70)
    print("Creating sample data with 5 features...")
    X_multi, y_multi, true_coeffs = create_sample_data_multi()
    print(f"Generated {len(X_multi)} samples with {X_multi.shape[1]} features")
    print()
    
    # Train model
    print("Training the model...")
    model_multi = MultipleLinearRegression()
    model_multi.fit(X_multi, y_multi)
    
    intercept_multi, coeffs_multi = model_multi.get_coefficients()
    print(f"✓ Model trained successfully!")
    print()
    print("Learned coefficients vs. True coefficients:")
    print(f"  Intercept: {intercept_multi:.2f} (True: 10.00)")
    for i, (learned, true) in enumerate(zip(coeffs_multi, true_coeffs)):
        print(f"  Feature {i+1}: {learned:6.2f} (True: {true:6.2f})")
    print()
    
    # Evaluate
    r2_multi = model_multi.score(X_multi, y_multi)
    print(f"R² Score: {r2_multi:.4f}")
    print()
    
    # Make some predictions
    print("Sample predictions:")
    X_test = X_multi[:5]
    y_test = y_multi[:5]
    y_pred = model_multi.predict(X_test)
    
    for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
        error = abs(actual - pred)
        print(f"  Sample {i+1}: Actual = {actual:7.2f}, Predicted = {pred:7.2f}, Error = {error:5.2f}")
    print()
    
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print("Generating plots... (close the plot windows to continue)")
    print()
    
    # Visualizations
    visualize_2d_regression(model_2d, X_2d, y_2d)
    
    print("Comparing simple vs multiple regression...")
    compare_simple_vs_multiple()
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Multiple regression handles multiple features: y = b₀ + b₁x₁ + ... + bₙxₙ")
    print("2. Uses matrix form (Xθ) for efficient computation")
    print("3. Normal Equation provides optimal solution in one step")
    print("4. Adding relevant features generally improves predictions")
    print("5. Feature coefficients show relative importance and direction")
    print("6. Can be visualized in 3D for 2 features")
    print("=" * 70)


if __name__ == "__main__":
    main()
