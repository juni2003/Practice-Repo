"""
Simple Linear Regression Implementation from Scratch

This module demonstrates simple linear regression with one feature.
Linear regression finds the best-fitting line through data points by
minimizing the sum of squared errors.

Mathematical Formula:
    y = mx + b
    
    Where:
    - y: predicted value (dependent variable)
    - m: slope of the line (how much y changes for each unit change in x)
    - x: input feature (independent variable)
    - b: y-intercept (value of y when x is 0)

Learning Approach:
    We use the Normal Equation (closed-form solution) which directly
    calculates the optimal parameters without iteration:
    
    m = Σ[(x - x̄)(y - ȳ)] / Σ[(x - x̄)²]
    b = ȳ - m * x̄
    
    Where x̄ is the mean of x and ȳ is the mean of y.

Author: Practice Repository
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class SimpleLinearRegression:
    """
    Simple Linear Regression model with one feature.
    
    Attributes:
        slope (float): The slope (m) of the regression line
        intercept (float): The y-intercept (b) of the regression line
        is_fitted (bool): Whether the model has been trained
    """
    
    def __init__(self):
        """Initialize the model with no parameters."""
        self.slope = None
        self.intercept = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleLinearRegression':
        """
        Fit the linear regression model using the Normal Equation.
        
        The Normal Equation provides a closed-form solution:
        - Calculate means of X and y
        - Compute slope: m = Σ[(x - x̄)(y - ȳ)] / Σ[(x - x̄)²]
        - Compute intercept: b = ȳ - m * x̄
        
        Args:
            X (np.ndarray): Training features, shape (n_samples,) or (n_samples, 1)
            y (np.ndarray): Training targets, shape (n_samples,)
        
        Returns:
            self: The fitted model
        
        Example:
            >>> X = np.array([1, 2, 3, 4, 5])
            >>> y = np.array([2, 4, 5, 4, 5])
            >>> model = SimpleLinearRegression()
            >>> model.fit(X, y)
        """
        # Ensure X is 1D array
        if X.ndim > 1:
            X = X.flatten()
        
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (m) using the formula
        # m = Σ[(x - x̄)(y - ȳ)] / Σ[(x - x̄)²]
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        # Prevent division by zero
        if denominator == 0:
            raise ValueError("Cannot fit model: all X values are identical")
        
        self.slope = numerator / denominator
        
        # Calculate intercept (b) using the formula
        # b = ȳ - m * x̄
        self.intercept = y_mean - self.slope * x_mean
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Formula: y_pred = mx + b
        
        Args:
            X (np.ndarray): Features to predict, shape (n_samples,) or (n_samples, 1)
        
        Returns:
            np.ndarray: Predicted values, shape (n_samples,)
        
        Raises:
            ValueError: If model hasn't been fitted yet
        
        Example:
            >>> X_test = np.array([6, 7, 8])
            >>> predictions = model.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # Ensure X is 1D array
        if X.ndim > 1:
            X = X.flatten()
        
        # Apply the linear equation: y = mx + b
        return self.slope * X + self.intercept
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the R² (coefficient of determination) score.
        
        R² represents the proportion of variance in the dependent variable
        that is predictable from the independent variable.
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res = Σ(y - ŷ)² (residual sum of squares)
        - SS_tot = Σ(y - ȳ)² (total sum of squares)
        
        R² ranges from 0 to 1 (can be negative for poor fits):
        - 1.0: Perfect fit
        - 0.5: Model explains 50% of variance
        - 0.0: Model no better than predicting the mean
        
        Args:
            X (np.ndarray): Features, shape (n_samples,)
            y (np.ndarray): True target values, shape (n_samples,)
        
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        
        # Calculate residual sum of squares: Σ(y - ŷ)²
        ss_res = np.sum((y - y_pred) ** 2)
        
        # Calculate total sum of squares: Σ(y - ȳ)²
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Calculate R² score
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def get_params(self) -> Tuple[float, float]:
        """
        Get the model parameters.
        
        Returns:
            Tuple[float, float]: (slope, intercept)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        return self.slope, self.intercept


def create_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data with a linear relationship plus some noise.
    
    This generates synthetic data following: y ≈ 2x + 3 + noise
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) training data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate X values from 0 to 10
    X = np.linspace(0, 10, 50)
    
    # Generate y values with a linear relationship plus random noise
    # True relationship: y = 2x + 3
    y = 2 * X + 3 + np.random.randn(50) * 2  # Adding Gaussian noise
    
    return X, y


def visualize_regression(model: SimpleLinearRegression, X: np.ndarray, y: np.ndarray):
    """
    Visualize the regression line and data points.
    
    Args:
        model (SimpleLinearRegression): Fitted model
        X (np.ndarray): Feature values
        y (np.ndarray): Target values
    """
    # Create predictions for plotting
    X_line = np.linspace(X.min(), X.max(), 100)
    y_pred_line = model.predict(X_line)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot actual data points
    plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data', s=50)
    
    # Plot regression line
    plt.plot(X_line, y_pred_line, color='red', linewidth=2, label='Regression Line')
    
    # Add labels and title
    plt.xlabel('X (Feature)', fontsize=12)
    plt.ylabel('y (Target)', fontsize=12)
    plt.title(f'Simple Linear Regression\ny = {model.slope:.2f}x + {model.intercept:.2f}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add R² score as text
    r2 = model.score(X, y)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def demonstrate_overfitting_underfitting():
    """
    Demonstrate how different slopes might fit (or not fit) the data.
    This helps understand the importance of finding the optimal line.
    """
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2 * X + 3 + np.random.randn(50) * 2
    
    # Fit our model
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    # Create some arbitrary lines for comparison
    X_line = np.linspace(X.min(), X.max(), 100)
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Show different possible lines
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data', s=50)
    plt.plot(X_line, model.predict(X_line), 'r-', linewidth=2, label='Optimal Fit')
    plt.plot(X_line, 1.5 * X_line + 5, 'g--', linewidth=2, label='Steeper slope')
    plt.plot(X_line, 2.5 * X_line + 1, 'm--', linewidth=2, label='Different intercept')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Comparing Different Line Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Show residuals for optimal fit
    plt.subplot(1, 2, 2)
    y_pred = model.predict(X)
    residuals = y - y_pred
    plt.scatter(X, residuals, color='purple', alpha=0.6, s=50)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Residuals (y - ŷ)')
    plt.title('Residual Plot (Optimal Fit)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating simple linear regression.
    """
    print("=" * 70)
    print("SIMPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create sample data
    print("Step 1: Creating sample data...")
    X, y = create_sample_data()
    print(f"Generated {len(X)} data points")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # Initialize and fit the model
    print("Step 2: Training the model...")
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    slope, intercept = model.get_params()
    print(f"✓ Model trained successfully!")
    print(f"  Learned equation: y = {slope:.4f}x + {intercept:.4f}")
    print(f"  (True relationship was: y = 2x + 3 + noise)")
    print()
    
    # Make predictions
    print("Step 3: Making predictions...")
    X_test = np.array([2.5, 5.0, 7.5, 10.0])
    y_pred = model.predict(X_test)
    
    print("Sample predictions:")
    for x_val, y_val in zip(X_test, y_pred):
        print(f"  X = {x_val:5.1f} → Predicted y = {y_val:6.2f}")
    print()
    
    # Evaluate the model
    print("Step 4: Evaluating the model...")
    r2_score = model.score(X, y)
    print(f"R² Score: {r2_score:.4f}")
    print(f"This means the model explains {r2_score * 100:.2f}% of the variance in the data")
    print()
    
    # Interpretation
    print("Step 5: Interpreting the results...")
    print(f"Slope (m = {slope:.4f}):")
    print(f"  → For every 1 unit increase in X, y increases by {slope:.4f} units")
    print(f"Intercept (b = {intercept:.4f}):")
    print(f"  → When X = 0, the predicted y value is {intercept:.4f}")
    print()
    
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print("Generating plots... (close the plot windows to continue)")
    print()
    
    # Visualize the results
    visualize_regression(model, X, y)
    
    # Demonstrate different fits
    print("Showing comparison of different line fits...")
    demonstrate_overfitting_underfitting()
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Simple linear regression finds the best line: y = mx + b")
    print("2. The Normal Equation gives us the optimal m and b directly")
    print("3. R² score tells us how well the line fits the data")
    print("4. Residuals should be randomly distributed around zero")
    print("5. The model learns from data to minimize prediction errors")
    print("=" * 70)


if __name__ == "__main__":
    main()
