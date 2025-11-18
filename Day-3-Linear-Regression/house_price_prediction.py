"""
Real-World Example: House Price Prediction

This module demonstrates a complete end-to-end linear regression pipeline
using house price prediction as a practical example. It brings together all
the concepts learned in this module:

- Data preparation and exploration
- Feature scaling (from Day-1)
- Model training using both Normal Equation and Gradient Descent
- Cost function monitoring
- Model evaluation with multiple metrics
- Predictions on new data
- Visualization of results

Pipeline Steps:
    1. Generate synthetic house data
    2. Explore and visualize the data
    3. Split into training and testing sets
    4. Scale features for better performance
    5. Train models
    6. Evaluate and compare models
    7. Make predictions on new houses
    8. Interpret results

Author: Practice Repository
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    """
    Complete house price prediction model with data preprocessing.
    """
    
    def __init__(self, use_gradient_descent: bool = False, learning_rate: float = 0.01, 
                 n_iterations: int = 1000):
        """
        Initialize the predictor.
        
        Args:
            use_gradient_descent (bool): If True, use gradient descent; else use Normal Equation
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of iterations for gradient descent
        """
        self.use_gradient_descent = use_gradient_descent
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.feature_means = None
        self.feature_stds = None
        self.cost_history = []
        self.is_fitted = False
        self.feature_names = None
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardize features (zero mean, unit variance).
        This is crucial for gradient descent convergence.
        
        Args:
            X (np.ndarray): Features to scale
            fit (bool): If True, compute and store scaling parameters
        
        Returns:
            np.ndarray: Scaled features
        """
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # Prevent division by zero
            self.feature_stds[self.feature_stds == 0] = 1
        
        X_scaled = (X - self.feature_means) / self.feature_stds
        return X_scaled
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to feature matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Solve for optimal parameters using Normal Equation.
        θ = (X'X)^(-1)X'y
        """
        self.coefficients = np.linalg.solve(X.T @ X, X.T @ y)
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """
        Optimize parameters using gradient descent.
        """
        m, n = X.shape
        self.coefficients = np.zeros(n)
        
        for iteration in range(self.n_iterations):
            # Compute predictions
            predictions = X @ self.coefficients
            
            # Compute cost
            cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = (1 / m) * (X.T @ (predictions - y))
            
            # Update parameters
            self.coefficients -= self.learning_rate * gradients
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Fit the model to training data.
        
        Args:
            X (np.ndarray): Training features, shape (m, n)
            y (np.ndarray): Training targets, shape (m,)
            feature_names (list): Names of features for interpretation
        """
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Add bias term
        X_with_bias = self._add_bias(X_scaled)
        
        # Train model
        if self.use_gradient_descent:
            self._gradient_descent(X_with_bias, y)
        else:
            self._normal_equation(X_with_bias, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features, shape (m, n)
        
        Returns:
            np.ndarray: Predictions, shape (m,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Scale features using training statistics
        X_scaled = self._scale_features(X, fit=False)
        
        # Add bias term
        X_with_bias = self._add_bias(X_scaled)
        
        # Make predictions
        return X_with_bias @ self.coefficients
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (coefficient magnitudes).
        
        Returns:
            Dict[str, float]: Feature names and their coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = {}
        
        # Intercept
        importance['Intercept'] = self.coefficients[0]
        
        # Feature coefficients
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                importance[name] = self.coefficients[i + 1]
        else:
            for i in range(len(self.coefficients) - 1):
                importance[f'Feature_{i+1}'] = self.coefficients[i + 1]
        
        return importance


def generate_house_data(n_samples: int = 200, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate synthetic house price data.
    
    Features:
        1. Size (square feet): 1000-4000
        2. Bedrooms: 1-5
        3. Age (years): 0-50
        4. Distance to city center (miles): 1-30
    
    Price formula (approximately):
        Price = 50000 + 150*Size + 20000*Bedrooms - 1000*Age - 3000*Distance + noise
    
    Args:
        n_samples (int): Number of house samples to generate
        random_state (int): Random seed for reproducibility
    
    Returns:
        Tuple[np.ndarray, np.ndarray, list]: Features, prices, feature names
    """
    np.random.seed(random_state)
    
    # Generate features
    size = np.random.uniform(1000, 4000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    distance = np.random.uniform(1, 30, n_samples)
    
    # Combine into feature matrix
    X = np.column_stack([size, bedrooms, age, distance])
    
    # Generate prices with some realistic relationships
    base_price = 50000
    price_per_sqft = 150
    price_per_bedroom = 20000
    depreciation_per_year = 1000
    distance_penalty = 3000
    
    prices = (base_price + 
              price_per_sqft * size + 
              price_per_bedroom * bedrooms - 
              depreciation_per_year * age - 
              distance_penalty * distance +
              np.random.randn(n_samples) * 30000)  # Add noise
    
    # Ensure no negative prices
    prices = np.maximum(prices, 50000)
    
    feature_names = ['Size (sqft)', 'Bedrooms', 'Age (years)', 'Distance (miles)']
    
    return X, prices, feature_names


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Targets
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
    
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Random shuffle
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {'R²': r2, 'RMSE': rmse, 'MAE': mae}


def visualize_results(model: HousePricePredictor, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray, feature_names: list):
    """
    Create comprehensive visualization of results.
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Actual vs Predicted (Training)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_train, y_train_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Price ($)', fontsize=10)
    ax1.set_ylabel('Predicted Price ($)', fontsize=10)
    ax1.set_title(f'Training Set\nR² = {train_metrics["R²"]:.4f}', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Testing)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5, color='orange')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual Price ($)', fontsize=10)
    ax2.set_ylabel('Predicted Price ($)', fontsize=10)
    ax2.set_title(f'Test Set\nR² = {test_metrics["R²"]:.4f}', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    ax3 = plt.subplot(2, 3, 3)
    importance = model.get_feature_importance()
    
    # Exclude intercept from visualization
    feature_imp = {k: v for k, v in importance.items() if k != 'Intercept'}
    names = list(feature_imp.keys())
    values = list(feature_imp.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax3.barh(names, values, color=colors, alpha=0.7)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Coefficient Value', fontsize=10)
    ax3.set_title(f'Feature Importance\nIntercept = ${importance["Intercept"]:,.0f}', 
                 fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Residuals
    ax4 = plt.subplot(2, 3, 4)
    residuals_test = y_test - y_test_pred
    ax4.scatter(y_test_pred, residuals_test, alpha=0.6, s=30, edgecolors='k', 
               linewidth=0.5, color='purple')
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Price ($)', fontsize=10)
    ax4.set_ylabel('Residuals ($)', fontsize=10)
    ax4.set_title('Residual Plot (Test Set)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Error Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals_test, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals ($)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title(f'Error Distribution\nMean = ${np.mean(residuals_test):,.0f}', 
                 fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Cost convergence (if gradient descent)
    ax6 = plt.subplot(2, 3, 6)
    if model.use_gradient_descent and len(model.cost_history) > 0:
        ax6.plot(model.cost_history, linewidth=2)
        ax6.set_xlabel('Iteration', fontsize=10)
        ax6.set_ylabel('Cost', fontsize=10)
        ax6.set_title('Cost Function Convergence', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.axis('off')
        summary_text = f"""
        Model Performance Summary
        {'=' * 40}
        
        Training Set:
          R² Score: {train_metrics['R²']:.4f}
          RMSE:     ${train_metrics['RMSE']:,.2f}
          MAE:      ${train_metrics['MAE']:,.2f}
        
        Test Set:
          R² Score: {test_metrics['R²']:.4f}
          RMSE:     ${test_metrics['RMSE']:,.2f}
          MAE:      ${test_metrics['MAE']:,.2f}
        
        Method: Normal Equation
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating complete house price prediction pipeline.
    """
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: Generate data
    print("Step 1: Generating synthetic house data...")
    print("-" * 70)
    X, y, feature_names = generate_house_data(n_samples=200)
    print(f"✓ Generated {len(X)} house records")
    print(f"✓ Features: {', '.join(feature_names)}")
    print(f"✓ Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print()
    
    # Step 2: Explore data
    print("Step 2: Data exploration...")
    print("-" * 70)
    print(f"{'Feature':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)
    for i, name in enumerate(feature_names):
        print(f"{name:<20} {X[:, i].mean():<12.2f} {X[:, i].std():<12.2f} "
              f"{X[:, i].min():<12.2f} {X[:, i].max():<12.2f}")
    print(f"{'Price':<20} {y.mean():<12.2f} {y.std():<12.2f} {y.min():<12.2f} {y.max():<12.2f}")
    print()
    
    # Step 3: Split data
    print("Step 3: Splitting data into train/test sets...")
    print("-" * 70)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples:     {len(X_test)}")
    print()
    
    # Step 4: Train with Normal Equation
    print("Step 4: Training model (Normal Equation)...")
    print("-" * 70)
    model_normal = HousePricePredictor(use_gradient_descent=False)
    model_normal.fit(X_train, y_train, feature_names)
    print("✓ Model trained successfully!")
    print()
    
    # Step 5: Train with Gradient Descent
    print("Step 5: Training model (Gradient Descent)...")
    print("-" * 70)
    model_gd = HousePricePredictor(use_gradient_descent=True, learning_rate=0.01, n_iterations=1000)
    model_gd.fit(X_train, y_train, feature_names)
    print("✓ Model trained successfully!")
    print(f"✓ Final cost: {model_gd.cost_history[-1]:.4f}")
    print()
    
    # Step 6: Evaluate models
    print("Step 6: Evaluating models...")
    print("-" * 70)
    
    # Normal Equation
    y_train_pred_normal = model_normal.predict(X_train)
    y_test_pred_normal = model_normal.predict(X_test)
    train_metrics_normal = evaluate_model(y_train, y_train_pred_normal)
    test_metrics_normal = evaluate_model(y_test, y_test_pred_normal)
    
    # Gradient Descent
    y_train_pred_gd = model_gd.predict(X_train)
    y_test_pred_gd = model_gd.predict(X_test)
    train_metrics_gd = evaluate_model(y_train, y_train_pred_gd)
    test_metrics_gd = evaluate_model(y_test, y_test_pred_gd)
    
    print("Normal Equation Results:")
    print(f"  Training   - R²: {train_metrics_normal['R²']:.4f}, RMSE: ${train_metrics_normal['RMSE']:,.2f}")
    print(f"  Test       - R²: {test_metrics_normal['R²']:.4f}, RMSE: ${test_metrics_normal['RMSE']:,.2f}")
    print()
    
    print("Gradient Descent Results:")
    print(f"  Training   - R²: {train_metrics_gd['R²']:.4f}, RMSE: ${train_metrics_gd['RMSE']:,.2f}")
    print(f"  Test       - R²: {test_metrics_gd['R²']:.4f}, RMSE: ${test_metrics_gd['RMSE']:,.2f}")
    print()
    
    # Step 7: Feature importance
    print("Step 7: Interpreting feature importance...")
    print("-" * 70)
    importance = model_normal.get_feature_importance()
    print("Feature Coefficients (after scaling):")
    for feature, coeff in importance.items():
        if feature == 'Intercept':
            print(f"  {feature:<20}: ${coeff:,.2f}")
        else:
            direction = "increases" if coeff > 0 else "decreases"
            print(f"  {feature:<20}: {coeff:8.2f} (price {direction})")
    print()
    
    # Step 8: Make predictions on new houses
    print("Step 8: Making predictions on new houses...")
    print("-" * 70)
    
    new_houses = np.array([
        [2000, 3, 10, 5],   # Medium house, newer, close to city
        [3500, 4, 2, 15],   # Large house, very new, moderate distance
        [1200, 2, 40, 25],  # Small house, old, far from city
    ])
    
    predictions = model_normal.predict(new_houses)
    
    print(f"{'Size':<6} {'Beds':<6} {'Age':<6} {'Dist':<6} {'Predicted Price':<20}")
    print("-" * 70)
    for i, (house, price) in enumerate(zip(new_houses, predictions)):
        print(f"{house[0]:<6.0f} {house[1]:<6.0f} {house[2]:<6.0f} {house[3]:<6.0f} ${price:,.2f}")
    print()
    
    # Visualization
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print("Generating comprehensive results visualization...")
    print("(close the plot window to continue)")
    print()
    
    visualize_results(model_normal, X_train, y_train, X_test, y_test, feature_names)
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Complete pipeline: Data → Preprocessing → Training → Evaluation")
    print("2. Feature scaling is crucial for gradient descent convergence")
    print("3. Both Normal Equation and Gradient Descent give similar results")
    print("4. Train/test split helps evaluate generalization")
    print("5. R² and RMSE provide different insights into performance")
    print("6. Feature coefficients reveal relationships between features and price")
    print("7. Residual analysis helps identify model weaknesses")
    print("=" * 70)
    print()
    print("🎉 Congratulations! You've completed the Linear Regression module!")
    print()


if __name__ == "__main__":
    main()
