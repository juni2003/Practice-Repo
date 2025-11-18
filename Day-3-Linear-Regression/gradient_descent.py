"""
Gradient Descent Algorithm Implementation from Scratch

Gradient Descent is an iterative optimization algorithm used to find the
minimum of a function. In machine learning, we use it to minimize the cost
function and find optimal model parameters.

Key Concepts:
    1. Start with random parameter values
    2. Calculate the gradient (slope) of the cost function
    3. Move in the opposite direction of the gradient
    4. Repeat until convergence

Mathematical Update Rule:
    θⱼ := θⱼ - α * (∂J/∂θⱼ)
    
    Where:
    - θⱼ: parameter j
    - α: learning rate (step size)
    - ∂J/∂θⱼ: partial derivative of cost function with respect to θⱼ

For Linear Regression:
    ∂J/∂θⱼ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾ⱼ
    
    Where:
    - m: number of training examples
    - h(x): hypothesis function (prediction)
    - x⁽ⁱ⁾ⱼ: j-th feature of i-th training example

Author: Practice Repository
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class GradientDescentRegression:
    """
    Linear Regression trained using Gradient Descent.
    
    Unlike the Normal Equation which computes parameters in one step,
    Gradient Descent iteratively improves parameters to minimize cost.
    
    Attributes:
        learning_rate (float): Step size for parameter updates (α)
        n_iterations (int): Number of iterations to run
        coefficients (np.ndarray): Model parameters (θ)
        cost_history (list): History of cost function values
        is_fitted (bool): Whether the model has been trained
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the Gradient Descent model.
        
        Args:
            learning_rate (float): Learning rate (α). Common values: 0.001, 0.01, 0.1
                - Too high: may overshoot and diverge
                - Too low: slow convergence
            n_iterations (int): Number of iterations to train
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.cost_history = []
        self.is_fitted = False
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute the Mean Squared Error cost function.
        
        J(θ) = (1/2m) * Σ(h(x) - y)²
        
        The factor of 1/2 is for mathematical convenience (derivative becomes cleaner).
        
        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n+1)
            y (np.ndarray): Target values, shape (m,)
            theta (np.ndarray): Current parameters, shape (n+1,)
        
        Returns:
            float: Cost value
        """
        m = len(y)
        predictions = X @ theta
        squared_errors = (predictions - y) ** 2
        cost = (1 / (2 * m)) * np.sum(squared_errors)
        return cost
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the cost function.
        
        ∂J/∂θⱼ = (1/m) * Σ(h(x) - y) * xⱼ
        
        In vectorized form:
        ∇J(θ) = (1/m) * Xᵀ(Xθ - y)
        
        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n+1)
            y (np.ndarray): Target values, shape (m,)
            theta (np.ndarray): Current parameters, shape (n+1,)
        
        Returns:
            np.ndarray: Gradients for each parameter, shape (n+1,)
        """
        m = len(y)
        predictions = X @ theta
        errors = predictions - y
        gradients = (1 / m) * (X.T @ errors)
        return gradients
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'GradientDescentRegression':
        """
        Fit the model using Batch Gradient Descent.
        
        Algorithm:
        1. Initialize parameters (usually to zeros or small random values)
        2. For each iteration:
           a. Compute predictions
           b. Calculate cost
           c. Compute gradients
           d. Update parameters: θ := θ - α∇J(θ)
        3. Repeat until convergence or max iterations
        
        Args:
            X (np.ndarray): Training features, shape (m, n)
            y (np.ndarray): Training targets, shape (m,)
            verbose (bool): If True, print cost every 100 iterations
        
        Returns:
            self: The fitted model
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        m, n = X.shape
        
        # Add bias term (column of ones)
        X_with_bias = np.c_[np.ones(m), X]
        
        # Initialize parameters to zeros
        self.coefficients = np.zeros(n + 1)
        
        # Gradient Descent loop
        for iteration in range(self.n_iterations):
            # Compute cost (for tracking progress)
            cost = self._compute_cost(X_with_bias, y, self.coefficients)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = self._compute_gradients(X_with_bias, y, self.coefficients)
            
            # Update parameters
            self.coefficients = self.coefficients - self.learning_rate * gradients
            
            # Print progress
            if verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration:4d}: Cost = {cost:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Features, shape (m, n)
        
        Returns:
            np.ndarray: Predictions, shape (m,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        m = X.shape[0]
        X_with_bias = np.c_[np.ones(m), X]
        
        return X_with_bias @ self.coefficients
    
    def get_coefficients(self) -> Tuple[float, np.ndarray]:
        """Get model parameters."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        return self.coefficients[0], self.coefficients[1:]


def visualize_cost_convergence(models: List[Tuple[str, GradientDescentRegression]]):
    """
    Visualize how cost decreases over iterations for different learning rates.
    
    Args:
        models (List[Tuple[str, GradientDescentRegression]]): List of (name, model) tuples
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Cost vs Iterations (Linear scale)
    plt.subplot(1, 2, 1)
    for name, model in models:
        plt.plot(model.cost_history, label=name, linewidth=2)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Cost J(θ)', fontsize=11)
    plt.title('Cost Function Convergence\n(Linear Scale)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost vs Iterations (Log scale)
    plt.subplot(1, 2, 2)
    for name, model in models:
        plt.plot(model.cost_history, label=name, linewidth=2)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Cost J(θ)', fontsize=11)
    plt.title('Cost Function Convergence\n(Log Scale)', fontsize=12, fontweight='bold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_learning_rates():
    """
    Demonstrate the impact of different learning rates on convergence.
    """
    # Generate sample data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = 2 * X + 3 + np.random.randn(100) * 2
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    models = []
    
    print("\nComparing Different Learning Rates:")
    print("-" * 70)
    
    for lr in learning_rates:
        model = GradientDescentRegression(learning_rate=lr, n_iterations=500)
        model.fit(X, y)
        
        intercept, slope = model.get_coefficients()
        final_cost = model.cost_history[-1]
        
        print(f"Learning Rate = {lr:5.3f}:")
        print(f"  Final Cost: {final_cost:.4f}")
        print(f"  Parameters: y = {slope[0]:.4f}x + {intercept:.4f}")
        print(f"  Expected:   y = 2.0000x + 3.0000")
        print()
        
        models.append((f"α = {lr}", model))
    
    visualize_cost_convergence(models)


def visualize_gradient_descent_path():
    """
    Visualize how gradient descent navigates the cost surface.
    Shows the path taken by the algorithm in parameter space.
    """
    # Simple 1D example for visualization
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    
    # Create grid for cost surface
    m_vals = np.linspace(0, 3, 100)
    b_vals = np.linspace(-1, 5, 100)
    M, B = np.meshgrid(m_vals, b_vals)
    
    # Compute cost for each (m, b) combination
    costs = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            m, b = M[i, j], B[i, j]
            predictions = m * X + b
            costs[i, j] = np.mean((predictions - y) ** 2) / 2
    
    # Run gradient descent and track path
    model = GradientDescentRegression(learning_rate=0.1, n_iterations=50)
    
    # Manually track parameter history
    param_history = []
    X_train = X.reshape(-1, 1)
    
    # Initialize
    m, n = X_train.shape
    X_with_bias = np.c_[np.ones(m), X_train]
    theta = np.zeros(2)
    
    # Run gradient descent manually to track path
    for _ in range(50):
        param_history.append(theta.copy())
        predictions = X_with_bias @ theta
        errors = predictions - y
        gradients = (1 / m) * (X_with_bias.T @ errors)
        theta = theta - 0.1 * gradients
    
    param_history = np.array(param_history)
    
    # Create visualization
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Contour plot with gradient descent path
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(M, B, costs, levels=20, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(param_history[:, 1], param_history[:, 0], 'ro-', linewidth=2, 
             markersize=5, label='GD Path')
    ax1.plot(param_history[0, 1], param_history[0, 0], 'go', markersize=10, 
             label='Start')
    ax1.plot(param_history[-1, 1], param_history[-1, 0], 'r*', markersize=15, 
             label='End')
    ax1.set_xlabel('Slope (m)', fontsize=11)
    ax1.set_ylabel('Intercept (b)', fontsize=11)
    ax1.set_title('Gradient Descent Path\non Cost Surface', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter values over iterations
    ax2 = fig.add_subplot(122)
    ax2.plot(param_history[:, 1], label='Slope (m)', linewidth=2)
    ax2.plot(param_history[:, 0], label='Intercept (b)', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Parameter Value', fontsize=11)
    ax2.set_title('Parameter Convergence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating gradient descent.
    """
    print("=" * 70)
    print("GRADIENT DESCENT ALGORITHM DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create sample data
    print("Creating sample data...")
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = 2 * X + 3 + np.random.randn(100) * 2
    print(f"Generated {len(X)} samples")
    print(f"True relationship: y = 2x + 3 + noise")
    print()
    
    # Train model with gradient descent
    print("Training model with Gradient Descent...")
    print("-" * 70)
    model = GradientDescentRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y, verbose=True)
    print("-" * 70)
    print()
    
    # Get results
    intercept, slope = model.get_coefficients()
    print(f"✓ Training complete!")
    print(f"  Learned equation: y = {slope[0]:.4f}x + {intercept:.4f}")
    print(f"  Expected:        y = 2.0000x + 3.0000")
    print(f"  Initial cost: {model.cost_history[0]:.4f}")
    print(f"  Final cost:   {model.cost_history[-1]:.4f}")
    print()
    
    # Predictions
    print("Making predictions...")
    X_test = np.array([2.5, 5.0, 7.5, 10.0])
    y_pred = model.predict(X_test)
    
    print("Sample predictions:")
    for x_val, y_val in zip(X_test, y_pred):
        print(f"  X = {x_val:5.1f} → Predicted y = {y_val:6.2f}")
    print()
    
    print("=" * 70)
    print("UNDERSTANDING LEARNING RATES")
    print("=" * 70)
    
    # Compare different learning rates
    compare_learning_rates()
    
    print("\nVisualizing gradient descent path...")
    visualize_gradient_descent_path()
    
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Gradient Descent iteratively optimizes parameters")
    print("2. Learning rate (α) controls step size:")
    print("   - Too large: may overshoot and diverge")
    print("   - Too small: slow convergence")
    print("3. Cost should decrease with each iteration")
    print("4. Converges to same solution as Normal Equation")
    print("5. More scalable for large datasets than Normal Equation")
    print("6. Feature scaling helps convergence speed")
    print("=" * 70)


if __name__ == "__main__":
    main()
