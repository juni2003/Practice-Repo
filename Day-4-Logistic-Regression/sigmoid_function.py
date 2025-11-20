import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Compute the sigmoid (logistic) function.
    
    The sigmoid function maps any real number to a value between 0 and 1.
    Formula: σ(z) = 1 / (1 + e^(-z))
    
    Parameters:
    -----------
    z : float or numpy array
        Input value(s) to the sigmoid function
        
    Returns:
    --------
    float or numpy array
        Sigmoid value(s) between 0 and 1
        
    Mathematical Properties:
    -----------------------
    - σ(0) = 0.5
    - σ(z) approaches 1 as z approaches infinity
    - σ(z) approaches 0 as z approaches negative infinity
    - Derivative: σ'(z) = σ(z) * (1 - σ(z))
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Compute the derivative of the sigmoid function.
    
    This is useful for gradient descent optimization.
    Formula: σ'(z) = σ(z) * (1 - σ(z))
    
    Parameters:
    -----------
    z : float or numpy array
        Input value(s)
        
    Returns:
    --------
    float or numpy array
        Derivative value(s)
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def visualize_sigmoid():
    """
    Visualize the sigmoid function and its derivative.
    
    This helps understand:
    - How sigmoid squashes values to (0, 1)
    - The S-shaped curve
    - Where the function changes most rapidly
    """
    z = np.linspace(-10, 10, 200)
    
    sig_values = sigmoid(z)
    sig_derivative_values = sigmoid_derivative(z)
    
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Sigmoid Function
    plt.subplot(1, 2, 1)
    plt.plot(z, sig_values, 'b-', linewidth=2, label='Sigmoid Function')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold (0.5)')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='z = 0')
    plt.grid(True, alpha=0.3)
    plt.xlabel('z (input)', fontsize=12)
    plt.ylabel('σ(z)', fontsize=12)
    plt.title('Sigmoid Function: σ(z) = 1 / (1 + e^(-z))', fontsize=14)
    plt.legend()
    plt.ylim(-0.1, 1.1)
    
    # Annotate key points
    plt.plot(0, 0.5, 'ro', markersize=8)
    plt.annotate('(0, 0.5)', xy=(0, 0.5), xytext=(1, 0.3),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Sigmoid Derivative
    plt.subplot(1, 2, 2)
    plt.plot(z, sig_derivative_values, 'r-', linewidth=2, label='Sigmoid Derivative')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Maximum at z = 0')
    plt.grid(True, alpha=0.3)
    plt.xlabel('z (input)', fontsize=12)
    plt.ylabel("σ'(z)", fontsize=12)
    plt.title('Sigmoid Derivative: σ\'(z) = σ(z) * (1 - σ(z))', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sigmoid_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sigmoid function visualization saved as 'sigmoid_visualization.png'")


def demonstrate_sigmoid_properties():
    """
    Demonstrate key properties of the sigmoid function with examples.
    """
    print("=" * 60)
    print("SIGMOID FUNCTION PROPERTIES")
    print("=" * 60)
    
    test_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    
    print("\nSigmoid outputs for various inputs:")
    print("-" * 60)
    print(f"{'Input (z)':<15} {'Sigmoid(z)':<20} {'Interpretation':<25}")
    print("-" * 60)
    
    for z in test_values:
        sig_val = sigmoid(z)
        
        if sig_val < 0.3:
            interpretation = "Strong Class 0"
        elif sig_val < 0.45:
            interpretation = "Likely Class 0"
        elif sig_val < 0.55:
            interpretation = "Uncertain"
        elif sig_val < 0.7:
            interpretation = "Likely Class 1"
        else:
            interpretation = "Strong Class 1"
            
        print(f"{z:<15} {sig_val:<20.6f} {interpretation:<25}")
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. Output is ALWAYS between 0 and 1 (perfect for probabilities)")
    print("2. At z = 0, output is exactly 0.5 (decision boundary)")
    print("3. Large positive z values give output close to 1")
    print("4. Large negative z values give output close to 0")
    print("5. Function is symmetric around the point (0, 0.5)")


def compare_with_step_function():
    """
    Compare sigmoid with a step function to show why sigmoid is better.
    """
    z = np.linspace(-5, 5, 200)
    sigmoid_values = sigmoid(z)
    
    step_values = np.where(z >= 0, 1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid_values, 'b-', linewidth=2, label='Sigmoid (Smooth)', alpha=0.8)
    plt.plot(z, step_values, 'r-', linewidth=2, label='Step Function (Not Smooth)', alpha=0.8)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('z', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.title('Sigmoid vs Step Function', fontsize=14)
    plt.legend(fontsize=11)
    plt.ylim(-0.1, 1.1)
    
    plt.text(2, 0.2, 'Sigmoid is differentiable\n(good for gradient descent)', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.text(-4, 0.7, 'Step function is not differentiable\n(bad for optimization)', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sigmoid_vs_step.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison saved as 'sigmoid_vs_step.png'")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIGMOID FUNCTION EXPLORATION")
    print("="*60 + "\n")
    
    demonstrate_sigmoid_properties()
    
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60 + "\n")
    
    visualize_sigmoid()
    compare_with_step_function()
    
    print("\n" + "="*60)
    print("WHY SIGMOID FOR LOGISTIC REGRESSION?")
    print("="*60)
    print("1. Outputs are in range (0, 1) - interpretable as probabilities")
    print("2. Smooth and differentiable - works with gradient descent")
    print("3. Non-linear - can model complex relationships")
    print("4. Has nice mathematical properties for derivation")
    print("="*60 + "\n")
