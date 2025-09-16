"""
Visual Comparison of Feature Scaling Techniques

This script creates visual comparisons of different feature scaling methods
to help understand their effects on data distribution.

Features:
- Before/after scaling visualizations
- Side-by-side comparisons of different scalers
- Distribution plots showing the effect of each scaler
- Real dataset examples

Usage:
    python visual_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')

def create_sample_data():
    """Create sample data with different scales and outliers for demonstration."""
    np.random.seed(42)
    
    # Feature 1: Normal distribution, small scale
    feature1 = np.random.normal(5, 1, 1000)
    
    # Feature 2: Normal distribution, large scale with outliers
    feature2 = np.random.normal(1000, 200, 1000)
    feature2[np.random.choice(1000, 50)] += np.random.normal(0, 1000, 50)  # Add outliers
    
    # Feature 3: Uniform distribution, medium scale
    feature3 = np.random.uniform(20, 80, 1000)
    
    return np.column_stack([feature1, feature2, feature3])

def plot_distributions_comparison(data, feature_names=None):
    """Plot original vs scaled distributions for different scalers."""
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(data.shape[1])]
    
    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }
    
    fig, axes = plt.subplots(len(scalers), data.shape[1], figsize=(15, 12))
    fig.suptitle('Feature Scaling Comparison: Distribution Changes', fontsize=16, fontweight='bold')
    
    for row, (scaler_name, scaler) in enumerate(scalers.items()):
        if scaler is None:
            scaled_data = data
        else:
            scaled_data = scaler.fit_transform(data)
        
        for col in range(data.shape[1]):
            ax = axes[row, col]
            
            # Plot histogram and KDE
            ax.hist(scaled_data[:, col], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add statistics text
            mean_val = np.mean(scaled_data[:, col])
            std_val = np.std(scaled_data[:, col])
            min_val = np.min(scaled_data[:, col])
            max_val = np.max(scaled_data[:, col])
            
            stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nmin={min_val:.2f}\nmax={max_val:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
            
            # Set titles
            if row == 0:
                ax.set_title(f'{feature_names[col]}', fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{scaler_name}', fontweight='bold', rotation=90)
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_2d_scaling_effects():
    """Show how scaling affects 2D data visualization."""
    # Create 2D sample data
    np.random.seed(42)
    X = np.random.multivariate_normal([10, 1000], [[1, 0.5], [0.5, 40000]], 500)
    
    scalers = {
        'Original Data': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('2D Visualization: Effect of Different Scalers', fontsize=16, fontweight='bold')
    
    for idx, (name, scaler) in enumerate(scalers.items()):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        if scaler is None:
            X_scaled = X
        else:
            X_scaled = scaler.fit_transform(X)
        
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, s=20)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        
        # Add range information
        x_range = f'X range: [{X_scaled[:, 0].min():.2f}, {X_scaled[:, 0].max():.2f}]'
        y_range = f'Y range: [{X_scaled[:, 1].min():.2f}, {X_scaled[:, 1].max():.2f}]'
        ax.text(0.02, 0.98, f'{x_range}\n{y_range}', transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_real_dataset_example():
    """Demonstrate scaling effects on a real dataset."""
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data[:, :4]  # Use first 4 features for visualization
    feature_names = data.feature_names[:4]
    
    print("Real Dataset Example: Breast Cancer Dataset (first 4 features)")
    print("=" * 60)
    
    # Show original statistics
    print("\nOriginal Data Statistics:")
    print("-" * 30)
    for i, name in enumerate(feature_names):
        print(f"{name:25}: mean={X[:, i].mean():.2f}, std={X[:, i].std():.2f}, "
              f"range=[{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    
    # Plot distributions
    plot_distributions_comparison(X, feature_names)

def demonstrate_outlier_sensitivity():
    """Show how different scalers handle outliers."""
    np.random.seed(42)
    
    # Create data with outliers
    normal_data = np.random.normal(0, 1, 950)
    outliers = np.random.normal(0, 1, 50) * 10  # Extreme outliers
    data_with_outliers = np.concatenate([normal_data, outliers])
    data_with_outliers = data_with_outliers.reshape(-1, 1)
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Outlier Sensitivity Comparison', fontsize=16, fontweight='bold')
    
    # Original data
    axes[0].boxplot(data_with_outliers.flatten())
    axes[0].set_title('Original Data\n(with outliers)', fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Scaled data
    for idx, (name, scaler) in enumerate(scalers.items()):
        scaled_data = scaler.fit_transform(data_with_outliers)
        axes[idx + 1].boxplot(scaled_data.flatten())
        axes[idx + 1].set_title(f'{name}\n(scaled)', fontweight='bold')
        axes[idx + 1].grid(True, alpha=0.3)
        
        # Add quartile information
        q1, median, q3 = np.percentile(scaled_data.flatten(), [25, 50, 75])
        axes[idx + 1].text(0.02, 0.98, f'Q1: {q1:.2f}\nMedian: {median:.2f}\nQ3: {q3:.2f}', 
                          transform=axes[idx + 1].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("\nOutlier Sensitivity Analysis:")
    print("-" * 40)
    print("• StandardScaler: Sensitive to outliers (moves mean and std)")
    print("• MinMaxScaler: Very sensitive to outliers (compresses normal data)")
    print("• RobustScaler: Less sensitive (uses median and IQR)")

def main():
    """Main function to run all visualizations."""
    print("Feature Scaling Visual Comparison")
    print("=" * 40)
    print("This script will show you how different scaling methods affect your data.\n")
    
    print("1. Creating sample data with different scales...")
    sample_data = create_sample_data()
    
    print("2. Plotting distribution comparisons...")
    plot_distributions_comparison(sample_data)
    
    print("3. Showing 2D scaling effects...")
    plot_2d_scaling_effects()
    
    print("4. Demonstrating outlier sensitivity...")
    demonstrate_outlier_sensitivity()
    
    print("5. Real dataset example...")
    plot_real_dataset_example()
    
    print("\nKey Takeaways:")
    print("-" * 20)
    print("• StandardScaler: Centers data around 0 with unit variance")
    print("• MinMaxScaler: Scales data to [0,1] range")
    print("• RobustScaler: Uses median and IQR, less sensitive to outliers")
    print("• MaxAbsScaler: Scales by maximum absolute value")
    print("\nChoose based on your data characteristics and algorithm requirements!")

if __name__ == "__main__":
    main()