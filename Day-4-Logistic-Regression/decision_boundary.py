import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression_from_scratch import LogisticRegressionScratch

def plot_decision_boundary(model, X, y, title="Decision Boundary", filename="decision_boundary.png"):
    """
    Visualize the decision boundary of a logistic regression model.
    
    The decision boundary is where the model predicts probability = 0.5,
    separating the two classes.
    
    Parameters:
    -----------
    model : LogisticRegressionScratch
        Trained logistic regression model
    X : numpy array, shape (m, 2)
        Feature matrix (must be 2D for visualization)
    y : numpy array, shape (m,)
        True labels
    title : str
        Plot title
    filename : str
        Output filename
    """
    h = 0.02
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(contour, label='Predicted Probability')
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='dashed')
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=100, alpha=0.9)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(f'{title}\n(Dashed line = Decision Boundary)', fontsize=13)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    contour_hard = plt.contourf(xx, yy, (Z >= 0.5).astype(int), 
                                levels=1, cmap='RdYlBu', alpha=0.6)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
               edgecolors='black', s=100, alpha=0.9)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Hard Classification\n(Blue = Class 0, Red = Class 1)', fontsize=13)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Decision boundary visualization saved as '{filename}'")


def linear_separable_data():
    """
    Create and visualize linearly separable data.
    
    This is the ideal case for logistic regression.
    """
    print("\n" + "="*60)
    print("CASE 1: LINEARLY SEPARABLE DATA")
    print("="*60 + "\n")
    
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.05,
        class_sep=2.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training logistic regression...")
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nPlotting decision boundary...")
    X_scaled = scaler.transform(X)
    plot_decision_boundary(
        model, X_scaled, y,
        title="Linear Decision Boundary (Ideal Case)",
        filename="decision_boundary_linear.png"
    )
    
    return model, X_scaled, y


def non_linear_data_circles():
    """
    Create and visualize non-linearly separable data (circles).
    
    This shows the limitation of logistic regression with linear decision boundary.
    """
    print("\n" + "="*60)
    print("CASE 2: NON-LINEAR DATA - CIRCLES")
    print("="*60 + "\n")
    
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training logistic regression...")
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Note: Low accuracy because data is NOT linearly separable!")
    
    print("\nPlotting decision boundary...")
    X_scaled = scaler.transform(X)
    plot_decision_boundary(
        model, X_scaled, y,
        title="Linear Boundary on Non-Linear Data (Poor Fit)",
        filename="decision_boundary_circles.png"
    )
    
    print("\nObservation: Linear decision boundary cannot separate circular data!")


def non_linear_data_moons():
    """
    Create and visualize non-linearly separable data (moons).
    
    Another example showing limitation of linear decision boundary.
    """
    print("\n" + "="*60)
    print("CASE 3: NON-LINEAR DATA - MOONS")
    print("="*60 + "\n")
    
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training logistic regression...")
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Note: Moderate accuracy - data has some non-linear pattern!")
    
    print("\nPlotting decision boundary...")
    X_scaled = scaler.transform(X)
    plot_decision_boundary(
        model, X_scaled, y,
        title="Linear Boundary on Moon-Shaped Data",
        filename="decision_boundary_moons.png"
    )


def demonstrate_threshold_effect():
    """
    Show how changing the decision threshold affects classification.
    """
    print("\n" + "="*60)
    print("CASE 4: EFFECT OF DECISION THRESHOLD")
    print("="*60 + "\n")
    
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.1,
        class_sep=1.5,
        random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=1000)
    model.fit(X_scaled, y)
    
    thresholds = [0.3, 0.5, 0.7]
    
    plt.figure(figsize=(15, 4))
    
    for idx, threshold in enumerate(thresholds):
        predictions = model.predict(X_scaled, threshold=threshold)
        accuracy = np.mean(predictions == y)
        
        plt.subplot(1, 3, idx + 1)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=predictions, 
                   cmap='RdYlBu', edgecolors='black', s=100, alpha=0.7)
        plt.xlabel('Feature 1', fontsize=11)
        plt.ylabel('Feature 2', fontsize=11)
        plt.title(f'Threshold = {threshold}\nAccuracy: {accuracy:.2f}', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Threshold comparison saved as 'threshold_comparison.png'")
    print("\nKey Insight: Changing threshold trades off between classes")
    print("- Lower threshold: More samples classified as Class 1")
    print("- Higher threshold: More samples classified as Class 0")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DECISION BOUNDARY VISUALIZATION")
    print("="*60)
    
    linear_separable_data()
    
    non_linear_data_circles()
    
    non_linear_data_moons()
    
    demonstrate_threshold_effect()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Logistic regression creates a LINEAR decision boundary")
    print("2. Works best when data is linearly separable")
    print("3. Struggles with circular or complex non-linear patterns")
    print("4. For non-linear data, consider:")
    print("   - Feature engineering (polynomial features)")
    print("   - Kernel methods (SVM)")
    print("   - Non-linear models (Decision Trees, Neural Networks)")
    print("5. Decision threshold can be adjusted based on problem needs")
    print("="*60 + "\n")
