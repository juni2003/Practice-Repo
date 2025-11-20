import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch using gradient descent.
    
    This class implements binary logistic regression to classify data into
    two categories (0 or 1).
    
    Key Components:
    ---------------
    1. Sigmoid function: Maps linear output to probability (0, 1)
    2. Cost function: Binary Cross-Entropy (Log Loss)
    3. Gradient Descent: Optimizes weights and bias
    4. Prediction: Uses 0.5 threshold for classification
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Step size for gradient descent
        num_iterations : int, default=1000
            Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """
        Compute sigmoid function.
        
        Parameters:
        -----------
        z : numpy array
            Linear combination of weights and features
            
        Returns:
        --------
        numpy array
            Sigmoid values between 0 and 1
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute Binary Cross-Entropy cost function.
        
        Formula: J = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
        
        This cost function is convex, ensuring gradient descent finds
        the global minimum.
        
        Parameters:
        -----------
        y_true : numpy array
            True labels (0 or 1)
        y_pred : numpy array
            Predicted probabilities
            
        Returns:
        --------
        float
            Cost value
        """
        m = len(y_true)
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Algorithm:
        ----------
        1. Initialize weights and bias to zeros
        2. For each iteration:
           a. Compute predictions using sigmoid(X * w + b)
           b. Compute cost (log loss)
           c. Compute gradients
           d. Update weights: w = w - α * dw
           e. Update bias: b = b - α * db
        3. Store cost history for visualization
        
        Parameters:
        -----------
        X : numpy array, shape (m, n)
            Training features
        y : numpy array, shape (m,)
            Training labels (0 or 1)
        """
        m, n = X.shape
        
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.num_iterations}, Cost: {cost:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probabilities for input samples.
        
        Parameters:
        -----------
        X : numpy array, shape (m, n)
            Input features
            
        Returns:
        --------
        numpy array, shape (m,)
            Predicted probabilities between 0 and 1
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input samples.
        
        Decision Rule:
        - If probability >= threshold: predict 1
        - If probability < threshold: predict 0
        
        Parameters:
        -----------
        X : numpy array, shape (m, n)
            Input features
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        numpy array, shape (m,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """
        Plot the cost function over iterations.
        
        A decreasing cost indicates successful learning.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (Log Loss)', fontsize=12)
        plt.title('Cost Function vs Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cost_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Cost history plot saved as 'cost_history.png'")


def generate_classification_data(n_samples=200, random_state=42):
    """
    Generate synthetic binary classification data.
    
    Creates two classes with some overlap to make the problem realistic.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : numpy array
        Features
    y : numpy array
        Labels
    """
    np.random.seed(random_state)
    
    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0
    
    X_class_0 = np.random.randn(n_class_0, 2) + np.array([2, 2])
    X_class_1 = np.random.randn(n_class_1, 2) + np.array([5, 5])
    
    X = np.vstack([X_class_0, X_class_1])
    y = np.hstack([np.zeros(n_class_0), np.ones(n_class_1)])
    
    return X, y


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model : LogisticRegressionScratch
        Trained model
    X_test : numpy array
        Test features
    y_test : numpy array
        True test labels
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = np.mean(y_pred == y_test)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nSample Predictions:")
    print("-"*60)
    print(f"{'True Label':<15} {'Predicted Prob':<20} {'Predicted Label':<20}")
    print("-"*60)
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:<15} {y_pred_proba[i]:<20.4f} {y_pred[i]:<20}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION FROM SCRATCH")
    print("="*60 + "\n")
    
    print("Step 1: Generating synthetic classification data...")
    X, y = generate_classification_data(n_samples=300)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class 0: {np.sum(y == 0)} samples")
    print(f"Class 1: {np.sum(y == 1)} samples")
    
    print("\nStep 2: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nStep 3: Feature scaling (important for gradient descent)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled to mean=0, std=1")
    
    print("\nStep 4: Training Logistic Regression model...")
    print("-"*60)
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    print("\nStep 5: Visualizing cost function decrease...")
    model.plot_cost_history()
    
    print("\nStep 6: Evaluating model on test data...")
    evaluate_model(model, X_test_scaled, y_test)
    
    print("="*60)
    print("LEARNED PARAMETERS")
    print("="*60)
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")
    print("="*60 + "\n")
