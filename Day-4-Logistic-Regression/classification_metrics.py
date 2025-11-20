import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression_from_scratch import LogisticRegressionScratch

def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.
    
    Confusion Matrix Structure:
                    Predicted
                    0       1
    Actual  0      TN      FP
            1      FN      TP
    
    Parameters:
    -----------
    y_true : numpy array
        True labels
    y_pred : numpy array
        Predicted labels
        
    Returns:
    --------
    dict
        Dictionary with TP, TN, FP, FN counts
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def plot_confusion_matrix(cm_dict, filename="confusion_matrix.png"):
    """
    Visualize confusion matrix as a heatmap.
    
    Parameters:
    -----------
    cm_dict : dict
        Dictionary with confusion matrix values
    filename : str
        Output filename
    """
    cm_array = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    for i in range(2):
        for j in range(2):
            value = cm_array[i, j]
            if i == 0 and j == 0:
                label = f'TN\n{value}'
            elif i == 0 and j == 1:
                label = f'FP\n{value}'
            elif i == 1 and j == 0:
                label = f'FN\n{value}'
            else:
                label = f'TP\n{value}'
            plt.text(j + 0.5, i + 0.7, label, ha='center', va='center', 
                    fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved as '{filename}'")


def accuracy(cm_dict):
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    
    Accuracy measures overall correctness of predictions.
    
    WARNING: Can be misleading with imbalanced datasets!
    Example: If 95% of data is class 0, always predicting 0 gives 95% accuracy.
    """
    TP, TN, FP, FN = cm_dict['TP'], cm_dict['TN'], cm_dict['FP'], cm_dict['FN']
    return (TP + TN) / (TP + TN + FP + FN)


def precision(cm_dict):
    """
    Calculate precision: TP / (TP + FP)
    
    Precision answers: "Of all positive predictions, how many were correct?"
    
    High precision means few false positives.
    Important when false positives are costly.
    Example: Email spam detection (don't want to mark important emails as spam)
    """
    TP, FP = cm_dict['TP'], cm_dict['FP']
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall(cm_dict):
    """
    Calculate recall (sensitivity): TP / (TP + FN)
    
    Recall answers: "Of all actual positives, how many did we find?"
    
    High recall means few false negatives.
    Important when false negatives are costly.
    Example: Disease detection (don't want to miss sick patients)
    """
    TP, FN = cm_dict['TP'], cm_dict['FN']
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def f1_score(cm_dict):
    """
    Calculate F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    
    F1 Score is the harmonic mean of precision and recall.
    
    Use F1 when you need a balance between precision and recall.
    Particularly useful for imbalanced datasets.
    
    Range: 0 to 1 (higher is better)
    """
    prec = precision(cm_dict)
    rec = recall(cm_dict)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def specificity(cm_dict):
    """
    Calculate specificity (True Negative Rate): TN / (TN + FP)
    
    Specificity answers: "Of all actual negatives, how many did we correctly identify?"
    
    Useful in medical testing and fraud detection.
    """
    TN, FP = cm_dict['TN'], cm_dict['FP']
    if TN + FP == 0:
        return 0.0
    return TN / (TN + FP)


def print_all_metrics(cm_dict):
    """
    Print all classification metrics in a formatted table.
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    print("\nConfusion Matrix Components:")
    print("-"*60)
    print(f"True Positives (TP):   {cm_dict['TP']:<10} (Correctly predicted positive)")
    print(f"True Negatives (TN):   {cm_dict['TN']:<10} (Correctly predicted negative)")
    print(f"False Positives (FP):  {cm_dict['FP']:<10} (Type I Error)")
    print(f"False Negatives (FN):  {cm_dict['FN']:<10} (Type II Error)")
    
    print("\nPerformance Metrics:")
    print("-"*60)
    
    acc = accuracy(cm_dict)
    prec = precision(cm_dict)
    rec = recall(cm_dict)
    f1 = f1_score(cm_dict)
    spec = specificity(cm_dict)
    
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:   {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:      {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score:    {f1:.4f} ({f1*100:.2f}%)")
    print(f"Specificity: {spec:.4f} ({spec*100:.2f}%)")
    
    print("\n" + "="*60)


def plot_metrics_comparison(metrics_dict, filename="metrics_comparison.png"):
    """
    Create a bar chart comparing different metrics.
    """
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Baseline')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Metrics comparison saved as '{filename}'")


def demonstrate_imbalanced_dataset():
    """
    Show why accuracy alone is insufficient for imbalanced datasets.
    """
    print("\n" + "="*60)
    print("IMBALANCED DATASET EXAMPLE")
    print("="*60 + "\n")
    
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        flip_y=0.05,
        random_state=42
    )
    
    print(f"Class 0 samples: {np.sum(y == 0)} (90%)")
    print(f"Class 1 samples: {np.sum(y == 1)} (10%)")
    print("\nThis is an IMBALANCED dataset!")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining logistic regression...")
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=500)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    cm_dict = confusion_matrix(y_test, y_pred)
    print_all_metrics(cm_dict)
    plot_confusion_matrix(cm_dict, "confusion_matrix_imbalanced.png")
    
    metrics = {
        'Accuracy': accuracy(cm_dict),
        'Precision': precision(cm_dict),
        'Recall': recall(cm_dict),
        'F1 Score': f1_score(cm_dict),
        'Specificity': specificity(cm_dict)
    }
    
    plot_metrics_comparison(metrics, "metrics_comparison_imbalanced.png")
    
    print("\nKey Observation:")
    print("Even with high accuracy, the model might perform poorly on minority class!")
    print("Always check Precision, Recall, and F1 Score for imbalanced data.")


def demonstrate_balanced_dataset():
    """
    Show metrics on a balanced dataset for comparison.
    """
    print("\n" + "="*60)
    print("BALANCED DATASET EXAMPLE")
    print("="*60 + "\n")
    
    X, y = make_classification(
        n_samples=400,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],
        flip_y=0.05,
        random_state=42
    )
    
    print(f"Class 0 samples: {np.sum(y == 0)} (50%)")
    print(f"Class 1 samples: {np.sum(y == 1)} (50%)")
    print("\nThis is a BALANCED dataset!")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining logistic regression...")
    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=500)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    cm_dict = confusion_matrix(y_test, y_pred)
    print_all_metrics(cm_dict)
    plot_confusion_matrix(cm_dict, "confusion_matrix_balanced.png")
    
    metrics = {
        'Accuracy': accuracy(cm_dict),
        'Precision': precision(cm_dict),
        'Recall': recall(cm_dict),
        'F1 Score': f1_score(cm_dict),
        'Specificity': specificity(cm_dict)
    }
    
    plot_metrics_comparison(metrics, "metrics_comparison_balanced.png")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS DEEP DIVE")
    print("="*60)
    
    demonstrate_balanced_dataset()
    
    demonstrate_imbalanced_dataset()
    
    print("\n" + "="*60)
    print("WHEN TO USE WHICH METRIC?")
    print("="*60)
    print("\nACCURACY:")
    print("  Use when: Classes are balanced")
    print("  Avoid when: Imbalanced datasets")
    
    print("\nPRECISION:")
    print("  Use when: False positives are costly")
    print("  Example: Spam detection (don't mark important emails as spam)")
    
    print("\nRECALL:")
    print("  Use when: False negatives are costly")
    print("  Example: Disease detection (don't miss sick patients)")
    
    print("\nF1 SCORE:")
    print("  Use when: Need balance between precision and recall")
    print("  Example: Most real-world imbalanced problems")
    
    print("\nSPECIFICITY:")
    print("  Use when: Need to measure true negative rate")
    print("  Example: Medical screening tests")
    print("="*60 + "\n")
