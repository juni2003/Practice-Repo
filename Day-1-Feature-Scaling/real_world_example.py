"""
Real-World Feature Scaling Examples

This script demonstrates practical applications of feature scaling using real-world
scenarios and datasets. Each example shows why scaling is important and how to
apply it correctly in production-like environments.

Examples covered:
1. Customer segmentation (K-Means clustering)
2. House price prediction (Linear regression with mixed features)
3. Credit card fraud detection (Imbalanced classification)
4. Image classification (Neural network with pixel data)

Usage:
    python real_world_example.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')

def customer_segmentation_example():
    """
    Real-world scenario: Customer segmentation for an e-commerce company
    Features with different scales: age (20-80), income ($20K-$200K), purchases (1-100)
    """
    print("="*60)
    print("EXAMPLE 1: CUSTOMER SEGMENTATION")
    print("="*60)
    print("Scenario: E-commerce company wants to segment customers")
    print("Features: Age, Annual Income, Purchase Frequency")
    print("Algorithm: K-Means Clustering (distance-based → needs scaling)")
    
    # Create realistic customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Age: 18-70 years
    age = np.random.normal(35, 12, n_customers)
    age = np.clip(age, 18, 70)
    
    # Annual Income: $25K-$150K
    income = np.random.lognormal(np.log(50000), 0.5, n_customers)
    income = np.clip(income, 25000, 150000)
    
    # Purchase Frequency: 1-50 purchases per year
    # Correlated with income
    purchase_freq = (income / 3000) + np.random.normal(0, 5, n_customers)
    purchase_freq = np.clip(purchase_freq, 1, 50)
    
    # Create DataFrame
    customer_data = pd.DataFrame({
        'age': age,
        'annual_income': income,
        'purchase_frequency': purchase_freq
    })
    
    print(f"\nDataset: {len(customer_data)} customers")
    print("\nOriginal data statistics:")
    print(customer_data.describe())
    
    # Show the problem: features on very different scales
    print(f"\nScale differences:")
    print(f"Age range: {customer_data['age'].min():.1f} - {customer_data['age'].max():.1f}")
    print(f"Income range: ${customer_data['annual_income'].min():,.0f} - ${customer_data['annual_income'].max():,.0f}")
    print(f"Purchase frequency range: {customer_data['purchase_frequency'].min():.1f} - {customer_data['purchase_frequency'].max():.1f}")
    
    # K-Means without scaling (income dominates)
    kmeans_unscaled = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_unscaled = kmeans_unscaled.fit_predict(customer_data)
    silhouette_unscaled = silhouette_score(customer_data, clusters_unscaled)
    
    # K-Means with scaling
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data)
    kmeans_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_scaled = kmeans_scaled.fit_predict(customer_data_scaled)
    silhouette_scaled = silhouette_score(customer_data_scaled, clusters_scaled)
    
    print(f"\nClustering Quality (Silhouette Score):")
    print(f"Without scaling: {silhouette_unscaled:.3f}")
    print(f"With scaling: {silhouette_scaled:.3f}")
    print(f"Improvement: {((silhouette_scaled - silhouette_unscaled) / silhouette_unscaled * 100):+.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Unscaled clustering
    scatter1 = axes[0].scatter(customer_data['annual_income'], customer_data['age'], 
                              c=clusters_unscaled, cmap='viridis', alpha=0.6)
    axes[0].set_title(f'Without Scaling\nSilhouette Score: {silhouette_unscaled:.3f}')
    axes[0].set_xlabel('Annual Income ($)')
    axes[0].set_ylabel('Age')
    
    # Scaled clustering
    scatter2 = axes[1].scatter(customer_data['annual_income'], customer_data['age'], 
                              c=clusters_scaled, cmap='viridis', alpha=0.6)
    axes[1].set_title(f'With Scaling\nSilhouette Score: {silhouette_scaled:.3f}')
    axes[1].set_xlabel('Annual Income ($)')
    axes[1].set_ylabel('Age')
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Key Insight: Scaling improved clustering by allowing all features to contribute equally!")

def house_price_prediction_example():
    """
    Real-world scenario: House price prediction with mixed feature scales
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: HOUSE PRICE PREDICTION")
    print("="*60)
    print("Scenario: Predicting house prices with features of different scales")
    print("Algorithm: Linear Regression (gradient-based → benefits from scaling)")
    
    # Load California housing dataset
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target  # House prices in hundreds of thousands
    
    print(f"\nDataset: {len(X)} houses")
    print("\nFeature scales:")
    for col in X.columns:
        print(f"{col:15}: {X[col].min():8.2f} - {X[col].max():8.2f} (range: {X[col].max() - X[col].min():8.2f})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models without and with scaling
    models = {
        'No Scaling': LinearRegression(),
        'StandardScaler': Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())]),
        'MinMaxScaler': Pipeline([('scaler', MinMaxScaler()), ('lr', LinearRegression())]),
        'RobustScaler': Pipeline([('scaler', RobustScaler()), ('lr', LinearRegression())])
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error')
        rmse_cv = np.sqrt(-cv_scores.mean())
        
        # Test set performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {'cv_rmse': rmse_cv, 'test_rmse': rmse_test}
    
    print(f"\nModel Performance (RMSE - lower is better):")
    print(f"{'Model':<15} {'CV RMSE':<10} {'Test RMSE':<10} {'Improvement':<12}")
    print("-" * 50)
    
    baseline_cv = results['No Scaling']['cv_rmse']
    baseline_test = results['No Scaling']['test_rmse']
    
    for name, scores in results.items():
        cv_improvement = ((baseline_cv - scores['cv_rmse']) / baseline_cv * 100) if name != 'No Scaling' else 0
        test_improvement = ((baseline_test - scores['test_rmse']) / baseline_test * 100) if name != 'No Scaling' else 0
        
        print(f"{name:<15} {scores['cv_rmse']:<10.3f} {scores['test_rmse']:<10.3f} {test_improvement:+6.1f}%")
    
    print("\n💡 Key Insight: Scaling helps linear models by ensuring all features contribute proportionally!")

def fraud_detection_example():
    """
    Real-world scenario: Credit card fraud detection with imbalanced data
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: CREDIT CARD FRAUD DETECTION")
    print("="*60)
    print("Scenario: Detect fraudulent transactions (imbalanced dataset)")
    print("Algorithm: Random Forest vs Neural Network")
    
    # Create synthetic fraud detection data
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.99, 0.01],  # 1% fraud (imbalanced)
        flip_y=0.01,
        random_state=42
    )
    
    # Add features with different scales (like real transaction data)
    X[:, 0] *= 1000  # Transaction amount ($1-$1000)
    X[:, 1] *= 24    # Hour of day (0-24)
    X[:, 2] *= 7     # Day of week (0-7)
    
    feature_names = ['amount', 'hour', 'day_of_week'] + [f'feature_{i}' for i in range(3, 20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"\nDataset: {len(X)} transactions")
    print(f"Fraud rate: {y.mean()*100:.1f}%")
    print(f"\nFeature scales:")
    for i, col in enumerate(['amount', 'hour', 'day_of_week']):
        print(f"{col:15}: {X[:, i].min():8.2f} - {X[:, i].max():8.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Compare tree-based (doesn't need scaling) vs neural network (needs scaling)
    models = {
        'Random Forest (no scaling)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Random Forest (with scaling)': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Neural Network (no scaling)': MLPClassifier(hidden_layer_sizes=(100, 50), 
                                                     max_iter=500, random_state=42),
        'Neural Network (with scaling)': Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
        ])
    }
    
    print(f"\nModel Performance:")
    print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = (y_pred == y_test).mean()
        # F1-score for fraud class (class 1)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name:<30} {accuracy:<10.3f} {f1:<10.3f}")
    
    print("\n💡 Key Insights:")
    print("- Tree-based models: Scaling doesn't help (they're scale-invariant)")
    print("- Neural Networks: Scaling is crucial for proper convergence")

def image_classification_example():
    """
    Real-world scenario: Digit recognition with pixel intensity features
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: HANDWRITTEN DIGIT RECOGNITION")
    print("="*60)
    print("Scenario: Classify handwritten digits (0-9)")
    print("Features: Pixel intensities (0-16 scale)")
    print("Algorithm: Neural Network")
    
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"\nDataset: {len(X)} digit images")
    print(f"Image size: 8x8 pixels ({X.shape[1]} features)")
    print(f"Pixel intensity range: {X.min():.0f} - {X.max():.0f}")
    print(f"Classes: {len(np.unique(y))} digits (0-9)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Compare different scaling approaches for neural networks
    scalers = {
        'No Scaling': None,
        'MinMax [0,1]': MinMaxScaler(),
        'MinMax [-1,1]': MinMaxScaler(feature_range=(-1, 1)),
        'StandardScaler': StandardScaler()
    }
    
    results = {}
    for scaler_name, scaler in scalers.items():
        if scaler is None:
            pipeline = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                                   random_state=42, early_stopping=True)
        else:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                                   random_state=42, early_stopping=True))
            ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        results[scaler_name] = accuracy
    
    print(f"\nNeural Network Performance:")
    print(f"{'Scaling Method':<20} {'Accuracy':<10} {'Improvement':<12}")
    print("-" * 45)
    
    baseline = results['No Scaling']
    for name, accuracy in results.items():
        improvement = ((accuracy - baseline) / baseline * 100) if name != 'No Scaling' else 0
        print(f"{name:<20} {accuracy:<10.3f} {improvement:+6.1f}%")
    
    # Visualize sample digits
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Sample Handwritten Digits', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        digit_image = X[i].reshape(8, 8)
        ax.imshow(digit_image, cmap='gray')
        ax.set_title(f'Digit: {y[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Key Insights:")
    print("- MinMax scaling to [0,1] is ideal for neural networks with image data")
    print("- StandardScaler can also work well, especially for normalized images")
    print("- Proper scaling significantly improves convergence and final performance")

def production_pipeline_example():
    """
    Show how to build a production-ready pipeline with proper scaling
    """
    print("\n" + "="*60)
    print("PRODUCTION PIPELINE EXAMPLE")
    print("="*60)
    print("Best practices for deploying models with feature scaling")
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                               random_state=42)
    
    # Split data properly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    print("\n✅ CORRECT: Production-ready pipeline")
    print("-" * 40)
    
    # Build pipeline
    production_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train pipeline
    production_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = production_pipeline.score(X_train, y_train)
    test_score = production_pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Simulate new data prediction
    new_sample = np.random.randn(1, 10)  # New data point
    prediction = production_pipeline.predict(new_sample)
    prediction_proba = production_pipeline.predict_proba(new_sample)
    
    print(f"\nNew sample prediction: {prediction[0]}")
    print(f"Prediction confidence: {prediction_proba[0].max():.3f}")
    
    print("\n💡 Pipeline Benefits:")
    print("- Automatic scaling of new data")
    print("- No risk of data leakage")
    print("- Easy to deploy and maintain")
    print("- Consistent preprocessing")
    
    # Show what the pipeline does internally
    print(f"\nPipeline steps:")
    for i, (name, transformer) in enumerate(production_pipeline.steps):
        print(f"{i+1}. {name}: {type(transformer).__name__}")

def main():
    """Run all real-world examples."""
    print("🌍 REAL-WORLD FEATURE SCALING EXAMPLES")
    print("This script demonstrates practical applications of feature scaling")
    print("in common machine learning scenarios.\n")
    
    # Run all examples
    customer_segmentation_example()
    house_price_prediction_example()
    fraud_detection_example()
    image_classification_example()
    production_pipeline_example()
    
    print("\n" + "="*60)
    print("SUMMARY OF KEY TAKEAWAYS")
    print("="*60)
    print("1. 🎯 Algorithm Choice Matters:")
    print("   - Distance-based (KNN, K-Means): Always scale")
    print("   - Gradient-based (Neural Networks, Linear): Usually scale")
    print("   - Tree-based (Random Forest, XGBoost): Rarely need scaling")
    
    print("\n2. 📊 Data Characteristics Guide Scaler Choice:")
    print("   - Normal distribution → StandardScaler")
    print("   - Need [0,1] range → MinMaxScaler")
    print("   - Has outliers → RobustScaler")
    print("   - Sparse data → MaxAbsScaler")
    
    print("\n3. 🔧 Production Best Practices:")
    print("   - Use sklearn Pipeline for consistency")
    print("   - Fit scaler only on training data")
    print("   - Scale new data with same parameters")
    print("   - Include scaling in cross-validation")
    
    print("\n4. 📈 Performance Impact:")
    print("   - Can improve accuracy by 5-20%+ for sensitive algorithms")
    print("   - Critical for neural network convergence")
    print("   - Essential for clustering quality")
    
    print("\nRemember: Feature scaling is not just a preprocessing step—")
    print("it's a crucial decision that can make or break your model's performance! 🚀")

if __name__ == "__main__":
    main()