"""
Train Gradient Boosting model using heart.csv data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_model2 import train_and_save_sklearn_model, evaluate_model

def load_and_preprocess_data():
    """
    Load heart.csv data and preprocess it for training.
    """
    # Load the data
    print("Loading heart.csv data...")
    df = pd.read_csv('data/heart.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Display target distribution
    print("\nTarget distribution (HeartDisease):")
    print(df['HeartDisease'].value_counts())
    print(f"Percentage of heart disease cases: {df['HeartDisease'].mean():.2%}")
    
    return df

def prepare_features(df):
    """
    Prepare features for training based on training_columns.txt
    """
    print("\nPreparing features...")
    
    # Create dummy variables for categorical columns
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Select features based on training_columns.txt
    feature_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_F', 'Sex_M',
        'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
        'ExerciseAngina_N', 'ExerciseAngina_Y',
        'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    # Check which columns are available
    available_columns = [col for col in feature_columns if col in df_encoded.columns]
    missing_columns = [col for col in feature_columns if col not in df_encoded.columns]
    
    print(f"Available features: {len(available_columns)}")
    print(f"Missing features: {missing_columns}")
    
    # Select features and target
    X = df_encoded[available_columns]
    y = df['HeartDisease']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, available_columns

def scale_features(X_train, X_test):
    """
    Scale numerical features
    """
    scaler = StandardScaler()
    
    # Identify numerical columns (excluding dummy variables)
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    available_numerical = [col for col in numerical_cols if col in X_train.columns]
    
    # Scale numerical features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if available_numerical:
        X_train_scaled[available_numerical] = scaler.fit_transform(X_train[available_numerical])
        X_test_scaled[available_numerical] = scaler.transform(X_test[available_numerical])
        print(f"Scaled numerical features: {available_numerical}")
    
    return X_train_scaled, X_test_scaled, scaler

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance
    """
    importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importance - Gradient Boosting')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('ml/feature_importance_gradient_boost.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 most important features:")
    print(feature_importance_df.head(10))

def main():
    """
    Main training function
    """
    print("=== Gradient Boosting Model Training ===")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split the data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model with default parameters
    print("\n=== Training Gradient Boosting Model (Default Parameters) ===")
    model_default = train_and_save_sklearn_model(
        X_train_scaled, y_train, 
        "ml/gradient_boost_default.pkl"
    )
    
    # Evaluate default model
    print("\n=== Evaluating Default Model ===")
    results_default = evaluate_model(model_default, X_test_scaled, y_test)
    
    # Plot feature importance
    plot_feature_importance(model_default, feature_names)
    
    # Train model with hyperparameter tuning (optional - takes longer)
    print("\n=== Training Gradient Boosting Model (With Hyperparameter Tuning) ===")
    print("This will take longer due to grid search...")
    
    try:
        model_tuned = train_and_save_sklearn_model(
            X_train_scaled, y_train, 
            "ml/gradient_boost_tuned.pkl",
            hyperparameter_tuning=True
        )
        
        # Evaluate tuned model
        print("\n=== Evaluating Tuned Model ===")
        results_tuned = evaluate_model(model_tuned, X_test_scaled, y_test)
        
        # Plot feature importance for tuned model
        plot_feature_importance(model_tuned, feature_names)
        
        # Compare models
        print("\n=== Model Comparison ===")
        print(f"Default model accuracy: {results_default['accuracy']:.4f}")
        print(f"Tuned model accuracy: {results_tuned['accuracy']:.4f}")
        
        if results_tuned['accuracy'] > results_default['accuracy']:
            print("Tuned model performs better!")
        else:
            print("Default model performs better or equally well.")
            
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Using default model only.")
    
    print("\n=== Training Complete ===")
    print("Models saved:")
    print("- ml/gradient_boost_default.pkl")
    print("- ml/gradient_boost_tuned.pkl (if successful)")
    print("- ml/feature_importance_gradient_boost.png")

if __name__ == "__main__":
    main() 