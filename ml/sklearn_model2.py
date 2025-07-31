"""
sklearn_model2.py

Gradient Boosting model implementation using sklearn.
"""

# For sklearn models
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

class GradientBoostModel(BaseEstimator):
    """
    Gradient Boosting model class using sklearn's GradientBoostingClassifier.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        """Train the gradient boosting model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_

def train_and_save_sklearn_model(X_train, y_train, save_path, hyperparameter_tuning=False):
    """
    Train and save a gradient boosting model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        save_path: Path to save the model
        hyperparameter_tuning: Whether to perform hyperparameter tuning
    """
    if hyperparameter_tuning:
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create base model
        base_model = GradientBoostingClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = GradientBoostModel(**grid_search.best_params_)
        model.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    else:
        # Use default parameters
        model = GradientBoostModel()
        model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, save_path)
    print(f"Gradient Boosting model saved to {save_path}")
    
    return model

def load_sklearn_model(load_path):
    """
    Load a saved gradient boosting model.
    """
    return joblib.load(load_path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the gradient boosting model.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# Example usage (uncomment and adapt as needed):
# X_train, y_train = ... # Load your data
# X_test, y_test = ... # Load your test data
# 
# # Train with default parameters
# train_and_save_sklearn_model(X_train, y_train, "ml/gradient_boost_model.pkl")
# 
# # Train with hyperparameter tuning
# train_and_save_sklearn_model(X_train, y_train, "ml/gradient_boost_tuned.pkl", hyperparameter_tuning=True)
# 
# # Load and evaluate model
# model = load_sklearn_model("ml/gradient_boost_model.pkl")
# results = evaluate_model(model, X_test, y_test)