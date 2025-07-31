"""
ml_model_template.py

A template for creating, training, saving, and loading new ML models.
Copy this file and adapt for your specific model and framework.
"""

# For sklearn models
from sklearn.base import BaseEstimator
import joblib

class MySklearnModel(BaseEstimator):
    """
    Example sklearn model class.
    Replace with your own model and parameters.
    """
    def __init__(self, **kwargs):
        # Set model parameters here
        pass

    def fit(self, X, y):
        # Train your model here
        pass

    def predict(self, X):
        # Predict using your model here
        pass

def train_and_save_sklearn_model(X_train, y_train, save_path):
    """
    Train and save an sklearn model.
    """
    model = MySklearnModel()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def load_sklearn_model(load_path):
    """
    Load a saved sklearn model.
    """
    return joblib.load(load_path)



# Example usage (uncomment and adapt as needed):
# X_train, y_train = ... # Load your data
# train_and_save_sklearn_model(X_train, y_train, "ml/my_sklearn_model.pkl")
# model = load_sklearn_model("ml/my_sklearn_model.pkl")
#
# train_and_save_torch_model(X_train, y_train, input_dim=20, save_path="ml/my_torch_model.pt")
# model = load_torch_model("ml/my_torch_model.pt", input_dim=20)