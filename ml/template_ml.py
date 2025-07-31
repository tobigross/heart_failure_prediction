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
       """
       Initialize your model with parameters.
       
       Args:
           **kwargs: Model-specific parameters
       """
       # TODO: Set model parameters here
       # Example: self.param1 = kwargs.get('param1', default_value)
        pass

    def fit(self, X, y):
         """
        Train your model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
        """
        # TODO: Implement model training logic
        pass

    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predictions
        """
        # TODO: Implement prediction logic
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

# For PyTorch models
import torch
import torch.nn as nn

class MyTorchModel(nn.Module):
    """
    Example PyTorch model class.
    Replace with your own architecture.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

def train_and_save_torch_model(X_train, y_train, input_dim, save_path, epochs=10):
    """
    Train and save a PyTorch model.
    """
    model = MyTorchModel(input_dim)
    # Add your training loop here
    # Example: optimizer, loss, etc.
    # for epoch in range(epochs):
    #     ...
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_torch_model(load_path, input_dim):
    """
    Load a saved PyTorch model.
    """
    model = MyTorchModel(input_dim)
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Example usage (uncomment and adapt as needed):
# X_train, y_train = ... # Load your data
# train_and_save_sklearn_model(X_train, y_train, "ml/my_sklearn_model.pkl")
# model = load_sklearn_model("ml/my_sklearn_model.pkl")
#
# train_and_save_torch_model(X_train, y_train, input_dim=20, save_path="ml/my_torch_model.pt")
# model = load_torch_model("ml/my_torch_model.pt", input_dim=20)