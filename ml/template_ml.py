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
        """
        Initialize the model with optional parameters.
        
        Parameters:
            **kwargs: Arbitrary keyword arguments for model configuration.
        """
        pass

    def fit(self, X, y):
        # Train your model here
        """
        Fits the model to the provided training data.
        
        Parameters:
            X: Training input features.
            y: Target values corresponding to the input features.
        """
        pass

    def predict(self, X):
        # Predict using your model here
        """
        Generate predictions for the input data using the model.
        
        Parameters:
            X: Input features for which predictions are to be made.
        
        Returns:
            Predicted values corresponding to the input data.
        """
        pass

def train_and_save_sklearn_model(X_train, y_train, save_path):
    """
    Train a scikit-learn model on the provided data and save it to disk.
    
    Parameters:
    	X_train: Training feature data.
    	y_train: Training target labels.
    	save_path (str): File path where the trained model will be saved.
    """
    model = MySklearnModel()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def load_sklearn_model(load_path):
    """
    Load a scikit-learn model from the specified file path.
    
    Parameters:
        load_path (str): Path to the saved model file.
    
    Returns:
        model: The loaded scikit-learn model instance.
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
        """
        Initialize a simple feedforward neural network with one hidden layer.
        
        Parameters:
            input_dim (int): The number of input features for the model.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        Performs a forward pass through the neural network using the input tensor.
        
        Parameters:
            x (torch.Tensor): Input tensor to the model.
        
        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        return self.layers(x)

def train_and_save_torch_model(X_train, y_train, input_dim, save_path, epochs=10):
    """
    Trains a PyTorch model on the provided data and saves its state dictionary to disk.
    
    Parameters:
        X_train: Training input data.
        y_train: Training target data.
        input_dim: The number of input features for the model.
        save_path: File path where the trained model's state dictionary will be saved.
        epochs: Number of training epochs (default is 10).
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
    Load a saved PyTorch model from disk and prepare it for inference.
    
    Parameters:
        load_path (str): Path to the saved model state dictionary file.
        input_dim (int): Input dimension required to initialize the model architecture.
    
    Returns:
        MyTorchModel: The loaded PyTorch model set to evaluation mode.
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