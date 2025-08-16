import pandas as pd
import torch
import joblib
import os
from model import HeartModel
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

    
sk_model = joblib.load("ml/model_sklearn.pkl")

data = pd.read_csv("data/heart.csv")
data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))

    # Load training columns
with open("ml/training_columns.txt") as f:
    training_columns = [line.strip() for line in f if line.strip()]

X = data_encoded.reindex(columns=training_columns, fill_value=0).astype(float)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_dim=20
torch_model = HeartModel(input_dim=input_dim)
torch_model.load_state_dict(torch.load("ml/model_pytorch.pt", map_location=torch.device('cpu')))
torch_model.eval()
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.int64))

X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.int64))
with torch.no_grad():
    outputs = torch_model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)    
y_pred = predicted.numpy()
print(classification_report(y_test, sk_model.predict(X_test)))
print("ROC-AUC sklearn:", roc_auc_score(y_test, sk_model.predict_proba(X_test)[:, 1]))
print("ROC-AUC torch:", roc_auc_score(y_test, y_pred))