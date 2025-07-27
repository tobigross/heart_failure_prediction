from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from ml.model import HeartModel
# -------- API Setup --------
app = FastAPI()

# -------- Modelldefinition & Laden --------
input_dim = 20  # Anpassen an deine Eingabefeatures
model = HeartModel(input_dim=input_dim)
model.load_state_dict(torch.load("ml/model_pytorch.pt", map_location=torch.device('cpu')))
model.eval()
#Laden der Features im encoding
with open("ml/training_columns.txt") as f:
    TRAINING_COLUMNS = [line.strip() for line in f]
# -------- Input-Schema für API --------
class PatientData(BaseModel):
    Age: float
    Sex: int
    ChestPainType: int
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: int
    MaxHR: float
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int
    # weitere Features je nach Datensatzstruktur
@app.get("/")
def root():
    return {"message": "API läuft"}
@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
     # One-hot-Encoding wie beim Training
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    # Reihenfolge & Spaltenstruktur ggf. anpassen!
    tensor = torch.tensor(df_encoded.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}