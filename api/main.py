from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import sys
import os
import getpass
import joblib
import psycopg2
from datetime import datetime
sys.path.append(os.path.abspath(".."))
from ml.model import HeartModel
# -------- API Setup --------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Database Config --------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("DB_NAME", "patients"),
    "user": os.environ.get("DB_USER", "postgres"),
   "password": os.environ.get("DB_PASSWORD", "")
}


# -------- Modelldefinition & Laden --------
input_dim = 20
pytorch_model = HeartModel(input_dim=input_dim)
pytorch_model.load_state_dict(torch.load("ml/model_pytorch.pt", map_location=torch.device('cpu')))
pytorch_model.eval()

# Sklearn models
sklearn_model1 = joblib.load("ml/sklearn_model1.pkl")  # Update path
sklearn_model2 = joblib.load("ml/sklearn_model2.pkl")
sklearn_model3 = joblib.load("ml/sklearn_model3.pkl")
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
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
     # One-hot-Encoding wie beim Training
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    # Reihenfolge & Spaltenstruktur ggf. anpassen!
    tensor = torch.tensor(df_encoded.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        output =pytorch_model(tensor)
        pytorch_pred= torch.argmax(output, dim=1).item()
    sklearn1_pred = sklearn_model1.predict(df_encoded)[0]
    sklearn2_pred = sklearn_model2.predict(df_encoded)[0]
    sklearn3_pred = sklearn_model3.predict(df_encoded)[0]
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Insert into your existing table (adjust column names to match your table)
        cur.execute("""
    INSERT INTO your_table_name (
        age, sex, chest_pain_type, resting_bp, cholesterol, 
        fasting_bs, resting_ecg, max_hr, exercise_angina, 
        oldpeak, st_slope, pytorch_prediction, 
        sklearn_model1_prediction, sklearn_model2_prediction,
        sklearn_model3_prediction,
        created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (
    data.Age, data.Sex, data.ChestPainType, data.RestingBP, 
    data.Cholesterol, data.FastingBS, data.RestingECG, 
    data.MaxHR, data.ExerciseAngina, data.Oldpeak, 
    data.ST_Slope, pytorch_pred, int(sklearn1_pred), 
    int(sklearn2_pred), int(sklearn3_pred), datetime.now()
))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
    
    return {
        "prediction_pytorch": pytorch_pred,
        "prediction_sklearn1":sklearn1_pred,
        "prediction_sklearn2":sklearn2_pred,
        "prediction_sklearn3": sklearn3_pred
    }