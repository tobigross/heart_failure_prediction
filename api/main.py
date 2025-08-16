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
sklearn_model1 = joblib.load("ml/model_sklearn.pkl") 
sklearn_model2 = joblib.load("ml/gradient_boost_default.pkl")
sklearn_model3 = joblib.load("ml/gradient_boost_tuned.pkl")
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
        #convert to string for DB insertion
        sex_string = "M" if data.Sex == 1 else "F"

        chest_pain_map = {0: "TA", 1: "ATA", 2: "NAP", 3: "ASY"}
        chest_pain_string = chest_pain_map.get(data.ChestPainType, "Unknown")
    
        ecg_map = {0: "Normal", 1: "ST", 2: "LVM"}
        ecg_string = ecg_map.get(data.RestingECG, "Unknown")
    
        slope_map = {0: "Up", 1: "Flat", 2: "Down"}
        slope_string = slope_map.get(data.ST_Slope, "Unknown")
    
        exercise_angina_string = "Y" if data.ExerciseAngina == 1 else "N"
   

        # Insert into your existing table (adjust column names to match your table)
        cur.execute("""
    INSERT INTO heart_predictions_mini (
                "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
                "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
                "Oldpeak", "ST_Slope", "prediction_pytorch", 
                "prediction_sklearn", "prediction_gradient_boost_default",
                "prediction_gradient_boost_tuned"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
        data.Age, 
        sex_string,          
        chest_pain_string,    
        data.RestingBP, 
        data.Cholesterol,
        data.FastingBS,    
        ecg_string,           
        data.MaxHR, 
        exercise_angina_string, 
        data.Oldpeak,
        slope_string,         
        pytorch_pred, 
        int(sklearn1_pred),
        int(sklearn2_pred), 
        int(sklearn3_pred)
    ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
    
    return {
        "prediction_pytorch": int(pytorch_pred),
        "prediction_sklearn1":int(sklearn1_pred),
        "prediction_sklearn2":int(sklearn2_pred),
        "prediction_sklearn3": int(sklearn3_pred)
    }