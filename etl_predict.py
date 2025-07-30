import pandas as pd
import torch
import os
from ml.model import HeartModel
import psycopg2
from psycopg2 import sql

# --- Load model and columns ---
input_dim = 20  # Number of input features after encoding
model = HeartModel(input_dim=input_dim)
model.load_state_dict(torch.load("ml/model_pytorch.pt", map_location=torch.device('cpu')))
model.eval()

with open("ml/training_columns.txt") as f:
    TRAINING_COLUMNS = [line.strip() for line in f if line.strip()]

# --- Load data ---
data = pd.read_csv("data/heart.csv")

# --- Feature engineering (one-hot encoding, align columns) ---
data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))
data_encoded = data_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
data_encoded = data_encoded.astype(float)  # Ensure all columns are float for torch

# --- Predict ---
tensor = torch.tensor(data_encoded.to_numpy(), dtype=torch.float32)
with torch.no_grad():
    outputs = model(tensor)
    predictions = torch.argmax(outputs, dim=1).numpy()

# --- Prepare results for SQL ---
result = data.copy()
result["prediction"] = predictions

# --- SQL Connection Details ---
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "patients"
DB_USER = "postgres"
DB_PASS = "xamyadt123"
TABLE_NAME = "heart_predictions_pytorch"

# --- Create table and insert data ---
def create_table_and_insert(df):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cur = conn.cursor()
    # Build CREATE TABLE statement
    columns = list(df.columns)
    col_types = []
    for col in columns:
        if col.lower() == "age" or col.lower() == "oldpeak":
            col_types.append(f'"{col}" FLOAT')
        elif col.lower() in ["restingbp", "cholesterol", "fastingbs", "maxhr", "heartdisease", "prediction"]:
            col_types.append(f'"{col}" INTEGER')
        else:
            col_types.append(f'"{col}" TEXT')
    create_stmt = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (id SERIAL PRIMARY KEY, {', '.join(col_types)})"
    cur.execute(create_stmt)
    conn.commit()
    # Insert data
    placeholders = ', '.join(['%s'] * len(columns))
    columns_sql = sql.SQL(', ').join([sql.Identifier(col) for col in columns])
    insert_stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(TABLE_NAME),
        columns_sql,
        sql.SQL(placeholders)
    )
    for _, row in df.iterrows():
        cur.execute(insert_stmt, tuple(row))
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(df)} rows into {TABLE_NAME}.")

if __name__ == "__main__":
    create_table_and_insert(result)
    print("ETL process complete.")