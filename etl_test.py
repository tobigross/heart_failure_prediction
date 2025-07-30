import pandas as pd
import torch
import os
import logging
from pathlib import Path
from typing import List, Tuple
import psycopg2
from psycopg2 import sql
from ml.model import HeartModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartDiseaseETL:
    """ETL pipeline for heart disease predictions"""
    
    def __init__(self):
        # Configuration
        self.INPUT_DIM = 20
        self.MODEL_PATH = "ml/model_pytorch.pt"
        self.COLUMNS_PATH = "ml/training_columns.txt"
        self.DATA_PATH = "data/heart.csv"
        
        # Database configuration
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "patients",
            "user": "postgres",
            "password": "xamyadt123"  # Consider using env variables
        }
        self.table_name = "heart_predictions_pytorch"
        
    def extract_data(self) -> Tuple[torch.nn.Module, List[str], pd.DataFrame]:
        """Extract model, training columns, and data"""
        logger.info("Starting data extraction...")
        
        # Load model
        model = HeartModel(input_dim=self.INPUT_DIM)
        model.load_state_dict(torch.load(self.MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully")
        
        # Load training columns
        with open(self.COLUMNS_PATH) as f:
            training_columns = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(training_columns)} training columns")
        
        # Load data
        data = pd.read_csv(self.DATA_PATH)
        logger.info(f"Loaded {len(data)} rows of data")
        
        return model, training_columns, data
    
    def transform_data(self, data: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
        """Transform data for model prediction"""
        logger.info("Starting data transformation...")
        
        # Feature engineering
        data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))
        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        data_encoded = data_encoded.astype(float)
        
        logger.info(f"Data encoded with {data_encoded.shape[1]} features")
        return data_encoded
    
    def make_predictions(self, model: torch.nn.Module, data_encoded: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using the model"""
        logger.info("Generating predictions...")
        
        tensor = torch.tensor(data_encoded.to_numpy(), dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def load_to_database(self, data: pd.DataFrame, predictions) -> None:
        """Load results to PostgreSQL database"""
        logger.info("Loading data to database...")
        
        # Prepare results
        result = data.copy()
        result["prediction"] = predictions
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Create table
            self._create_table(cur, result.columns)
            
            # Insert data
            self._insert_data(cur, result)
            
            conn.commit()
            logger.info(f"Successfully inserted {len(result)} rows into {self.table_name}")
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
    
    def _create_table(self, cursor, columns: List[str]) -> None:
        """Create database table with appropriate column types"""
        col_types = []
        for col in columns:
            if col.lower() in ["age", "oldpeak"]:
                col_types.append(f'"{col}" FLOAT')
            elif col.lower() in ["restingbp", "cholesterol", "fastingbs", "maxhr", "heartdisease", "prediction"]:
                col_types.append(f'"{col}" INTEGER')
            else:
                col_types.append(f'"{col}" TEXT')
        
        create_stmt = f"CREATE TABLE IF NOT EXISTS {self.table_name} (id SERIAL PRIMARY KEY, {', '.join(col_types)})"
        cursor.execute(create_stmt)
        logger.info(f"Table {self.table_name} created/verified")
    
    def _insert_data(self, cursor, df: pd.DataFrame) -> None:
        """Insert data into the database table"""
        columns = list(df.columns)
        placeholders = ', '.join(['%s'] * len(columns))
        columns_sql = sql.SQL(', ').join([sql.Identifier(col) for col in columns])
        
        insert_stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(self.table_name),
            columns_sql,
            sql.SQL(placeholders)
        )
        
        # Use executemany for better performance
        data_tuples = [tuple(row) for row in df.to_numpy()]
        cursor.executemany(insert_stmt, data_tuples)
    
    def run_etl(self) -> None:
        """Execute the complete ETL pipeline"""
        try:
            # Extract
            model, training_columns, data = self.extract_data()
            
            # Transform
            data_encoded = self.transform_data(data, training_columns)
            predictions = self.make_predictions(model, data_encoded)
            
            # Load
            self.load_to_database(data, predictions)
            
            logger.info("ETL process completed successfully!")
            
        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            raise

def main():
    """Main execution function"""
    etl = HeartDiseaseETL()
    etl.run_etl()

if __name__ == "__main__":
    main()