import pandas as pd
import torch
import logging
import joblib
from ml.model import HeartModel
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#add load and predict function here,,,,
def load_pytorch_model(model_path, input_dim=20):
    """
    Load a PyTorch HeartModel from a file and prepare it for inference.
    
    Parameters:
        model_path (str): Path to the saved model weights file.
        input_dim (int, optional): Number of input features for the model. Defaults to 20.
    
    Returns:
        HeartModel: The loaded PyTorch model set to evaluation mode.
    """
    model = HeartModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_pytorch(model, data):
    """Predict using a PyTorch model."""
    tensor = torch.tensor(data.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(tensor)
        return torch.argmax(outputs, dim=1).numpy().tolist()

def load_sklearn_model(model_path):
    """Load a scikit-learn model from file."""
    return joblib.load(model_path)

def predict_sklearn(model, data):
    """Predict using a scikit-learn model."""
    return model.predict(data).tolist()

class MiniModelETL:
    """Minimal ETL pipeline for comparing PyTorch and sklearn model predictions."""
    def __init__(self):
        self.COLUMNS_PATH = "ml/training_columns.txt"
        self.DATA_PATH = "data/heart.csv"
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "patients",
            "user": "postgres",
            "password": "xamyadt123"
        }
        self.table_name = "heart_predictions_mini"
        # SQLAlchemy connection string
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        # To add more models, define load/predict functions and add an entry below.
        self.model_configs = {
            "pytorch": {
                "load": lambda: load_pytorch_model("ml/model_pytorch.pt", input_dim=20),
                "predict": predict_pytorch
            },
            "sklearn": {
                "load": lambda: load_sklearn_model("ml/model_sklearn.pkl"),
                "predict": predict_sklearn
            }
        }
    def extract_data(self):
        """Load models, training columns, and data."""
        logger.info("Extracting models and data...")
        models = {name: cfg["load"]() for name, cfg in self.model_configs.items()}
        with open(self.COLUMNS_PATH) as f:
            training_columns = [line.strip() for line in f if line.strip()]
        data = pd.read_csv(self.DATA_PATH)
        return models, training_columns, data
    def transform_data(self, data, training_columns):
        """One-hot encode and align columns."""
        data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))
        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        return data_encoded.astype(float)
    def generate_predictions(self, models, data_encoded):
        """Generate predictions for all models."""
        preds = {}
        for name, model in models.items():
            predict_fn = self.model_configs[name]["predict"]
            preds[name] = predict_fn(model, data_encoded)
        return preds
    def add_column_if_not_exists(self, column_name, col_type="INTEGER"):
        """Add a column to the table if it does not exist, using SQLAlchemy."""
        stmt = text(f'''
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name=:table_name AND column_name=:column_name
                ) THEN
                    EXECUTE 'ALTER TABLE ' || quote_ident(:table_name) || ' ADD COLUMN ' || quote_ident(:column_name) || ' {col_type}';
                END IF;
            END
            $$;
        ''')
        with self.engine.begin() as conn:
            conn.execute(stmt, {"table_name": self.table_name, "column_name": column_name})
    def load_to_database(self, data, predictions):
        """Save results to PostgreSQL using pandas to_sql."""
        result = data.copy()
        for name, preds in predictions.items():
            result[f"prediction_{name}"] = preds
        # Add columns if needed
        for name in predictions.keys():
            self.add_column_if_not_exists(f"prediction_{name}", "INTEGER")
        # Use pandas to_sql for insertion
        result.to_sql(self.table_name, self.engine, if_exists='append', index=False, method='multi')
        logger.info(f"Inserted {len(result)} rows into {self.table_name}")
    def run(self):
        """Run the minimal ETL pipeline."""
        models, training_columns, data = self.extract_data()
        data_encoded = self.transform_data(data, training_columns)
        predictions = self.generate_predictions(models, data_encoded)
        self.load_to_database(data, predictions)
        logger.info("Mini ETL pipeline complete.")

def main():
    """Entry point for the mini ETL pipeline."""
    etl = MiniModelETL()
    etl.run()

if __name__ == "__main__":
    main()