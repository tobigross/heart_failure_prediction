import pandas as pd
import torch
import logging
import joblib
import warnings
import os
from ml.model import HeartModel
from sqlalchemy import create_engine, text

# Set logging to WARNING to reduce output (or ERROR for even less)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def load_pytorch_model(model_path, input_dim=20):
    """Load a PyTorch model from file."""
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
    if not os.path.exists(model_path):
        return None
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

def predict_sklearn(model, data):
    """Predict using a scikit-learn model."""
    if model is None:
        return [None] * len(data)
    
    try:
        predictions = model.predict(data)
        return [int(pred) for pred in predictions]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return [None] * len(data)

class MiniModelETL:
    """Minimal ETL pipeline for comparing multiple model predictions."""
    
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
        
        # Model configurations
        self.model_configs = {
            "pytorch": {
                "load": lambda: load_pytorch_model("ml/model_pytorch.pt", input_dim=20),
                "predict": predict_pytorch
            },
            "sklearn": {
                "load": lambda: load_sklearn_model("ml/model_sklearn.pkl"),
                "predict": predict_sklearn
            },
            "gradient_boost_default": {
                "load": lambda: load_sklearn_model("ml/gradient_boost_default.pkl"),
                "predict": predict_sklearn
            },
            "gradient_boost_tuned": {
                "load": lambda: load_sklearn_model("ml/gradient_boost_tuned.pkl"),
                "predict": predict_sklearn
            }
        }
    
    def extract_data(self):
        """Load models, training columns, and data."""
        print("Loading models and data...")
        
        # Load models
        models = {}
        for name, cfg in self.model_configs.items():
            try:
                model = cfg["load"]()
                models[name] = model
                if model is not None:
                    print(f"✅ {name}")
                else:
                    print(f"❌ {name} (failed to load)")
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                models[name] = None
        
        # Load training columns
        with open(self.COLUMNS_PATH) as f:
            training_columns = [line.strip() for line in f if line.strip()]
        
        # Load data
        data = pd.read_csv(self.DATA_PATH)
        print(f"Loaded {len(data)} rows of data")
        
        return models, training_columns, data
    
    def transform_data(self, data, training_columns):
        """One-hot encode and align columns."""
        data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))
        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        return data_encoded.astype(float)
    
    def generate_predictions(self, models, data_encoded):
        """Generate predictions for all models."""
        print("Generating predictions...")
        preds = {}
        
        for name, model in models.items():
            if model is None:
                preds[name] = [None] * len(data_encoded)
                continue
            
            try:
                predict_fn = self.model_configs[name]["predict"]
                model_preds = predict_fn(model, data_encoded)
                preds[name] = model_preds
            except Exception as e:
                logger.error(f"Error generating predictions for {name}: {e}")
                preds[name] = [None] * len(data_encoded)
        
        return preds
    
    def add_column_if_not_exists(self, column_name, col_type="INTEGER"):
        """Add a column to the table if it does not exist."""
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
        
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt, {"table_name": self.table_name, "column_name": column_name})
        except Exception as e:
            logger.error(f"Error adding column {column_name}: {e}")
    
    def load_to_database(self, data, predictions):
        """Save results to PostgreSQL."""
        print("Saving to database...")
        
        result = data.copy()
        for name, preds in predictions.items():
            column_name = f"prediction_{name}"
            result[column_name] = preds
        
        # Add columns if needed
        for name in predictions:
            self.add_column_if_not_exists(f"prediction_{name}", "INTEGER")
        
        # Insert data
        with self.engine.begin() as conn:
            result.to_sql(self.table_name, conn, if_exists='append', index=False, method='multi')
        
        print(f"✅ Inserted {len(result)} rows")
    
    def run(self):
        """Run the ETL pipeline."""
        try:
            print("Starting ETL pipeline...")
            
            # Extract
            models, training_columns, data = self.extract_data()
            
            # Transform
            data_encoded = self.transform_data(data, training_columns)
            
            # Generate predictions
            predictions = self.generate_predictions(models, data_encoded)
            
            # Load to database
            self.load_to_database(data, predictions)
            
            # Summary
            successful_models = [name for name, model in models.items() if model is not None]
            failed_models = [name for name, model in models.items() if model is None]
            
            print("✅ Pipeline completed successfully!")
            print(f"📊 Processed models: {', '.join(successful_models)}")
            if failed_models:
                print(f"⚠️ Failed models: {', '.join(failed_models)}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Entry point for the ETL pipeline."""
    etl = MiniModelETL()
    etl.run()

if __name__ == "__main__":
    main()