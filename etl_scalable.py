import pandas as pd
import torch
import logging
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import psycopg2
from psycopg2 import sql
import joblib
from ml.model import HeartModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader(ABC):
    """Abstract base class for model loaders"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load and return the model"""
        pass
    
    @abstractmethod
    def predict(self, model: Any, data: pd.DataFrame) -> List[int]:
        """Generate predictions using the model"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type name"""
        pass

class PyTorchModelLoader(ModelLoader):
    """PyTorch model loader and predictor"""
    
    def __init__(self, input_dim: int = 20):
        self.input_dim = input_dim
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        model = HeartModel(input_dim=self.input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def predict(self, model: torch.nn.Module, data: pd.DataFrame) -> List[int]:
        tensor = torch.tensor(data.to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            outputs = model(tensor)
            predictions = torch.argmax(outputs, dim=1).numpy().tolist()
        return predictions
    
    @property
    def model_type(self) -> str:
        return "pytorch"

class SklearnModelLoader(ModelLoader):
    """Scikit-learn model loader and predictor"""
    
    def load_model(self, model_path: str) -> Any:
        return joblib.load(model_path)
    
    def predict(self, model: Any, data: pd.DataFrame) -> List[int]:
        return model.predict(data).tolist()
    
    @property
    def model_type(self) -> str:
        return "sklearn"



class ScalableModelComparison:
    """Scalable ETL pipeline for comparing predictions from multiple ML models"""
    
    def __init__(self):
        # Data configuration
        self.COLUMNS_PATH = "ml/training_columns.txt"
        self.DATA_PATH = "data/heart.csv"
        
        # Database configuration
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "patients",
            "user": "postgres",
            "password": "xamyadt123"
        }
        self.table_name = "heart_predictions_multi_model"
        
        # Model configuration - sklearn and pytorch models
        self.model_configs = {
            "pytorch": {
                "path": "ml/model_pytorch.pt",
                "loader": PyTorchModelLoader(input_dim=20)
            },
            "sklearn": {
                "path": "ml/model_sklearn.pkl",
                "loader": SklearnModelLoader()
            }
        }
    
    def add_model(self, name: str, model_path: str, loader: ModelLoader) -> None:
        """Dynamically add a new model to the comparison"""
        self.model_configs[name] = {
            "path": model_path,
            "loader": loader
        }
        logger.info(f"Added model '{name}' to comparison pipeline")
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the comparison"""
        if name in self.model_configs:
            del self.model_configs[name]
            logger.info(f"Removed model '{name}' from comparison pipeline")
        else:
            logger.warning(f"Model '{name}' not found in configuration")
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all configured models and data"""
        logger.info("Starting data extraction...")
        
        # Load models
        models = {}
        for name, config in self.model_configs.items():
            try:
                model = config["loader"].load_model(config["path"])
                models[name] = {
                    "model": model,
                    "loader": config["loader"]
                }
                logger.info(f"✅ {name} model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load {name} model: {e}")
                models[name] = None
        
        # Load training columns
        with open(self.COLUMNS_PATH) as f:
            training_columns = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(training_columns)} training columns")
        
        # Load data
        data = pd.read_csv(self.DATA_PATH)
        logger.info(f"Loaded {len(data)} rows of data")
        
        return {
            "models": models,
            "training_columns": training_columns,
            "data": data
        }
    
    def transform_data(self, data: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
        """Transform data for model predictions"""
        logger.info("Starting data transformation...")
        
        data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))
        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        data_encoded = data_encoded.astype(float)
        
        logger.info(f"Data encoded with {data_encoded.shape[1]} features")
        return data_encoded
    
    def generate_all_predictions(self, models: Dict[str, Any], data_encoded: pd.DataFrame) -> Dict[str, List]:
        """Generate predictions from all available models"""
        predictions = {}
        
        for name, model_info in models.items():
            if model_info is None:
                logger.warning(f"Skipping {name} - model not loaded")
                predictions[name] = [None] * len(data_encoded)
                continue
            
            try:
                logger.info(f"Generating {name} predictions...")
                preds = model_info["loader"].predict(model_info["model"], data_encoded)
                predictions[name] = preds
                logger.info(f"✅ Generated {len(preds)} {name} predictions")
                
            except Exception as e:
                logger.error(f"❌ Error generating {name} predictions: {e}")
                predictions[name] = [None] * len(data_encoded)
        
        return predictions
    
    def analyze_model_agreement(self, predictions: Dict[str, List]) -> Dict[str, Any]:
        """Analyze agreement between all models"""
        # Filter out models that failed
        valid_predictions = {k: v for k, v in predictions.items() if None not in v}
        
        if len(valid_predictions) < 2:
            logger.warning("Need at least 2 successful models for comparison")
            return {"comparison_available": False}
        
        model_names = list(valid_predictions.keys())
        total_predictions = len(list(valid_predictions.values())[0])
        
        # Calculate pairwise agreements
        pairwise_agreements = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                agreements = sum(1 for p1, p2 in zip(valid_predictions[model1], valid_predictions[model2]) if p1 == p2)
                agreement_rate = agreements / total_predictions
                pairwise_agreements[f"{model1}_vs_{model2}"] = {
                    "agreements": agreements,
                    "total": total_predictions,
                    "rate": agreement_rate
                }
        
        # Calculate overall consensus (all models agree)
        consensus_count = 0
        for i in range(total_predictions):
            prediction_set = {valid_predictions[model][i] for model in model_names}
            if len(prediction_set) == 1:  # All predictions are the same
                consensus_count += 1
        
        consensus_rate = consensus_count / total_predictions
        
        return {
            "comparison_available": True,
            "models_compared": model_names,
            "total_predictions": total_predictions,
            "pairwise_agreements": pairwise_agreements,
            "consensus_count": consensus_count,
            "consensus_rate": consensus_rate
        }
    
    def load_to_database(self, data: pd.DataFrame, predictions: Dict[str, List]) -> None:
        """Load results to PostgreSQL database"""
        logger.info("Loading multi-model comparison data to database...")
        
        # Prepare results
        result = data.copy()
        for model_name, preds in predictions.items():
            result[f"prediction_{model_name}"] = preds
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            self._create_table(cur, result.columns)
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
            elif (col.lower() in ["restingbp", "cholesterol", "fastingbs", "maxhr", "heartdisease"] or 
                  col.lower().startswith("prediction_")):
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
        
        data_tuples = []
        for row in df.to_numpy():
            processed_row = tuple(None if pd.isna(val) else val for val in row)
            data_tuples.append(processed_row)
        
        cursor.executemany(insert_stmt, data_tuples)
    
    def run_multi_model_etl(self) -> Dict[str, Any]:
        """Execute the complete multi-model comparison ETL pipeline"""
        try:
            # Extract
            extracted_data = self.extract_data()
            
            # Transform
            data_encoded = self.transform_data(extracted_data["data"], extracted_data["training_columns"])
            
            # Generate predictions from all models
            predictions = self.generate_all_predictions(extracted_data["models"], data_encoded)
            
            # Analyze model agreements
            analysis = self.analyze_model_agreement(predictions)
            
            # Load to database
            self.load_to_database(extracted_data["data"], predictions)
            
            logger.info("Multi-model comparison ETL process completed successfully!")
            return analysis
            
        except Exception as e:
            logger.error(f"Multi-model ETL process failed: {e}")
            raise

def main():
    """Main execution function"""
    etl = ScalableModelComparison()
    
    # Example: Add additional sklearn models if needed
    # from sklearn.ensemble import RandomForestClassifier
    # rf_loader = SklearnModelLoader()
    # etl.add_model("random_forest", "ml/rf_model.pkl", rf_loader)
    
    # Run the comparison
    analysis = etl.run_multi_model_etl()
    
    # Display results
    if analysis and analysis.get("comparison_available"):
        print(f"\n🔍 Multi-Model Comparison Results:")
        print(f"   Models compared: {', '.join(analysis['models_compared'])}")
        print(f"   Overall consensus rate: {analysis['consensus_rate']:.2%}")
        print(f"   Consensus predictions: {analysis['consensus_count']}/{analysis['total_predictions']}")
        
        print(f"\n📊 Pairwise Agreement Rates:")
        for pair, stats in analysis['pairwise_agreements'].items():
            print(f"   {pair.replace('_vs_', ' vs ')}: {stats['rate']:.2%}")
    else:
        print("\n⚠️  Model comparison not available - need at least 2 working models")

if __name__ == "__main__":
    main()