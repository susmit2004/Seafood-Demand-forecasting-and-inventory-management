import sys
import os
import mlflow
import joblib
import pandas as pd
from typing import Dict, Any, List
import logging

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        
    def get_best_model(self, experiment_name: str, metric: str = "rmse") -> Dict[str, Any]:
        """Get the best model from MLflow registry"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.warning(f"Experiment {experiment_name} not found")
                return self._create_dummy_best_model()
                
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} ASC"]
            )
            
            if runs.empty:
                logger.warning("No runs found in experiment")
                return self._create_dummy_best_model()
                
            best_run = runs.iloc[0]
            
            return {
                'run_id': best_run.run_id,
                'model_uri': f"runs:/{best_run.run_id}/model",
                'metrics': {
                    'rmse': best_run.get('metrics.rmse', 0),
                    'mae': best_run.get('metrics.mae', 0),
                    'mape': best_run.get('metrics.mape', 0)
                },
                'params': best_run.filter(regex='params.*').to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return self._create_dummy_best_model()
    
    def _create_dummy_best_model(self) -> Dict[str, Any]:
        """Create a dummy best model for testing"""
        return {
            'run_id': 'dummy_run_id',
            'model_uri': 'dummy_uri',
            'metrics': {'rmse': 100, 'mae': 80, 'mape': 15},
            'params': {}
        }
    
    def load_production_model(self, model_name: str) -> Any:
        """Load production model from local storage"""
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                f"models/saved_models/{model_name}_model.pkl"
            )
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Model {model_name} loaded successfully")
                return model
            else:
                logger.warning(f"Model {model_name} not found at {model_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def register_model(self, run_id: str, model_name: str):
        """Register model in MLflow model registry"""
        try:
            mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name
            )
            logger.info(f"Model {model_name} registered successfully")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
    
    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """Transition model to different stage"""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} version {version} transitioned to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")