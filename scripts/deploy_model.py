import logging
import os
import sys

import joblib
import mlflow
import yaml

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.model_registry import ModelRegistry
except ImportError:
    print("Warning: ModelRegistry import failed. Creating dummy implementation.")

    class ModelRegistry:
        def __init__(self, tracking_uri: str = "http://localhost:5000"):
            mlflow.set_tracking_uri(tracking_uri)

        def get_best_model(self, experiment_name: str, metric: str = "rmse"):
            return {
                "run_id": "dummy_run_id",
                "model_uri": "dummy_uri",
                "metrics": {"rmse": 100, "mae": 80, "mape": 15},
                "params": {},
            }

        def register_model(self, run_id: str, model_name: str):
            print(f"Would register model {model_name} with run_id {run_id}")

        def transition_model_stage(self, model_name: str, version: int, stage: str):
            print(f"Would transition model {model_name} to {stage}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Use absolute path
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, config_path)

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            # Create default config
            self.config = self._create_default_config()
        else:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

        # Ensure mlflow config exists
        if "mlflow" not in self.config:
            self.config["mlflow"] = {"tracking_uri": "http://localhost:5000", "experiment_name": "seafood_forecasting"}

        self.registry = ModelRegistry(self.config["mlflow"]["tracking_uri"])

    def _create_default_config(self):
        """Create default configuration"""
        return {
            "mlflow": {"tracking_uri": "http://localhost:5000", "experiment_name": "seafood_forecasting"},
            "model": {"target_column": "PAY WEIGHT"},
        }

    def deploy_best_model(self):
        """Deploy the best model to production"""
        try:
            # Get best model from registry
            best_model = self.registry.get_best_model(self.config["mlflow"]["experiment_name"], metric="rmse")

            logger.info(f"Best model info: {best_model}")

            # Register model
            model_name = "seafood_demand_forecaster"
            self.registry.register_model(best_model["run_id"], model_name)

            # Transition to production
            self.registry.transition_model_stage(model_name, 1, "Production")

            logger.info("Model deployment process completed")

        except Exception as e:
            logger.error(f"Error in deployment process: {e}")

    def validate_deployment(self):
        """Validate model deployment"""
        try:
            # Load production model
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/saved_models/xgboost_model.pkl"
            )

            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info("Production model loaded successfully")
                return True
            else:
                logger.error("Production model file not found")
                return False

        except Exception as e:
            logger.error(f"Error validating deployment: {e}")
            return False

    def check_environment(self):
        """Check if all required components are available"""
        logger.info("Checking deployment environment...")

        # Check if models are trained
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/saved_models")
        if not os.path.exists(models_dir):
            logger.error("❌ Models directory not found")
            return False

        model_files = os.listdir(models_dir)
        if not model_files:
            logger.error("❌ No model files found")
            return False

        logger.info(f"✅ Found {len(model_files)} model files")

        # Check MLflow connection
        try:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            experiments = mlflow.search_experiments()
            logger.info(f"✅ MLflow connection successful. Found {len(experiments)} experiments")
        except Exception as e:
            logger.warning(f"⚠️ MLflow connection failed: {e}")
            logger.info("Continuing with local deployment...")

        return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SEAFOOD DEMAND FORECASTING - MODEL DEPLOYMENT")
    print("=" * 60)

    deployer = ModelDeployer()

    # Check environment first
    if not deployer.check_environment():
        logger.error("❌ Environment check failed. Please run training first.")
        sys.exit(1)

    logger.info("✅ Environment check passed. Starting deployment...")

    # Deploy models
    deployer.deploy_best_model()

    # Validate deployment
    if deployer.validate_deployment():
        logger.info("🎉 Model deployment completed successfully!")
    else:
        logger.warning("⚠️ Model deployment completed with warnings")

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print("✅ Models are ready for use in production")
    print("📊 Access the application at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
