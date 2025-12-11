import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        with open("config/model_config.yaml", "r") as file:
            self.model_config = yaml.safe_load(file)

        # Create results directory
        os.makedirs("models/saved_models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load processed data"""
        if not os.path.exists(self.config["data"]["processed_path"]):
            raise FileNotFoundError(f"Processed data not found at {self.config['data']['processed_path']}")

        df = pd.read_parquet(self.config["data"]["processed_path"])
        logger.info(f"Loaded data with shape: {df.shape}")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List]:
        """Prepare features and target"""
        # Remove rows with missing target
        df = df.dropna(subset=[self.config["model"]["target_column"]])

        # Get feature columns
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                self.config["model"]["date_column"],
                self.config["model"]["center_column"],
                self.config["model"]["item_column"],
                self.config["model"]["target_column"],
            ]
        ]

        # Fill missing values
        df[feature_cols] = df[feature_cols].fillna(0)

        X = df[feature_cols]
        y = df[self.config["model"]["target_column"]]

        logger.info(f"Prepared data: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Number of features: {len(feature_cols)}")

        return X, y, feature_cols

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")

        # Simple XGBoost parameters - no early stopping
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")

        # Simple LightGBM parameters - no early stopping
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)

        # Handle zero values for MAPE
        non_zero_mask = y_test > 0
        if non_zero_mask.sum() > 0:
            y_test_nonzero = y_test[non_zero_mask]
            y_pred_nonzero = y_pred[non_zero_mask]
            mape = np.mean(np.abs((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)) * 100
        else:
            mape = float("inf")

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mse": mean_squared_error(y_test, y_pred),
            "mape": mape,
        }

        logger.info(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")

        return metrics

    def train_models(self):
        """Train all models"""
        try:
            # Load data
            df = self.load_data()
            X, y, feature_cols = self.prepare_data(df)

            # Simple train-test split (no validation for simplicity)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config["model"]["test_size"], random_state=42, shuffle=False  # Important for time series
            )

            logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

            # Train models
            models = {}
            metrics = {}

            logger.info("\n" + "=" * 50)
            logger.info("STARTING MODEL TRAINING")
            logger.info("=" * 50)

            xgb_model = self.train_xgboost(X_train, y_train)
            lgb_model = self.train_lightgbm(X_train, y_train)

            # Evaluate models
            logger.info("\n" + "=" * 50)
            logger.info("MODEL EVALUATION")
            logger.info("=" * 50)

            xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            lgb_metrics = self.evaluate_model(lgb_model, X_test, y_test, "LightGBM")

            models["xgboost"] = xgb_model
            models["lightgbm"] = lgb_model
            metrics["xgboost"] = xgb_metrics
            metrics["lightgbm"] = lgb_metrics

            # Save models
            self._save_models(models, feature_cols)

            # Save results
            model_info = {
                "training_date": datetime.now().isoformat(),
                "models_trained": list(models.keys()),
                "metrics": metrics,
                "feature_columns": feature_cols,
                "best_model": min(metrics.items(), key=lambda x: x[1]["rmse"])[0],
            }

            with open("results/training_info.json", "w") as f:
                json.dump(model_info, f, indent=2)

            # Print best model
            best_model_name, best_model_metrics = min(metrics.items(), key=lambda x: x[1]["rmse"])
            logger.info("\n" + "=" * 50)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best RMSE: {best_model_metrics['rmse']:.4f}")
            logger.info(f"Best MAE: {best_model_metrics['mae']:.4f}")
            logger.info(f"Models saved to: models/saved_models/")

            return models, metrics

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def _save_models(self, models: Dict, feature_cols: List):
        """Save trained models"""
        for name, model in models.items():
            joblib.dump(model, f"models/saved_models/{name}_model.pkl")
            logger.info(f"Saved {name} model")

        # Save feature columns
        joblib.dump(feature_cols, "models/saved_models/feature_columns.pkl")
        logger.info("Saved feature columns")


if __name__ == "__main__":
    try:
        print("\nStarting Model Training...")
        trainer = ModelTrainer()
        models, metrics = trainer.train_models()
        print("\n[SUCCESS] Model training completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Error in model training: {e}")
        import traceback

        traceback.print_exc()
