import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Load models and features
        self.models = {}
        self.feature_columns = []
        self.data = None
        self.model_info = {}

        self._load_models()
        self._load_data()
        self._load_model_info()

    def _load_models(self):
        """Load trained models"""
        try:
            self.models["xgboost"] = joblib.load("models/saved_models/xgboost_model.pkl")
            self.models["lightgbm"] = joblib.load("models/saved_models/lightgbm_model.pkl")
            self.feature_columns = joblib.load("models/saved_models/feature_columns.pkl")
            logger.info("✅ Models loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Some models not found: {e}")

    def _load_data(self):
        """Load processed data"""
        try:
            self.data = pd.read_parquet(self.config["data"]["processed_path"])
            logger.info("✅ Data loaded successfully")
        except FileNotFoundError:
            logger.error("❌ Processed data not found")

    def _load_model_info(self):
        """Load model training information"""
        try:
            with open("results/training_info.json", "r") as f:
                self.model_info = json.load(f)
            logger.info("✅ Model info loaded successfully")
        except FileNotFoundError:
            logger.warning("Model info not found")

    def get_available_centers(self) -> List[str]:
        """Get list of available centers"""
        if self.data is not None:
            return sorted(self.data[self.config["model"]["center_column"]].unique())
        return []

    def get_available_items(self, center: Optional[str] = None) -> List[str]:
        """Get available items for a center"""
        if self.data is not None:
            if center:
                filtered_data = self.data[self.data[self.config["model"]["center_column"]] == center]
            else:
                filtered_data = self.data
            return sorted(filtered_data[self.config["model"]["item_column"]].unique())
        return []

    def generate_forecast(
        self, centers: List[str], items: List[str], forecast_days: int = 30, model_type: str = "xgboost"
    ) -> Tuple[Dict, Dict, Optional[Dict]]:
        """Generate forecasts for specified centers and items"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not available")

        model = self.models[model_type]
        forecasts = {}
        all_predictions = []
        all_actuals = []

        logger.info(f"Generating forecasts for {len(centers)} centers and {len(items)} items...")

        for center in centers:
            forecasts[center] = {}
            for item in items:
                # Get historical data for this combination
                historical = self.data[
                    (self.data[self.config["model"]["center_column"]] == center)
                    & (self.data[self.config["model"]["item_column"]] == item)
                ].copy()

                if historical.empty:
                    logger.warning(f"No data found for {center} - {item}")
                    continue

                # Generate future dates
                last_date = historical[self.config["model"]["date_column"]].max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq="D")

                # Create future dataframe with features
                future_df = self._create_future_dataframe(historical, future_dates, center, item)

                # Make predictions
                predictions = model.predict(future_df[self.feature_columns])

                # Store forecasts
                center_item_forecasts = []
                for i, date in enumerate(future_dates):
                    center_item_forecasts.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "forecast": float(predictions[i]),
                            "lower_bound": float(max(0, predictions[i] * 0.8)),  # 20% lower bound
                            "upper_bound": float(predictions[i] * 1.2),  # 20% upper bound
                        }
                    )

                forecasts[center][item] = center_item_forecasts

                # For metrics calculation (using last 30 days as test)
                if len(historical) > 30:
                    test_data = historical.tail(30)
                    if all(col in test_data.columns for col in self.feature_columns):
                        test_predictions = model.predict(test_data[self.feature_columns])
                        all_predictions.extend(test_predictions)
                        all_actuals.extend(test_data[self.config["model"]["target_column"]].values)

        # Calculate metrics
        metrics = {}
        if all_predictions and all_actuals:
            metrics = self._calculate_metrics(all_actuals, all_predictions)

        # Get feature importance
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            feature_importance = {
                k: v for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            }  # Top 10 features

        logger.info("✅ Forecast generation completed")
        return forecasts, metrics, feature_importance

    def _create_future_dataframe(
        self, historical: pd.DataFrame, future_dates: pd.DatetimeIndex, center: str, item: str
    ) -> pd.DataFrame:
        """Create future dataframe with features"""
        from scripts.data_pipeline import DataPipeline

        # Create base future dataframe
        future_df = pd.DataFrame(
            {
                self.config["model"]["date_column"]: future_dates,
                self.config["model"]["center_column"]: center,
                self.config["model"]["item_column"]: item,
            }
        )

        # Create date features
        pipeline = DataPipeline()
        future_df = pipeline._create_date_features(future_df)

        # Add lag and rolling features (using last available values)
        last_values = historical.tail(30)  # Use last 30 days for reference

        for lag in self.config["features"]["lag_features"]:
            future_df[f"lag_{lag}"] = last_values[self.config["model"]["target_column"]].mean()

        for window in self.config["features"]["rolling_windows"]:
            future_df[f"rolling_mean_{window}"] = last_values[self.config["model"]["target_column"]].mean()
            future_df[f"rolling_std_{window}"] = last_values[self.config["model"]["target_column"]].std()

        # Add seasonal features
        if self.config["features"]["seasonal_features"]:
            future_df = pipeline._create_seasonal_features(future_df)

        # Fill missing values
        future_df = future_df.fillna(0)

        return future_df

    def _calculate_metrics(self, actuals: List[float], predictions: List[float]) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        # Remove zeros to avoid division by zero in MAPE
        mask = actuals > 0
        actuals_filtered = actuals[mask]
        predictions_filtered = predictions[mask]

        metrics = {
            "mae": mean_absolute_error(actuals, predictions),
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "mse": mean_squared_error(actuals, predictions),
        }

        if len(actuals_filtered) > 0:
            metrics["mape"] = np.mean(np.abs((actuals_filtered - predictions_filtered) / actuals_filtered)) * 100

        return metrics

    def get_historical_data(self, center: str, item: str, start_date: str, end_date: str) -> List[Dict]:
        """Get historical data for a center and item"""
        if self.data is None:
            return []

        filtered_data = self.data[
            (self.data[self.config["model"]["center_column"]] == center)
            & (self.data[self.config["model"]["item_column"]] == item)
            & (self.data[self.config["model"]["date_column"]] >= pd.to_datetime(start_date))
            & (self.data[self.config["model"]["date_column"]] <= pd.to_datetime(end_date))
        ]

        historical_data = []
        for _, row in filtered_data.iterrows():
            historical_data.append(
                {
                    "date": row[self.config["model"]["date_column"]].strftime("%Y-%m-%d"),
                    "demand": row[self.config["model"]["target_column"]],
                    "amount": row.get("AMOUNT", 0),
                    "rate": row.get("RATE", 0),
                }
            )

        return historical_data

    def get_model_metrics(self) -> Dict:
        """Get overall model performance metrics"""
        return self.model_info.get("metrics", {})

    def get_inventory_recommendations(self, center: str, item: str, current_stock: float, lead_time: int = 1) -> Dict:
        """Get inventory recommendations based on forecasts"""
        try:
            # Get forecast for lead time period
            forecasts, _, _ = self.generate_forecast([center], [item], forecast_days=lead_time + 7)

            if center in forecasts and item in forecasts[center]:
                item_forecasts = forecasts[center][item]
                lead_time_demand = sum([f["forecast"] for f in item_forecasts[:lead_time]])
                weekly_demand = sum([f["forecast"] for f in item_forecasts[:7]])

                # Calculate safety stock (assuming 20% variability)
                safety_stock = weekly_demand * 0.2
                reorder_point = lead_time_demand + safety_stock
                recommended_order = max(0, reorder_point - current_stock)

                # Calculate stock-out risk
                stock_out_risk = (
                    max(0, (lead_time_demand - current_stock) / lead_time_demand * 100) if lead_time_demand > 0 else 0
                )

                return {
                    "recommended_order": round(recommended_order, 2),
                    "reorder_point": round(reorder_point, 2),
                    "safety_stock": round(safety_stock, 2),
                    "expected_demand": round(weekly_demand, 2),
                    "stock_out_risk": round(stock_out_risk, 2),
                }

        except Exception as e:
            logger.error(f"Error generating inventory recommendations: {e}")

        return {"recommended_order": 0, "reorder_point": 0, "safety_stock": 0, "expected_demand": 0, "stock_out_risk": 0}
