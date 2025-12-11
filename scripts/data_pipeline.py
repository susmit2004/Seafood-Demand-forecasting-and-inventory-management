import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.data_config = self.config["data"]
        self.model_config = self.config["model"]
        self.feature_config = self.config["features"]

    def load_raw_data(self) -> pd.DataFrame:
        """Load and preprocess raw data"""
        logger.info("Loading raw data...")

        # Check if file exists
        if not os.path.exists(self.data_config["raw_path"]):
            raise FileNotFoundError(f"Raw data file not found at {self.data_config['raw_path']}")

        df = pd.read_csv(self.data_config["raw_path"])

        # Print column names for debugging
        logger.info(f"Columns in raw data: {df.columns.tolist()}")

        # Convert date column
        df[self.model_config["date_column"]] = pd.to_datetime(df[self.model_config["date_column"]], errors="coerce")

        # Handle missing dates
        df = df.dropna(subset=[self.model_config["date_column"]])

        # Handle missing values
        df = self._handle_missing_values(df)

        logger.info(f"Loaded {len(df)} rows of data")
        logger.info(
            f"Date range: {df[self.model_config['date_column']].min()} to {df[self.model_config['date_column']].max()}"
        )

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numeric columns with 0
        numeric_cols = ["BOX", "NET", "ICE/WATER", "PAY WEIGHT", "RATE", "AMOUNT"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Fill categorical columns with 'Unknown'
        categorical_cols = ["VEHICLE NO", "CENTER NAME", "PARTY NAME", "ITEM"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        return df

    def create_forecasting_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series dataset for forecasting"""
        logger.info("Creating forecasting dataset...")

        # Check if required columns exist
        required_cols = [
            self.model_config["date_column"],
            self.model_config["center_column"],
            self.model_config["item_column"],
            self.model_config["target_column"],
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Aggregate data by date, center, and item
        agg_df = (
            df.groupby(
                [self.model_config["date_column"], self.model_config["center_column"], self.model_config["item_column"]]
            )
            .agg({self.model_config["target_column"]: "sum", "AMOUNT": "sum", "RATE": "mean", "BOX": "sum"})
            .reset_index()
        )

        logger.info(f"Aggregated data shape: {agg_df.shape}")

        # Create complete date range
        min_date = agg_df[self.model_config["date_column"]].min()
        max_date = agg_df[self.model_config["date_column"]].max()
        all_dates = pd.date_range(min_date, max_date, freq="D")

        # Get unique centers and items
        centers = agg_df[self.model_config["center_column"]].unique()
        items = agg_df[self.model_config["item_column"]].unique()

        logger.info(f"Unique centers: {len(centers)}, Unique items: {len(items)}")

        # Create complete panel data using merge (more memory efficient)
        date_center_item_combinations = pd.MultiIndex.from_product(
            [all_dates, centers, items],
            names=[self.model_config["date_column"], self.model_config["center_column"], self.model_config["item_column"]],
        ).to_frame(index=False)

        # Merge with actual data
        complete_df = date_center_item_combinations.merge(
            agg_df,
            on=[self.model_config["date_column"], self.model_config["center_column"], self.model_config["item_column"]],
            how="left",
        )

        # Fill missing values
        complete_df[self.model_config["target_column"]] = complete_df[self.model_config["target_column"]].fillna(0)
        complete_df["AMOUNT"] = complete_df["AMOUNT"].fillna(0)
        complete_df["RATE"] = complete_df["RATE"].fillna(complete_df["RATE"].mean())
        complete_df["BOX"] = complete_df["BOX"].fillna(0)

        logger.info(f"Complete dataset shape: {complete_df.shape}")

        return complete_df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for forecasting"""
        logger.info("Creating features...")

        # Sort by date
        df = df.sort_values(
            [self.model_config["date_column"], self.model_config["center_column"], self.model_config["item_column"]]
        )

        # Date-based features
        df = self._create_date_features(df)

        # Lag features
        df = self._create_lag_features(df)

        # Rolling statistics
        df = self._create_rolling_features(df)

        # Seasonal features
        if self.feature_config["seasonal_features"]:
            df = self._create_seasonal_features(df)

        logger.info(f"Final dataset with features shape: {df.shape}")
        logger.info(
            f"Feature columns: {[col for col in df.columns if col not in [self.model_config['date_column'], self.model_config['center_column'], self.model_config['item_column'], self.model_config['target_column']]]}"
        )

        return df

    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-based features"""
        df["year"] = df[self.model_config["date_column"]].dt.year
        df["month"] = df[self.model_config["date_column"]].dt.month
        df["day"] = df[self.model_config["date_column"]].dt.day
        df["day_of_week"] = df[self.model_config["date_column"]].dt.dayofweek
        df["day_of_year"] = df[self.model_config["date_column"]].dt.dayofyear
        df["week_of_year"] = df[self.model_config["date_column"]].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = df[self.model_config["date_column"]].dt.is_month_start.astype(int)
        df["is_month_end"] = df[self.model_config["date_column"]].dt.is_month_end.astype(int)
        df["quarter"] = df[self.model_config["date_column"]].dt.quarter

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        logger.info("Creating lag features...")

        # Sort to ensure proper lag calculation
        df = df.sort_values(
            [self.model_config["center_column"], self.model_config["item_column"], self.model_config["date_column"]]
        )

        for lag in self.feature_config["lag_features"]:
            df[f"lag_{lag}"] = df.groupby([self.model_config["center_column"], self.model_config["item_column"]])[
                self.model_config["target_column"]
            ].shift(lag)

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics features"""
        logger.info("Creating rolling features...")

        for window in self.feature_config["rolling_windows"]:
            df[f"rolling_mean_{window}"] = df.groupby([self.model_config["center_column"], self.model_config["item_column"]])[
                self.model_config["target_column"]
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

            df[f"rolling_std_{window}"] = df.groupby([self.model_config["center_column"], self.model_config["item_column"]])[
                self.model_config["target_column"]
            ].transform(lambda x: x.rolling(window=window, min_periods=1).std())

            df[f"rolling_min_{window}"] = df.groupby([self.model_config["center_column"], self.model_config["item_column"]])[
                self.model_config["target_column"]
            ].transform(lambda x: x.rolling(window=window, min_periods=1).min())

            df[f"rolling_max_{window}"] = df.groupby([self.model_config["center_column"], self.model_config["item_column"]])[
                self.model_config["target_column"]
            ].transform(lambda x: x.rolling(window=window, min_periods=1).max())

        return df

    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal features"""
        # Monthly seasonality
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Weekly seasonality
        df["week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Yearly seasonality
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        return df

    def run_pipeline(self) -> pd.DataFrame:
        """Run complete data pipeline"""
        try:
            # Load raw data
            raw_df = self.load_raw_data()

            # Create forecasting dataset
            forecasting_df = self.create_forecasting_dataset(raw_df)

            # Create features
            feature_df = self.create_features(forecasting_df)

            # Save processed data
            os.makedirs(os.path.dirname(self.data_config["processed_path"]), exist_ok=True)
            feature_df.to_parquet(self.data_config["processed_path"], index=False)

            # Print summary statistics
            self._print_summary_statistics(feature_df)

            logger.info("Data pipeline completed successfully!")
            logger.info(f"Processed data saved to {self.data_config['processed_path']}")

            return feature_df

        except Exception as e:
            logger.error(f"Error in data pipeline: {e}")
            raise

    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics of the processed data"""
        logger.info("\n" + "=" * 50)
        logger.info("DATA PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total records: {len(df):,}")
        logger.info(
            f"Date range: {df[self.model_config['date_column']].min()} to {df[self.model_config['date_column']].max()}"
        )
        logger.info(f"Number of centers: {df[self.model_config['center_column']].nunique()}")
        logger.info(f"Number of items: {df[self.model_config['item_column']].nunique()}")
        logger.info(f"Total demand: {df[self.model_config['target_column']].sum():,.0f} kg")
        logger.info(f"Average daily demand: {df[self.model_config['target_column']].mean():.2f} kg")
        logger.info(f"Number of features: {len(df.columns) - 4}")  # Excluding date, center, item, target

        # Top items by demand
        top_items = df.groupby(self.model_config["item_column"])[self.model_config["target_column"]].sum().nlargest(5)
        logger.info("\nTop 5 items by demand:")
        for item, demand in top_items.items():
            logger.info(f"  {item}: {demand:,.0f} kg")

        # Top centers by demand
        top_centers = df.groupby(self.model_config["center_column"])[self.model_config["target_column"]].sum().nlargest(5)
        logger.info("\nTop 5 centers by demand:")
        for center, demand in top_centers.items():
            logger.info(f"  {center}: {demand:,.0f} kg")


if __name__ == "__main__":
    try:
        pipeline = DataPipeline()
        processed_data = pipeline.run_pipeline()
        print("\n[SUCCESS] Data pipeline completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Error in data pipeline: {e}")
        import traceback

        traceback.print_exc()
