import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.data_pipeline import DataPipeline


def test_data_loading():
    """Test data loading functionality"""
    pipeline = DataPipeline()
    df = pipeline.load_raw_data()

    assert not df.empty, "Data should not be empty"
    assert "DATE" in df.columns, "Should have DATE column"
    assert "PAY_WEIGHT" in df.columns, "Should have PAY_WEIGHT column"


def test_feature_creation():
    """Test feature creation"""
    pipeline = DataPipeline()
    df = pipeline.load_raw_data()
    forecasting_df = pipeline.create_forecasting_dataset(df)
    feature_df = pipeline.create_features(forecasting_df)

    # Check if features are created
    expected_features = ["year", "month", "day_of_week", "lag_1", "rolling_mean_7"]
    for feature in expected_features:
        assert feature in feature_df.columns, f"Should have {feature} column"

    assert not feature_df.isnull().all().any(), "No column should have all null values"


if __name__ == "__main__":
    pytest.main([__file__])
