from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.main import forecast_engine
from scripts.utils import ForecastEngine

router = APIRouter()


class ForecastRequest(BaseModel):
    centers: List[str]
    items: List[str]
    forecast_days: int = 30
    model_type: str = "xgboost"


class ForecastResponse(BaseModel):
    forecasts: Dict[str, Dict[str, List[Dict]]]
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]


@router.get("/centers")
async def get_centers():
    """Get all available centers"""
    try:
        centers = forecast_engine.get_available_centers()
        return {"centers": centers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/items")
async def get_items(center: Optional[str] = None):
    """Get available items for a center"""
    try:
        items = forecast_engine.get_available_items(center)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate forecasts for specified centers and items"""
    try:
        forecasts, metrics, feature_importance = forecast_engine.generate_forecast(
            centers=request.centers, items=request.items, forecast_days=request.forecast_days, model_type=request.model_type
        )

        return ForecastResponse(forecasts=forecasts, metrics=metrics, feature_importance=feature_importance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical")
async def get_historical_data(center: str, item: str, start_date: date, end_date: date):
    """Get historical data for a center and item"""
    try:
        historical_data = forecast_engine.get_historical_data(
            center=center, item=item, start_date=start_date, end_date=end_date
        )
        return historical_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    try:
        metrics = forecast_engine.get_model_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inventory-recommendations")
async def get_inventory_recommendations(center: str, item: str, current_stock: float, lead_time: int = 1):
    """Get inventory recommendations"""
    try:
        recommendations = forecast_engine.get_inventory_recommendations(
            center=center, item=item, current_stock=current_stock, lead_time=lead_time
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
