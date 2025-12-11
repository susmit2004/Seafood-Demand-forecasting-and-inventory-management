from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    date: date
    forecast: float
    lower_bound: float
    upper_bound: float


class CenterItemForecast(BaseModel):
    center: str
    item: str
    forecasts: List[ForecastPoint]
    historical: List[Dict]


class InventoryRecommendation(BaseModel):
    recommended_order: float
    reorder_point: float
    safety_stock: float
    expected_demand: float
    stock_out_risk: float
