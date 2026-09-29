from datetime import date
from pydantic import BaseModel, Field


class DemandForecastPoint(BaseModel):
    date: date
    predicted_demand: float = Field(ge=0)


class DemandForecastResponse(BaseModel):
    model: str
    model_label: str
    horizon: int
    last_historical_date: date
    forecast_start_date: date
    forecast_end_date: date
    total_predicted_demand: float
    average_daily_demand: float
    peak_date: date
    peak_demand: float
    historical_rows: int
    metrics: dict[str, float]
    forecast: list[DemandForecastPoint]
