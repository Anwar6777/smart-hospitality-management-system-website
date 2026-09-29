from fastapi import APIRouter, HTTPException, Query
from app.schemas.demand import DemandForecastResponse
from app.services.demand_service import forecast_demand

router = APIRouter(prefix="/demand", tags=["Demand Forecasting"])


@router.get("/forecast", response_model=DemandForecastResponse)
def get_demand_forecast(
    horizon: int = Query(7, ge=1, le=30, description="Forecast horizon in days"),
    model: str = Query("random_forest", description="Demand model to use"),
):
    try:
        return forecast_demand(horizon, model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Demand forecast failed: {exc}") from exc
