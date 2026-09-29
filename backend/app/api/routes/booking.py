from fastapi import APIRouter, HTTPException
from app.schemas.booking import BookingPredictionRequest, BookingPredictionResponse
from app.services.booking_service import predict

router = APIRouter(prefix="/booking", tags=["Module 1 - Booking"])


@router.post("/predict", response_model=BookingPredictionResponse)
def booking_prediction(request: BookingPredictionRequest):
    payload = request.model_dump()
    model_name = payload.pop("model_name")
    try:
        return predict(payload, model_name=model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
