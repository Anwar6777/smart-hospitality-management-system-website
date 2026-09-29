from fastapi import APIRouter
from app.schemas.sentiment import SentimentPredictionRequest, SentimentPredictionResponse
from app.services.sentiment_service import get_sentiment_service

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("/predict", response_model=SentimentPredictionResponse)
def predict_sentiment(payload: SentimentPredictionRequest):
    return get_sentiment_service().predict(payload.text, payload.model)
