from typing import Literal
from pydantic import BaseModel, Field


SentimentModel = Literal[
    "tfidf_logistic_regression",
    "bilstm",
    "bilstm_glove",
]


class SentimentPredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10000, description="Hotel review text")
    model: SentimentModel = Field(
        default="tfidf_logistic_regression",
        description="Sentiment model to use",
    )


class SentimentPredictionResponse(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"]
    confidence: float
    probabilities: dict[str, float]
    model: str
