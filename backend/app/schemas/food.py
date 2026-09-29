from pydantic import BaseModel, Field
from typing import List

class FoodPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

class FoodPredictionResponse(BaseModel):
    prediction: FoodPrediction
    top_predictions: List[FoodPrediction]
    model: str
    artifact_available: bool
