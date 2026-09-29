from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.food import FoodPredictionResponse
from app.services.food_service import predict

router = APIRouter(prefix="/food", tags=["Food Classification"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 10 * 1024 * 1024

@router.post("/predict", response_model=FoodPredictionResponse)
async def food_predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, WEBP, or BMP image.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        return predict(data)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")
