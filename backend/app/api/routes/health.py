from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter(tags=["System"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "smart-hospitality-api",
        "environment": "development",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/version")
def version():
    return {
        "api_version": "0.1.0",
        "phase": 0,
    }
