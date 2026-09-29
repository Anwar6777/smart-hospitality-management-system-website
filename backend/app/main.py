from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes.health import router as health_router
from app.api.routes.booking import router as booking_router
from app.api.routes.sentiment import router as sentiment_router
from app.api.routes.demand import router as demand_router
from app.api.routes.food import router as food_router

app = FastAPI(
    title=settings.app_name,
    version="0.4.0",
    description="Backend API for the Smart Hospitality Management System.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(booking_router, prefix=settings.api_prefix)
app.include_router(sentiment_router, prefix=settings.api_prefix)
app.include_router(demand_router, prefix=settings.api_prefix)
app.include_router(food_router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "status": "running",
        "phase": 4,
        "docs": "/docs",
    }
