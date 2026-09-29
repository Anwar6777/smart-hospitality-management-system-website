from typing import Literal
from pydantic import BaseModel, Field


class BookingPredictionRequest(BaseModel):
    model_name: Literal["random_forest", "ann"] = "random_forest"
    hotel: Literal["City Hotel", "Resort Hotel"] = "City Hotel"
    lead_time: int = Field(30, ge=0, le=1000)
    arrival_date_year: int = Field(2017, ge=2015, le=2035)
    arrival_date_month: str = "July"
    arrival_date_week_number: int = Field(27, ge=1, le=53)
    arrival_date_day_of_month: int = Field(15, ge=1, le=31)
    stays_in_weekend_nights: int = Field(1, ge=0, le=50)
    stays_in_week_nights: int = Field(2, ge=0, le=100)
    adults: int = Field(2, ge=0, le=50)
    children: int = Field(0, ge=0, le=20)
    babies: int = Field(0, ge=0, le=10)
    meal: str = "BB"
    country: str = "PRT"
    market_segment: str = "Online TA"
    distribution_channel: str = "TA/TO"
    is_repeated_guest: int = Field(0, ge=0, le=1)
    previous_cancellations: int = Field(0, ge=0, le=100)
    previous_bookings_not_canceled: int = Field(0, ge=0, le=100)
    reserved_room_type: str = "A"
    assigned_room_type: str = "A"
    booking_changes: int = Field(0, ge=0, le=100)
    deposit_type: str = "No Deposit"
    days_in_waiting_list: int = Field(0, ge=0, le=1000)
    customer_type: str = "Transient"
    adr: float = Field(100, ge=0, le=10000)
    required_car_parking_spaces: int = Field(0, ge=0, le=10)
    total_of_special_requests: int = Field(0, ge=0, le=20)
    has_agent: int = Field(1, ge=0, le=1)


class BookingPredictionResponse(BaseModel):
    prediction: Literal["likely_to_cancel", "likely_to_stay"]
    cancellation_probability: float
    stay_probability: float
    risk_level: Literal["low", "medium", "high"]
    model: str
    model_key: Literal["random_forest", "ann"]
