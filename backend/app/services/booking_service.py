from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE / "modules" / "module-1-booking" / "models" / "model.pkl"

BUNDLE = joblib.load(MODEL_PATH)
FEATURE_COLUMNS: list[str] = BUNDLE["preprocessing"]["feature_columns"]
SCALER = BUNDLE["preprocessing"]["scaler"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_MAP = {month: i + 1 for i, month in enumerate(MONTHS)}
TOP_COUNTRIES = set(BUNDLE["metadata"]["top_countries"])


def _transform(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["room_type_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    df["total_previous_bookings"] = (
        df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    )
    df["prev_cancellation_rate"] = (
        df["previous_cancellations"] / (df["total_previous_bookings"] + 1)
    )
    df["is_returning_guest"] = df["is_repeated_guest"]
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP).fillna(1)

    def season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Autumn"

    df["season"] = df["arrival_month_num"].apply(season)
    df["adr_per_guest"] = df["adr"] / df["total_guests"].replace(0, 1)
    df["country_grouped"] = df["country"].apply(
        lambda value: value if value in TOP_COUNTRIES else "Other"
    )

    drop = [
        "country", "arrival_date_month", "arrival_date_year", "arrival_date_week_number",
        "arrival_date_day_of_month", "reserved_room_type", "assigned_room_type",
        "previous_cancellations", "previous_bookings_not_canceled", "is_repeated_guest",
        "stays_in_weekend_nights", "stays_in_week_nights",
    ]
    df = df.drop(columns=drop)
    df = pd.get_dummies(
        df,
        columns=df.select_dtypes(include="object").columns.tolist(),
        drop_first=True,
    )
    return df.reindex(columns=FEATURE_COLUMNS, fill_value=0)


@lru_cache(maxsize=1)
def _load_ann():
    """Load the exact ANN stored inside model.pkl only when ANN is requested."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "ANN inference requires TensorFlow. Install the backend requirements first."
        ) from exc

    keras_bytes = BUNDLE["models"]["ann"]["keras_model_bytes"]

    # Keras 3 expects a filesystem path for this .keras archive.
    # Keep the only persistent model artifact as model.pkl, and materialize
    # the embedded ANN bytes only temporarily during model loading.
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(keras_bytes)
        tmp_path = tmp.name

    try:
        return tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _risk(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def predict(payload: dict, model_name: str = "random_forest") -> dict:
    if model_name not in {"random_forest", "ann"}:
        raise ValueError("model_name must be 'random_forest' or 'ann'")

    X = _transform(payload)

    if model_name == "random_forest":
        estimator = BUNDLE["models"]["random_forest"]["estimator"]
        probability = float(estimator.predict_proba(X)[0, 1])
        display_name = "Random Forest"
    else:
        ann = _load_ann()
        X_scaled = SCALER.transform(X).astype(np.float32)
        probability = float(np.asarray(ann.predict(X_scaled, verbose=0)).reshape(-1)[0])
        display_name = "ANN"

    threshold = float(BUNDLE["models"][model_name].get("threshold", 0.5))
    prediction = "likely_to_cancel" if probability >= threshold else "likely_to_stay"

    return {
        "prediction": prediction,
        "cancellation_probability": round(probability, 4),
        "stay_probability": round(1 - probability, 4),
        "risk_level": _risk(probability),
        "model": display_name,
        "model_key": model_name,
    }
