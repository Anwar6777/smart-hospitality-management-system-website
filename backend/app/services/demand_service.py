from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3] / "modules" / "module-3-demand"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "daily_demand.csv"

BUNDLE = None
HISTORY = None

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "linear_regression": "Linear Regression",
    "lstm": "LSTM",
    "lstm_improved": "Improved LSTM",
}


def _load_bundle():
    global BUNDLE, HISTORY
    if BUNDLE is None:
        BUNDLE = joblib.load(MODEL_PATH)
        HISTORY = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def available_models() -> list[str]:
    _load_bundle()
    return list(MODEL_LABELS)


def _features_for_date(date: pd.Timestamp, values: list[float]) -> dict[str, float]:
    dow = date.dayofweek
    month = date.month
    row = {
        "day_of_month": date.day,
        "is_weekend": int(dow in (5, 6)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
    }
    for lag in (1, 2, 3, 7, 14):
        row[f"lag_{lag}"] = float(values[-lag])
    recent7 = np.asarray(values[-7:], dtype=float)
    recent14 = np.asarray(values[-14:], dtype=float)
    recent30 = np.asarray(values[-30:], dtype=float)
    row["rolling_mean_7"] = float(recent7.mean())
    row["rolling_std_7"] = float(recent7.std(ddof=1)) if len(recent7) > 1 else 0.0
    row["rolling_mean_14"] = float(recent14.mean())
    row["rolling_mean_30"] = float(recent30.mean())
    return row


def _predict_sklearn(model_name: str, feature_dict: dict[str, float]) -> float:
    estimator = BUNDLE["models"][model_name]["estimator"]
    columns = BUNDLE["preprocessing"]["feature_columns"]
    row = pd.DataFrame([feature_dict])[columns]
    return float(estimator.predict(row)[0])


def _load_keras_model(model_name: str):
    import tensorflow as tf

    model_bytes = BUNDLE["models"][model_name]["keras_model_bytes"]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as handle:
            handle.write(model_bytes)
            temp_path = handle.name
        return tf.keras.models.load_model(temp_path, compile=False)
    finally:
        # The caller loads the model into memory before this path is removed.
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _inverse_target(scaled_target, scaler, n_features: int):
    dummy = np.zeros((len(np.atleast_1d(scaled_target)), n_features), dtype=float)
    dummy[:, 0] = np.atleast_1d(scaled_target)
    return scaler.inverse_transform(dummy)[:, 0]


def _calendar_features(date: pd.Timestamp) -> list[float]:
    dow = date.dayofweek
    month = date.month
    return [
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ]


def _forecast_lstm(model_name: str, values: list[float], last_date: pd.Timestamp, horizon: int) -> list[float]:
    model = _load_keras_model(model_name)
    scaler_key = "lstm_scaler" if model_name == "lstm" else "lstm_improved_scaler"
    scaler = BUNDLE["preprocessing"][scaler_key]
    window_size = 14
    predictions = []

    # Build the most recent 14 historical feature rows.
    feature_rows: list[list[float]] = []
    if model_name == "lstm":
        for idx in range(len(values) - window_size, len(values)):
            date = HISTORY["date"].iloc[idx]
            feature_rows.append([values[idx], *_calendar_features(date)])
    else:
        # Residual = demand - trailing 14-day causal average.
        for idx in range(len(values) - window_size, len(values)):
            if idx < 14:
                raise ValueError("Not enough historical rows for improved LSTM")
            local_avg = float(np.mean(values[idx - 14:idx]))
            feature_rows.append([values[idx] - local_avg, *_calendar_features(HISTORY["date"].iloc[idx])])

    for step in range(1, horizon + 1):
        target_date = last_date + pd.Timedelta(days=step)
        scaled_window = scaler.transform(np.asarray(feature_rows[-window_size:], dtype=float))
        scaled_pred = float(model.predict(scaled_window[np.newaxis, ...], verbose=0).ravel()[0])
        raw_pred = float(_inverse_target([scaled_pred], scaler, 5)[0])

        if model_name == "lstm_improved":
            local_avg = float(np.mean(values[-14:]))
            prediction = max(0.0, raw_pred + local_avg)
            residual = prediction - local_avg
            new_feature_row = [residual, *_calendar_features(target_date)]
        else:
            prediction = max(0.0, raw_pred)
            new_feature_row = [prediction, *_calendar_features(target_date)]

        values.append(prediction)
        feature_rows.append(new_feature_row)
        predictions.append(prediction)

    return predictions


def forecast_demand(horizon: int, model_name: str = "random_forest"):
    _load_bundle()
    if not 1 <= horizon <= 30:
        raise ValueError("horizon must be between 1 and 30 days")
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported demand model: {model_name}")

    history = HISTORY.copy()
    values = history["Total"].astype(float).tolist()
    last_date = history["date"].iloc[-1]

    if model_name in {"lstm", "lstm_improved"}:
        predictions = _forecast_lstm(model_name, values, last_date, horizon)
        points = [
            {"date": (last_date + pd.Timedelta(days=i)).date(), "predicted_demand": round(float(value), 2)}
            for i, value in enumerate(predictions, start=1)
        ]
    else:
        points = []
        for step in range(1, horizon + 1):
            target_date = last_date + pd.Timedelta(days=step)
            feature_dict = _features_for_date(target_date, values)
            prediction = max(0.0, _predict_sklearn(model_name, feature_dict))
            values.append(prediction)
            points.append({"date": target_date.date(), "predicted_demand": round(prediction, 2)})

    forecast_values = np.array([p["predicted_demand"] for p in points], dtype=float)
    peak_idx = int(np.argmax(forecast_values))
    metrics = BUNDLE["metadata"]["metrics"].get(model_name, {})

    return {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "horizon": horizon,
        "last_historical_date": last_date.date(),
        "forecast_start_date": points[0]["date"],
        "forecast_end_date": points[-1]["date"],
        "total_predicted_demand": round(float(forecast_values.sum()), 2),
        "average_daily_demand": round(float(forecast_values.mean()), 2),
        "peak_date": points[peak_idx]["date"],
        "peak_demand": round(float(forecast_values[peak_idx]), 2),
        "historical_rows": int(len(history)),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "forecast": points,
    }


def demand_history(limit: int = 90):
    _load_bundle()
    data = HISTORY.tail(limit)
    return [{"date": row.date.date(), "actual_demand": float(row.Total)} for row in data.itertuples()]
