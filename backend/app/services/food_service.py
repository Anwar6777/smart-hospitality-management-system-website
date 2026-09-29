from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "modules" / "module-4-food" / "models" / "model.pkl"

_bundle: dict[str, Any] | None = None
_model: Any = None
_model_load_attempted = False


def _load_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Food classification model bundle is missing: modules/module-4-food/models/model.pkl")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    bundle = _load_bundle()
    keras_bytes = bundle.get("model", {}).get("keras_model_bytes")
    if not keras_bytes:
        raise RuntimeError("The food model bundle does not contain the trained MobileNetV2 artifact.")

    temp_path = None
    try:
        import tensorflow as tf
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(keras_bytes)
            temp_path = tmp.name
        _model = tf.keras.models.load_model(temp_path, compile=False)
    except Exception as exc:
        _model = None
        raise RuntimeError(f"Could not load the embedded MobileNetV2 model: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
    return _model


def artifact_available() -> bool:
    try:
        return MODEL_PATH.exists() and bool(_load_bundle().get("model", {}).get("keras_model_bytes"))
    except Exception:
        return False


def predict(image_bytes: bytes) -> dict:
    model = _load_model()
    bundle = _load_bundle()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)[None, ...]
    # The supplied trained model includes its MobileNetV2 preprocessing in the graph.
    probs = np.asarray(model.predict(arr, verbose=0)[0], dtype=float)
    order = np.argsort(probs)[::-1][:5]
    names = bundle.get("class_names") or [str(i) for i in range(len(probs))]
    top = [{"label": names[int(i)], "confidence": float(probs[int(i)])} for i in order]
    return {
        "prediction": top[0],
        "top_predictions": top,
        "model": "MobileNetV2 transfer learning",
        "artifact_available": True,
    }
