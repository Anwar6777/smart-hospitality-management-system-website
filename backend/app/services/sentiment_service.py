from __future__ import annotations

import io
import os
import pickle
import tempfile
from pathlib import Path

import joblib
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:  # TensorFlow is only required for the BiLSTM variants.
    tf = None
    pad_sequences = None

import re


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "modules" / "module-2-sentiment" / "models" / "model.pkl"

STOPWORDS = set("""a an the and or but if while of at by for with about against
between into through during before after above below to from up down in out on off
over under again further then once here there all any both each few more most other
some such no nor not only own same so than too very s t can will just don should now
i me my myself we our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs what which who whom
this that these those am is are was were be been being have has had having do does did doing
would could should ought i'm you're he's she's it's we're they're""".split())


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def _load_pickle_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


class SentimentService:
    def __init__(self) -> None:
        self.bundle = joblib.load(MODEL_PATH)

        preprocessing = self.bundle["preprocessing"]
        self.vectorizer = _load_pickle_bytes(preprocessing["tfidf_vectorizer_bytes"])
        self.label_encoder = _load_pickle_bytes(preprocessing["label_encoder_bytes"])

        self.default_model = self.bundle.get(
            "default_model", "tfidf_logistic_regression"
        )
        self.sequence_length = int(
            self.bundle.get("metadata", {}).get("sequence_length", 150)
        )

        self._lr_model = None
        self._keras_models: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}

    def _get_lr_model(self):
        if self._lr_model is None:
            data = self.bundle["models"]["tfidf_logistic_regression"]["estimator_bytes"]
            self._lr_model = _load_pickle_bytes(data)
        return self._lr_model

    def _get_keras_model(self, model_name: str):
        if tf is None:
            raise RuntimeError(
                "TensorFlow is required for BiLSTM sentiment models. "
                "Install the backend requirements before selecting a BiLSTM model."
            )

        if model_name in self._keras_models:
            return self._keras_models[model_name]

        data = self.bundle["models"][model_name]["keras_model_bytes"]

        # Keras 3 expects a filesystem path for .keras archives in this environment.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".keras", delete=False
            ) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name

            model = tf.keras.models.load_model(tmp_path, compile=False)
            self._keras_models[model_name] = model
            return model
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _get_tokenizer(self, model_name: str):
        if model_name not in self._tokenizers:
            data = self.bundle["models"][model_name]["tokenizer_bytes"]
            self._tokenizers[model_name] = _load_pickle_bytes(data)
        return self._tokenizers[model_name]

    def _probability_map(self, probabilities) -> dict[str, float]:
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        classes = [str(x) for x in self.label_encoder.classes_]
        return {
            label: round(float(prob), 6)
            for label, prob in zip(classes, probabilities)
        }

    def _predict_lr(self, text: str):
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.strip().lower()

        features = self.vectorizer.transform([cleaned])
        probabilities = self._get_lr_model().predict_proba(features)[0]
        return np.asarray(probabilities, dtype=float)

    def _predict_bilstm(self, text: str, model_name: str):
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.strip().lower()

        tokenizer = self._get_tokenizer(model_name)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(
            sequence,
            maxlen=self.sequence_length,
            padding="post",
            truncating="post",
        )

        output = self._get_keras_model(model_name).predict(padded, verbose=0)
        return np.asarray(output[0], dtype=float)

    def predict(self, text: str, model_name: str | None = None) -> dict:
        selected_model = model_name or self.default_model
        available = self.bundle["metadata"]["available_models"]

        if selected_model not in available:
            raise ValueError(
                f"Unknown sentiment model '{selected_model}'. "
                f"Available models: {', '.join(available)}"
            )

        if selected_model == "tfidf_logistic_regression":
            probabilities = self._predict_lr(text)
        else:
            probabilities = self._predict_bilstm(text, selected_model)

        predicted_index = int(np.argmax(probabilities))
        sentiment = str(
            self.label_encoder.inverse_transform([predicted_index])[0]
        )

        return {
            "sentiment": sentiment,
            "confidence": round(float(probabilities[predicted_index]), 6),
            "probabilities": self._probability_map(probabilities),
            "model": selected_model,
        }


_service: SentimentService | None = None


def get_sentiment_service() -> SentimentService:
    global _service
    if _service is None:
        _service = SentimentService()
    return _service
