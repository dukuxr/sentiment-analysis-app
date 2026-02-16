"""
Core logic for the sentiment analysis app.
Keep ML/model concerns here; UI stays in app.py / app_ui.py.
"""

from pathlib import Path
import pickle
import re

import numpy as np
import nltk
from nltk.corpus import stopwords

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def ensure_stopwords():
    """Ensure NLTK stopwords are available."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def accuracy_to_percentage(raw_accuracy, default=88.30):
    """Normalize accuracy values stored as either ratio (0-1) or percent (0-100)."""
    try:
        value = float(raw_accuracy)
    except (TypeError, ValueError):
        return default

    return value * 100 if value <= 1 else value


def load_ml_model():
    """Load the trained ML model and vectorizer."""
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    try:
        with open(MODELS_DIR / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    except Exception:
        metadata = {"accuracy": 0.883, "model_type": "logistic"}

    return model, vectorizer, metadata


def preprocess_text(text):
    """Clean and preprocess text (same as training)."""
    stop_words = set(stopwords.words("english"))
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    words = text.split()
    text = " ".join([word for word in words if word not in stop_words])
    return text


def analyze_with_ml(text, model, vectorizer):
    """Analyze sentiment using the trained ML model."""
    if model is None or vectorizer is None:
        return None

    clean_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([clean_text])

    prediction = model.predict(text_vectorized)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vectorized)[0]
        classes = list(getattr(model, "classes_", []))
        if classes and prediction in classes:
            prediction_index = classes.index(prediction)
            confidence = probabilities[prediction_index] * 100
        else:
            confidence = np.max(probabilities) * 100
    else:
        decision = model.decision_function(text_vectorized)[0]
        confidence = min(95, 50 + abs(decision) * 10)

    if isinstance(prediction, str):
        is_positive = prediction.strip().lower() in {"positive", "pos", "1", "true"}
    else:
        is_positive = bool(prediction == 1)

    sentiment = "Positive" if is_positive else "Negative"
    emoji = ""
    color = "#10b981" if is_positive else "#ef4444"

    word_count = len(text.split())
    exclamations = text.count("!")
    questions = text.count("?")

    return {
        "sentiment": sentiment,
        "prediction": prediction,
        "confidence": confidence,
        "emoji": emoji,
        "color": color,
        "word_count": word_count,
        "exclamations": exclamations,
        "questions": questions,
    }


def normalize_sentiment_label(value):
    """Map incoming labels to Positive/Negative/None."""
    if value is None:
        return None

    label = str(value).strip().lower()
    if label in {"1", "positive", "pos", "true", "yes", "p"}:
        return "Positive"
    if label in {"0", "negative", "neg", "false", "no", "n"}:
        return "Negative"
    return None


def sentiment_to_binary(label):
    """Convert Positive/Negative labels to 1/0."""
    normalized = normalize_sentiment_label(label)
    if normalized == "Positive":
        return 1
    if normalized == "Negative":
        return 0
    return None
