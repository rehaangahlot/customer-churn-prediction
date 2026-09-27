"""
Flask Web Application for Customer Churn Prediction.
Serves responsive web interface and REST prediction API.
"""

import os
import json
import logging
from datetime import datetime, timezone
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

from src.preprocessing import transform_single_customer, FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base directory paths (platform-independent)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model", "metadata.json")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global variables for model and scaler
model = None
scaler = None
metadata = {}


def load_artifacts():
    """Loads the trained model, scaler, and metadata into memory."""
    global model, scaler, metadata

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.warning("Model or scaler not found. Running train.py may be required.")
        return False

    try:
        logger.info(f"Loading ANN model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)

        logger.info(f"Loading scaler from: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info("Artifacts loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}", exc_info=True)
        return False


# Attempt loading artifacts on application startup
load_artifacts()


@app.route("/", methods=["GET"])
def home():
    """Renders the main customer churn prediction dashboard."""
    metrics = metadata.get("metrics", {})
    return render_template(
        "index.html",
        accuracy=round(metrics.get("test_accuracy", 0.77) * 100, 1),
        precision=round(metrics.get("precision", 0.62) * 100, 1),
        recall=round(metrics.get("recall", 0.53) * 100, 1),
        f1=round(metrics.get("f1_score", 0.57) * 100, 1)
    )


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for cloud monitoring, containers, and deployment status.
    Returns: {"status": "healthy"}
    """
    is_ready = model is not None and scaler is not None
    status_str = "healthy" if is_ready else "degraded (model not loaded)"
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "ready": is_ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": os.environ.get("FLASK_ENV", "production")
    }), (200 if is_ready else 503)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction API endpoint.
    Accepts customer profile in JSON format and returns churn probabilities.
    """
    global model, scaler

    # Lazy-load if not already in memory
    if model is None or scaler is None:
        if not load_artifacts():
            return jsonify({
                "error": "Model artifacts not loaded. Please ensure train.py has been executed."
            }), 503

    try:
        # Support both application/json and form-encoded data
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()

        if not data:
            return jsonify({"error": "No customer data provided in request."}), 400

        # Transform inputs into scaled vector
        feature_vector = transform_single_customer(data, scaler)

        # Make prediction
        prediction_raw = model.predict(feature_vector, verbose=0)
        churn_prob = float(prediction_raw[0][0])
        non_churn_prob = float(1.0 - churn_prob)

        # Class decision threshold (0.5)
        is_churn = churn_prob >= 0.5
        prediction_label = "Churn" if is_churn else "No Churn"

        # Determine risk level
        if churn_prob >= 0.70:
            risk_level = "High"
        elif churn_prob >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        response = {
            "prediction": prediction_label,
            "churn_probability": round(churn_prob, 4),
            "non_churn_probability": round(non_churn_prob, 4),
            "risk_level": risk_level,
            "threshold": 0.5
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to process prediction.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    # Support dynamic PORT environment variable for cloud hosting
    port = int(os.environ.get("PORT", 5000))
    # Host 0.0.0.0 enables local network & container access
    app.run(host="0.0.0.0", port=port, debug=False)

