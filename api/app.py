"""
ScamShield Flask API — Production prediction endpoint.

Endpoints:
  POST /predict  — Single message prediction
  POST /batch    — Batch prediction (up to 100 messages)
  GET  /health   — Health check
  GET  /info     — Model info and supported languages

Usage:
    python api/app.py [--port 5000] [--models-dir models/]
"""

import os
import sys
import json
import time
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, request, jsonify
from src.predict import ScamDetector


app = Flask(__name__)
detector = None  # Initialized on startup


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None,
    })


@app.route("/info", methods=["GET"])
def info():
    """Model information endpoint."""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify({
        "model": "ScamShield",
        "version": "1.0.0",
        "supported_languages": ["en", "hi", "mr", "te", "kn"],
        "features": 32,
        "thresholds": detector.thresholds,
        "categories": [
            "otp_fraud", "lottery", "kyc_scam", "investment",
            "phishing", "impersonation", "job_scam", "romance",
            "tech_support", "customs_package", "charity", "safe"
        ],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single message prediction.

    Request body:
        { "text": "Your message here" }

    Response:
        {
            "verdict": "scam",
            "probability": 0.97,
            "category": "lottery",
            "language": "en",
            "threshold": 0.42,
            "signals": ["urgency_detected", "money_language"],
            "inference_ms": 5.2
        }
    """
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"]
    if not isinstance(text, str) or len(text.strip()) == 0:
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    start_time = time.time()
    result = detector.predict(text)
    inference_ms = (time.time() - start_time) * 1000

    # Remove verbose features from API response
    response = {
        "verdict": result["verdict"],
        "probability": result["probability"],
        "category": result["category"],
        "language": result["language"],
        "threshold": result["threshold"],
        "signals": result["signals"],
        "has_url": result["has_url"],
        "has_phone": result["has_phone"],
        "inference_ms": round(inference_ms, 2),
    }

    return jsonify(response)


@app.route("/batch", methods=["POST"])
def batch_predict():
    """
    Batch prediction (up to 100 messages).

    Request body:
        { "texts": ["message1", "message2", ...] }

    Response:
        { "results": [...], "total": 5, "inference_ms": 25.3 }
    """
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 messages per batch"}), 400

    start_time = time.time()
    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            r = detector.predict(text)
            results.append({
                "verdict": r["verdict"],
                "probability": r["probability"],
                "category": r["category"],
                "language": r["language"],
                "signals": r["signals"],
            })
        else:
            results.append({"verdict": "safe", "probability": 0.0, "error": "invalid input"})

    inference_ms = (time.time() - start_time) * 1000

    return jsonify({
        "results": results,
        "total": len(results),
        "inference_ms": round(inference_ms, 2),
    })


def create_app(models_dir: str = "models"):
    """Create and configure the Flask app."""
    global detector
    try:
        detector = ScamDetector(models_dir)
        print(f"Model loaded from {models_dir}/")
        print(f"Thresholds: {detector.thresholds}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but predictions will fail until models are trained.")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScamShield API Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.chdir(project_root)
    create_app(args.models_dir)
    print(f"\nScamShield API running on http://localhost:{args.port}")
    print(f"  POST /predict  — Single prediction")
    print(f"  POST /batch    — Batch prediction")
    print(f"  GET  /health   — Health check")
    print(f"  GET  /info     — Model info")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
