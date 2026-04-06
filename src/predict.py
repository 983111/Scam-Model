"""
ScamShield Predictor — Production inference module.

Loads trained models and provides prediction with:
  - Scam probability
  - Verdict (scam/safe) based on per-language threshold
  - Top contributing signals (explainability)
  - Detected language
  - Scam category
"""

import os
import json
import joblib
import numpy as np

from src.preprocessor import ScamPreprocessor
from src.feature_extractor import (
    extract_all_features, extract_all_32_features, FEATURE_NAMES
)
from src.labeler import categorize_scam


class ScamDetector:
    """
    Production-ready scam detector.

    Usage:
        detector = ScamDetector("models/")
        result = detector.predict("You've won a lottery! Click here to claim!")
        print(result)
        # {
        #   "verdict": "scam",
        #   "probability": 0.97,
        #   "category": "lottery",
        #   "language": "en",
        #   "threshold": 0.42,
        #   "signals": ["urgency", "money_language", "exclamation"],
        #   "features": {...}
        # }
    """

    def __init__(self, models_dir: str = "models"):
        self.preprocessor = ScamPreprocessor()

        # Load n-gram model
        ngram_path = os.path.join(models_dir, "ngram_model.pkl")
        if os.path.exists(ngram_path):
            self.ngram_model = joblib.load(ngram_path)
        else:
            raise FileNotFoundError(f"N-gram model not found: {ngram_path}")

        # Load main classifier
        model_path = os.path.join(models_dir, "scam_detector_final.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Main model not found: {model_path}")

        # Load per-language thresholds
        thresholds_path = os.path.join(models_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {"en": 0.5, "hi": 0.5, "mr": 0.5, "te": 0.5, "kn": 0.5}

        self.default_threshold = 0.5

    def predict(self, text: str) -> dict:
        """
        Predict whether a message is scam or safe.

        Args:
            text: Raw message text

        Returns:
            dict with verdict, probability, category, language, signals, etc.
        """
        # Clean text
        cleaned = self.preprocessor.clean(text)
        if not cleaned:
            return self._empty_result()

        # Detect language
        language = self.preprocessor.detect_language(cleaned)

        # Extract features
        features_dict = extract_all_features(cleaned, self.ngram_model)
        features_vector = np.array([features_dict[name] for name in FEATURE_NAMES]).reshape(1, -1)

        # Get probability
        proba = float(self.model.predict_proba(features_vector)[0][1])

        # Get threshold for this language
        threshold = self.thresholds.get(language, self.default_threshold)

        # Verdict
        is_scam = proba >= threshold
        verdict = "scam" if is_scam else "safe"

        # Category
        category = categorize_scam(cleaned) if is_scam else "safe"

        # Extract explainability signals
        signals = self._extract_signals(features_dict, proba)

        return {
            "verdict": verdict,
            "probability": round(proba, 4),
            "category": category,
            "language": language,
            "threshold": round(threshold, 4),
            "signals": signals,
            "features": features_dict,
            "text_length": len(cleaned),
            "has_url": self.preprocessor.has_url(cleaned),
            "has_phone": self.preprocessor.has_phone(cleaned),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict for a batch of texts."""
        return [self.predict(text) for text in texts]

    def _extract_signals(self, features: dict, proba: float) -> list:
        """Extract the most important signals from features for explainability."""
        signals = []

        if features.get("f1_has_urgency"):
            signals.append("urgency_detected")
        if features.get("f2_has_money"):
            signals.append("money_language")
        if features.get("f3_has_sensitive"):
            signals.append("credential_request")
        if features.get("f4_has_off_platform"):
            signals.append("off_platform_redirect")
        if features.get("f5_has_threat"):
            signals.append("threat_language")
        if features.get("f8_exclamation_count", 0) >= 2:
            signals.append("excessive_exclamation")
        if features.get("f10_uppercase_ratio", 0) > 0.3:
            signals.append("high_uppercase_ratio")
        if features.get("f20_has_ip_url"):
            signals.append("ip_based_url")
        if features.get("f21_has_shortener"):
            signals.append("url_shortener")
        if features.get("f22_has_risky_tld"):
            signals.append("risky_domain")
        if features.get("f23_has_domain_spoof"):
            signals.append("domain_spoofing")
        if features.get("f31_script_mismatch", 0) > 0.3:
            signals.append("script_mismatch")
        if features.get("f32_ngram_score", 0) > 0.7:
            signals.append("ngram_scam_pattern")
        if features.get("f6_has_legitimacy_marker"):
            signals.append("legitimacy_marker")

        return signals

    def _empty_result(self) -> dict:
        return {
            "verdict": "safe",
            "probability": 0.0,
            "category": "safe",
            "language": "other",
            "threshold": self.default_threshold,
            "signals": [],
            "features": {},
            "text_length": 0,
            "has_url": False,
            "has_phone": False,
        }
