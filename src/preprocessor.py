"""
ScamShield Preprocessor — Preserves scam signals during text cleaning.

Unlike general NLP preprocessing, this pipeline does NOT:
  - Remove punctuation (exclamation marks are a scam signal)
  - Lowercase everything (uppercase ratio is a feature)
  - Remove URLs (URL features are critical for phishing detection)
"""

import re
import unicodedata


class ScamPreprocessor:
    """
    Preprocessing pipeline that preserves scam-indicative signals.
    """

    # Marathi-specific markers for Hindi vs Marathi disambiguation
    MARATHI_MARKERS = ["आहे", "करा", "आता", "होते", "होईल", "मिळवा", "तुम्ही", "आम्ही", "कृपया"]

    def clean(self, text: str) -> str:
        """
        Clean text while preserving scam signals.
        - Normalizes Unicode (important for Indic scripts)
        - Removes zero-width characters (used in homoglyph attacks)
        - Normalizes whitespace but preserves structure
        """
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode (NFC form — important for Indic scripts)
        text = unicodedata.normalize("NFC", text)

        # Remove zero-width characters (used in homoglyph/evasion attacks)
        text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)

        # Normalize whitespace (but preserve structure)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_urls(self, text: str) -> list:
        """Extract all URLs from text."""
        url_pattern = r"https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/[^\s]*)?"
        return re.findall(url_pattern, text)

    def extract_phone_numbers(self, text: str) -> list:
        """Extract phone numbers (Indian and international formats)."""
        patterns = [
            r"\+91[\s-]?\d{10}",           # Indian +91
            r"\b0\d{10}\b",                 # Indian 0-prefix
            r"\b[6-9]\d{9}\b",             # Indian 10-digit mobile
            r"\+1[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}",  # US
            r"\b1[\s-]?800[\s-]?\d{3}[\s-]?\d{4}\b",        # Toll-free
        ]
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text))
        return numbers

    def strip_urls_for_lang_detection(self, text: str) -> str:
        """
        Remove URLs before language detection — prevents injected English
        URLs in native-script messages from causing misclassification.
        """
        return re.sub(
            r"https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/[^\s]*)?",
            " ", text
        )

    def detect_language(self, text: str) -> str:
        """
        Zero-dependency language detection using Unicode block counting.
        Supports: English (en), Hindi (hi), Marathi (mr), Telugu (te), Kannada (kn).
        """
        stripped = self.strip_urls_for_lang_detection(text)

        # Count characters in each Unicode block
        deva = len(re.findall(r"[\u0900-\u097F]", stripped))   # Devanagari
        telu = len(re.findall(r"[\u0C00-\u0C7F]", stripped))   # Telugu
        kann = len(re.findall(r"[\u0C80-\u0CFF]", stripped))   # Kannada
        latin = len(re.findall(r"[a-zA-Z]", stripped))          # Latin

        total = deva + telu + kann + latin
        if total == 0:
            return "other"

        scores = {"devanagari": deva, "telugu": telu, "kannada": kann, "latin": latin}
        dominant = max(scores, key=scores.get)

        if dominant == "devanagari":
            # Disambiguate Hindi vs Marathi using marker words
            return "mr" if any(m in stripped for m in self.MARATHI_MARKERS) else "hi"

        return {"telugu": "te", "kannada": "kn", "latin": "en"}.get(dominant, "other")

    def has_url(self, text: str) -> bool:
        """Check if text contains any URL."""
        return bool(self.extract_urls(text))

    def has_phone(self, text: str) -> bool:
        """Check if text contains any phone number."""
        return bool(self.extract_phone_numbers(text))
