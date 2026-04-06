"""
ScamShield Labeler — 3-tier auto-labeling with heuristic scam signal detection.

Tier 1: Rule-confirmed (confidence = 1.0)
  - Messages from PhishTank, known spam corpora, verified scam URLs

Tier 2: Heuristic-confirmed (confidence = 0.85)
  - Messages with 3+ scam signals (urgency + credential request + shortener)

Tier 3: Human-review queue (confidence = 0.0)
  - Ambiguous messages that need manual annotation
"""

import re


# Scam signal word lists
URGENCY_WORDS = [
    "urgent", "immediately", "asap", "expire", "suspend", "block", "now",
    "hurry", "limited", "deadline", "last chance", "final notice",
    "तुरंत", "अभी", "जल्दी", "ताबडतोब", "आत्ताच", "వెంటనే", "ತಕ್ಷಣ",
]

CREDENTIAL_WORDS = [
    "otp", "password", "cvv", "pin", "account number", "bank account",
    "credit card", "aadhaar", "pan card", "social security",
    "ओटीपी", "पासवर्ड", "ఓటీపీ", "ಒಟಿಪಿ",
]

SHORTENERS = [
    "bit.ly", "tinyurl", "goo.gl", "t.me", "wa.me", "is.gd",
    "cutt.ly", "ow.ly", "rb.gy", "short.io",
]

MONEY_WORDS = [
    "lottery", "winner", "prize", "won", "jackpot", "reward",
    "guaranteed", "free money", "cash prize", "earn",
    "लॉटरी", "जीत", "इनाम", "బహుమతి", "ಬಹುಮಾನ",
]

THREAT_WORDS = [
    "suspended", "deleted", "blocked", "security alert",
    "unauthorized", "unusual activity", "frozen", "closed",
    "खाता बंद", "संदिग्ध", "ఖాతా నిలిపివేయబడుతుంది", "ಖಾತೆ ಮುಚ್ಚಲಾಗುವುದು",
]


def auto_label_tier2(text: str) -> dict:
    """
    Tier 2 heuristic labeling — assigns label based on scam signal count.

    Returns:
        dict with keys: label, confidence, annotator, signals
    """
    text_lower = text.lower()
    scam_signals = 0
    detected_signals = []

    # Check urgency (weight: 1)
    if any(w in text_lower for w in URGENCY_WORDS):
        scam_signals += 1
        detected_signals.append("urgency")

    # Check credential requests (weight: 2 — most dangerous)
    if any(w in text_lower for w in CREDENTIAL_WORDS):
        scam_signals += 2
        detected_signals.append("credential_request")

    # Check URL shorteners (weight: 1)
    if any(w in text_lower for w in SHORTENERS):
        scam_signals += 1
        detected_signals.append("url_shortener")

    # Check money/prize language (weight: 1)
    if any(w in text_lower for w in MONEY_WORDS):
        scam_signals += 1
        detected_signals.append("money_language")

    # Check threats (weight: 1)
    if any(w in text_lower for w in THREAT_WORDS):
        scam_signals += 1
        detected_signals.append("threat")

    # Check excessive exclamation marks (weight: 1)
    if text.count("!") >= 2:
        scam_signals += 1
        detected_signals.append("excessive_exclamation")

    # Check for suspicious URLs (weight: 1)
    risky_tlds = [".tk", ".ml", ".ga", ".cf", ".pw", ".xyz", ".top", ".click"]
    if any(tld in text_lower for tld in risky_tlds):
        scam_signals += 1
        detected_signals.append("risky_tld")

    # Check for IP-based URLs (weight: 1)
    if re.search(r"https?://\d+\.\d+\.\d+\.\d+", text):
        scam_signals += 1
        detected_signals.append("ip_url")

    # Decision
    if scam_signals >= 3:
        return {
            "label": 1,
            "confidence": 0.85,
            "annotator": "auto_tier2",
            "signals": detected_signals,
        }
    elif scam_signals >= 1:
        return {
            "label": None,  # Needs human review
            "confidence": 0.0,
            "annotator": "needs_review",
            "signals": detected_signals,
        }
    else:
        return {
            "label": 0,
            "confidence": 0.70,
            "annotator": "auto_tier2",
            "signals": [],
        }


def categorize_scam(text: str) -> str:
    """Auto-categorize a scam message based on content analysis."""
    text_lower = text.lower()

    # OTP fraud
    if any(w in text_lower for w in ["otp", "pin", "cvv", "password", "ओटीपी", "पासवर्ड"]):
        if any(w in text_lower for w in ["share", "send", "provide", "शेयर", "भेजें", "द्या"]):
            return "otp_fraud"

    # KYC scam
    if any(w in text_lower for w in ["kyc", "केवाईसी"]):
        return "kyc_scam"

    # Lottery
    if any(w in text_lower for w in ["lottery", "winner", "won", "prize", "jackpot",
                                      "लॉटरी", "जीत", "లాటరీ", "ಲಾಟರಿ"]):
        return "lottery"

    # Investment
    if any(w in text_lower for w in ["invest", "return", "guaranteed", "crypto", "trading",
                                      "निवेश", "रिटर्न", "గుంతవణూక"]):
        return "investment"

    # Job scam
    if any(w in text_lower for w in ["job", "earn", "work from home", "hiring",
                                      "कमाएं", "नौकरी", "సంపాదించండి", "ಗಳಿಸಿ"]):
        return "job_scam"

    # Impersonation
    if any(w in text_lower for w in ["government", "sbi", "hdfc", "icici", "rbi",
                                      "सरकार", "ప్రభుత్వం"]):
        return "impersonation"

    # Tech support
    if any(w in text_lower for w in ["virus", "hacked", "infected", "वायरस"]):
        return "tech_support"

    # Package/customs
    if any(w in text_lower for w in ["package", "customs", "delivery fee", "held at"]):
        return "customs_package"

    # Phishing (default for URL-heavy messages)
    if any(w in text_lower for w in ["http", "www", "click", "verify", "login"]):
        return "phishing"

    return "phishing"  # Default category for uncategorized scams
