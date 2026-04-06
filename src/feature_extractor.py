"""
ScamShield Feature Extractor — 32 handcrafted features across 5 groups.

Feature Groups:
  f1–f6:   Binary keyword features (urgency, money, sensitive, off-platform, threat, legitimacy)
  f7–f14:  Statistical text features (length, entropy, ratios, densities)
  f15–f17: Keyword density features
  f18–f24: URL features (shorteners, risky TLDs, IP URLs, domain spoofing)
  f25–f32: Multilingual features (language encoding, script mismatch, n-gram score)
"""

import re
import math
from collections import Counter
from urllib.parse import urlparse

from src.preprocessor import ScamPreprocessor


# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORD LEXICONS — Multilingual scam signal dictionaries
# ═══════════════════════════════════════════════════════════════════════════════

KEYWORD_LEXICONS = {
    "urgency": {
        "en": ["urgent", "immediately", "asap", "expires", "final notice", "act now",
               "last chance", "24 hours", "suspended", "blocked", "hurry", "limited time",
               "deadline", "right away", "don't delay", "time sensitive", "expiring"],
        "hi": ["तुरंत", "अभी", "जल्दी", "खाता बंद", "तत्काल", "फौरन", "समय सीमा"],
        "mr": ["ताबडतोब", "आत्ताच", "खाते बंद", "लगेच"],
        "te": ["వెంటనే", "ఇప్పుడే", "ఖాతా మూసివేయబడుతుంది", "త్వరగా"],
        "kn": ["ತಕ್ಷಣ", "ಈಗಲೇ", "ಖಾತೆ ಮುಚ್ಚಲಾಗುವುದು", "ಬೇಗ"],
    },
    "money": {
        "en": ["lottery", "winner", "prize", "earn", "guaranteed", "crypto",
               "free money", "cash prize", "reward", "won", "jackpot", "million",
               "billion", "profit", "investment return", "double your money",
               "risk free", "100%", "200%", "300%", "unlimited income"],
        "hi": ["लॉटरी", "बधाई हो", "इनाम", "कमाएं", "मुफ्त", "जीत", "पुरस्कार",
               "करोड़", "लाख", "गारंटीड रिटर्न"],
        "mr": ["लॉटरी", "बक्षीस", "जिंकले", "कमवा", "मोफत"],
        "te": ["లాటరీ", "బహుమతి", "గెలిచారు", "సంపాదించండి", "ఉచితం"],
        "kn": ["ಲಾಟರಿ", "ಬಹುಮಾನ", "ಗೆದ್ದಿದ್ದೀರಿ", "ಗಳಿಸಿ", "ಉಚಿತ"],
    },
    "sensitive": {
        "en": ["password", "cvv", "pin", "otp", "social security", "bank account",
               "credit card", "verify now", "confirm identity", "card number",
               "expiry date", "security code", "login credentials", "ssn",
               "aadhaar", "pan card", "pan number"],
        "hi": ["पासवर्ड", "ओटीपी", "खाता नंबर", "सत्यापित करें", "आधार",
               "पैन कार्ड", "बैंक खाता", "क्रेडिट कार्ड"],
        "mr": ["पासवर्ड", "ओटीपी", "खाते क्रमांक", "आधार", "पॅन कार्ड"],
        "te": ["పాస్వర్డ్", "ఓటీపీ", "ఖాతా సంఖ్య", "ఆధార్", "పాన్ కార్డ్"],
        "kn": ["ಪಾಸ್ವರ್ಡ್", "ಒಟಿಪಿ", "ಖಾತೆ ಸಂಖ್ಯೆ", "ಆಧಾರ್", "ಪ್ಯಾನ್ ಕಾರ್ಡ್"],
    },
    "off_platform": {
        "en": ["telegram", "whatsapp", "signal", "dm me", "call this number",
               "contact us at", "reach us on", "join our group", "click the link",
               "visit this website", "send money to"],
        "hi": ["व्हाट्सएप", "टेलीग्राम", "कॉल करें", "लिंक पर क्लिक करें"],
        "mr": ["व्हॉट्सअॅप", "टेलिग्राम", "कॉल करा", "लिंक वर क्लिक करा"],
        "te": ["వాట్సాప్", "టెలిగ్రామ్", "కాల్ చేయండి", "లింక్ క్లిక్ చేయండి"],
        "kn": ["ವಾಟ್ಸಾಪ್", "ಟೆಲಿಗ್ರಾಮ್", "ಕಾಲ್ ಮಾಡಿ", "ಲಿಂಕ್ ಕ್ಲಿಕ್ ಮಾಡಿ"],
    },
    "threat": {
        "en": ["will be suspended", "will be deleted", "blocked", "security alert",
               "unauthorized access", "unusual activity", "account will be closed",
               "legal action", "police complaint", "arrest warrant", "court order",
               "account frozen", "permanently disabled"],
        "hi": ["खाता बंद होगा", "संदिग्ध गतिविधि", "कानूनी कार्रवाई", "गिरफ्तारी"],
        "mr": ["खाते बंद होईल", "कायदेशीर कारवाई"],
        "te": ["ఖాతా నిలిపివేయబడుతుంది", "చట్టపరమైన చర్య"],
        "kn": ["ಖಾತೆ ಮುಚ್ಚಲಾಗುವುದು", "ಕಾನೂನು ಕ್ರಮ"],
    },
    "legitimacy": {
        "en": ["regards", "sincerely", "please find attached", "as discussed",
               "meeting", "schedule", "documentation", "invoice", "thank you for",
               "we appreciate", "your reference number", "transaction successful",
               "order confirmed", "delivery scheduled"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# URL CLASSIFICATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd",
    "cutt.ly", "ow.ly", "rb.gy", "short.io", "tiny.cc",
    "rebrand.ly", "bl.ink", "soo.gd",
}

RISKY_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".pw", ".xyz", ".top",
    ".click", ".download", ".work", ".loan", ".gq",
    ".win", ".date", ".racing", ".review", ".stream",
}

VERIFIED_DOMAINS = {
    "google.com", "apple.com", "amazon.com", "amazon.in",
    "microsoft.com", "github.com", "flipkart.com",
    "sbi.co.in", "hdfcbank.com", "icicibank.com",
    "paytm.com", "razorpay.com", "phonepe.com",
    "gpay.app", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com",
}

BRAND_KEYWORDS = [
    "paypal", "amazon", "google", "sbi", "hdfc", "icici",
    "axis", "kotak", "paytm", "phonepe", "netflix",
    "apple", "microsoft", "facebook", "instagram",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 1: Binary Keyword Features (f1–f6)
# ═══════════════════════════════════════════════════════════════════════════════

def _has_keyword(text_lower: str, group: str, lang: str) -> bool:
    """Check if text contains any keyword from a lexicon group."""
    lexicon = KEYWORD_LEXICONS.get(group, {})
    all_words = lexicon.get("en", []) + lexicon.get(lang, [])
    return any(w.lower() in text_lower for w in all_words)


def extract_binary_features(text: str, lang: str) -> dict:
    """Extract 6 binary keyword-presence features."""
    text_lower = text.lower()
    return {
        "f1_has_urgency":           int(_has_keyword(text_lower, "urgency", lang)),
        "f2_has_money":             int(_has_keyword(text_lower, "money", lang)),
        "f3_has_sensitive":         int(_has_keyword(text_lower, "sensitive", lang)),
        "f4_has_off_platform":      int(_has_keyword(text_lower, "off_platform", lang)),
        "f5_has_threat":            int(_has_keyword(text_lower, "threat", lang)),
        "f6_has_legitimacy_marker": int(_has_keyword(text_lower, "legitimacy", lang)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 2: Statistical Features (f7–f14)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_statistical_features(text: str) -> dict:
    """Extract 8 statistical text features."""
    words = text.split()
    chars = list(text)
    n = max(len(chars), 1)

    # Character entropy — scam messages often have unusual distributions
    freq = Counter(chars)
    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)

    return {
        "f7_text_length":          len(text),
        "f8_exclamation_count":    text.count("!"),
        "f9_question_count":       text.count("?"),
        "f10_uppercase_ratio":     sum(1 for c in text if c.isupper()) / n,
        "f11_digit_ratio":         sum(1 for c in text if c.isdigit()) / n,
        "f12_char_entropy":        round(entropy, 6),
        "f13_avg_word_length":     round(sum(len(w) for w in words) / max(len(words), 1), 4),
        "f14_punctuation_density": round(sum(1 for c in text if c in "!?.,;:") / n, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 3: Keyword Density Features (f15–f17)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_density_features(text: str, lang: str) -> dict:
    """Extract 3 keyword density features — ratio of scam keywords to total words."""
    words = text.lower().split()
    n = max(len(words), 1)

    urgency_words = set(
        KEYWORD_LEXICONS["urgency"].get("en", []) +
        KEYWORD_LEXICONS["urgency"].get(lang, [])
    )
    money_words = set(
        KEYWORD_LEXICONS["money"].get("en", []) +
        KEYWORD_LEXICONS["money"].get(lang, [])
    )
    sensitive_words = set(
        KEYWORD_LEXICONS["sensitive"].get("en", []) +
        KEYWORD_LEXICONS["sensitive"].get(lang, [])
    )

    return {
        "f15_urgency_density":   round(sum(1 for w in words if w in urgency_words) / n, 6),
        "f16_money_density":     round(sum(1 for w in words if w in money_words) / n, 6),
        "f17_sensitive_density": round(sum(1 for w in words if w in sensitive_words) / n, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 4: URL Features (f18–f24)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_url_features(text: str) -> dict:
    """Extract 7 URL-based features for phishing/scam detection."""
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", text)
    n_urls = len(urls)
    n_words = max(len(text.split()), 1)

    has_ip_url = False
    has_shortener = False
    has_risky_tld = False
    has_spoof = False
    has_verified = False

    for url in urls:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        domain = parsed.netloc.lower().replace("www.", "")

        # IP address as domain
        if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
            has_ip_url = True

        # URL shortener
        if domain in URL_SHORTENERS or "link.in" in domain:
            has_shortener = True

        # Risky TLD
        if any(url.lower().endswith(tld) or tld + "/" in url.lower() for tld in RISKY_TLDS):
            has_risky_tld = True

        # Domain spoofing (e.g., paypa1.com, sbi-verify.com)
        for brand in BRAND_KEYWORDS:
            if brand in domain and domain not in VERIFIED_DOMAINS:
                has_spoof = True
                break

        # Verified domain
        if domain in VERIFIED_DOMAINS:
            has_verified = True

    return {
        "f18_num_urls":         n_urls,
        "f19_url_density":      round(n_urls / n_words, 6),
        "f20_has_ip_url":       int(has_ip_url),
        "f21_has_shortener":    int(has_shortener),
        "f22_has_risky_tld":    int(has_risky_tld),
        "f23_has_domain_spoof": int(has_spoof),
        "f24_has_verified_dom": int(has_verified),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 5: Multilingual Features (f25–f32)
# ═══════════════════════════════════════════════════════════════════════════════

def _has_keyword_ml(group: str, lang: str, text: str) -> bool:
    """Multilingual keyword check — searches both native-language and English lexicons."""
    text_lower = text.lower()
    lexicon = KEYWORD_LEXICONS.get(group, {})
    keywords = lexicon.get(lang, []) + lexicon.get("en", [])
    return any(kw.lower() in text_lower for kw in keywords)


def extract_multilingual_features(text: str, lang: str, ngram_model=None) -> dict:
    """Extract 8 multilingual features including script mismatch and n-gram score."""
    # Script mismatch: Roman chars injected into native-script message
    native_count = len(re.findall(r"[\u0900-\u0CFF]", text))  # Devanagari + Telugu + Kannada
    latin_count = len(re.findall(r"[a-zA-Z]", text))

    # Subtract URL latin chars to avoid false positives
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+|[a-z0-9.-]+\.[a-z]{2,6}", text.lower())
    url_latin_count = sum(len(re.findall(r"[a-zA-Z]", u)) for u in urls)
    adjusted_latin = max(latin_count - url_latin_count, 0)

    script_mismatch = 0.0
    if native_count > 0 and adjusted_latin > 0:
        script_mismatch = adjusted_latin / (native_count + adjusted_latin)

    # f32: char n-gram model probability output
    ngram_score = 0.0
    if ngram_model is not None:
        try:
            ngram_score = float(ngram_model.predict_proba([text])[0][1])
        except Exception:
            ngram_score = 0.0

    lang_to_int = {"en": 0, "hi": 1, "mr": 2, "te": 3, "kn": 4, "other": 5}

    return {
        "f25_lang_int":         lang_to_int.get(lang, 5),
        "f26_has_urgency_ml":   int(_has_keyword_ml("urgency", lang, text)),
        "f27_has_money_ml":     int(_has_keyword_ml("money", lang, text)),
        "f28_has_sensitive_ml": int(_has_keyword_ml("sensitive", lang, text)),
        "f29_has_offplatf_ml":  int(_has_keyword_ml("off_platform", lang, text)),
        "f30_has_threat_ml":    int(_has_keyword_ml("threat", lang, text)),
        "f31_script_mismatch":  round(script_mismatch, 6),
        "f32_ngram_score":      round(ngram_score, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER EXTRACTION — All 32 features
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "f1_has_urgency", "f2_has_money", "f3_has_sensitive",
    "f4_has_off_platform", "f5_has_threat", "f6_has_legitimacy_marker",
    "f7_text_length", "f8_exclamation_count", "f9_question_count",
    "f10_uppercase_ratio", "f11_digit_ratio", "f12_char_entropy",
    "f13_avg_word_length", "f14_punctuation_density",
    "f15_urgency_density", "f16_money_density", "f17_sensitive_density",
    "f18_num_urls", "f19_url_density", "f20_has_ip_url",
    "f21_has_shortener", "f22_has_risky_tld", "f23_has_domain_spoof",
    "f24_has_verified_dom",
    "f25_lang_int", "f26_has_urgency_ml", "f27_has_money_ml",
    "f28_has_sensitive_ml", "f29_has_offplatf_ml", "f30_has_threat_ml",
    "f31_script_mismatch", "f32_ngram_score",
]


def extract_all_features(text: str, ngram_model=None) -> dict:
    """
    Extract all 32 features from a text message.

    Returns a dict with feature names as keys.
    """
    preprocessor = ScamPreprocessor()
    cleaned = preprocessor.clean(text)
    lang = preprocessor.detect_language(cleaned)

    f1_6 = extract_binary_features(cleaned, lang)
    f7_14 = extract_statistical_features(cleaned)
    f15_17 = extract_density_features(cleaned, lang)
    f18_24 = extract_url_features(cleaned)
    f25_32 = extract_multilingual_features(cleaned, lang, ngram_model)

    return {**f1_6, **f7_14, **f15_17, **f18_24, **f25_32}


def extract_all_32_features(text: str, ngram_model=None) -> list:
    """
    Extract all 32 features as a flat list (for model input).
    """
    features = extract_all_features(text, ngram_model)
    return [features[name] for name in FEATURE_NAMES]
