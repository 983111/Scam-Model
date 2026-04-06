"""
Tests for ScamShield Feature Extraction — verifies all 5 feature groups.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import ScamPreprocessor
from src.feature_extractor import (
    extract_binary_features,
    extract_statistical_features,
    extract_density_features,
    extract_url_features,
    extract_multilingual_features,
    extract_all_features,
    extract_all_32_features,
    FEATURE_NAMES,
)


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.pp = ScamPreprocessor()

    def test_clean_removes_zero_width(self):
        text = "Hello\u200bWorld"
        self.assertEqual(self.pp.clean(text), "HelloWorld")

    def test_clean_normalizes_whitespace(self):
        text = "Hello   World\n\nFoo"
        self.assertEqual(self.pp.clean(text), "Hello World Foo")

    def test_detect_language_english(self):
        self.assertEqual(self.pp.detect_language("Hello this is a test"), "en")

    def test_detect_language_hindi(self):
        self.assertEqual(self.pp.detect_language("यह एक परीक्षा है"), "hi")

    def test_detect_language_telugu(self):
        self.assertEqual(self.pp.detect_language("ఇది ఒక పరీక్ష"), "te")

    def test_detect_language_kannada(self):
        self.assertEqual(self.pp.detect_language("ಇದು ಒಂದು ಪರೀಕ್ಷೆ"), "kn")

    def test_detect_language_marathi(self):
        # Contains Marathi marker "आहे"
        self.assertEqual(self.pp.detect_language("हे एक चाचणी आहे"), "mr")

    def test_extract_urls(self):
        text = "Click http://evil.com/phish and www.scam.xyz/claim"
        urls = self.pp.extract_urls(text)
        self.assertGreaterEqual(len(urls), 2)

    def test_has_phone(self):
        self.assertTrue(self.pp.has_phone("Call +919876543210 now"))
        self.assertFalse(self.pp.has_phone("Hello world"))


class TestBinaryFeatures(unittest.TestCase):
    def test_urgency_detected(self):
        features = extract_binary_features("URGENT: Act now before it expires!", "en")
        self.assertEqual(features["f1_has_urgency"], 1)

    def test_money_detected(self):
        features = extract_binary_features("You've won a lottery prize!", "en")
        self.assertEqual(features["f2_has_money"], 1)

    def test_sensitive_detected(self):
        features = extract_binary_features("Share your OTP and password", "en")
        self.assertEqual(features["f3_has_sensitive"], 1)

    def test_off_platform_detected(self):
        features = extract_binary_features("Contact us on WhatsApp", "en")
        self.assertEqual(features["f4_has_off_platform"], 1)

    def test_threat_detected(self):
        features = extract_binary_features("Your account will be suspended", "en")
        self.assertEqual(features["f5_has_threat"], 1)

    def test_legitimacy_detected(self):
        features = extract_binary_features("Best regards, John", "en")
        self.assertEqual(features["f6_has_legitimacy_marker"], 1)

    def test_safe_message(self):
        features = extract_binary_features("Hey, are you free for lunch?", "en")
        self.assertEqual(features["f1_has_urgency"], 0)
        self.assertEqual(features["f2_has_money"], 0)
        self.assertEqual(features["f3_has_sensitive"], 0)

    def test_hindi_urgency(self):
        features = extract_binary_features("तुरंत OTP शेयर करें", "hi")
        self.assertEqual(features["f1_has_urgency"], 1)
        self.assertEqual(features["f3_has_sensitive"], 1)


class TestStatisticalFeatures(unittest.TestCase):
    def test_exclamation_count(self):
        features = extract_statistical_features("WOW!!! Amazing!!")
        self.assertEqual(features["f8_exclamation_count"], 5)

    def test_uppercase_ratio(self):
        features = extract_statistical_features("HELLO world")
        self.assertGreater(features["f10_uppercase_ratio"], 0.3)

    def test_text_length(self):
        features = extract_statistical_features("Hello")
        self.assertEqual(features["f7_text_length"], 5)

    def test_entropy(self):
        features = extract_statistical_features("aaaa")
        self.assertEqual(features["f12_char_entropy"], 0.0)

        features2 = extract_statistical_features("abcd")
        self.assertGreater(features2["f12_char_entropy"], 0)


class TestURLFeatures(unittest.TestCase):
    def test_url_count(self):
        features = extract_url_features("Visit http://evil.com and http://bad.org")
        self.assertEqual(features["f18_num_urls"], 2)

    def test_ip_url(self):
        features = extract_url_features("Click http://192.168.1.1/phish")
        self.assertEqual(features["f20_has_ip_url"], 1)

    def test_shortener(self):
        features = extract_url_features("Click http://bit.ly/scam123")
        self.assertEqual(features["f21_has_shortener"], 1)

    def test_risky_tld(self):
        features = extract_url_features("Visit http://bank-verify.xyz/login")
        self.assertEqual(features["f22_has_risky_tld"], 1)

    def test_domain_spoof(self):
        features = extract_url_features("Login at http://paypal-verify.com/auth")
        self.assertEqual(features["f23_has_domain_spoof"], 1)

    def test_verified_domain(self):
        features = extract_url_features("Visit https://google.com/search")
        self.assertEqual(features["f24_has_verified_dom"], 1)

    def test_no_urls(self):
        features = extract_url_features("No links here, just text.")
        self.assertEqual(features["f18_num_urls"], 0)


class TestFullExtraction(unittest.TestCase):
    def test_returns_32_features(self):
        features = extract_all_32_features("Test message with no scam signals")
        self.assertEqual(len(features), 32)

    def test_feature_names_match(self):
        self.assertEqual(len(FEATURE_NAMES), 32)

    def test_scam_message_has_signals(self):
        scam = "URGENT: Your SBI account will be suspended! Share OTP 123456 immediately at http://sbi-verify.xyz/login"
        features = extract_all_features(scam)
        self.assertEqual(features["f1_has_urgency"], 1)
        self.assertEqual(features["f3_has_sensitive"], 1)
        self.assertEqual(features["f5_has_threat"], 1)
        self.assertEqual(features["f22_has_risky_tld"], 1)

    def test_safe_message_clean(self):
        safe = "Hey, are you coming to the party tonight?"
        features = extract_all_features(safe)
        self.assertEqual(features["f1_has_urgency"], 0)
        self.assertEqual(features["f2_has_money"], 0)
        self.assertEqual(features["f3_has_sensitive"], 0)


if __name__ == "__main__":
    unittest.main()
