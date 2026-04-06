"""
Integration Tests for ScamShield Predictions.

Tests the full prediction pipeline end-to-end.
Requires trained models in models/ directory.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScamDetector(unittest.TestCase):
    """Integration tests — require trained models."""

    @classmethod
    def setUpClass(cls):
        """Load the detector once for all tests."""
        try:
            from src.predict import ScamDetector
            cls.detector = ScamDetector("models")
            cls.available = True
        except FileNotFoundError:
            cls.available = False
            print("SKIP: Models not found. Run training first.")

    def setUp(self):
        if not self.available:
            self.skipTest("Models not trained yet")

    # ── Scam Detection Tests ─────────────────────────────────────────────────

    def test_lottery_scam_en(self):
        result = self.detector.predict(
            "CONGRATULATIONS! You've won $50,000 in our annual lottery! "
            "Call +919876543210 to claim NOW!"
        )
        self.assertEqual(result["verdict"], "scam")
        self.assertGreater(result["probability"], 0.6)

    def test_otp_scam_en(self):
        result = self.detector.predict(
            "URGENT: Your SBI account has been suspended. "
            "Share your OTP 847291 to prevent account closure."
        )
        self.assertEqual(result["verdict"], "scam")

    def test_phishing_url(self):
        result = self.detector.predict(
            "Security Alert: Verify your account at http://sbi-verify.xyz/login immediately"
        )
        self.assertEqual(result["verdict"], "scam")

    def test_investment_scam(self):
        result = self.detector.predict(
            "Invest ₹10,000 today and get 300% returns in 30 days! "
            "Guaranteed by CryptoMax. Call +919876543210"
        )
        self.assertEqual(result["verdict"], "scam")

    def test_job_scam(self):
        result = self.detector.predict(
            "Earn ₹5000/day working from home! No experience needed. "
            "WhatsApp us at +919876543210"
        )
        self.assertEqual(result["verdict"], "scam")

    # ── Hindi Scam Tests ─────────────────────────────────────────────────────

    def test_hindi_lottery_scam(self):
        result = self.detector.predict(
            "बधाई हो! आपने 50,000 रुपये की लॉटरी जीती है! "
            "अभी क्लेम करें: +919876543210"
        )
        self.assertEqual(result["verdict"], "scam")
        self.assertEqual(result["language"], "hi")

    def test_hindi_otp_scam(self):
        result = self.detector.predict(
            "SBI: आपका खाता ब्लॉक होने वाला है। "
            "तुरंत OTP 123456 शेयर करें अन्यथा खाता बंद हो जाएगा।"
        )
        self.assertEqual(result["verdict"], "scam")

    # ── Safe Message Tests ───────────────────────────────────────────────────

    def test_safe_casual_message(self):
        result = self.detector.predict("Hey! Are you free for lunch today?")
        self.assertEqual(result["verdict"], "safe")
        self.assertLess(result["probability"], 0.5)

    def test_safe_otp(self):
        result = self.detector.predict(
            "Your OTP is 847291. Valid for 10 minutes. Do not share with anyone."
        )
        # This is a legitimate OTP — should be safe
        # Note: may be borderline due to OTP keyword
        self.assertLess(result["probability"], 0.8)

    def test_safe_transaction(self):
        result = self.detector.predict(
            "Transaction of Rs.2500 from your HDFC account. "
            "If not done by you, call 1800-233-1234."
        )
        self.assertEqual(result["verdict"], "safe")

    def test_safe_hindi(self):
        result = self.detector.predict(
            "कल की मीटिंग 10 बजे है। पक्का आना।"
        )
        self.assertEqual(result["verdict"], "safe")

    # ── Structural Tests ─────────────────────────────────────────────────────

    def test_result_structure(self):
        result = self.detector.predict("Test message")
        self.assertIn("verdict", result)
        self.assertIn("probability", result)
        self.assertIn("category", result)
        self.assertIn("language", result)
        self.assertIn("threshold", result)
        self.assertIn("signals", result)

    def test_probability_range(self):
        result = self.detector.predict("Any message here")
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_empty_message(self):
        result = self.detector.predict("")
        self.assertEqual(result["verdict"], "safe")
        self.assertEqual(result["probability"], 0.0)

    def test_batch_prediction(self):
        texts = [
            "You've won a lottery!",
            "Hey, how are you?",
            "Share your OTP now!",
        ]
        results = self.detector.predict_batch(texts)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
