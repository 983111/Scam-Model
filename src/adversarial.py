"""
ScamShield Adversarial Robustness Testing — 4 attack types.

Attack Types:
  1. Synonym Substitution — Replace scam keywords with synonyms not in lexicon
  2. Homoglyph Attack — Replace chars with visually similar Unicode chars
  3. URL Obfuscation — Wrap malicious URLs in redirect chains
  4. Script Swap — Romanize native-script scam keywords
"""

import re
import random
import numpy as np
from sklearn.metrics import recall_score

from src.feature_extractor import extract_all_32_features


class AdversarialAttacks:
    """Collection of adversarial attack methods for scam message evasion."""

    @staticmethod
    def synonym_substitution(text: str) -> str:
        """Replace scam keywords with synonyms not in the detection lexicon."""
        replacements = {
            "urgent": "important",
            "immediately": "soon",
            "verify": "confirm",
            "suspended": "paused",
            "lottery": "competition",
            "prize": "gift",
            "otp": "verification code",
            "blocked": "restricted",
            "password": "passcode",
            "winner": "selected participant",
            "free": "complimentary",
            "earn": "make",
            "guaranteed": "assured",
            "expires": "ends",
            "click": "tap",
            "account": "profile",
            "security alert": "notice",
            "unauthorized": "unrecognized",
        }
        result = text
        for orig, sub in replacements.items():
            result = result.replace(orig, sub)
            result = result.replace(orig.title(), sub.title())
            result = result.replace(orig.upper(), sub.upper())
        return result

    @staticmethod
    def homoglyph_attack(text: str, rate: float = 0.3) -> str:
        """Replace characters with visually similar Unicode characters."""
        homoglyphs = {
            "a": "\u0430",  # Cyrillic а
            "e": "\u0435",  # Cyrillic е
            "o": "\u043e",  # Cyrillic о
            "p": "\u0440",  # Cyrillic р
            "c": "\u0441",  # Cyrillic с
            "i": "\u0456",  # Ukrainian і
            "s": "\u0455",  # Cyrillic ѕ
            "x": "\u0445",  # Cyrillic х
            "y": "\u0443",  # Cyrillic у
        }
        random.seed(42)
        result = []
        for char in text:
            lower = char.lower()
            if lower in homoglyphs and random.random() < rate:
                replacement = homoglyphs[lower]
                result.append(replacement.upper() if char.isupper() else replacement)
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def url_obfuscation(text: str) -> str:
        """Wrap malicious URLs in legitimate-looking redirect chains."""
        url_pattern = r"https?://[^\s]+"
        redirectors = [
            "https://google.com/url?q=",
            "https://www.google.com/amp/s/",
            "https://l.facebook.com/l.php?u=",
            "https://t.co/redirect?url=",
        ]

        def obfuscate(m):
            original = m.group(0)
            redirector = random.choice(redirectors)
            return f"{redirector}{original}"

        random.seed(42)
        return re.sub(url_pattern, obfuscate, text)

    @staticmethod
    def script_swap(text: str) -> str:
        """Romanize key scam words in native-script messages."""
        native_to_roman = {
            # Hindi
            "तुरंत": "turant",
            "अभी": "abhi",
            "ओटीपी": "otp",
            "पासवर्ड": "password",
            "लॉटरी": "lottery",
            "खाता बंद": "account band",
            "जल्दी": "jaldi",
            "कमाएं": "kamayen",
            "सत्यापित": "verify",
            "बधाई हो": "congratulations",
            # Telugu
            "వెంటనే": "ventane",
            "ఓటీపీ": "otp",
            "లాటరీ": "lottery",
            "గెలిచారు": "won",
            "సంపాదించండి": "earn",
            # Kannada
            "ತಕ್ಷಣ": "takshana",
            "ಒಟಿಪಿ": "otp",
            "ಲಾಟರಿ": "lottery",
            "ಗೆದ್ದಿದ್ದೀರಿ": "won",
            "ಗಳಿಸಿ": "earn",
            # Marathi
            "ताबडतोब": "tabadtob",
            "आत्ताच": "attacha",
            "लगेच": "lagech",
        }
        result = text
        for native, roman in native_to_roman.items():
            result = result.replace(native, roman)
        return result

    @staticmethod
    def char_insertion(text: str, rate: float = 0.1) -> str:
        """Insert invisible/decorative characters to break keyword matching."""
        insertions = ["\u200b", "\u200c", "\u200d", "\ufeff"]  # Zero-width chars
        random.seed(42)
        result = []
        for char in text:
            result.append(char)
            if char.isalpha() and random.random() < rate:
                result.append(random.choice(insertions))
        return "".join(result)


def run_adversarial_evaluation(model, ngram_model, test_texts, test_labels):
    """
    Run all adversarial attacks and measure recall degradation.

    Only tests on scam samples (label=1) since adversarial attacks
    simulate a scammer trying to evade detection.
    """
    attacks = AdversarialAttacks()
    attack_fns = {
        "Synonym Substitution": attacks.synonym_substitution,
        "Homoglyph Attack":     attacks.homoglyph_attack,
        "URL Obfuscation":      attacks.url_obfuscation,
        "Script Swap":          attacks.script_swap,
        "Char Insertion":       attacks.char_insertion,
    }

    # Filter to scam-only samples
    scam_texts = [t for t, l in zip(test_texts, test_labels) if l == 1]
    scam_labels = [1] * len(scam_texts)

    if not scam_texts:
        print("No scam samples in test set for adversarial evaluation.")
        return {}

    print(f"\n{'=' * 55}")
    print(f"  Adversarial Robustness Evaluation")
    print(f"  Testing on {len(scam_texts)} scam samples")
    print(f"{'=' * 55}")
    print(f"  {'Attack':<25} {'Recall':>8} {'Δ Recall':>10}")
    print(f"  {'-' * 45}")

    # Baseline (clean data)
    baseline_feats = np.array([extract_all_32_features(t, ngram_model) for t in scam_texts])
    baseline_preds = model.predict(baseline_feats)
    baseline_recall = recall_score(scam_labels, baseline_preds)
    print(f"  {'Clean (no attack)':<25} {baseline_recall:>8.4f} {'—':>10}")

    results = {"clean": baseline_recall}

    for attack_name, attack_fn in attack_fns.items():
        attacked = [attack_fn(t) for t in scam_texts]
        feats = np.array([extract_all_32_features(t, ngram_model) for t in attacked])
        preds = model.predict(feats)
        recall = recall_score(scam_labels, preds)
        delta = recall - baseline_recall
        results[attack_name] = recall
        print(f"  {attack_name:<25} {recall:>8.4f} {delta:>+10.4f}")

    # Summary
    avg_adversarial = np.mean([v for k, v in results.items() if k != "clean"])
    print(f"\n  Average adversarial recall: {avg_adversarial:.4f}")
    print(f"  Average recall drop: {avg_adversarial - baseline_recall:+.4f}")

    return results
