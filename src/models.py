"""
ScamShield Model Definitions — 3-layer ensemble architecture.

Layer 1: Char N-gram Model (TF-IDF + Logistic Regression) — generates f32 score
Layer 2: Sentence Embeddings (optional, paraphrase-multilingual-MiniLM-L12-v2)
Layer 3: Calibrated Gradient Boosting Machine (primary classifier)
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: Char N-gram Model (f32 generator)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ngram_model():
    """
    Build a character n-gram model that captures subword scam patterns.

    Architecture:
      TF-IDF (char 3–5grams, 8K features) → StandardScaler → Logistic Regression

    This model serves dual purpose:
      1. Standalone baseline classifier
      2. Its predict_proba output becomes feature f32 in the ensemble
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",       # Word-boundary character n-grams
            ngram_range=(3, 5),       # 3-to-5 character grams
            max_features=8000,        # Keeps model under 600 KB
            sublinear_tf=True,        # log(tf) scaling — reduces impact of very common chars
            strip_accents=None,       # Preserve Indic script accents
            decode_error="replace",   # Handle encoding errors gracefully
        )),
        ("scaler", StandardScaler(with_mean=False)),  # Sparse-compatible scaling
        ("clf", LogisticRegression(
            C=1.0,
            solver="liblinear",       # Best for small-medium datasets
            class_weight="balanced",  # Handle class imbalance
            max_iter=500,
        )),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: Sentence Embedding (optional — for server-side deployment)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingFeatureExtractor:
    """
    Sentence embedding feature extractor using paraphrase-multilingual-MiniLM-L12-v2.

    This is a 118MB model that produces 384-dim embeddings.
    Supports Hindi, Telugu, Kannada out of the box.

    Usage:
        extractor = EmbeddingFeatureExtractor()
        embeddings = extractor.transform(["Hello world", "This is a test"])
        # Shape: (2, 384)

    Note: Requires sentence-transformers package (pip install sentence-transformers).
          This adds ~500MB of dependencies including PyTorch.
          Only use for server-side deployment, not on-device/mobile.
    """

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            print("[WARNING] sentence-transformers not installed. "
                  "Embeddings will not be used. Install with: "
                  "pip install sentence-transformers")
            self.model = None
            self.available = False

    def transform(self, texts: list) -> np.ndarray:
        """Encode texts into 384-dim embeddings."""
        if not self.available:
            # Return zeros if library not available
            return np.zeros((len(texts), 384))
        return self.model.encode(texts, batch_size=64, show_progress_bar=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: Calibrated GBM (primary classifier)
# ═══════════════════════════════════════════════════════════════════════════════

def build_gbm_model():
    """
    Build a Calibrated Gradient Boosting Classifier.

    Architecture:
      GBM (200 trees, depth 5) → Isotonic Calibration (5-fold)

    Calibration ensures predict_proba outputs true probabilities,
    not just relative confidence scores. This is critical for:
      - Setting per-language detection thresholds
      - Reporting Brier score
      - User-facing confidence displays
    """
    base = GradientBoostingClassifier(
        n_estimators=200,       # More trees = better generalization
        max_depth=5,            # Deeper for real-world complexity
        learning_rate=0.04,     # Lower LR + more trees = better performance
        min_samples_leaf=5,
        subsample=0.8,          # Stochastic gradient boosting
        max_features="sqrt",    # Random feature subset per tree
        random_state=42,
    )
    # Isotonic calibration is better than sigmoid for tree ensembles
    return CalibratedClassifierCV(base, method="isotonic", cv=5)


def build_baseline_lr():
    """Build a simple Logistic Regression baseline for comparison."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
        )),
    ])


def build_baseline_rf():
    """Build a Random Forest baseline for comparison."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
