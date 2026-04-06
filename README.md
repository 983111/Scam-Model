# ScamShield — Multilingual Scam Detection ML Pipeline

> A production-quality, multilingual scam detection system with 32 handcrafted features, a 3-layer ensemble model, adversarial robustness testing, and a Flask prediction API.

**Supports:** English, Hindi, Marathi, Telugu, Kannada  
**Categories:** OTP Fraud, Lottery, KYC Scam, Investment, Phishing, Impersonation, Job Scam, Romance, Tech Support, Customs/Package, Charity

---

## Table of Contents

- [Quick Start (Full Pipeline)](#quick-start-full-pipeline)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Setup & Install](#1-setup--install)
  - [2. Build Dataset](#2-build-dataset)
  - [3. Train Model](#3-train-model)
  - [4. Evaluate Model](#4-evaluate-model)
  - [5. Run Ablation Study](#5-run-ablation-study)
  - [6. Interactive Prediction](#6-interactive-prediction)
  - [7. Start API Server](#7-start-api-server)
  - [8. Run Tests](#8-run-tests)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Feature Engineering (32 Features)](#feature-engineering-32-features)
- [Adding Real Data](#adding-real-data)
- [Integration with Your App](#integration-with-your-app)
- [Performance Targets](#performance-targets)

---

## Quick Start (Full Pipeline)

Run the entire pipeline (build dataset → train model → evaluate) with a single command:

```bash
# 1. Clone/navigate to the project
cd scam-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything
python run.py all
```

That's it! This will:
- Download the UCI SMS dataset (5,574 real messages)
- Generate Hindi, Marathi, Telugu, Kannada samples (~13,000 messages)
- Train the char n-gram model + Calibrated GBM ensemble
- Run full evaluation with all metrics + adversarial testing

---

## Step-by-Step Guide

### 1. Setup & Install

```bash
# Make sure you have Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

**Dependencies installed:**
- `scikit-learn` — ML models, evaluation, cross-validation
- `numpy`, `pandas` — Data processing
- `scipy` — Statistical distributions for hyperparameter tuning
- `imbalanced-learn` — SMOTE for class imbalance (optional)
- `statsmodels` — McNemar's test for statistical significance
- `flask` — Prediction API server
- `requests` — Dataset downloading
- `joblib` — Model serialization

---

### 2. Build Dataset

```bash
python run.py build
```

**What this does:**
1. **Downloads the UCI SMS Spam Collection** (5,574 real English SMS messages) from UCI ML Repository
   - If the download fails, it auto-generates realistic English templates as fallback
2. **Generates Indian language messages** using realistic scam/safe templates:
   - Hindi: 4,000 messages (2,000 scam + 2,000 safe)
   - Marathi: 3,000 messages (1,500 + 1,500)
   - Telugu: 3,000 messages (1,500 + 1,500)
   - Kannada: 3,000 messages (1,500 + 1,500)
3. **Merges, validates, and cleans** all data
4. **Creates stratified train/test split** (80/20, stratified on label + language)

**Output files:**
```
data/
├── raw/
│   ├── uci_sms.csv           # English SMS data
│   └── indic_messages.csv    # Hindi/Marathi/Telugu/Kannada
├── processed/
│   ├── full_dataset.csv      # Complete merged dataset
│   ├── train.csv             # 80% training split
│   └── test.csv              # 20% test split
```

**Expected output:**
```
Total dataset: ~18,000 samples
  Scam: ~9,000 (50%)
  Safe: ~9,000 (50%)
  en: ~5,500  |  hi: ~4,000  |  mr: ~3,000  |  te: ~3,000  |  kn: ~3,000
```

---

### 3. Train Model

```bash
python run.py train
```

**What this does (6 steps):**

1. **Trains char n-gram model** (TF-IDF char 3-5grams → LogisticRegression)
   - Captures subword scam patterns across all 5 languages
   - Outputs: `models/ngram_model.pkl` (~500 KB)

2. **Extracts 32 handcrafted features** for every message:
   - Binary keyword features (urgency, money, sensitive, off-platform, threat, legitimacy)
   - Statistical features (length, entropy, uppercase ratio, digit ratio)
   - Keyword density features
   - URL features (shorteners, risky TLDs, IP URLs, domain spoofing)
   - Multilingual features (script mismatch, n-gram score)

3. **Skips sentence embeddings** by default (add `--use-embeddings` flag if needed)

4. **5-fold stratified cross-validation** of Calibrated GBM

5. **Trains final model** on full dataset with balanced sample weights
   - Outputs: `models/scam_detector_final.pkl` (~1-2 MB)

6. **Computes per-language detection thresholds** (optimized for 92% recall)
   - Outputs: `models/thresholds.json`

**Advanced training options:**
```bash
# Train with sentence embeddings (requires: pip install sentence-transformers)
python run.py train --use-embeddings

# Train on a custom dataset
python run.py train --data-path path/to/your/data.csv
```

**Expected output:**
```
CV Results (5-fold):
  F1:        0.95+ (depends on data quality)
  ROC-AUC:   0.98+
  Precision:  0.94+
  Recall:     0.95+

Models saved to models/
  ngram_model.pkl          (~500 KB)
  scam_detector_final.pkl  (~1-2 MB)
  thresholds.json
```

---

### 4. Evaluate Model

```bash
python run.py evaluate
```

**What this does:**
1. Loads trained model and test set
2. Computes **7 metrics**: F1 (macro + binary), ROC-AUC, MCC, Brier Score, Recall, Precision
3. Prints confusion matrix and classification report
4. Runs **per-language evaluation** (F1 for each language)
5. Runs **adversarial robustness testing** (5 attack types):
   - Synonym Substitution
   - Homoglyph Attack (Cyrillic lookalike chars)
   - URL Obfuscation (redirect wrapping)
   - Script Swap (Romanization of native keywords)
   - Char Insertion (zero-width characters)
6. Runs **error analysis** (false negatives + false positives)

**Custom test set:**
```bash
python run.py evaluate --test-path data/processed/test.csv
```

---

### 5. Run Ablation Study

```bash
python run.py ablation
```

Drops each of the 5 feature groups and measures F1 impact. Shows which feature groups matter most for your model's accuracy.

---

### 6. Interactive Prediction

```bash
python run.py predict
```

Type any message and get instant scam/safe prediction with probability, category, language detection, and explainability signals:

```
> You've won 50,000 in our lottery! Call +919876543210 NOW!

  Verdict:     SCAM
  Probability: 0.9734
  Category:    lottery
  Language:    en
  Threshold:   0.4200
  Signals:     urgency_detected, money_language, excessive_exclamation

> Hey, are you free for lunch today?

  Verdict:     SAFE
  Probability: 0.0312
  Category:    safe
  Language:    en
```

---

### 7. Start API Server

```bash
python run.py serve
# Or with custom port:
python run.py serve --port 8080
```

Then test with curl:

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: Your account will be suspended. Share OTP now!"}'

# Batch prediction
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["You won a lottery!", "Hey how are you?"]}'

# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/info
```

---

### 8. Run Tests

```bash
python run.py test
```

Runs unit tests for:
- Preprocessor (language detection, URL extraction, phone detection)
- Binary features (urgency, money, sensitive keywords in 5 languages)
- Statistical features (entropy, ratios, counts)
- URL features (shorteners, risky TLDs, IP URLs, domain spoofing)
- Full 32-feature extraction
- End-to-end predictions (requires trained model)

---

## API Reference

### `POST /predict`

Predict a single message.

**Request:**
```json
{
  "text": "Your SBI account will be blocked. Share OTP 847291 now!"
}
```

**Response:**
```json
{
  "verdict": "scam",
  "probability": 0.9562,
  "category": "otp_fraud",
  "language": "en",
  "threshold": 0.42,
  "signals": ["urgency_detected", "credential_request", "threat_language"],
  "has_url": false,
  "has_phone": false,
  "inference_ms": 4.7
}
```

### `POST /batch`

Predict up to 100 messages at once.

**Request:**
```json
{
  "texts": [
    "You've won a lottery!",
    "Meeting at 3pm tomorrow",
    "Share your OTP now!"
  ]
}
```

### `GET /health`

Returns `{"status": "healthy", "model_loaded": true}`

### `GET /info`

Returns model metadata, supported languages, thresholds, and categories.

---

## Project Structure

```
scam-detection/
├── run.py                          # Master runner (use this!)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── raw/                        # Downloaded/generated raw data
│   └── processed/                  # Train/test splits
│
├── models/                         # Trained model files
│   ├── ngram_model.pkl             # Char n-gram (~500 KB)
│   ├── scam_detector_final.pkl     # Calibrated GBM (~1-2 MB)
│   └── thresholds.json             # Per-language thresholds
│
├── src/
│   ├── preprocessor.py             # Text cleaning, language detection
│   ├── feature_extractor.py        # 32 features, keyword lexicons
│   ├── models.py                   # Model architectures
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Evaluation metrics
│   ├── adversarial.py              # 5 adversarial attack types
│   ├── predict.py                  # ScamDetector class
│   ├── labeler.py                  # Auto-labeling heuristics
│   ├── tune.py                     # Hyperparameter tuning
│   └── collectors/
│       ├── uci_collector.py        # UCI SMS dataset downloader
│       ├── indic_generator.py      # Indian language generator
│       ├── reddit_collector.py     # Reddit collector (needs API key)
│       └── phishtank_collector.py  # PhishTank collector
│
├── api/
│   └── app.py                      # Flask API server
│
├── scripts/
│   ├── build_dataset.py            # Dataset builder
│   ├── run_evaluation.py           # Full evaluation runner
│   ├── ablation_study.py           # Feature group ablation
│   └── release_dataset.py          # HuggingFace/Zenodo release
│
└── tests/
    ├── test_features.py            # Feature extraction tests
    └── test_predictions.py         # Prediction integration tests
```

---

## Architecture

```
Input Text
    │
    ├─── 32 Handcrafted Features ─────────┐
    │    (urgency, money, URLs, entropy)   │
    │                                      │
    ├─── Char 3-5gram TF-IDF ─────────────┤
    │    (outputs f32 n-gram score)        │
    │                                      ▼
    └─── [Optional] Sentence Embedding    Feature Fusion
         (paraphrase-MiniLM-L12)           │
                                           ▼
                                Calibrated GBM (200 trees)
                                  + Isotonic Calibration
                                           │
                                           ▼
                                Verdict + Probability + Signals
                                  + Per-Language Threshold
```

---

## Feature Engineering (32 Features)

| Group | Features | What They Capture |
|-------|----------|-------------------|
| Binary Keywords (f1-f6) | urgency, money, sensitive, off-platform, threat, legitimacy | Scam keyword presence in 5 languages |
| Statistical (f7-f14) | length, exclamation/question count, uppercase ratio, digit ratio, entropy, avg word length, punctuation density | Text style patterns |
| Keyword Density (f15-f17) | urgency/money/sensitive density | Keyword concentration |
| URL Features (f18-f24) | URL count, density, IP URL, shortener, risky TLD, domain spoof, verified domain | Phishing URL analysis |
| Multilingual (f25-f32) | language encoding, per-language keywords, script mismatch, n-gram score | Cross-language scam patterns |

---

## Adding Real Data

### Replace generated data with real collected data:

1. **UCI SMS (already included)** — downloads automatically during build

2. **Reddit (requires API key):**
   ```bash
   python -m src.collectors.reddit_collector \
       --client-id YOUR_REDDIT_ID \
       --client-secret YOUR_REDDIT_SECRET
   ```

3. **PhishTank (free, no auth for basic list):**
   ```bash
   python -m src.collectors.phishtank_collector
   ```

4. **Your own data:** Add CSV files to `data/raw/` with these columns:
   ```
   text, label (0/1), language (en/hi/mr/te/kn), category, source
   ```
   Then rebuild: `python run.py build && python run.py train`

---

## Integration with Your App

### Python Integration

```python
from src.predict import ScamDetector

# Load once at app startup
detector = ScamDetector("models/")

# Predict
result = detector.predict("Share your OTP to unlock account")
print(result["verdict"])      # "scam"
print(result["probability"])  # 0.97
print(result["signals"])      # ["credential_request", "urgency_detected"]
print(result["category"])     # "otp_fraud"
print(result["language"])     # "en"
```

### REST API Integration

```python
import requests

response = requests.post("http://localhost:5000/predict", json={
    "text": "You've won a lottery! Call now!"
})
data = response.json()
# data = {"verdict": "scam", "probability": 0.95, "category": "lottery", ...}
```

### Android/Mobile Integration

The model files are small enough for on-device ML:
- `ngram_model.pkl` — ~500 KB
- `scam_detector_final.pkl` — ~1-2 MB
- `thresholds.json` — ~1 KB

Use the feature extraction logic from `src/feature_extractor.py` (pure Python, no heavy dependencies) and port to your mobile platform. Or use the Flask API for server-side inference.

---

## Performance Targets

| Metric | Expected | Exceptional |
|--------|----------|-------------|
| F1 (your data) | 0.89–0.95 | > 0.95 |
| ROC-AUC | 0.96–0.99 | > 0.995 |
| MCC | 0.82–0.91 | > 0.93 |
| Brier Score | 0.04–0.09 | < 0.04 |
| Adversarial recall | 0.60–0.75 | > 0.80 |
| Model size | < 2 MB | < 500 KB |
| Inference | < 10 ms | < 3 ms |

---

## Hyperparameter Tuning

For better results, run a proper hyperparameter search:

```bash
python -m src.tune --data-path data/processed/full_dataset.csv --n-iter 50
```

This runs RandomizedSearchCV over GBM parameters (n_estimators, max_depth, learning_rate, etc.) with 5-fold CV.

---

## Dataset Release

To release your dataset on HuggingFace:

```bash
# Prepare release files (CSV, JSON, dataset card)
python scripts/release_dataset.py

# Push to HuggingFace (requires token)
python scripts/release_dataset.py --push --hf-token YOUR_TOKEN --repo-name your-username/scam-detection
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `python run.py all` | Full pipeline: build → train → evaluate |
| `python run.py build` | Build dataset from sources |
| `python run.py train` | Train the ensemble model |
| `python run.py evaluate` | Run full evaluation + adversarial testing |
| `python run.py ablation` | Feature group ablation study |
| `python run.py predict` | Interactive prediction mode |
| `python run.py serve` | Start Flask API (default port 5000) |
| `python run.py test` | Run unit tests |

---

*Built on ScamShield (DOI: 10.5281/zenodo.18988170)*
