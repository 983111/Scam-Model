"""
ScamShield Training Pipeline — Full ensemble training with cross-validation.

Usage:
    python -m src.train [--use-embeddings] [--data-path data/processed/full_dataset.csv]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    precision_recall_curve, classification_report
)
from sklearn.utils import class_weight

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_ngram_model, build_gbm_model, EmbeddingFeatureExtractor
from src.feature_extractor import extract_all_32_features


def get_sample_weights(y):
    """Compute balanced sample weights for class imbalance handling."""
    classes = np.unique(y)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
    weight_map = dict(zip(classes, weights))
    return np.array([weight_map[int(label)] for label in y])


def find_optimal_threshold(model, X_val, y_val, target_recall=0.92):
    """Find threshold that achieves target recall with maximum precision."""
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    valid_idx = [i for i, r in enumerate(recalls) if r >= target_recall]
    if not valid_idx:
        return 0.5

    best_idx = max(valid_idx, key=lambda i: precisions[i])
    if best_idx >= len(thresholds):
        return 0.5
    return float(thresholds[best_idx])


def train_full_pipeline(data_path: str, use_embeddings: bool = False,
                        models_dir: str = "models"):
    """
    Full training pipeline:
      1. Train char n-gram model → save
      2. Extract 32 features using trained n-gram model
      3. Optionally compute sentence embeddings
      4. Cross-validate GBM
      5. Train final model on full data
      6. Compute per-language thresholds
    """
    os.makedirs(models_dir, exist_ok=True)

    # Load dataset
    print("=" * 60)
    print("ScamShield Training Pipeline")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"\nDataset: {len(df)} samples")
    print(f"  Scam:  {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
    print(f"  Safe:  {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")
    print(f"  Languages: {df['language'].value_counts().to_dict()}")

    texts = df["text"].tolist()
    labels = df["label"].values

    # ── Step 1: Train char n-gram model ──────────────────────────────────────
    print("\n[1/6] Training char n-gram model...")
    ngram_model = build_ngram_model()
    ngram_model.fit(texts, labels)

    ngram_path = os.path.join(models_dir, "ngram_model.pkl")
    joblib.dump(ngram_model, ngram_path)
    ngram_size = os.path.getsize(ngram_path) / 1024
    print(f"  Saved: {ngram_path} ({ngram_size:.0f} KB)")

    # Quick n-gram validation
    ngram_preds = ngram_model.predict(texts[:100])
    ngram_acc = (ngram_preds == labels[:100]).mean()
    print(f"  N-gram train accuracy (first 100): {ngram_acc:.4f}")

    # ── Step 2: Extract 32-feature vectors ───────────────────────────────────
    print("\n[2/6] Extracting 32-feature vectors...")
    X_handcrafted = np.array([
        extract_all_32_features(text, ngram_model)
        for i, text in enumerate(texts)
    ])
    print(f"  Feature matrix shape: {X_handcrafted.shape}")

    # ── Step 3: Optionally compute sentence embeddings ───────────────────────
    if use_embeddings:
        print("\n[3/6] Computing sentence embeddings...")
        extractor = EmbeddingFeatureExtractor()
        if extractor.available:
            X_embeddings = extractor.transform(texts)
            X = np.hstack([X_handcrafted, X_embeddings])
            print(f"  Combined feature matrix: {X.shape}")
        else:
            print("  Embeddings not available, using handcrafted features only.")
            X = X_handcrafted
    else:
        print("\n[3/6] Skipping embeddings (use --use-embeddings to enable)")
        X = X_handcrafted

    y = labels

    # ── Step 4: Cross-validate ───────────────────────────────────────────────
    print("\n[4/6] Cross-validating Calibrated GBM (5-fold)...")
    model = build_gbm_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["f1", "roc_auc", "precision", "recall"],
        return_train_score=False,
        verbose=0,
    )

    print(f"\n  CV Results (5-fold):")
    print(f"    F1:        {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"    ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"    Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    print(f"    Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")

    # ── Step 5: Train final model on full data ───────────────────────────────
    print("\n[5/6] Training final model on full dataset...")
    sample_weights = get_sample_weights(y)
    model = build_gbm_model()
    model.fit(X, y, sample_weight=sample_weights)

    model_path = os.path.join(models_dir, "scam_detector_final.pkl")
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Saved: {model_path} ({model_size:.2f} MB)")

    # ── Step 6: Compute per-language thresholds ──────────────────────────────
    print("\n[6/6] Computing per-language thresholds...")

    # Use a held-out validation split
    strat_key = df["label"].astype(str) + "_" + df["language"]
    _, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=strat_key
    )

    X_val = X[val_idx]
    y_val = y[val_idx]
    val_langs = df.iloc[val_idx]["language"].values

    lang_thresholds = {}
    for lang in ["en", "hi", "mr", "te", "kn"]:
        lang_mask = val_langs == lang
        if lang_mask.sum() < 30:
            lang_thresholds[lang] = 0.50
            print(f"  {lang}: 0.500 (default — only {lang_mask.sum()} samples)")
            continue

        threshold = find_optimal_threshold(
            model, X_val[lang_mask], y_val[lang_mask], target_recall=0.92
        )
        lang_thresholds[lang] = threshold
        print(f"  {lang}: {threshold:.3f}")

    thresholds_path = os.path.join(models_dir, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(lang_thresholds, f, indent=2)
    print(f"  Saved: {thresholds_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_size = ngram_size / 1024 + model_size  # MB
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Total model size: {total_size:.2f} MB")
    print(f"  CV F1: {cv_results['test_f1'].mean():.4f}")
    print(f"  CV ROC-AUC: {cv_results['test_roc_auc'].mean():.4f}")
    print(f"  Models saved to: {models_dir}/")

    return model, ngram_model, lang_thresholds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ScamShield model")
    parser.add_argument("--data-path", default="data/processed/full_dataset.csv",
                        help="Path to training dataset CSV")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Include sentence embeddings (requires sentence-transformers)")
    parser.add_argument("--models-dir", default="models",
                        help="Directory to save trained models")
    args = parser.parse_args()

    train_full_pipeline(args.data_path, args.use_embeddings, args.models_dir)
