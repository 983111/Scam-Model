"""
Run Full Evaluation — Loads trained model and runs comprehensive evaluation.

Usage:
    python scripts/run_evaluation.py [--models-dir models/] [--test-path data/processed/test.csv]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.feature_extractor import extract_all_32_features
from src.evaluate import full_evaluation, per_language_evaluation, error_analysis, mcnemar_test
from src.adversarial import run_adversarial_evaluation
from src.models import build_ngram_model, build_baseline_lr, build_baseline_rf


def run_full_evaluation(models_dir: str = "models", test_path: str = "data/processed/test.csv"):
    """Run complete evaluation suite."""
    print("=" * 60)
    print("ScamShield Full Evaluation")
    print("=" * 60)

    # Load test data
    df_test = pd.read_csv(test_path)
    print(f"\nTest set: {len(df_test)} samples")

    # Load models
    ngram_model = joblib.load(os.path.join(models_dir, "ngram_model.pkl"))
    model = joblib.load(os.path.join(models_dir, "scam_detector_final.pkl"))

    # Extract features
    print("\nExtracting features for test set...")
    X_test = np.array([extract_all_32_features(t, ngram_model) for t in df_test["text"]])
    y_test = df_test["label"].values

    # ── Standard Evaluation ──────────────────────────────────────────────────
    metrics = full_evaluation(model, X_test, y_test, "ScamShield (Calibrated GBM)")

    # ── Per-Language Evaluation ──────────────────────────────────────────────
    lang_results = per_language_evaluation(
        model, X_test, y_test, df_test["language"].values, "ScamShield"
    )

    # ── Error Analysis ───────────────────────────────────────────────────────
    error_results = error_analysis(model, X_test, y_test, df_test["text"].tolist())

    # ── Baseline Comparisons ─────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Baseline Comparisons")
    print(f"{'=' * 55}")

    # Train baselines on full dataset for comparison
    df_full = pd.read_csv("data/processed/full_dataset.csv")
    X_full = np.array([extract_all_32_features(t, ngram_model) for t in df_full["text"]])
    y_full = df_full["label"].values

    # N-gram baseline
    print("\n  Training baselines for comparison...")
    ngram_preds = ngram_model.predict(df_test["text"].tolist())
    ngram_f1 = float(np.mean(ngram_preds == y_test))
    print(f"  N-gram accuracy on test: {ngram_f1:.4f}")

    # Logistic Regression baseline
    lr_model = build_baseline_lr()
    lr_model.fit(X_full, y_full)
    lr_metrics = full_evaluation(lr_model, X_test, y_test, "Baseline: Logistic Regression")

    # McNemar test: GBM vs LR
    y_pred_gbm = model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)
    mcnemar_test(y_test, y_pred_gbm, y_pred_lr, "GBM", "Logistic Regression")

    # ── Adversarial Evaluation ───────────────────────────────────────────────
    adversarial_results = run_adversarial_evaluation(
        model, ngram_model, df_test["text"].tolist(), y_test.tolist()
    )

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    args = parser.parse_args()

    os.chdir(project_root)
    run_full_evaluation(args.models_dir, args.test_path)
