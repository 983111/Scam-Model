"""
ScamShield Hyperparameter Tuning — RandomizedSearchCV over GBM params.

Usage:
    python -m src.tune --data-path data/processed/full_dataset.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_ngram_model, build_gbm_model
from src.feature_extractor import extract_all_32_features


def run_tuning(data_path: str, n_iter: int = 50, models_dir: str = "models"):
    """Run hyperparameter tuning with RandomizedSearchCV."""
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("ScamShield Hyperparameter Tuning")
    print("=" * 60)

    df = pd.read_csv(data_path)
    texts = df["text"].tolist()
    y = df["label"].values

    # Load or train n-gram model
    ngram_path = os.path.join(models_dir, "ngram_model.pkl")
    if os.path.exists(ngram_path):
        print("Loading existing n-gram model...")
        ngram_model = joblib.load(ngram_path)
    else:
        print("Training n-gram model...")
        ngram_model = build_ngram_model()
        ngram_model.fit(texts, y)
        joblib.dump(ngram_model, ngram_path)

    # Extract features
    print("Extracting features...")
    X = np.array([extract_all_32_features(t, ngram_model) for t in texts])

    # Parameter search space
    param_dist = {
        "estimator__n_estimators":     randint(100, 400),
        "estimator__max_depth":        randint(3, 8),
        "estimator__learning_rate":    uniform(0.01, 0.15),
        "estimator__min_samples_leaf": randint(3, 15),
        "estimator__subsample":        uniform(0.6, 0.4),
    }

    print(f"\nRunning RandomizedSearchCV ({n_iter} iterations, 5-fold CV)...")
    search = RandomizedSearchCV(
        build_gbm_model(),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X, y)

    print(f"\nBest CV F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Save best model
    best_path = os.path.join(models_dir, "scam_detector_tuned.pkl")
    joblib.dump(search.best_estimator_, best_path)
    print(f"Saved tuned model: {best_path}")

    return search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/full_dataset.csv")
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    run_tuning(args.data_path, args.n_iter, args.models_dir)
