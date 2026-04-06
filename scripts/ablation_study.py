"""
Ablation Study — Measures F1 impact of dropping each feature group.

Usage:
    python scripts/ablation_study.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.feature_extractor import extract_all_32_features, FEATURE_NAMES
from src.models import build_gbm_model


# Feature group definitions (indices into the 32-feature vector)
FEATURE_GROUPS = {
    "Binary Keywords (f1-f6)":     list(range(0, 6)),
    "Statistical (f7-f14)":        list(range(6, 14)),
    "Keyword Density (f15-f17)":   list(range(14, 17)),
    "URL Features (f18-f24)":      list(range(17, 24)),
    "Multilingual (f25-f32)":      list(range(24, 32)),
}


def run_ablation_study(data_path: str = "data/processed/full_dataset.csv",
                       models_dir: str = "models"):
    """
    Drop each feature group and measure F1 impact.
    """
    print("=" * 60)
    print("ScamShield Ablation Study")
    print("=" * 60)

    # Load data
    df = pd.read_csv(data_path)
    ngram_model = joblib.load(os.path.join(models_dir, "ngram_model.pkl"))

    print(f"\nExtracting full 32-feature vectors...")
    X_full = np.array([extract_all_32_features(t, ngram_model) for t in df["text"]])
    y = df["label"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Full model baseline
    print(f"\nTraining full model (all 32 features)...")
    full_scores = cross_val_score(build_gbm_model(), X_full, y, cv=cv, scoring="f1")
    full_f1 = full_scores.mean()
    print(f"  Full model F1: {full_f1:.4f} ± {full_scores.std():.4f}")

    # Ablation — drop each group
    print(f"\n  {'Feature Group':<30} {'F1':>8} {'Δ F1':>10} {'Impact':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'All 32 features':<30} {full_f1:>8.4f} {'—':>10} {'(baseline)':>10}")

    ablation_results = {}

    for group_name, group_indices in FEATURE_GROUPS.items():
        # Create feature matrix without this group
        keep_indices = [i for i in range(32) if i not in group_indices]
        X_ablated = X_full[:, keep_indices]

        scores = cross_val_score(build_gbm_model(), X_ablated, y, cv=cv, scoring="f1")
        ablated_f1 = scores.mean()
        delta = ablated_f1 - full_f1

        impact = "LOW" if abs(delta) < 0.01 else "MEDIUM" if abs(delta) < 0.03 else "HIGH"
        ablation_results[group_name] = {
            "f1": ablated_f1,
            "delta": delta,
            "impact": impact,
            "features_dropped": [FEATURE_NAMES[i] for i in group_indices],
        }

        print(f"  {f'w/o {group_name}':<30} {ablated_f1:>8.4f} {delta:>+10.4f} {impact:>10}")

    # Summary
    print(f"\n  Most impactful group: ", end="")
    most_impactful = min(ablation_results.items(), key=lambda x: x[1]["f1"])
    print(f"{most_impactful[0]} (Δ F1 = {most_impactful[1]['delta']:+.4f})")

    print(f"\n{'=' * 60}")
    print(f"Ablation Study Complete!")
    print(f"{'=' * 60}")

    return ablation_results


if __name__ == "__main__":
    os.chdir(project_root)
    run_ablation_study()
