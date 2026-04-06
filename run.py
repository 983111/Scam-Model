# -*- coding: utf-8 -*-
"""
ScamShield Master Runner — Single script to build, train, evaluate, and serve.

Usage:
    python run.py build      - Build the dataset
    python run.py train      - Train the model
    python run.py evaluate   - Run full evaluation
    python run.py ablation   - Run ablation study
    python run.py predict    - Interactive prediction mode
    python run.py serve      - Start Flask API server
    python run.py test       - Run unit tests
    python run.py all        - Build + Train + Evaluate (full pipeline)
"""

import os
import sys
import json
import argparse

# Fix Windows encoding for Indic text
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def cmd_build(args):
    """Build the dataset from all sources."""
    print("=" * 60)
    print("  STEP 1: Building Dataset")
    print("=" * 60)

    from src.collectors.uci_collector import download_uci_sms
    from src.collectors.indic_generator import generate_indic_dataset
    from src.preprocessor import ScamPreprocessor

    import uuid
    import pandas as pd
    from sklearn.model_selection import train_test_split

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Step 1: English data
    print("\n[1/4] Collecting English data (UCI SMS)...")
    df_english = download_uci_sms(raw_dir)

    # Step 2: Indian language data
    print("\n[2/4] Generating Indian language data...")
    df_indic = generate_indic_dataset(raw_dir)

    # Step 3: Merge
    print("\n[3/4] Merging and validating...")
    df = pd.concat([df_english, df_indic], ignore_index=True)

    required_cols = ["id", "text", "label", "language", "category", "source",
                     "collection_date", "annotator", "confidence", "has_url",
                     "has_phone", "reviewed"]
    for col in required_cols:
        if col not in df.columns:
            if col == "id":
                df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
            elif col in ("has_url", "has_phone", "reviewed"):
                df[col] = False
            elif col == "confidence":
                df[col] = 0.0
            else:
                df[col] = ""

    preprocessor = ScamPreprocessor()
    df["text"] = df["text"].apply(lambda x: preprocessor.clean(str(x)) if pd.notna(x) else "")
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["has_url"] = df["text"].apply(preprocessor.has_url)
    df["has_phone"] = df["text"].apply(preprocessor.has_phone)

    print(f"\n  Total dataset: {len(df)} samples")
    print(f"  Scam: {(df['label']==1).sum()} ({(df['label']==1).mean():.1%})")
    print(f"  Safe: {(df['label']==0).sum()} ({(df['label']==0).mean():.1%})")
    for lang, count in df["language"].value_counts().items():
        print(f"  {lang}: {count}")

    # Step 4: Split
    print("\n[4/4] Stratified train/test split...")
    df["strat_key"] = df["label"].astype(str) + "_" + df["language"]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["strat_key"])
    train_df = train_df.drop(columns=["strat_key"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strat_key"]).reset_index(drop=True)
    df = df.drop(columns=["strat_key"])

    df.to_csv(f"{processed_dir}/full_dataset.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(f"{processed_dir}/train.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(f"{processed_dir}/test.csv", index=False, encoding="utf-8-sig")

    print(f"\n  Saved: full_dataset.csv ({len(df)}), train.csv ({len(train_df)}), test.csv ({len(test_df)})")
    print("  Dataset build complete!\n")
    return df


def cmd_train(args):
    """Train the model."""
    print("=" * 60)
    print("  STEP 2: Training Model")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.metrics import precision_recall_curve
    from sklearn.utils import class_weight as cw_module

    from src.models import build_ngram_model, build_gbm_model
    from src.feature_extractor import extract_all_32_features

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    data_path = args.data_path if hasattr(args, 'data_path') and args.data_path else "data/processed/full_dataset.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    texts = df["text"].tolist()
    labels = df["label"].values

    print(f"\n  Dataset: {len(df)} samples | Scam: {(labels==1).sum()} | Safe: {(labels==0).sum()}")

    # 1. Train n-gram model
    print("\n[1/6] Training char n-gram model...")
    ngram_model = build_ngram_model()
    ngram_model.fit(texts, labels)
    joblib.dump(ngram_model, f"{models_dir}/ngram_model.pkl")
    size_kb = os.path.getsize(f"{models_dir}/ngram_model.pkl") / 1024
    print(f"  Saved ngram_model.pkl ({size_kb:.0f} KB)")

    # 2. Extract features
    print("\n[2/6] Extracting 32-feature vectors...")
    X = np.array([extract_all_32_features(t, ngram_model) for t in texts])
    y = labels
    print(f"  Feature matrix: {X.shape}")

    # 3. Skip embeddings (optional)
    print("\n[3/6] Embeddings skipped (use --use-embeddings to enable)")

    # 4. Cross-validate
    print("\n[4/6] Cross-validating Calibrated GBM (5-fold)...")
    model = build_gbm_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=cv,
                                scoring=["f1", "roc_auc", "precision", "recall"],
                                return_train_score=False)

    print(f"  F1:        {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}")
    print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} +/- {cv_results['test_precision'].std():.4f}")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} +/- {cv_results['test_recall'].std():.4f}")

    # 5. Train final model
    print("\n[5/6] Training final model on full data...")
    classes = np.unique(y)
    weights = cw_module.compute_class_weight("balanced", classes=classes, y=y)
    sample_weights = np.array([weights[int(l)] for l in y])

    model = build_gbm_model()
    model.fit(X, y, sample_weight=sample_weights)
    joblib.dump(model, f"{models_dir}/scam_detector_final.pkl")
    size_mb = os.path.getsize(f"{models_dir}/scam_detector_final.pkl") / (1024*1024)
    print(f"  Saved scam_detector_final.pkl ({size_mb:.2f} MB)")

    # 6. Per-language thresholds
    print("\n[6/6] Computing per-language thresholds...")
    strat_key = df["label"].astype(str) + "_" + df["language"]
    _, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42, stratify=strat_key)
    X_val, y_val = X[val_idx], y[val_idx]
    val_langs = df.iloc[val_idx]["language"].values

    thresholds = {}
    for lang in ["en", "hi", "mr", "te", "kn"]:
        mask = val_langs == lang
        if mask.sum() < 30:
            thresholds[lang] = 0.50
            print(f"  {lang}: 0.500 (default)")
            continue
        y_proba = model.predict_proba(X_val[mask])[:, 1]
        precs, recs, thresh = precision_recall_curve(y_val[mask], y_proba)
        valid = [i for i, r in enumerate(recs) if r >= 0.92]
        if valid:
            best = max(valid, key=lambda i: precs[i])
            t = float(thresh[best]) if best < len(thresh) else 0.5
        else:
            t = 0.5
        thresholds[lang] = round(t, 4)
        print(f"  {lang}: {thresholds[lang]}")

    with open(f"{models_dir}/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n  Training complete! Total model size: {size_kb/1024 + size_mb:.2f} MB")
    print(f"  CV F1: {cv_results['test_f1'].mean():.4f}")
    return model, ngram_model


def cmd_evaluate(args):
    """Run full evaluation."""
    print("=" * 60)
    print("  STEP 3: Full Evaluation")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import (f1_score, roc_auc_score, matthews_corrcoef,
                                 brier_score_loss, precision_score, recall_score,
                                 confusion_matrix, classification_report)

    from src.feature_extractor import extract_all_32_features
    from src.adversarial import run_adversarial_evaluation

    test_path = args.test_path if hasattr(args, 'test_path') and args.test_path else "data/processed/test.csv"
    df = pd.read_csv(test_path, encoding="utf-8-sig")
    ngram_model = joblib.load("models/ngram_model.pkl")
    model = joblib.load("models/scam_detector_final.pkl")

    print(f"\n  Test set: {len(df)} samples")
    print("  Extracting features...")
    X_test = np.array([extract_all_32_features(t, ngram_model) for t in df["text"]])
    y_test = df["label"].values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n  --- Main Metrics ---")
    print(f"  F1 (macro):    {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  F1 (binary):   {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:       {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  MCC:           {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"  Brier Score:   {brier_score_loss(y_test, y_proba):.4f}")
    print(f"  Recall:        {recall_score(y_test, y_pred):.4f}")
    print(f"  Precision:     {precision_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Pred Safe  Pred Scam")
    print(f"    Actual Safe   {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"    Actual Scam   {cm[1][0]:>6}     {cm[1][1]:>6}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Safe', 'Scam'])}")

    # Per-language
    print("  --- Per-Language F1 ---")
    for lang in sorted(df["language"].unique()):
        mask = df["language"].values == lang
        if mask.sum() < 10:
            continue
        lf1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
        print(f"  {lang}: F1={lf1:.4f} (n={mask.sum()})")

    # Adversarial
    run_adversarial_evaluation(model, ngram_model, df["text"].tolist(), y_test.tolist())

    # Error analysis
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    print(f"\n  --- Error Analysis ---")
    print(f"  False Negatives (missed scams): {fn}")
    print(f"  False Positives (false alarms): {fp}")

    print("\n  Evaluation complete!")


def cmd_predict(args):
    """Interactive prediction mode."""
    from src.predict import ScamDetector
    detector = ScamDetector("models")

    print("=" * 60)
    print("  ScamShield Interactive Predictor")
    print("  Type a message and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        result = detector.predict(text)
        verdict = "SCAM" if result["verdict"] == "scam" else "SAFE"
        color = "" # No ANSI on Windows cmd
        print(f"\n  Verdict:     {verdict}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Category:    {result['category']}")
        print(f"  Language:    {result['language']}")
        print(f"  Threshold:   {result['threshold']:.4f}")
        if result['signals']:
            print(f"  Signals:     {', '.join(result['signals'])}")


def cmd_serve(args):
    """Start Flask API server."""
    from api.app import create_app
    port = args.port if hasattr(args, 'port') and args.port else 5000
    app = create_app("models")
    print(f"\n  ScamShield API: http://localhost:{port}")
    print(f"  POST /predict  - Single prediction")
    print(f"  POST /batch    - Batch prediction")
    print(f"  GET  /health   - Health check")
    print(f"  GET  /info     - Model info\n")
    app.run(host="0.0.0.0", port=port, debug=False)


def cmd_test(args):
    """Run unit tests."""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def cmd_all(args):
    """Run full pipeline: build + train + evaluate."""
    cmd_build(args)
    cmd_train(args)
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser(
        description="ScamShield - Multilingual Scam Detection ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build      Build the dataset (download UCI + generate Indic samples)
  train      Train the 3-layer ensemble model
  evaluate   Run full evaluation with adversarial testing
  ablation   Run ablation study (drop each feature group)
  predict    Interactive prediction mode
  serve      Start Flask API server
  test       Run unit tests
  all        Run full pipeline (build + train + evaluate)

Examples:
  python run.py all                          # Full pipeline
  python run.py build                        # Just build dataset
  python run.py train                        # Just train model
  python run.py predict                      # Interactive mode
  python run.py serve --port 8080            # Start API on port 8080
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # Build
    subparsers.add_parser("build", help="Build the dataset")

    # Train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--data-path", default="data/processed/full_dataset.csv")
    p_train.add_argument("--use-embeddings", action="store_true")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run full evaluation")
    p_eval.add_argument("--test-path", default="data/processed/test.csv")

    # Ablation
    subparsers.add_parser("ablation", help="Run ablation study")

    # Predict
    subparsers.add_parser("predict", help="Interactive prediction")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Start API server")
    p_serve.add_argument("--port", type=int, default=5000)

    # Test
    subparsers.add_parser("test", help="Run unit tests")

    # All
    subparsers.add_parser("all", help="Full pipeline")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "build": cmd_build,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "ablation": lambda a: __import__('scripts.ablation_study', fromlist=['run_ablation_study']).run_ablation_study() if False else cmd_ablation(a),
        "predict": cmd_predict,
        "serve": cmd_serve,
        "test": cmd_test,
        "all": cmd_all,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


def cmd_ablation(args):
    """Run ablation study."""
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from src.feature_extractor import extract_all_32_features, FEATURE_NAMES
    from src.models import build_gbm_model

    FEATURE_GROUPS = {
        "Binary Keywords (f1-f6)":     list(range(0, 6)),
        "Statistical (f7-f14)":        list(range(6, 14)),
        "Keyword Density (f15-f17)":   list(range(14, 17)),
        "URL Features (f18-f24)":      list(range(17, 24)),
        "Multilingual (f25-f32)":      list(range(24, 32)),
    }

    df = pd.read_csv("data/processed/full_dataset.csv", encoding="utf-8-sig")
    ngram_model = joblib.load("models/ngram_model.pkl")
    X = np.array([extract_all_32_features(t, ngram_model) for t in df["text"]])
    y = df["label"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n  --- Ablation Study ---")
    full_f1 = cross_val_score(build_gbm_model(), X, y, cv=cv, scoring="f1").mean()
    print(f"  All 32 features: F1={full_f1:.4f}")
    print(f"\n  {'Dropped Group':<30} {'F1':>8} {'Delta':>10}")
    print(f"  {'-'*50}")

    for name, indices in FEATURE_GROUPS.items():
        keep = [i for i in range(32) if i not in indices]
        f1 = cross_val_score(build_gbm_model(), X[:, keep], y, cv=cv, scoring="f1").mean()
        print(f"  w/o {name:<25} {f1:>8.4f} {f1-full_f1:>+10.4f}")


if __name__ == "__main__":
    main()
