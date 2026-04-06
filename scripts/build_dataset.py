"""
Build Dataset Script — Orchestrates data collection, labeling, and splitting.

Usage:
    python scripts/build_dataset.py [--output-dir data/]
"""

import os
import sys
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.collectors.uci_collector import download_uci_sms
from src.collectors.indic_generator import generate_indic_dataset
from src.preprocessor import ScamPreprocessor


def build_dataset(output_dir: str = "data"):
    """
    Master dataset builder:
      1. Download/generate UCI SMS (English)
      2. Generate Indian language samples
      3. Merge and validate schema
      4. Stratified train/test split
      5. Save processed files
    """
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 60)
    print("ScamShield Dataset Builder")
    print("=" * 60)

    # ── Step 1: English data ─────────────────────────────────────────────────
    print("\n[1/4] Collecting English data (UCI SMS)...")
    df_english = download_uci_sms(raw_dir)

    # ── Step 2: Indian language data ─────────────────────────────────────────
    print("\n[2/4] Generating Indian language data...")
    df_indic = generate_indic_dataset(raw_dir)

    # ── Step 3: Merge and validate ───────────────────────────────────────────
    print("\n[3/4] Merging and validating dataset...")
    df = pd.concat([df_english, df_indic], ignore_index=True)

    # Ensure all required columns exist
    required_cols = [
        "id", "text", "label", "language", "category", "source",
        "collection_date", "annotator", "confidence", "has_url",
        "has_phone", "reviewed"
    ]
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

    # Clean texts
    preprocessor = ScamPreprocessor()
    df["text"] = df["text"].apply(lambda x: preprocessor.clean(str(x)) if pd.notna(x) else "")

    # Remove empty texts
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    # Ensure labels are int
    df["label"] = df["label"].astype(int)

    # Update URL/phone detection
    df["has_url"] = df["text"].apply(preprocessor.has_url)
    df["has_phone"] = df["text"].apply(preprocessor.has_phone)

    print(f"\n  Total dataset: {len(df)} samples")
    print(f"  Label distribution:")
    print(f"    Scam: {(df['label']==1).sum()} ({(df['label']==1).mean():.1%})")
    print(f"    Safe: {(df['label']==0).sum()} ({(df['label']==0).mean():.1%})")
    print(f"  Language distribution:")
    for lang, count in df["language"].value_counts().items():
        print(f"    {lang}: {count}")
    print(f"  Category distribution:")
    for cat, count in df["category"].value_counts().head(12).items():
        print(f"    {cat}: {count}")

    # ── Step 4: Stratified split ─────────────────────────────────────────────
    print("\n[4/4] Creating stratified train/test split...")

    # Stratify on label + language
    df["strat_key"] = df["label"].astype(str) + "_" + df["language"]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["strat_key"]
    )

    # Drop stratification key
    train_df = train_df.drop(columns=["strat_key"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strat_key"]).reset_index(drop=True)
    df = df.drop(columns=["strat_key"])

    # Save
    full_path = os.path.join(processed_dir, "full_dataset.csv")
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n  Saved:")
    print(f"    Full:  {full_path} ({len(df)} samples)")
    print(f"    Train: {train_path} ({len(train_df)} samples)")
    print(f"    Test:  {test_path} ({len(test_df)} samples)")

    print(f"\n{'=' * 60}")
    print(f"Dataset build complete!")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    build_dataset(args.output_dir)
