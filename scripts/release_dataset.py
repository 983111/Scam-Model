"""
Dataset Release Utility — Prepares dataset for HuggingFace/Zenodo publishing.

Usage:
    python scripts/release_dataset.py [--hf-token YOUR_TOKEN] [--push]
"""

import os
import sys
import argparse
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def prepare_release(data_path: str = "data/processed/full_dataset.csv",
                    output_dir: str = "data/release"):
    """Prepare dataset for public release."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # Select release columns (no internal columns)
    release_cols = ["text", "label", "language", "category", "source",
                    "collection_date", "confidence"]
    df_release = df[[c for c in release_cols if c in df.columns]].copy()

    # Save as CSV and JSON
    csv_path = os.path.join(output_dir, "scam_detection_dataset.csv")
    json_path = os.path.join(output_dir, "scam_detection_dataset.json")
    df_release.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_release.to_json(json_path, orient="records", lines=True, force_ascii=False)

    # Generate dataset card
    card = generate_dataset_card(df_release)
    card_path = os.path.join(output_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"Release files saved to {output_dir}/:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {card_path}")

    return df_release


def generate_dataset_card(df: pd.DataFrame) -> str:
    """Generate a HuggingFace dataset card."""
    # Language stats
    lang_stats = []
    for lang in sorted(df["language"].unique()):
        total = (df["language"] == lang).sum()
        scam = ((df["language"] == lang) & (df["label"] == 1)).sum()
        safe = ((df["language"] == lang) & (df["label"] == 0)).sum()
        lang_stats.append(f"| {lang} | {total} | {scam} | {safe} |")

    categories = ", ".join(sorted(df["category"].unique()))

    return f"""# Indian Scam Detection Dataset

## Dataset Description
A multi-language scam message detection dataset covering English,
Hindi, Marathi, Telugu, and Kannada. Built for training ML models
to detect SMS/messaging scams common in India.

## Statistics
| Language | Total | Scam | Safe |
|----------|-------|------|------|
{chr(10).join(lang_stats)}

**Total samples:** {len(df)}
**Scam ratio:** {(df['label']==1).mean():.1%}

## Scam Categories
{categories}

## Features
- Multi-language coverage (5 languages)
- 12 scam categories
- Real-world scam patterns
- Confidence scores for each label

## Schema
- `text` (string): Raw message text
- `label` (int): 1 = scam, 0 = safe
- `language` (string): ISO language code
- `category` (string): Scam category
- `source` (string): Data source
- `collection_date` (string): ISO date
- `confidence` (float): Label confidence (0.0–1.0)

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("YOUR_USERNAME/indian-scam-detection")
```

## Licensing
CC BY 4.0 — free to use for research and commercial purposes with attribution.

## Citation
```bibtex
@dataset{{scam_detection_2025,
  title={{Indian Scam Detection Dataset}},
  year={{2025}},
  license={{CC BY 4.0}}
}}
```
"""


def push_to_huggingface(data_path: str, hf_token: str, repo_name: str):
    """Push dataset to HuggingFace Hub."""
    try:
        from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
    except ImportError:
        print("ERROR: datasets library not installed. Run: pip install datasets huggingface_hub")
        return

    df = pd.read_csv(data_path)

    features = Features({
        "text":            Value("string"),
        "label":           ClassLabel(names=["safe", "scam"]),
        "language":        Value("string"),
        "category":        Value("string"),
        "source":          Value("string"),
        "collection_date": Value("string"),
        "confidence":      Value("float32"),
    })

    release_cols = ["text", "label", "language", "category", "source",
                    "collection_date", "confidence"]
    df_release = df[[c for c in release_cols if c in df.columns]].copy()

    dataset = Dataset.from_pandas(df_release, features=features)
    train_test = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "test": train_test["test"]
    })

    dataset_dict.push_to_hub(repo_name, token=hf_token)
    print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/full_dataset.csv")
    parser.add_argument("--output-dir", default="data/release")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token")
    parser.add_argument("--repo-name", default="scam-detection-dataset")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace")
    args = parser.parse_args()

    os.chdir(project_root)
    prepare_release(args.data_path, args.output_dir)

    if args.push and args.hf_token:
        push_to_huggingface(args.data_path, args.hf_token, args.repo_name)
