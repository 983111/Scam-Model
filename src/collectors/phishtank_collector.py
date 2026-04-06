"""
PhishTank Phishing URL Collector.

Downloads verified phishing URLs from PhishTank and formats them
as simulated SMS messages for the scam detection dataset.

No authentication needed for the basic list.
Source: https://phishtank.org/developer_info.php
"""

import os
import uuid
import pandas as pd
import requests
from datetime import datetime


PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.json"


def collect_phishtank(output_dir: str = "data/raw", max_records: int = 2000) -> pd.DataFrame:
    """Download verified phishing URLs from PhishTank."""
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading PhishTank verified phishing data...")
    try:
        response = requests.get(
            PHISHTANK_URL,
            headers={"User-Agent": "ScamDatasetCollector/1.0"},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  PhishTank download failed: {e}")
        return pd.DataFrame()

    sms_templates = [
        "Click here to verify your account: {url}",
        "URGENT: Verify your identity now: {url}",
        "Your account has been compromised. Secure it: {url}",
        "Action required: Update your details at {url}",
        "Security alert — confirm your login: {url}",
    ]

    import random
    random.seed(42)

    records = []
    for entry in data[:max_records]:
        template = random.choice(sms_templates)
        text = template.format(url=entry["url"])
        records.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": 1,
            "language": "en",
            "category": "phishing",
            "source": "phishtank",
            "collection_date": entry.get("submission_time", datetime.now().isoformat()),
            "annotator": "auto_tier1",
            "confidence": 1.0,
            "has_url": True,
            "has_phone": False,
            "reviewed": True,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        output_path = os.path.join(output_dir, "phishtank.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} phishing records to {output_path}")
    return df


if __name__ == "__main__":
    df = collect_phishtank()
    print(f"Collected {len(df)} records")
