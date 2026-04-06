"""
Reddit Scam Message Collector (via PRAW).

Requires Reddit API credentials. Get them at:
  https://www.reddit.com/prefs/apps/

Usage:
  python -m src.collectors.reddit_collector \
      --client-id YOUR_ID \
      --client-secret YOUR_SECRET
"""

import os
import uuid
import argparse
import pandas as pd
from datetime import datetime


def collect_reddit_scams(client_id: str, client_secret: str,
                         output_dir: str = "data/raw",
                         limit: int = 500) -> pd.DataFrame:
    """Collect scam messages from Reddit using PRAW."""
    try:
        import praw
    except ImportError:
        print("ERROR: praw not installed. Run: pip install praw")
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="scam-dataset-collector/1.0"
    )

    subreddits = ["Scams", "phishing", "fraudalert", "cybersecurity", "india"]
    records = []

    for sub_name in subreddits:
        print(f"  Searching r/{sub_name}...")
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.search("scam message text", limit=limit, time_filter="year"):
                text = post.selftext.strip()
                if len(text) > 30:
                    records.append({
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "label": 1,  # Will need manual review
                        "language": "en",
                        "category": "unknown",
                        "source": f"reddit/{sub_name}",
                        "collection_date": datetime.now().isoformat(),
                        "annotator": "needs_review",
                        "confidence": 0.0,
                        "has_url": bool("http" in text.lower()),
                        "has_phone": False,
                        "reviewed": False,
                    })
        except Exception as e:
            print(f"  Error with r/{sub_name}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        output_path = os.path.join(output_dir, "reddit_scams.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    collect_reddit_scams(args.client_id, args.client_secret, limit=args.limit)
