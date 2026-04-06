"""
UCI SMS Spam Collection Downloader & Parser.

Downloads the real UCI SMS Spam Collection (5,574 messages) and converts
it to the ScamShield standard schema.

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
License: Public domain for research use.
"""

import os
import io
import csv
import uuid
import zipfile
import requests
import pandas as pd
from datetime import datetime


UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
BACKUP_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"


def download_uci_sms(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download and parse the UCI SMS Spam Collection.

    Returns DataFrame with standard schema columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "uci_sms.csv")

    # Try downloading from UCI
    print("Downloading UCI SMS Spam Collection...")
    records = []

    try:
        response = requests.get(UCI_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # The zip contains 'SMSSpamCollection' (tab-separated)
            for name in zf.namelist():
                if "SMSSpamCollection" in name or name.endswith(".txt"):
                    with zf.open(name) as f:
                        content = f.read().decode("utf-8", errors="replace")
                        for line in content.strip().split("\n"):
                            parts = line.split("\t", 1)
                            if len(parts) == 2:
                                label_str, text = parts
                                label = 1 if label_str.strip().lower() == "spam" else 0
                                records.append(_make_record(text.strip(), label))
                    break
        print(f"  Downloaded {len(records)} messages from UCI zip")

    except Exception as e:
        print(f"  UCI download failed ({e}). Trying backup URL...")
        try:
            response = requests.get(BACKUP_URL, timeout=30)
            response.raise_for_status()
            for line in response.text.strip().split("\n"):
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    label_str, text = parts
                    label = 1 if label_str.strip().lower() == "spam" else 0
                    records.append(_make_record(text.strip(), label))
            print(f"  Downloaded {len(records)} messages from backup")
        except Exception as e2:
            print(f"  Backup download also failed ({e2}).")
            print("  Generating synthetic English SMS dataset instead...")
            records = _generate_fallback_english()

    if not records:
        print("  No records obtained. Generating fallback dataset...")
        records = _generate_fallback_english()

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} records to {output_path}")
    print(f"  Scam: {(df['label']==1).sum()}, Safe: {(df['label']==0).sum()}")
    return df


def _make_record(text: str, label: int) -> dict:
    """Create a standardized record from a UCI SMS entry."""
    category = "safe"
    if label == 1:
        text_lower = text.lower()
        if any(w in text_lower for w in ["prize", "won", "winner", "lottery", "cash"]):
            category = "lottery"
        elif any(w in text_lower for w in ["click", "http", "www", "url", "link"]):
            category = "phishing"
        elif any(w in text_lower for w in ["call", "ring", "contact", "claim"]):
            category = "impersonation"
        elif any(w in text_lower for w in ["free", "offer", "discount", "deal"]):
            category = "lottery"
        else:
            category = "phishing"  # Default scam category

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "label": label,
        "language": "en",
        "category": category,
        "source": "uci_sms_spam",
        "collection_date": "2012-01-01",
        "annotator": "auto_tier1",
        "confidence": 1.0,
        "has_url": bool("http" in text.lower() or "www" in text.lower()),
        "has_phone": bool(any(c.isdigit() for c in text) and sum(c.isdigit() for c in text) >= 7),
        "reviewed": True,
    }


def _generate_fallback_english() -> list:
    """Generate a synthetic English SMS dataset as fallback if download fails."""
    import random
    random.seed(42)

    scam_templates = [
        # Lottery/Prize
        "CONGRATULATIONS! You've won ${amount} in our {event}! Call {phone} to claim NOW!",
        "WINNER!! You've been selected for a {amount} prize. Txt {code} to {phone} to claim!",
        "Ur awarded a £{amount} prize! To collect call {phone}. Hurry, offer expires {date}!",
        "You have won £{amount}! To claim reply PRIZE to {phone} with your bank details.",
        # Phishing
        "URGENT: Your {bank} account has been suspended. Verify at http://{fake_domain}/verify",
        "Alert: Unusual activity on your {bank} account. Login at http://{fake_domain}/secure to verify",
        "Your {bank} debit card has been blocked. Click http://{fake_domain}/unblock to reactivate.",
        # OTP/Credential
        "Your OTP is {otp}. Share this with our agent to verify your identity and unlock your account.",
        "Security Alert: Share your OTP {otp} with our executive to prevent account closure.",
        # Job scam
        "Earn ${amount}/day working from home! No experience needed. Apply now: http://{fake_domain}/jobs",
        "HIRING NOW! Part-time job from home. Earn {amount} weekly. WhatsApp us at {phone}",
        # Investment
        "Invest {amount} today and get 300% returns in 30 days! Guaranteed by {company}. Call {phone}",
        "CRYPTO ALERT: Bitcoin will hit ${amount}! Invest now at http://{fake_domain}/invest",
        # Impersonation
        "{bank}: Your account will be blocked in 24hrs. Complete KYC now: http://{fake_domain}/kyc",
        "From {bank}: Suspicious transaction of ${amount} detected. Call {phone} immediately.",
        # Tech support
        "WARNING: Your device has been infected with {count} viruses! Call {phone} for immediate help.",
        "Microsoft Security Alert: Your PC is at risk. Call {phone} to fix now. Code: {code}",
    ]

    safe_templates = [
        "Hey! Are you free for lunch today? Let me know.",
        "Meeting at {time} has been moved to {time2}. Please confirm.",
        "Your order #{code} has been shipped. Track at {bank}.com/track",
        "Reminder: Your appointment is scheduled for {date} at {time}.",
        "Hi, just checking in. How's your project going?",
        "Thanks for dinner last night! Let's do it again soon.",
        "Can you pick up milk on your way home? Thanks!",
        "Happy birthday! Hope you have an amazing day! 🎂",
        "Your OTP is {otp}. Valid for 10 minutes. Do not share.",
        "Transaction of Rs.{amount} from your {bank} account. If not done by you, call {phone}.",
        "Your {bank} statement for {date} is ready. Login to view.",
        "Exam results are out. Check at {bank}.edu/results",
        "Flight {code} to {city} departs at {time}. Web check-in open.",
        "Package delivered to your doorstep. Order #{code}",
        "Your subscription will renew on {date}. Manage at account settings.",
        "Weather alert: Heavy rain expected tomorrow. Stay safe!",
        "Library book due on {date}. Renew online if needed.",
        "Your monthly phone bill of Rs.{amount} is due on {date}.",
    ]

    banks = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "BOB", "PNB"]
    fake_domains = ["sbi-secure.xyz", "hdfc-verify.tk", "icici-update.ml",
                    "bank-secure.pw", "account-verify.top", "secure-login.click"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]
    companies = ["TrustInvest", "CryptoMax", "ProfitGuru", "WealthPrime"]

    records = []

    # Generate scams
    for _ in range(2000):
        template = random.choice(scam_templates)
        text = template.format(
            amount=random.choice(["1000", "5000", "10000", "50000", "100000", "1,00,000"]),
            phone=f"+91{random.randint(7000000000, 9999999999)}",
            code=f"{random.randint(1000, 9999)}",
            otp=f"{random.randint(100000, 999999)}",
            bank=random.choice(banks),
            fake_domain=random.choice(fake_domains),
            event=random.choice(["annual draw", "lucky dip", "mega lottery", "grand prize"]),
            date=f"{random.randint(1,28)}/{random.randint(1,12)}/2025",
            company=random.choice(companies),
            count=random.randint(3, 47),
            time=f"{random.randint(8,20)}:{random.choice(['00','15','30','45'])}",
        )
        rec = _make_record(text, 1)
        rec["source"] = "generated_english"
        rec["annotator"] = "auto_tier1"
        records.append(rec)

    # Generate safe messages
    for _ in range(2500):
        template = random.choice(safe_templates)
        text = template.format(
            amount=random.choice(["500", "1200", "2500", "899", "1499"]),
            phone=f"1800-{random.randint(100,999)}-{random.randint(1000,9999)}",
            code=f"{random.randint(10000, 99999)}",
            otp=f"{random.randint(100000, 999999)}",
            bank=random.choice(banks),
            date=f"{random.randint(1,28)}/{random.randint(1,12)}/2025",
            time=f"{random.randint(8,20)}:{random.choice(['00','15','30','45'])}",
            time2=f"{random.randint(8,20)}:{random.choice(['00','15','30','45'])}",
            city=random.choice(cities),
        )
        rec = _make_record(text, 0)
        rec["source"] = "generated_english"
        rec["annotator"] = "auto_tier1"
        rec["category"] = "safe"
        records.append(rec)

    random.shuffle(records)
    return records


if __name__ == "__main__":
    df = download_uci_sms()
    print(f"\nDataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
