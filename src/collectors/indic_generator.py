"""
Indian Language Scam & Safe Message Generator.

Generates realistic Hindi, Marathi, Telugu, and Kannada scam and safe
messages using templates based on real scam patterns observed in India.

Categories covered: OTP fraud, lottery, KYC, investment, phishing,
impersonation, job scam, tech support, customs/package, charity.
"""

import os
import uuid
import random
import pandas as pd
from datetime import datetime

random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# HINDI TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

HINDI_SCAM_TEMPLATES = {
    "otp_fraud": [
        "प्रिय ग्राहक, आपके खाते में संदिग्ध गतिविधि पाई गई है। कृपया अपना OTP {otp} हमारे अधिकारी को शेयर करें।",
        "SBI: आपका खाता ब्लॉक होने वाला है। तुरंत OTP {otp} शेयर करें अन्यथा खाता बंद हो जाएगा।",
        "HDFC Bank: सुरक्षा सत्यापन के लिए अपना OTP {otp} भेजें। यह 5 मिनट में समाप्त होगा।",
        "आपके बैंक खाते से {amount} रुपये का संदिग्ध लेनदेन। रोकने के लिए OTP {otp} शेयर करें।",
        "ICICI: आपका डेबिट कार्ड ब्लॉक कर दिया गया है। अनब्लॉक करने के लिए OTP {otp} हमारे एजेंट को दें।",
    ],
    "lottery": [
        "बधाई हो! आपने {amount} रुपये की लॉटरी जीती है! अभी क्लेम करें: {phone}",
        "जिओ लकी ड्रॉ: आपका नंबर चुना गया है! {amount} लाख जीतने के लिए {phone} पर कॉल करें।",
        "KBC: बधाई हो! आप 25 लाख के विजेता हैं। अभी क्लेम करें। WhatsApp: {phone}",
        "Amazon Lucky Draw: आपने iPhone 15 Pro जीता है! डिलीवरी के लिए {amount} रुपये प्रोसेसिंग फीस भेजें।",
        "Flipkart Mega Prize: {amount} रुपये का cashback आपके नाम! कॉल करें: {phone}",
    ],
    "kyc_scam": [
        "URGENT: आपका {bank} KYC एक्सपायर हो गया है। 24 घंटे में अपडेट करें अन्यथा खाता फ्रीज हो जाएगा। लिंक: {url}",
        "{bank}: KYC अपडेट अनिवार्य है। आज ही पूरा करें: {url}। समय सीमा: कल शाम 6 बजे।",
        "आपका {bank} खाता KYC pending के कारण बंद होने वाला है। तुरंत अपडेट करें: {url}",
    ],
    "investment": [
        "क्रिप्टो में निवेश करें और 30 दिन में 300% रिटर्न पाएं! गारंटीड! WhatsApp: {phone}",
        "शेयर मार्केट टिप: आज {company} के शेयर खरीदें। 500% रिटर्न गारंटीड। कॉल: {phone}",
        "मुफ्त ट्रेडिंग टिप्स! रोज़ ₹{amount} कमाएं। हमारे Telegram ग्रुप जॉइन करें: {url}",
    ],
    "phishing": [
        "आपका {bank} अकाउंट लॉक हो गया है। अनलॉक करने के लिए क्लिक करें: {url}",
        "{bank} सिक्योरिटी अलर्ट: आपके खाते में अनधिकृत लॉगिन। सत्यापित करें: {url}",
        "आपका PAN कार्ड {bank} खाते से लिंक नहीं है। लिंक करें: {url}। समय सीमा: आज।",
    ],
    "impersonation": [
        "यह संदेश भारत सरकार की ओर से है। आपकी सब्सिडी ₹{amount} तैयार है। क्लेम: {phone}",
        "PM किसान योजना: ₹{amount} आपके खाते में आने वाले हैं। सत्यापन के लिए कॉल करें: {phone}",
        "EPFO: आपका PF निकासी ₹{amount} अप्रूव हो गया है। प्रोसेसिंग के लिए: {phone}",
    ],
    "job_scam": [
        "घर बैठे कमाएं ₹{amount} रोज़! कोई अनुभव नहीं चाहिए। WhatsApp: {phone}",
        "Amazon/Flipkart में पार्ट-टाइम जॉब। ₹{amount} प्रतिदिन। अभी अप्लाई करें: {url}",
        "डेटा एंट्री जॉब - ₹{amount}/महीना। कोई इन्वेस्टमेंट नहीं। कॉल: {phone}",
    ],
    "tech_support": [
        "चेतावनी: आपके फोन में {count} वायरस मिले हैं! तुरंत स्कैन करें: {url}",
        "आपका WhatsApp हैक हो गया है! सुरक्षित करने के लिए: {url}",
    ],
}

HINDI_SAFE_TEMPLATES = [
    "आपका OTP {otp} है। 10 मिनट के लिए वैध है। किसी को शेयर न करें।",
    "आपके {bank} खाते से ₹{amount} का लेनदेन सफल। Bal: ₹{balance}",
    "आपका {bank} स्टेटमेंट तैयार है। नेट बैंकिंग पर देखें।",
    "बिजली बिल ₹{amount} का भुगतान सफल। रसीद: {code}",
    "आपकी ट्रेन {code} का स्टेटस: समय पर। प्लेटफॉर्म {count}.",
    "कल की मीटिंग {time} बजे है। पक्का आना।",
    "दूध लाना मत भूलना घर आते वक्त!",
    "जन्मदिन मुबारक! बहुत सारी शुभकामनाएं! 🎂",
    "आपका Aadhaar सफलतापूर्वक अपडेट हो गया है।",
    "आपके ऑर्डर #{code} की डिलीवरी कल तक हो जाएगी।",
    "रिमाइंडर: EMI ₹{amount} कल ड्यू है। खाते में बैलेंस रखें।",
    "आज का मौसम: बारिश की संभावना। छाता लेकर जाएं।",
    "स्कूल कल से बंद है। अगला नोटिस जल्द आएगा।",
    "LPG सिलेंडर बुकिंग #{code} कन्फर्म। डिलीवरी: {date}",
    "आपका {bank} FD mature हो गया है। ₹{amount} खाते में जमा।",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MARATHI TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

MARATHI_SCAM_TEMPLATES = {
    "otp_fraud": [
        "प्रिय ग्राहक, तुमच्या खात्यात संशयास्पद व्यवहार आढळला आहे। कृपया OTP {otp} शेअर करा।",
        "{bank}: तुमचे खाते ब्लॉक होणार आहे। ताबडतोब OTP {otp} शेअर करा।",
        "तुमच्या खात्यातून ₹{amount} चा संशयास्पद व्यवहार. थांबवण्यासाठी OTP {otp} द्या.",
    ],
    "lottery": [
        "अभिनंदन! तुम्ही ₹{amount} ची लॉटरी जिंकली आहे! आत्ताच क्लेम करा: {phone}",
        "जिओ लकी ड्रॉ: तुमचा नंबर निवडला गेला आहे! ₹{amount} जिंकण्यासाठी कॉल करा: {phone}",
    ],
    "kyc_scam": [
        "URGENT: तुमचे {bank} KYC एक्सपायर झाले आहे. 24 तासांत अपडेट करा: {url}",
        "{bank}: KYC अपडेट अनिवार्य आहे. आज पूर्ण करा: {url}",
    ],
    "investment": [
        "शेअर बाजारात गुंतवणूक करा. 30 दिवसांत 300% परतावा! कॉल करा: {phone}",
        "क्रिप्टो गुंतवणूक - दररोज ₹{amount} कमवा. Telegram जॉइन करा: {url}",
    ],
    "phishing": [
        "तुमचे {bank} खाते लॉक झाले आहे. अनलॉक करण्यासाठी: {url}",
    ],
    "job_scam": [
        "घरबसल्या कमवा ₹{amount} रोज! अनुभव नको. WhatsApp: {phone}",
        "Amazon मध्ये पार्ट-टाइम जॉब. ₹{amount}/दिवस. अप्लाय करा: {url}",
    ],
}

MARATHI_SAFE_TEMPLATES = [
    "तुमचा OTP {otp} आहे. 10 मिनिटांसाठी वैध. कोणालाही शेअर करू नका.",
    "तुमच्या {bank} खात्यातून ₹{amount} चा व्यवहार यशस्वी. Bal: ₹{balance}",
    "वीज बिल ₹{amount} भरणा यशस्वी. पावती: {code}",
    "उद्याची मीटिंग {time} वाजता आहे. नक्की या.",
    "वाढदिवसाच्या हार्दिक शुभेच्छा! 🎂",
    "तुमची ऑर्डर #{code} उद्या डिलिव्हर होईल.",
    "आजचे हवामान: पावसाची शक्यता. छत्री घ्या.",
    "EMI ₹{amount} उद्या ड्यू आहे. खात्यात बॅलन्स ठेवा.",
    "LPG सिलिंडर बुकिंग #{code} कन्फर्म. डिलिव्हरी: {date}",
    "तुमचा {bank} FD mature झाला आहे. ₹{amount} खात्यात जमा.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TELUGU TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

TELUGU_SCAM_TEMPLATES = {
    "otp_fraud": [
        "ప్రియ కస్టమర్, మీ ఖాతాలో అనుమానాస్పద లావాదేవీ కనుగొనబడింది. దయచేసి OTP {otp} షేర్ చేయండి.",
        "{bank}: మీ ఖాతా బ్లాక్ అవుతుంది. వెంటనే OTP {otp} షేర్ చేయండి.",
    ],
    "lottery": [
        "అభినందనలు! మీరు ₹{amount} లాటరీ గెలిచారు! ఇప్పుడే క్లెయిమ్ చేయండి: {phone}",
        "జియో లక్కీ డ్రా: మీ నంబర్ ఎంపిక చేయబడింది! గెలవడానికి కాల్ చేయండి: {phone}",
    ],
    "kyc_scam": [
        "అత్యవసరం: మీ {bank} KYC గడువు ముగిసింది. 24 గంటల్లో అప్డేట్ చేయండి: {url}",
    ],
    "investment": [
        "క్రిప్టోలో పెట్టుబడి పెట్టండి. 30 రోజుల్లో 300% రిటర్న్! కాల్: {phone}",
        "షేర్ మార్కెట్ టిప్: రోజు ₹{amount} సంపాదించండి. Telegram చేరండి: {url}",
    ],
    "phishing": [
        "మీ {bank} ఖాతా లాక్ చేయబడింది. అన్‌లాక్ చేయడానికి: {url}",
    ],
    "job_scam": [
        "ఇంటి నుండి రోజు ₹{amount} సంపాదించండి! అనుభవం అవసరం లేదు. WhatsApp: {phone}",
    ],
}

TELUGU_SAFE_TEMPLATES = [
    "మీ OTP {otp}. 10 నిమిషాలు చెల్లుతుంది. ఎవరికీ షేర్ చేయకండి.",
    "మీ {bank} ఖాతా నుండి ₹{amount} లావాదేవీ విజయవంతం. Bal: ₹{balance}",
    "కరెంట్ బిల్ ₹{amount} చెల్లింపు విజయవంతం. రసీదు: {code}",
    "రేపటి మీటింగ్ {time} కి ఉంది. తప్పకుండా రండి.",
    "పుట్టినరోజు శుభాకాంక్షలు! 🎂",
    "మీ ఆర్డర్ #{code} రేపు డెలివరీ అవుతుంది.",
    "EMI ₹{amount} రేపు ఆటోమేటిక్ డెబిట్. ఖాతాలో బ్యాలెన్స్ ఉంచండి.",
    "LPG సిలిండర్ బుకింగ్ #{code} కన్ఫర్మ్. డెలివరీ: {date}",
]


# ═══════════════════════════════════════════════════════════════════════════════
# KANNADA TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

KANNADA_SCAM_TEMPLATES = {
    "otp_fraud": [
        "ಆತ್ಮೀಯ ಗ್ರಾಹಕರೇ, ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ಅನುಮಾನಾಸ್ಪದ ಚಟುವಟಿಕೆ ಕಂಡುಬಂದಿದೆ. ದಯವಿಟ್ಟು OTP {otp} ಹಂಚಿ.",
        "{bank}: ನಿಮ್ಮ ಖಾತೆ ಬ್ಲಾಕ್ ಆಗುತ್ತದೆ. ತಕ್ಷಣ OTP {otp} ಹಂಚಿಕೊಳ್ಳಿ.",
    ],
    "lottery": [
        "ಅಭಿನಂದನೆಗಳು! ನೀವು ₹{amount} ಲಾಟರಿ ಗೆದ್ದಿದ್ದೀರಿ! ಈಗಲೇ ಕ್ಲೇಮ್ ಮಾಡಿ: {phone}",
        "ಜಿಯೋ ಲಕ್ಕಿ ಡ್ರಾ: ನಿಮ್ಮ ನಂಬರ್ ಆಯ್ಕೆಯಾಗಿದೆ! ₹{amount} ಗೆಲ್ಲಲು ಕಾಲ್ ಮಾಡಿ: {phone}",
    ],
    "kyc_scam": [
        "ತುರ್ತು: ನಿಮ್ಮ {bank} KYC ಅವಧಿ ಮೀರಿದೆ. 24 ಗಂಟೆಯೊಳಗೆ ನವೀಕರಿಸಿ: {url}",
    ],
    "investment": [
        "ಕ್ರಿಪ್ಟೋದಲ್ಲಿ ಹೂಡಿಕೆ ಮಾಡಿ. 30 ದಿನಗಳಲ್ಲಿ 300% ರಿಟರ್ನ್! ಕಾಲ್: {phone}",
    ],
    "phishing": [
        "ನಿಮ್ಮ {bank} ಖಾತೆ ಲಾಕ್ ಆಗಿದೆ. ಅನ್‌ಲಾಕ್ ಮಾಡಲು: {url}",
    ],
    "job_scam": [
        "ಮನೆಯಿಂದ ದಿನಕ್ಕೆ ₹{amount} ಗಳಿಸಿ! ಅನುಭವ ಅಗತ್ಯವಿಲ್ಲ. WhatsApp: {phone}",
    ],
}

KANNADA_SAFE_TEMPLATES = [
    "ನಿಮ್ಮ OTP {otp}. 10 ನಿಮಿಷಗಳವರೆಗೆ ಮಾನ್ಯ. ಯಾರಿಗೂ ಹಂಚಬೇಡಿ.",
    "ನಿಮ್ಮ {bank} ಖಾತೆಯಿಂದ ₹{amount} ವಹಿವಾಟು ಯಶಸ್ವಿ. Bal: ₹{balance}",
    "ವಿದ್ಯುತ್ ಬಿಲ್ ₹{amount} ಪಾವತಿ ಯಶಸ್ವಿ. ರಸೀದಿ: {code}",
    "ನಾಳೆಯ ಸಭೆ {time} ಕ್ಕೆ ಇದೆ. ಖಂಡಿತ ಬನ್ನಿ.",
    "ಹುಟ್ಟುಹಬ್ಬದ ಶುಭಾಶಯಗಳು! 🎂",
    "ನಿಮ್ಮ ಆರ್ಡರ್ #{code} ನಾಳೆ ಡೆಲಿವರಿ ಆಗುತ್ತದೆ.",
    "EMI ₹{amount} ನಾಳೆ ಆಟೋ ಡೆಬಿಟ್. ಖಾತೆಯಲ್ಲಿ ಬ್ಯಾಲೆನ್ಸ್ ಇಡಿ.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "BOB", "PNB", "Canara"]
FAKE_URLS = [
    "http://sbi-kyc-update.xyz/verify", "http://hdfc-secure.tk/login",
    "http://icici-alert.ml/update", "http://bank-verify.pw/kyc",
    "http://secure-banking.top/auth", "http://account-alert.click/verify",
    "http://kyc-update.work/sbi", "http://bit.ly/bank-verify-now",
    "https://t.me/+invest_group_2025", "http://192.168.1.1/phish",
]
COMPANIES = ["TrustInvest", "CryptoMax", "ProfitGuru", "WealthPrime", "QuickGains"]


def _fill_template(template: str) -> str:
    """Fill a template with randomized realistic values."""
    return template.format(
        amount=random.choice(["1000", "5000", "10000", "25000", "50000",
                              "1,00,000", "5,00,000", "10,00,000", "25 लाख"]),
        phone=f"+91{random.randint(7000000000, 9999999999)}",
        otp=f"{random.randint(100000, 999999)}",
        code=f"{random.randint(10000, 99999)}",
        bank=random.choice(BANKS),
        url=random.choice(FAKE_URLS),
        company=random.choice(COMPANIES),
        count=random.randint(3, 47),
        date=f"{random.randint(1,28)}/{random.randint(1,12)}/2025",
        time=f"{random.randint(8,20)}:{random.choice(['00','15','30','45'])}",
        balance=f"{random.randint(1000, 500000)}",
    )


def _generate_for_language(
    lang_code: str,
    scam_templates: dict,
    safe_templates: list,
    n_scam: int = 1500,
    n_safe: int = 1500,
) -> list:
    """Generate scam and safe records for one language."""
    records = []

    # Generate scam messages
    all_scam_cats = list(scam_templates.keys())
    for _ in range(n_scam):
        category = random.choice(all_scam_cats)
        template = random.choice(scam_templates[category])
        text = _fill_template(template)
        records.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": 1,
            "language": lang_code,
            "category": category,
            "source": f"generated_{lang_code}",
            "collection_date": datetime.now().strftime("%Y-%m-%d"),
            "annotator": "auto_tier1",
            "confidence": 0.95,
            "has_url": bool("http" in text or "www" in text),
            "has_phone": bool("+91" in text),
            "reviewed": False,
        })

    # Generate safe messages
    for _ in range(n_safe):
        template = random.choice(safe_templates)
        text = _fill_template(template)
        records.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": 0,
            "language": lang_code,
            "category": "safe",
            "source": f"generated_{lang_code}",
            "collection_date": datetime.now().strftime("%Y-%m-%d"),
            "annotator": "auto_tier1",
            "confidence": 0.95,
            "has_url": bool("http" in text or "www" in text),
            "has_phone": bool("+91" in text),
            "reviewed": False,
        })

    return records


def generate_indic_dataset(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Generate multilingual Indian scam detection dataset.

    Returns combined DataFrame for Hindi, Marathi, Telugu, Kannada.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_records = []

    # Hindi: 2000 scam + 2000 safe
    print("Generating Hindi dataset...")
    all_records.extend(_generate_for_language(
        "hi", HINDI_SCAM_TEMPLATES, HINDI_SAFE_TEMPLATES, n_scam=2000, n_safe=2000
    ))

    # Marathi: 1500 scam + 1500 safe
    print("Generating Marathi dataset...")
    all_records.extend(_generate_for_language(
        "mr", MARATHI_SCAM_TEMPLATES, MARATHI_SAFE_TEMPLATES, n_scam=1500, n_safe=1500
    ))

    # Telugu: 1500 scam + 1500 safe
    print("Generating Telugu dataset...")
    all_records.extend(_generate_for_language(
        "te", TELUGU_SCAM_TEMPLATES, TELUGU_SAFE_TEMPLATES, n_scam=1500, n_safe=1500
    ))

    # Kannada: 1500 scam + 1500 safe
    print("Generating Kannada dataset...")
    all_records.extend(_generate_for_language(
        "kn", KANNADA_SCAM_TEMPLATES, KANNADA_SAFE_TEMPLATES, n_scam=1500, n_safe=1500
    ))

    random.shuffle(all_records)
    df = pd.DataFrame(all_records)

    output_path = os.path.join(output_dir, "indic_messages.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} Indic messages to {output_path}")
    print(f"Language distribution:\n{df['language'].value_counts().to_string()}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    df = generate_indic_dataset()
    print(f"\nDataset shape: {df.shape}")
