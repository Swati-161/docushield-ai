import re
from typing import Optional


# ── Regex patterns ─────────────────────────────────────────
# PAN card: 5 letters, 4 digits, 1 letter  e.g. ABCDE1234F
PAN_PATTERN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')

# Indian mobile numbers
PHONE_PATTERN = re.compile(r'\b[6-9]\d{9}\b')

# Dates in common formats: 12/01/2024  12-01-2024  12 Jan 2024
DATE_PATTERN = re.compile(
    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
    r'|(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|'
    r'Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
    re.IGNORECASE
)

# Indian rupee amounts: Rs. 75,000  ₹5,00,000  INR 12000
AMOUNT_PATTERN = re.compile(
    r'(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d{1,2})?',
    re.IGNORECASE
)

# Aadhaar: 12 digits (sometimes spaced as 4-4-4)
AADHAAR_PATTERN = re.compile(
    r'\b\d{4}\s?\d{4}\s?\d{4}\b'
)

# Names heuristic: 2-4 capitalised words in a row
# (catches "Rahul Kumar Sharma" etc.)
NAME_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
)


def _clean_amount(raw: str) -> Optional[float]:
    """Converts 'Rs. 5,00,000' → 500000.0"""
    digits = re.sub(r'[^\d.]', '', raw.replace(',', ''))
    try:
        return float(digits) if digits else None
    except ValueError:
        return None

def extract_entities(text: str) -> dict:
    """
    Extracts structured fields from raw OCR text.

    Args:
        text : the full_text string returned by OCR extractor

    Returns:
        dict of lists — each field has all values found
    """
    pan_numbers = list(set(PAN_PATTERN.findall(text)))

    raw_dates = DATE_PATTERN.findall(text)
    dates = list(set(
        d[0] if d[0] else d[1]
        for d in raw_dates
        if d[0] or d[1]
    ))

    raw_amounts = AMOUNT_PATTERN.findall(text)
    amounts = []
    for raw in raw_amounts:
        val = _clean_amount(raw)
        if val is not None:
            amounts.append({
                "raw"  : raw.strip(),
                "value": val
            })

    phones   = list(set(PHONE_PATTERN.findall(text)))
    aadhaar  = list(set(AADHAAR_PATTERN.findall(text)))

    raw_names = NAME_PATTERN.findall(text)
    # replace the existing noise set and names extraction
    noise = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        "Date", "Name", "Bank", "Page", "Total",
        "Amount", "Dear", "Sir", "From", "The",
        # banking noise
        "Account", "Statement", "Balance", "Credit",
        "Debit", "Opening", "Closing", "Salary",
        "Canara", "Branch", "Number", "Period",
        "Description", "Arrears", "Place", "Delhi",
        "Record", "Certificate", "Owner", "Survey",
        "Registrar", "Office", "Registration", "Area",
        "Market", "Value", "Dwarka", "Certified",
        "Connaught", "Total Credits", "Account Holder",
        "Account Statement", "Account Number",
        "Salary Credit", "Salary Arrears",
        "Opening Balance", "Closing Balance",
        "New Delhi", "Canara Bank",
    }

    names = list(set(
        n for n in raw_names
        if n not in noise          # not a noise word
        and len(n) > 4             # not too short
        and "\n" not in n          # no OCR multi-line artifacts
        and len(n.split()) <= 4    # max 4 words (real names)
        and not any(
            noise_word in n
            for noise_word in noise
        )                          # doesn't contain noise words
    ))
    return {
        "pan_numbers": pan_numbers,
        "dates"      : dates,
        "amounts"    : amounts,
        "phones"     : phones,
        "aadhaar"    : aadhaar,
        "names"      : names,
    }