import re
from typing import Optional


def _extract_all_numbers(text: str) -> list[float]:
    """
    Pulls every number from text, ignoring currency symbols.
    e.g. "Rs. 75,000 and Rs. 25,000 Total: Rs. 1,00,000"
    → [75000.0, 25000.0, 100000.0]
    """
    cleaned = re.sub(r'Rs\.?|₹|INR', '', text, flags=re.IGNORECASE)
    raw     = re.findall(r'[\d,]+(?:\.\d{1,2})?', cleaned)
    numbers = []
    for r in raw:
        try:
            val = float(r.replace(',', ''))
            if val > 100:
                numbers.append(val)
        except ValueError:
            continue
    return numbers


def _find_total_keywords(text: str) -> list[float]:
    """
    Finds numbers that appear next to total/sum keywords.
    e.g. "Total: Rs. 1,00,000" or "Net Salary: Rs. 75,000"
    """
    pattern = re.compile(
        r'(?:total|net|gross|sum|balance|salary|income|credit)'
        r'\s*:?\s*(?:Rs\.?|₹|INR)?\s*([\d,]+(?:\.\d{1,2})?)',
        re.IGNORECASE
    )
    found = []
    for match in pattern.finditer(text):
        try:
            val = float(match.group(1).replace(',', ''))
            if val > 100:
                found.append(val)
        except ValueError:
            continue
    return found

def _check_round_number_bias(numbers: list[float]) -> dict:
    """
    Real financial data is messy — genuine salaries are
    things like 73,450 or 68,200, not neat round numbers.
    If too many numbers end in 000, it's suspicious.
    """
    if len(numbers) < 5:
        return {"flagged": False, "reason": "Too few numbers to analyse"}

    round_count = sum(
        1 for n in numbers
        if n >= 1000 and n % 1000 == 0
    )
    round_ratio = round_count / len(numbers)

    if round_ratio > 0.7:
        return {
            "flagged"    : True,
            "round_ratio": round(round_ratio, 2),
            "reason"     : (
                f"{round_count}/{len(numbers)} amounts are suspiciously "
                f"round numbers — possible fabrication"
            )
        }
    return {
        "flagged"    : False,
        "round_ratio": round(round_ratio, 2),
        "reason"     : "Amount distribution looks natural"
    }


def run_arithmetic_check(text: str) -> dict:
    """
    Runs arithmetic validation on text from a financial document.

    Checks:
      1. Round number bias
      2. Whether any stated total is inconsistent with
         the sum of individual line items

    Args:
        text : full OCR text from the document

    Returns:
        dict with score, flags, and interpretation
    """
    all_numbers   = _extract_all_numbers(text)
    total_numbers = _find_total_keywords(text)

    flags = []

    round_check = _check_round_number_bias(all_numbers)
    if round_check["flagged"]:
        flags.append({
            "type"       : "round_number_bias",
            "severity"   : "medium",
            "description": round_check["reason"]
        })

    if len(all_numbers) >= 3 and total_numbers:
        non_total = [
            n for n in all_numbers
            if n not in total_numbers
        ]

        for stated_total in total_numbers:
            for i in range(len(non_total)):
                for j in range(i + 1, len(non_total)):
                    computed = non_total[i] + non_total[j]
                    diff     = abs(computed - stated_total)
                    tolerance = stated_total * 0.02

                    if diff < tolerance and diff > 0:
                        flags.append({
                            "type"       : "arithmetic_mismatch",
                            "severity"   : "high",
                            "description": (
                                f"Stated total {stated_total:,.0f} does not "
                                f"match computed sum {computed:,.0f} "
                                f"(difference: {diff:,.0f})"
                            )
                        })

    high   = sum(1 for f in flags if f.get("severity") == "high")
    medium = sum(1 for f in flags if f.get("severity") == "medium")

    score = min((high * 0.5 + medium * 0.25), 1.0)

    if not flags:
        interpretation = "No arithmetic anomalies detected"
    elif high > 0:
        interpretation = f"Arithmetic errors found — {high} high severity flags"
    else:
        interpretation = f"Minor anomalies — {medium} medium severity flags"

    return {
        "detector"      : "arithmetic",
        "score"         : round(score, 4),
        "flags"         : flags,
        "numbers_found" : len(all_numbers),
        "interpretation": interpretation,
    }