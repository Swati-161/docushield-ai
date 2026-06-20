import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import tempfile
import os

from backend.detectors.entity_extractor     import extract_entities
from backend.detectors.arithmetic_validator  import run_arithmetic_check
from backend.detectors.consistency_checker   import run_consistency_check


SAMPLE_TEXT_CLEAN = """
Loan Application Form
Applicant Name: Rahul Sharma
PAN Number: ABCDE1234F
Date of Birth: 12/05/1990
Mobile: 9876543210
Monthly Income: Rs. 75,000
Loan Amount Requested: Rs. 5,00,000
Property Address: 123 MG Road, Delhi
Date: 15/01/2024
"""

SAMPLE_TEXT_FINANCIAL = """
Bank Statement - January 2024
Account Holder: Rahul Sharma
PAN: ABCDE1234F
Opening Balance: Rs. 1,00,000
Credit 01/01: Rs. 75,000
Credit 15/01: Rs. 75,000
Debit  10/01: Rs. 20,000
Debit  20/01: Rs. 15,000
Closing Balance: Rs. 2,15,000
Total Credits: Rs. 1,50,000
"""

SAMPLE_TEXT_MISMATCHED = """
Land Record
Owner Name: Rahul Kumar
PAN: ABCDE1234F
Survey Number: 123/A
Date of Registration: 10/01/2024
Market Value: Rs. 50,00,000
"""


def run_tests():
    print("=" * 55)
    print("Phase 4 — NLP Detectors Test")
    print("=" * 55)

    print("\n[1] Testing entity extractor...")
    entities = extract_entities(SAMPLE_TEXT_CLEAN)
    assert "ABCDE1234F" in entities["pan_numbers"], \
        "PAN not found"
    assert len(entities["dates"]) > 0, \
        "No dates found"
    assert len(entities["amounts"]) > 0, \
        "No amounts found"
    assert len(entities["names"]) > 0, \
        "No names found"
    print(f"    PANs found    : {entities['pan_numbers']}")
    print(f"    Dates found   : {entities['dates']}")
    print(f"    Amounts found : {[a['raw'] for a in entities['amounts']]}")
    print(f"    Names found   : {entities['names']}")
    print("    PASSED")

    print("\n[2] Testing arithmetic validator (clean doc)...")
    arith_clean = run_arithmetic_check(SAMPLE_TEXT_FINANCIAL)
    assert "score" in arith_clean, "Missing score"
    assert 0.0 <= arith_clean["score"] <= 1.0, "Score out of range"
    print(f"    Score : {arith_clean['score']}")
    print(f"    Flags : {arith_clean['flags']}")
    print(f"    → {arith_clean['interpretation']}")
    print("    PASSED")

    print("\n[3] Testing arithmetic validator (round numbers)...")
    round_text = """
    Salary: Rs. 50,000
    Bonus:  Rs. 10,000
    Total:  Rs. 60,000
    Other:  Rs. 20,000
    Grand Total: Rs. 80,000
    """
    arith_round = run_arithmetic_check(round_text)
    print(f"    Score : {arith_round['score']}")
    print(f"    Flags : {arith_round['flags']}")
    print("    PASSED")

    print("\n[4] Testing consistency checker (with mismatch)...")
    doc_entities = {
        "loan_form": {
            "names"      : ["Rahul Sharma"],
            "pan_numbers": ["ABCDE1234F"],
            "dates"      : ["15/01/2024"],
            "amounts"    : [{"raw": "Rs. 5,00,000", "value": 500000}]
        },
        "land_record": {
            "names"      : ["Rahul Kumar"],
            "pan_numbers": ["ABCDE1234F"],
            "dates"      : ["10/01/2024"],
            "amounts"    : [{"raw": "Rs. 50,00,000", "value": 5000000}]
        }
    }
    consistency = run_consistency_check(doc_entities)
    assert "score" in consistency, "Missing score"
    assert 0.0 <= consistency["score"] <= 1.0, "Score out of range"
    print(f"    Score            : {consistency['score']}")
    print(f"    Inconsistencies  : {len(consistency['inconsistencies'])}")
    for inc in consistency["inconsistencies"]:
        print(f"    → [{inc.get('severity','?').upper()}] "
              f"{inc.get('description','')}")
    if consistency.get("fallback"):
        print("    (fallback — Ollama not running, start with: ollama serve)")
    print("    PASSED")

    print("\n" + "=" * 55)
    print("All Phase 4 tests PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    run_tests()