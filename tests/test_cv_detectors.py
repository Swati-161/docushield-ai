import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import tempfile
import os

from backend.detectors.ela_detector  import run_ela
from backend.detectors.cnn_detector  import run_cnn
from backend.detectors.font_detector import run_font_detector


def create_test_image(path: str):
    """Creates a synthetic document image with text for testing."""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255

    for i, line in enumerate([
        "Test Document - DocuShield AI",
        "Applicant Name: Rahul Sharma",
        "PAN Number:     ABCDE1234F",
        "Income:         Rs. 75,000",
        "Loan Amount:    Rs. 5,00,000",
        "Date:           12-Jan-2024",
    ]):
        cv2.putText(img, line, (50, 100 + i * 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)

    cv2.imwrite(path, img)


def run_tests():
    print("=" * 55)
    print("Phase 3 — CV Detectors Test")
    print("=" * 55)

    with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False) as f:
        test_img_path = f.name

    try:
        print("\n[0] Creating test image...")
        create_test_image(test_img_path)
        print(f"    Created: {test_img_path}")

        print("\n[1] Testing ELA detector...")
        ela = run_ela(test_img_path, "test_job")
        assert "score" in ela,         "Missing score"
        assert 0.0 <= ela["score"] <= 1.0, "Score out of range"
        assert "heatmap_path" in ela,  "Missing heatmap_path"
        assert Path(ela["heatmap_path"]).exists(), "Heatmap not saved"
        print(f"    Score: {ela['score']}  →  {ela['interpretation']}")
        print("    PASSED")

        print("\n[2] Testing CNN detector...")
        cnn = run_cnn(test_img_path)
        assert "score" in cnn,          "Missing score"
        assert 0.0 <= cnn["score"] <= 1.0, "Score out of range"
        assert "prediction" in cnn,     "Missing prediction"
        print(f"    Score: {cnn['score']}  →  {cnn['interpretation']}")
        if cnn.get("fallback"):
            print("    (fallback mode — model not trained yet, that is OK)")
        print("    PASSED")

        print("\n[3] Testing font detector...")
        font = run_font_detector(test_img_path)
        assert "score" in font,              "Missing score"
        assert 0.0 <= font["score"] <= 1.0,  "Score out of range"
        assert "flagged_regions" in font,    "Missing flagged_regions"
        print(f"    Score: {font['score']}  →  {font['interpretation']}")
        print("    PASSED")

        print("\n" + "=" * 55)
        print("All Phase 3 tests PASSED!")
        print("=" * 55)

    finally:
        os.unlink(test_img_path)


if __name__ == "__main__":
    run_tests()