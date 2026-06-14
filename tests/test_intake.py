"""
Run this to verify Phase 2 works correctly.
It creates a test PDF, converts it to images, and preprocesses them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.pdfgen import canvas as rl_canvas
from backend.utils.pdf_converter import pdf_to_images
from backend.utils.preprocessor import preprocess_image
import tempfile
import os


def create_test_pdf(path: str):
    """Creates a simple 2-page test PDF."""
    c = rl_canvas.Canvas(path)
    c.setFont("Helvetica", 14)
    c.drawString(100, 750, "DocuShield Test Document - Page 1")
    c.drawString(100, 700, "Applicant Name: Rahul Sharma")
    c.drawString(100, 650, "PAN: ABCDE1234F")
    c.drawString(100, 600, "Income: Rs. 75,000")
    c.showPage()
    c.drawString(100, 750, "DocuShield Test Document - Page 2")
    c.drawString(100, 700, "Bank Statement Summary")
    c.drawString(100, 650, "Monthly Credit: Rs. 75,000")
    c.save()
    print(f"Created test PDF: {path}")


def run_tests():
    print("=" * 50)
    print("Phase 2 — Document Intake Pipeline Test")
    print("=" * 50)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        test_pdf_path = f.name

    try:
        print("\n[1] Creating test PDF...")
        create_test_pdf(test_pdf_path)
        print("    PASSED")

        print("\n[2] Converting PDF to images...")
        image_paths = pdf_to_images(test_pdf_path)
        assert len(image_paths) == 2, \
            f"Expected 2 pages, got {len(image_paths)}"
        for p in image_paths:
            assert Path(p).exists(), f"Image not found: {p}"
        print(f"    PASSED — {len(image_paths)} page(s) converted")
        for p in image_paths:
            print(f"    → {p}")

        print("\n[3] Preprocessing images...")
        for i, img_path in enumerate(image_paths, 1):
            img = preprocess_image(img_path)
            assert img is not None, "Preprocessor returned None"
            assert len(img.shape) == 3, "Image should be 3D (H x W x C)"
            h, w, c = img.shape
            print(f"    Page {i}: {w}x{h}px, {c} channels — PASSED")

        print("\n" + "=" * 50)
        print("All Phase 2 tests PASSED!")
        print("=" * 50)

    finally:
        os.unlink(test_pdf_path)


if __name__ == "__main__":
    run_tests()