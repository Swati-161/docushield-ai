from paddleocr import PaddleOCR
from pathlib import Path
import numpy as np

_ocr_engine = None


def _get_engine() -> PaddleOCR:
    """
    Loads PaddleOCR once and reuses it.
    Loading it per-request would be very slow (~3 seconds each time).
    """
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR] Loading PaddleOCR engine...")
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
        )
        print("[OCR] Engine ready")
    return _ocr_engine


def extract_text_from_image(image_path: str) -> dict:
    """
    Runs OCR on a single page image.

    Returns:
        {
          "full_text"  : "all text joined as one string",
          "blocks"     : [
              {
                "text"      : "Rahul Sharma",
                "confidence": 0.97,
                "box"       : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
              }, ...
          ],
          "line_count" : 42
        }
    """
    ocr = _get_engine()

    result = ocr.ocr(image_path)

    blocks = []
    lines  = []

    if result and result[0]:
        for line in result[0]:
            box        = line[0]
            text       = line[1][0]
            confidence = float(line[1][1])

            if confidence < 0.5:
                continue

            blocks.append({
                "text"      : text,
                "confidence": round(confidence, 3),
                "box"       : box
            })
            lines.append(text)

    full_text = "\n".join(lines)

    return {
        "full_text" : full_text,
        "blocks"    : blocks,
        "line_count": len(lines)
    }


def extract_text_from_pages(image_paths: list[str]) -> dict:
    """
    Runs OCR on multiple page images and combines results.
    This is what gets called for multi-page PDFs.

    Returns:
        {
          "full_text"      : "combined text from all pages",
          "pages"          : { "1": {...}, "2": {...} },
          "total_lines"    : 120
        }
    """
    all_text  = []
    pages_out = {}
    total     = 0

    for i, path in enumerate(image_paths, start=1):
        print(f"[OCR] Processing page {i}/{len(image_paths)}...")
        result = extract_text_from_image(path)
        pages_out[str(i)] = result
        all_text.append(result["full_text"])
        total += result["line_count"]

    return {
        "full_text"  : "\n\n--- PAGE BREAK ---\n\n".join(all_text),
        "pages"      : pages_out,
        "total_lines": total
    }