from pdf2image import convert_from_path
from pathlib import Path
import uuid
from backend.config import OUTPUTS_DIR


def pdf_to_images(pdf_path: str) -> list[str]:
    """
    Takes a PDF file path.
    Returns a list of image file paths, one per page.

    Example:
        a 3-page PDF → returns 3 image paths
        ["outputs/abc123_page_1.png",
         "outputs/abc123_page_2.png",
         "outputs/abc123_page_3.png"]
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(
        str(pdf_path),
        dpi=200,
        fmt="png"
    )

    saved_paths = []
    for i, page in enumerate(pages, start=1):
        out_path = job_dir / f"page_{i}.png"
        page.save(str(out_path), "PNG")
        saved_paths.append(str(out_path))
        print(f"  Saved page {i} → {out_path}")

    print(f"Converted {len(pages)} page(s) from: {pdf_path.name}")
    return saved_paths