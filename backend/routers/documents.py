from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
from backend.config import SAMPLE_DOCS_DIR, OUTPUTS_DIR
from backend.utils.pdf_converter import pdf_to_images
from backend.utils.preprocessor import preprocess_image, save_preprocessed
from backend.detectors.ela_detector  import run_ela
from backend.detectors.cnn_detector  import run_cnn
from backend.detectors.font_detector import run_font_detector


router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/tiff",
}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or image file upload.
    Returns a job_id and list of processed page image paths.

    The frontend calls this endpoint when the user drops a file.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: PDF, JPEG, PNG, TIFF"
        )

    job_id = str(uuid.uuid4())[:8]

    suffix = Path(file.filename).suffix
    saved_path = SAMPLE_DOCS_DIR / f"{job_id}{suffix}"
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[{job_id}] Saved upload: {file.filename} ({file.content_type})")

    try:
        if file.content_type == "application/pdf":
            image_paths = pdf_to_images(str(saved_path))
        else:
            job_dir = OUTPUTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(job_dir / "page_1.png")
            shutil.copy(str(saved_path), out_path)
            image_paths = [out_path]

        preprocessed_paths = []
        for img_path in image_paths:
            img = preprocess_image(img_path)
            clean_path = img_path.replace(".png", "_clean.png")
            save_preprocessed(img, clean_path)
            preprocessed_paths.append(clean_path)

        print(f"[{job_id}] Preprocessed {len(preprocessed_paths)} page(s)")

        return {
            "job_id": job_id,
            "filename": file.filename,
            "pages": len(preprocessed_paths),
            "image_paths": preprocessed_paths,
            "status": "ready_for_analysis"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/analyze-cv")
async def analyze_cv(payload: dict):
    """
    Runs all CV detectors on pre-processed page images.

    Expects JSON body:
        {
          "job_id": "a3f9b2c1",
          "image_paths": ["path/to/page_1_clean.png", ...]
        }

    Returns per-page results from ELA, CNN, and font detector.
    """
    job_id      = payload.get("job_id")
    image_paths = payload.get("image_paths", [])

    if not job_id or not image_paths:
        raise HTTPException(
            status_code=400,
            detail="job_id and image_paths are required"
        )

    all_results = []

    for page_num, img_path in enumerate(image_paths, start=1):
        if not Path(img_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {img_path}"
            )

        print(f"[{job_id}] Analysing page {page_num}...")

        ela_result  = run_ela(img_path, job_id)
        cnn_result  = run_cnn(img_path)
        font_result = run_font_detector(img_path)

        page_result = {
            "page"  : page_num,
            "ela"   : ela_result,
            "cnn"   : cnn_result,
            "font"  : font_result,
        }
        all_results.append(page_result)
        print(f"[{job_id}] Page {page_num} — "
              f"ELA: {ela_result['score']:.2f} | "
              f"CNN: {cnn_result['score']:.2f} | "
              f"Font: {font_result['score']:.2f}")

    worst_ela  = max(r["ela"]["score"]  for r in all_results)
    worst_cnn  = max(r["cnn"]["score"]  for r in all_results)
    worst_font = max(r["font"]["score"] for r in all_results)

    return {
        "job_id"          : job_id,
        "pages_analysed"  : len(all_results),
        "page_results"    : all_results,
        "summary": {
            "worst_ela_score" : round(worst_ela,  4),
            "worst_cnn_score" : round(worst_cnn,  4),
            "worst_font_score": round(worst_font, 4),
        }
    }