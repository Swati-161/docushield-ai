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
from backend.detectors.ocr_extractor import extract_text_from_pages
from backend.detectors.entity_extractor import extract_entities
from backend.detectors.consistency_checker import run_consistency_check
from backend.detectors.arithmetic_validator import run_arithmetic_check
from backend.utils.fusion_engine import compute_risk_score
from backend.database import save_result, get_result, get_all_results

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

@router.post("/analyze-nlp")
async def analyze_nlp(payload: dict):
    """
    Runs the full NLP pipeline on one or more documents.

    Expects JSON body:
    {
      "job_id": "a3f9b2c1",
      "documents": {
        "land_record": {
          "image_paths": ["path/to/page_1_clean.png"]
        },
        "bank_statement": {
          "image_paths": ["path/to/page_1_clean.png",
                          "path/to/page_2_clean.png"]
        }
      }
    }

    Returns OCR text, extracted entities, consistency
    flags, and arithmetic checks for each document.
    """
    job_id    = payload.get("job_id")
    documents = payload.get("documents", {})

    if not job_id or not documents:
        raise HTTPException(
            status_code=400,
            detail="job_id and documents are required"
        )

    doc_results  = {}
    all_entities = {}

    for doc_label, doc_data in documents.items():
        image_paths = doc_data.get("image_paths", [])
        if not image_paths:
            continue

        print(f"[{job_id}] OCR on: {doc_label}")

        ocr_result  = extract_text_from_pages(image_paths)
        entities    = extract_entities(ocr_result["full_text"])
        arith_check = run_arithmetic_check(ocr_result["full_text"])

        doc_results[doc_label] = {
            "ocr"        : ocr_result,
            "entities"   : entities,
            "arithmetic" : arith_check,
        }

        all_entities[doc_label] = {
            k: v for k, v in entities.items()
            if v
        }

    print(f"[{job_id}] Running cross-document consistency check...")
    consistency = run_consistency_check(all_entities)

    worst_arith = max(
        (doc_results[d]["arithmetic"]["score"]
         for d in doc_results),
        default=0.0
    )

    return {
        "job_id"          : job_id,
        "document_results": doc_results,
        "consistency"     : consistency,
        "summary": {
            "worst_arithmetic_score"  : round(worst_arith, 4),
            "consistency_score"       : consistency["score"],
            "total_inconsistencies"   : len(
                consistency.get("inconsistencies", [])
            ),
        }
    }

@router.post("/analyze")
async def analyze_document(
    file         : UploadFile = File(...),
    doc_label    : str        = "document",
):
    """
    Master endpoint — runs the complete DocuShield pipeline.

    Upload a PDF or image, get back a full risk analysis.

    Args:
        file      : the document file to analyse
        doc_label : what type of document this is
                    e.g. "bank_statement", "land_record", "itr"

    Returns the complete analysis result including:
        - risk_score (0-100)
        - risk_level (low / medium / high)
        - verdict
        - per-detector scores
        - inconsistencies list
        - heatmap image paths
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    job_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*50}")
    print(f"[{job_id}] New analysis — {file.filename} ({doc_label})")
    print(f"{'='*50}")

    # ── Step 1: Save uploaded file ──────────────────────
    suffix     = Path(file.filename).suffix
    saved_path = SAMPLE_DOCS_DIR / f"{job_id}{suffix}"
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ── Step 2: Convert to images ───────────────────────
    try:
        if file.content_type == "application/pdf":
            image_paths = pdf_to_images(str(saved_path))
        else:
            job_dir   = OUTPUTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            out_path  = str(job_dir / "page_1.png")
            shutil.copy(str(saved_path), out_path)
            image_paths = [out_path]
            
        preprocessed = []
        for img_path in image_paths:
            img        = preprocess_image(img_path)
            clean_path = img_path.replace(".png", "_clean.png")
            save_preprocessed(img, clean_path)
            preprocessed.append(clean_path)

        print(f"[{job_id}] Preprocessed {len(preprocessed)} page(s)")

        # ── Step 3: CV layer ────────────────────────────
        print(f"[{job_id}] Running CV detectors...")
        cv_page_results = []
        for i, img_path in enumerate(preprocessed, 1):
            ela_r  = run_ela(img_path, job_id)
            cnn_r  = run_cnn(img_path)
            font_r = run_font_detector(img_path)
            cv_page_results.append({
                "page": i,
                "ela" : ela_r,
                "cnn" : cnn_r,
                "font": font_r,
            })

        worst_ela  = max(r["ela"]["score"]  for r in cv_page_results)
        worst_cnn  = max(r["cnn"]["score"]  for r in cv_page_results)
        worst_font = max(r["font"]["score"] for r in cv_page_results)
        cnn_fallback = any(
            r["cnn"].get("fallback", False)
            for r in cv_page_results
        )

        worst_page_ela = max(
            cv_page_results,
            key=lambda r: r["ela"]["score"]
        )
        heatmap_path = worst_page_ela["ela"].get("heatmap_path")
        overlay_path = worst_page_ela["ela"].get("overlay_path")

        # ── Step 4: NLP layer ───────────────────────────
        print(f"[{job_id}] Running NLP detectors...")
        ocr_result  = extract_text_from_pages(preprocessed)
        entities    = extract_entities(ocr_result["full_text"])
        arith_check = run_arithmetic_check(ocr_result["full_text"])
        doc_entities = {
            doc_label: {
                k: v for k, v in entities.items() if v
            }
        }
        consistency = run_consistency_check(doc_entities)

        all_flags = list(arith_check.get("flags", []))
        all_flags += [
            {
                "type"       : inc.get("type", "inconsistency"),
                "severity"   : inc.get("severity", "medium"),
                "description": inc.get("description", ""),
            }
            for inc in consistency.get("inconsistencies", [])
        ]

        # ── Step 5: Fusion ──────────────────────────────
        print(f"[{job_id}] Computing final risk score...")
        fusion = compute_risk_score(
            ela_score         = worst_ela,
            cnn_score         = worst_cnn,
            font_score        = worst_font,
            consistency_score = consistency["score"],
            arithmetic_score  = arith_check["score"],
            cnn_is_fallback   = cnn_fallback,
        )

        print(f"[{job_id}] FINAL SCORE: {fusion['risk_score']} "
              f"({fusion['risk_level'].upper()})")

        # ── Step 6: Save to database ────────────────────
        save_result({
            "job_id"           : job_id,
            "filename"         : file.filename,
            "risk_score"       : fusion["risk_score"],
            "risk_level"       : fusion["risk_level"],
            "risk_label"       : fusion["risk_label"],
            "verdict"          : fusion["verdict"],
            "ela_score"        : worst_ela,
            "cnn_score"        : worst_cnn,
            "font_score"       : worst_font,
            "consistency_score": consistency["score"],
            "arithmetic_score" : arith_check["score"],
            "inconsistencies"  : consistency.get("inconsistencies", []),
            "flags"            : all_flags,
            "heatmap_path"     : heatmap_path,
            "overlay_path"     : overlay_path,
        })

        # ── Step 7: Return full result ──────────────────
        return {
            "job_id"      : job_id,
            "filename"    : file.filename,
            "doc_label"   : doc_label,
            "pages"       : len(preprocessed),
            "risk_score"  : fusion["risk_score"],
            "risk_level"  : fusion["risk_level"],
            "risk_label"  : fusion["risk_label"],
            "risk_color"  : fusion["risk_color"],
            "verdict"     : fusion["verdict"],
            "breakdown"   : fusion["breakdown"],
            "raw_scores"  : fusion["raw_scores"],
            "flags"       : all_flags,
            "inconsistencies"   : consistency.get("inconsistencies", []),
            "entities"          : entities,
            "heatmap_url" : f"/outputs/{job_id}/"
                            f"{Path(heatmap_path).name}"
                            if heatmap_path else None,
            "overlay_url" : f"/outputs/{job_id}/"
                            f"{Path(overlay_path).name}"
                            if overlay_path else None,
            "cv_page_results"   : cv_page_results,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{job_id}")
async def get_analysis_result(job_id: str):
    """
    Fetches a previously saved analysis result by job_id.
    Useful if the frontend needs to reload a past result.
    """
    result = get_result(job_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No result found for job_id: {job_id}"
        )
    return result

@router.get("/history")
async def get_history():
    """Returns the last 50 analysis results — for the dashboard history panel."""
    return {"results": get_all_results()}

@router.post("/analyze-multi")
async def analyze_multi(
    files     : list[UploadFile] = File(...),
    doc_labels: str = "bank_statement,land_record"
):
    """
    Accepts multiple documents at once and runs the full
    pipeline across all of them together.
    doc_labels: comma-separated labels matching file order
    e.g. "bank_statement,land_record"
    """
    labels = [l.strip() for l in doc_labels.split(",")]

    if len(files) != len(labels):
        raise HTTPException(
            status_code=400,
            detail=f"Number of files ({len(files)}) must match "
                   f"number of labels ({len(labels)})"
        )

    job_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*50}")
    print(f"[{job_id}] Multi-doc analysis — {len(files)} documents")
    print(f"{'='*50}")

    all_preprocessed  = {}
    all_cv_results    = {}
    all_ocr_texts     = {}
    all_entities      = {}
    all_flags         = []
    worst_ela         = 0.0
    worst_cnn         = 0.0
    worst_font        = 0.0
    worst_arith       = 0.0
    cnn_fallback      = False
    first_heatmap     = None
    first_overlay     = None

    for file, label in zip(files, labels):
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type for {file.filename}"
            )

        suffix     = Path(file.filename).suffix
        saved_path = SAMPLE_DOCS_DIR / f"{job_id}_{label}{suffix}"
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.content_type == "application/pdf":
            image_paths = pdf_to_images(str(saved_path))
        else:
            job_dir  = OUTPUTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(job_dir / f"{label}_page_1.png")
            shutil.copy(str(saved_path), out_path)
            image_paths = [out_path]

        preprocessed = []
        for img_path in image_paths:
            img        = preprocess_image(img_path)
            clean_path = img_path.replace(".png", "_clean.png")
            save_preprocessed(img, clean_path)
            preprocessed.append(clean_path)

        all_preprocessed[label] = preprocessed

        print(f"[{job_id}] CV on: {label}")
        for img_path in preprocessed:
            ela_r  = run_ela(img_path, job_id)
            cnn_r  = run_cnn(img_path)
            font_r = run_font_detector(img_path)

            worst_ela  = max(worst_ela,  ela_r["score"])
            worst_cnn  = max(worst_cnn,  cnn_r["score"])
            worst_font = max(worst_font, font_r["score"])
            if cnn_r.get("fallback"):
                cnn_fallback = True
            if not first_heatmap:
                first_heatmap = ela_r.get("heatmap_path")
                first_overlay = ela_r.get("overlay_path")

        all_cv_results[label] = {
            "ela" : ela_r,
            "cnn" : cnn_r,
            "font": font_r
        }

        print(f"[{job_id}] NLP on: {label}")
        ocr_result  = extract_text_from_pages(preprocessed)
        entities    = extract_entities(ocr_result["full_text"])
        arith_check = run_arithmetic_check(ocr_result["full_text"])

        worst_arith = max(worst_arith, arith_check["score"])
        all_flags  += arith_check.get("flags", [])
        all_ocr_texts[label]  = ocr_result["full_text"]
        all_entities[label]   = {k: v for k, v in entities.items() if v}

    print(f"[{job_id}] Entities being sent to LLM: {all_entities}")
    
    print(f"[{job_id}] Cross-document consistency check...")
    consistency = run_consistency_check(all_entities)
    all_flags  += [
        {
            "type"       : i.get("type", "inconsistency"),
            "severity"   : i.get("severity", "medium"),
            "description": i.get("description", ""),
        }
        for i in consistency.get("inconsistencies", [])
    ]

    print(f"[{job_id}] Raw scores — "
          f"ELA: {worst_ela:.3f} | CNN: {worst_cnn:.3f} | "
          f"Font: {worst_font:.3f} | "
          f"Consistency: {consistency['score']:.3f} | "
          f"Arithmetic: {worst_arith:.3f}")

    fusion = compute_risk_score(
        ela_score         = worst_ela,
        cnn_score         = worst_cnn,
        font_score        = worst_font,
        consistency_score = consistency["score"],
        arithmetic_score  = worst_arith,
        cnn_is_fallback   = cnn_fallback,
    )

    print(f"[{job_id}] FINAL SCORE: {fusion['risk_score']} "
          f"({fusion['risk_level'].upper()})")

    save_result({
        "job_id"           : job_id,
        "filename"         : " + ".join(f.filename for f in files),
        "risk_score"       : fusion["risk_score"],
        "risk_level"       : fusion["risk_level"],
        "risk_label"       : fusion["risk_label"],
        "verdict"          : fusion["verdict"],
        "ela_score"        : worst_ela,
        "cnn_score"        : worst_cnn,
        "font_score"       : worst_font,
        "consistency_score": consistency["score"],
        "arithmetic_score" : worst_arith,
        "inconsistencies"  : consistency.get("inconsistencies", []),
        "flags"            : all_flags,
        "heatmap_path"     : first_heatmap,
        "overlay_path"     : first_overlay,
    })

    return {
        "job_id"         : job_id,
        "documents"      : list(all_entities.keys()),
        "risk_score"     : fusion["risk_score"],
        "risk_level"     : fusion["risk_level"],
        "risk_label"     : fusion["risk_label"],
        "risk_color"     : fusion["risk_color"],
        "verdict"        : fusion["verdict"],
        "breakdown"      : fusion["breakdown"],
        "raw_scores"     : fusion["raw_scores"],
        "flags"          : all_flags,
        "inconsistencies": consistency.get("inconsistencies", []),
        "all_entities"   : all_entities,
        "heatmap_url"    : f"/outputs/{job_id}/"
                           f"{Path(first_heatmap).name}"
                           if first_heatmap else None,
        "overlay_url"    : f"/outputs/{job_id}/"
                           f"{Path(first_overlay).name}"
                           if first_overlay else None,
    }