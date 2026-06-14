import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
from backend.config import OUTPUTS_DIR


def run_ela(image_path: str, job_id: str, quality: int = 75) -> dict:
    """
    Runs Error Level Analysis on a document image.

    How it works:
      1. Re-save the image at a lower JPEG quality
      2. Subtract the re-saved version from the original
      3. Amplify the difference — tampered regions glow bright
      4. Compute a suspicion score from the brightness

    Args:
        image_path : path to the preprocessed page image
        job_id     : unique ID for this analysis job
        quality    : JPEG re-save quality (lower = more sensitive)

    Returns:
        dict with score, heatmap path, and interpretation
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pil_original = Image.fromarray(original_rgb)

    buffer = io.BytesIO()
    pil_original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer))

    original_f  = original_rgb.astype(np.float32)
    recomp_f    = recompressed.astype(np.float32)
    difference  = np.abs(original_f - recomp_f)

    amplified = np.clip(difference * 10, 0, 255).astype(np.uint8)

    ela_gray   = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)
    ela_color  = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), 0.6,
        ela_color, 0.4,
        0
    )

    out_dir = OUTPUTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    page_name  = Path(image_path).stem
    heatmap_path  = str(out_dir / f"{page_name}_ela_heatmap.png")
    overlay_path  = str(out_dir / f"{page_name}_ela_overlay.png")

    cv2.imwrite(heatmap_path, ela_color)
    cv2.imwrite(overlay_path, overlay)

    mean_brightness = float(np.mean(ela_gray))
    max_brightness  = float(np.max(ela_gray))
    bright_pixels   = float(np.sum(ela_gray > 128))
    total_pixels    = ela_gray.size
    bright_ratio    = bright_pixels / total_pixels

    raw_score = (
        (mean_brightness / 255) * 40 +
        (bright_ratio)           * 40 +
        (max_brightness / 255)  * 20
    )
    score = min(round(raw_score, 2), 1.0)

    if score < 0.25:
        interpretation = "Low suspicion — image appears unmodified"
    elif score < 0.55:
        interpretation = "Moderate suspicion — possible editing detected"
    else:
        interpretation = "High suspicion — significant tampering indicators found"

    return {
        "detector"       : "ela",
        "score"          : score,
        "interpretation" : interpretation,
        "mean_brightness": round(mean_brightness, 2),
        "bright_ratio"   : round(bright_ratio, 4),
        "heatmap_path"   : heatmap_path,
        "overlay_path"   : overlay_path,
    }
