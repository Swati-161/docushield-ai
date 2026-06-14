import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FontRegion:
    x: int
    y: int
    w: int
    h: int
    mean_intensity: float
    stroke_width:   float
    aspect_ratio:   float


def run_font_detector(image_path: str) -> dict:
    """
    Detects font inconsistencies in a document image.

    Strategy:
      1. Find all text-like connected components (blobs)
      2. Measure stroke width and intensity for each
      3. Compute the median values across all components
      4. Flag components that deviate significantly from median
         (these are likely replacement text)

    Returns:
        dict with score, flagged region count, and locations
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    h_img, w_img = gray.shape
    min_area = (h_img * w_img) * 0.00005
    max_area = (h_img * w_img) * 0.005
    regions: list[FontRegion] = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if not (min_area < area < max_area):
            continue

        aspect = w / h if h > 0 else 0
        if not (0.1 < aspect < 5.0):
            continue

        mask           = (labels == i).astype(np.uint8)
        component_gray = gray * mask
        nonzero_pixels = component_gray[component_gray > 0]

        if len(nonzero_pixels) < 5:
            continue

        mean_intensity = float(np.mean(nonzero_pixels))
        stroke_width   = float(np.mean(
            cv2.distanceTransform(mask, cv2.DIST_L2, 5)[mask > 0]
        )) if np.any(mask) else 0.0

        regions.append(FontRegion(
            x=x, y=y, w=w, h=h,
            mean_intensity=mean_intensity,
            stroke_width=stroke_width,
            aspect_ratio=aspect
        ))

    if len(regions) < 10:
        return {
            "detector"          : "font",
            "score"             : 0.0,
            "flagged_regions"   : 0,
            "total_regions"     : len(regions),
            "interpretation"    : "Not enough text regions to analyse",
            "suspicious_boxes"  : [],
        }

    intensities   = np.array([r.mean_intensity   for r in regions])
    stroke_widths = np.array([r.stroke_width      for r in regions])

    med_int    = np.median(intensities)
    med_stroke = np.median(stroke_widths)
    std_int    = np.std(intensities)    + 1e-6
    std_stroke = np.std(stroke_widths)  + 1e-6

    suspicious_boxes = []
    for r in regions:
        int_dev    = abs(r.mean_intensity - med_int)    / std_int
        stroke_dev = abs(r.stroke_width   - med_stroke) / std_stroke

        if int_dev > 2.5 or stroke_dev > 2.5:
            suspicious_boxes.append({
                "x": r.x, "y": r.y,
                "w": r.w, "h": r.h,
                "intensity_deviation": round(int_dev, 2),
                "stroke_deviation":    round(stroke_dev, 2),
            })

    flagged = len(suspicious_boxes)
    ratio   = flagged / len(regions) if regions else 0

    if ratio < 0.03:
        score = 0.1
        interpretation = "Font appears consistent throughout the document"
    elif ratio < 0.08:
        score = 0.4
        interpretation = f"Minor font inconsistencies found ({flagged} regions)"
    else:
        score = 0.8
        interpretation = (
            f"Significant font inconsistencies — {flagged} suspicious "
            f"regions out of {len(regions)} total. Possible text replacement."
        )

    return {
        "detector"        : "font",
        "score"           : round(score, 4),
        "flagged_regions" : flagged,
        "total_regions"   : len(regions),
        "interpretation"  : interpretation,
        "suspicious_boxes": suspicious_boxes[:20],
    }