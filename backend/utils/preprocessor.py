import cv2
import numpy as np
from pathlib import Path


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Reads an image from disk, cleans it up, and returns
    a numpy array (that's what OpenCV and PyTorch work with).

    Steps:
        1. Read the image
        2. Convert colour format
        3. Resize to standard size
        4. Deskew (straighten if tilted)
        5. Denoise
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    max_dimension = 1600
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_LANCZOS4)

    img = _deskew(img)

    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    return img


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Detects and corrects tilt in a scanned document.
    If the document is tilted by more than 0.5 degrees, it rotates it back.
    Skips correction for very large tilts (probably not a scan issue).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5 or abs(angle) > 30:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def save_preprocessed(img: np.ndarray, output_path: str) -> str:
    """Saves a numpy array image back to disk as a PNG."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)
    return output_path