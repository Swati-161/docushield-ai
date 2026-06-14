import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from backend.config import MODELS_DIR

IMG_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_model       = None
_device      = None
_model_ready = False


def _load_model():
    """
    Loads the trained EfficientNet model from disk.
    Called once on first use — subsequent calls reuse the loaded model.
    """
    global _model, _device, _model_ready

    model_path = MODELS_DIR / "efficientnet_forgery.pth"

    if not model_path.exists():
        print(f"[CNN] WARNING: Model file not found at {model_path}")
        print("[CNN] Returning fallback scores. Train the model first.")
        _model_ready = False
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CNN] Loading model on: {_device}")

    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(
        torch.load(str(model_path), map_location=_device)
    )
    model.eval()
    model.to(_device)

    _model       = model
    _model_ready = True
    print("[CNN] Model loaded successfully")


def run_cnn(image_path: str) -> dict:
    """
    Runs the CNN forgery classifier on a single image.

    Returns:
        dict with score (0.0–1.0), prediction, and confidence
        score close to 1.0 = likely tampered
        score close to 0.0 = likely authentic
    """
    global _model, _device, _model_ready

    if _model is None:
        _load_model()

    if not _model_ready:
        return {
            "detector"      : "cnn",
            "score"         : 0.5,
            "prediction"    : "unknown",
            "confidence"    : 0.0,
            "interpretation": "Model not loaded — train and add weights file",
            "fallback"      : True,
        }

    try:
        img = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(_device)

        with torch.no_grad():
            outputs     = _model(tensor)
            probs       = torch.softmax(outputs, dim=1)[0]
            tamper_prob = float(probs[1])
            auth_prob   = float(probs[0])

        prediction = "tampered" if tamper_prob > 0.5 else "authentic"
        confidence = max(tamper_prob, auth_prob)
        score      = round(tamper_prob, 4)

        if score < 0.35:
            interpretation = "Authentic — CNN sees no signs of manipulation"
        elif score < 0.60:
            interpretation = "Uncertain — CNN detects weak tampering signals"
        else:
            interpretation = "Tampered — CNN is confident this image was manipulated"

        return {
            "detector"      : "cnn",
            "score"         : score,
            "prediction"    : prediction,
            "confidence"    : round(confidence, 4),
            "interpretation": interpretation,
            "fallback"      : False,
        }

    except Exception as e:
        return {
            "detector"      : "cnn",
            "score"         : 0.5,
            "prediction"    : "error",
            "confidence"    : 0.0,
            "interpretation": f"CNN error: {str(e)}",
            "fallback"      : True,
        }