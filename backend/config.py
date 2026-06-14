from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR        = BASE_DIR / "data"
OUTPUTS_DIR     = DATA_DIR / "outputs"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
SEALS_DIR       = DATA_DIR / "reference_seals"
MODELS_DIR      = BASE_DIR / "models"

for d in [OUTPUTS_DIR, SAMPLE_DOCS_DIR, SEALS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_URL      = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:3b"

ELA_QUALITY     = 75
RISK_WEIGHTS = {
    "ela":           0.20,
    "cnn":           0.35,
    "font":          0.15,
    "inconsistency": 0.20,
    "arithmetic":    0.10,
}

RISK_THRESHOLDS = {
    "low":    30,
    "medium": 60,
}