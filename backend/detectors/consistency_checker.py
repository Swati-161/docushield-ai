import json
import requests
from backend.config import OLLAMA_URL, OLLAMA_MODEL


SYSTEM_PROMPT = """You are a fraud detection system for a bank.

You will receive a JSON object where each key is a document name
and each value contains entities extracted from that document.

YOUR ONLY JOB: Compare the "names" field across all documents.
If the same person should appear in all documents but the names differ,
that is fraud. Flag it.

STRICT RULES:
1. Compare names across documents character by character
2. "Rahul Kumar" and "Rahul Sharma" are DIFFERENT — flag it
3. "Rahul Sharma" and "Rahul Sharma" are the SAME — do not flag
4. Only output a JSON array — no explanation, no text before or after
5. Do not reference any document that is not in the input data
6. If all names match, return exactly: []

Example of a mismatch you MUST catch:
  bank_statement names: ["Rahul Kumar"]
  land_record names: ["Rahul Sharma"]
  → These are different people — this is fraud — flag it as high severity

Output format — return ONLY this JSON array:
[
  {
    "type": "name_mismatch",
    "severity": "high",
    "description": "Name in bank_statement is 'Rahul Kumar' but name in land_record is 'Rahul Sharma'",
    "documents": ["bank_statement", "land_record"]
  }
]

If no mismatch: []"""

def _python_name_check(documents: dict) -> bool:
    """
    Returns True if a name mismatch is detected across documents.
    Uses simple string comparison — no LLM needed.
    """
    all_names = {}
    for doc_label, entities in documents.items():
        names = entities.get("names", [])
        # only keep short names (1-3 words) — real person names
        person_names = [
            n.strip().lower()
            for n in names
            if len(n.split()) <= 3 and "\n" not in n
        ]
        if person_names:
            all_names[doc_label] = set(person_names)

    if len(all_names) < 2:
        return False

    # check if any name in one doc exists in another doc
    doc_labels = list(all_names.keys())
    for i in range(len(doc_labels)):
        for j in range(i + 1, len(doc_labels)):
            names_a = all_names[doc_labels[i]]
            names_b = all_names[doc_labels[j]]
            # if they share NO names at all — mismatch
            if names_a.isdisjoint(names_b):
                return True

    return False

def _call_ollama(prompt: str) -> str:
    payload = {
        "model" : OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,   # changed from 0.1 to 0.0 — fully deterministic
            "num_predict": 512,
        }
    }
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _parse_llm_response(raw: str) -> list[dict]:
    """
    Safely parses the LLM's JSON response.
    If the LLM adds extra text around the JSON, this still works.
    """
    raw = raw.strip()

    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []

    json_str = raw[start:end]
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        return []


def run_consistency_check(documents: dict) -> dict:
    if not documents:
        return {
            "detector"       : "consistency",
            "score"          : 0.0,
            "inconsistencies": [],
            "interpretation" : "No documents provided",
            "fallback"       : False,
        }

    if len(documents) < 2:
        return {
            "detector"       : "consistency",
            "score"          : 0.0,
            "inconsistencies": [],
            "interpretation" : "Single document — cross-document check requires at least 2",
            "fallback"       : False,
        }

    # ── Python pre-check before calling LLM ────────────
    mismatch_detected = _python_name_check(documents)

    if not mismatch_detected:
        # names match — no need to call LLM at all
        return {
            "detector"       : "consistency",
            "score"          : 0.0,
            "inconsistencies": [],
            "interpretation" : "Names consistent across all documents",
            "fallback"       : False,
        }

    # ── Only reach here if Python detected a mismatch ──
    # Now call LLM to get a detailed description
    prompt = f"""Compare the names across these documents and flag any mismatch:

{json.dumps(documents, indent=2)}

Look at the "names" field in each document.
Are the person names the same or different across documents?
Return a JSON array of mismatches. If all names match return []."""

    try:
        raw_response    = _call_ollama(prompt)
        inconsistencies = _parse_llm_response(raw_response)

        # if LLM returned empty despite Python detecting mismatch,
        # create the flag ourselves
        if not inconsistencies:
            doc_labels = list(documents.keys())
            inconsistencies = [{
                "type"       : "name_mismatch",
                "severity"   : "high",
                "description": f"Name mismatch detected across {', '.join(doc_labels)}",
                "documents"  : doc_labels
            }]

        high_count   = sum(1 for i in inconsistencies if i.get("severity") == "high")
        medium_count = sum(1 for i in inconsistencies if i.get("severity") == "medium")
        low_count    = sum(1 for i in inconsistencies if i.get("severity") == "low")

        score = min(
            (high_count * 0.4 + medium_count * 0.2 + low_count * 0.1),
            1.0
        )

        interpretation = (
            f"Name mismatch found — {high_count} high severity conflicts"
        )

        return {
            "detector"       : "consistency",
            "score"          : round(score, 4),
            "inconsistencies": inconsistencies,
            "high_count"     : high_count,
            "medium_count"   : medium_count,
            "low_count"      : low_count,
            "interpretation" : interpretation,
            "fallback"       : False,
        }

    except requests.exceptions.ConnectionError:
        # Ollama not running but Python detected mismatch — still flag it
        return {
            "detector"       : "consistency",
            "score"          : 0.4,
            "inconsistencies": [{
                "type"       : "name_mismatch",
                "severity"   : "high",
                "description": "Name mismatch detected (Ollama offline — basic check)",
                "documents"  : list(documents.keys())
            }],
            "interpretation" : "Name mismatch detected via basic check",
            "fallback"       : True,
        }
    except Exception as e:
        return {
            "detector"       : "consistency",
            "score"          : 0.0,
            "inconsistencies": [],
            "interpretation" : f"Consistency check error: {str(e)}",
            "fallback"       : True,
        }