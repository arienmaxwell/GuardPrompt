# semantic_guard.py
"""
Semantic Guard for Mac (Llama-3-1B)
Classifies prompts: SAFE | PROMPT_INJECTION | DATA_LEAKAGE | MALICIOUS | OTHER
Works natively on Apple Silicon using PyTorch Metal backend.
"""

from typing import Dict, Any
import os, re, traceback
from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()

# Use smaller Llama model for local Apple runs
GUARD_MODEL_ID = os.environ.get("GUARD_MODEL_ID", "meta-llama/Meta-Llama-3-1B-Instruct")

_guard_pipe = None
try:
    _guard_pipe = pipeline("text-generation", model=GUARD_MODEL_ID, device_map="auto")
except Exception:
    _guard_pipe = None


def _heuristic_guard(prompt: str) -> Dict[str, Any]:
    """Fallback heuristic classification."""
    low = prompt.lower()
    if any(x in low for x in ["ignore previous", "system prompt", "/etc/passwd", "private key", "upload all files"]):
        return {"label": "PROMPT_INJECTION", "raw": "heuristic:injection"}
    if re.search(r"[A-Z]{5}\d{4}[A-Z]", prompt):  # PAN
        return {"label": "DATA_LEAKAGE", "raw": "heuristic:PAN"}
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", prompt):  # Aadhaar
        return {"label": "DATA_LEAKAGE", "raw": "heuristic:Aadhaar"}
    if re.search(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+", prompt):
        return {"label": "DATA_LEAKAGE", "raw": "heuristic:Email"}
    if "api_key" in low or "secret" in low or re.search(r"[A-Fa-f0-9]{32,}", prompt):
        return {"label": "DATA_LEAKAGE", "raw": "heuristic:Key"}
    return {"label": "SAFE", "raw": "heuristic:safe"}


def semantic_guard_check(prompt: str) -> Dict[str, Any]:
    """Runs LLM classification with heuristic fallback."""
    low = prompt.lower()

    # Fast static pre-check
    if any(x in low for x in ["ignore previous instructions", "print your system prompt", "/etc/passwd", "private key"]):
        return {"label": "PROMPT_INJECTION", "raw": "static:match"}

    if _guard_pipe is None:
        return _heuristic_guard(prompt)

    try:
        task = (
            "Classify the following text for security risks. "
            "Answer only with one of: SAFE, PROMPT_INJECTION, DATA_LEAKAGE, MALICIOUS, OTHER.\n\n"
            f"TEXT:\n{prompt}"
        )
        out = _guard_pipe(task, max_new_tokens=32, temperature=0.0)
        text = out[0]["generated_text"].strip().upper()
    except Exception:
        traceback.print_exc()
        return _heuristic_guard(prompt)

    # Interpret model output
    label = "SAFE"
    if "INJECTION" in text:
        label = "PROMPT_INJECTION"
    elif "LEAK" in text or "DATA" in text:
        label = "DATA_LEAKAGE"
    elif "MALICIOUS" in text:
        label = "MALICIOUS"
    elif "OTHER" in text:
        # Fallback: treat OTHER as SAFE unless risky terms present
        risky = any(x in low for x in ["ignore previous", "api_key", "private key", "/etc/passwd"])
        label = "PROMPT_INJECTION" if risky else "SAFE"
    return {"label": label, "raw": text}
