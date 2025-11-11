# guardprompt.py
"""
GuardPrompt SDK
--------------------------------------
Provides:
 - Prompt sanitization (regex, keyword, entropy)
 - Liability detection and improved prompt generation
 - Risk scoring and policy enforcement
 - Guarded LLM call wrapper for secure generation
 - JSON audit logging
 - Safe demo fallback response
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import json
import time
import uuid
import unicodedata
import math
import base64
import difflib
import sys

# Optional spaCy for NER redaction (not required)
try:
    import spacy
    _SPACY_AVAILABLE = True
    _NLP = spacy
except Exception:
    _SPACY_AVAILABLE = False
    _NLP = None

# Default Policy
DEFAULT_POLICY = {
    "max_prompt_tokens": 4000,
    "disallowed_keywords": [
        "ssh_private_key", "-----BEGIN PRIVATE KEY-----", "PASSWORD=",
        "API_KEY", "SECRET", "token:", "bearer ", "private_key",
        "/etc/passwd", "send all files"
    ],
    "sensitive_regexes": [
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",  # Email
        r"(?:[A-Za-z]\:)?(?:\\|/)(?:[\w\-\s\.]+(?:\\|/))*[\w\-\s\.]+",  # File path
        r"[A-Fa-f0-9]{32,}",  # Long hex (API keys)
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"  # IP address
    ],
    "block_if_contains": [
        "exfiltrate", "ignore previous instructions", "upload all content",
        "send all files", "download /", "open /etc", "output the contents of"
    ],
    "allow_after_redaction": True,
    "max_redactions": 10,
    "min_risk_to_block": 75,
    "min_risk_to_review": 40
}

# Normalization
_INVISIBLE_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

def normalize_obfuscation(s: str) -> str:
    """Normalize unicode and remove invisible characters."""
    try:
        nfkd = unicodedata.normalize("NFKC", s)
        for c in _INVISIBLE_CHARS:
            nfkd = nfkd.replace(c, "")
        return nfkd
    except Exception:
        return s

def leet_normalize(s: str) -> str:
    try:
        return s.translate(_LEET_MAP)
    except Exception:
        return s

# Entropy & Base64
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def detect_high_entropy_tokens(text: str, min_len: int = 20, entropy_threshold: float = 3.5):
    tokens = re.findall(r"[A-Za-z0-9+/=]{%d,}" % min_len, text)
    results = []
    for t in tokens:
        e = shannon_entropy(t)
        if e >= entropy_threshold:
            results.append((t, round(e, 2)))
    return results

def detect_base64_strings(text: str, min_len: int = 40):
    candidates = re.findall(r"(?:[A-Za-z0-9+/]{%d,}={0,2})" % min_len, text)
    valids = []
    for c in candidates:
        try:
            base64.b64decode(c, validate=True)
            valids.append(c)
        except Exception:
            pass
    return valids

# Fuzzy Keyword Search
def fuzzy_find_keywords(text: str, keywords: List[str], cutoff: float = 0.8):
    found = []
    low = text.lower()
    for kw in keywords:
        if kw.lower() in low:
            found.append((kw, 1.0))
            continue
        for i in range(len(low.split())):
            window = " ".join(low.split()[i:i+6])
            score = difflib.SequenceMatcher(None, kw.lower(), window).ratio()
            if score >= cutoff:
                found.append((kw, score))
                break
    return found

# Sanitizer
class RedactionResult:
    def __init__(self, text: str, redactions: List[Tuple[str, str]]):
        self.text = text
        self.redactions = redactions

def sanitize_prompt(prompt: str, policy: Dict[str, Any]) -> RedactionResult:
    """Redact emails, file paths, hex tokens, etc."""
    redactions = []
    text = prompt
    norm = leet_normalize(normalize_obfuscation(prompt))

    # Regex redaction
    for idx, rx in enumerate(policy.get("sensitive_regexes", [])):
        try:
            pattern = re.compile(rx)
            for m in pattern.finditer(norm):
                found = m.group(0)
                token = f"<REDACTED_{idx}>"
                text = text.replace(found, token)
                redactions.append((found, f"regex:{rx}"))
        except re.error:
            continue

    # Keyword redaction
    for kw in policy.get("disallowed_keywords", []):
        text_new, n = re.subn(re.escape(kw), "<REDACTED_KEYWORD>", text, flags=re.IGNORECASE)
        if n > 0:
            text = text_new
            redactions.append((kw, "keyword_exact"))
        else:
            fuzzy = fuzzy_find_keywords(norm, [kw], 0.82)
            for _kw, score in fuzzy:
                text = re.sub(re.escape(_kw), "<REDACTED_KEYWORD>", text, flags=re.IGNORECASE)
                redactions.append((_kw, f"keyword_fuzzy:{score}"))

    # Entropy / Base64
    for t, e in detect_high_entropy_tokens(norm):
        text = text.replace(t, "<REDACTED_SECRET>")
        redactions.append((t, f"high_entropy:{e}"))
    for b in detect_base64_strings(norm):
        text = text.replace(b, "<REDACTED_BASE64>")
        redactions.append((b, "base64"))

    # spaCy NER (if available)
    if _SPACY_AVAILABLE:
        try:
            nlp = _NLP.load("en_core_web_sm")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "EMAIL"):
                    text = text.replace(ent.text, f"<REDACTED_{ent.label_}>")
                    redactions.append((ent.text, f"ner:{ent.label_}"))
        except Exception:
            pass

    return RedactionResult(text, redactions)

# Risk Scoring
def semantic_risk_score(original: str, sanitized: str, policy: Dict[str, Any]) -> int:
    score = 0
    low = original.lower()
    for b in policy.get("block_if_contains", []):
        if b in low:
            return 100
    redactions_count = len(re.findall(r"<REDACTED", sanitized))
    score += min(60, redactions_count * 5)
    if re.search(r"[A-Fa-f0-9]{32,}", original):
        score += 20
    if re.search(r"/etc/", original):
        score += 20
    if re.search(r"password|api[_-]?key|secret", original, re.IGNORECASE):
        score += 15
    return min(100, score)

# Decision & Evaluation
class Decision:
    def __init__(self, action: str, reason: str, risk_score: int, redactions: List[Tuple[str, str]]):
        self.action = action
        self.reason = reason
        self.risk_score = risk_score
        self.redactions = redactions
        self.review_id = str(uuid.uuid4()) if action == "REVIEW" else None

def evaluate_prompt(prompt: str, policy: Optional[Dict[str, Any]] = None) -> Decision:
    policy = DEFAULT_POLICY if policy is None else {**DEFAULT_POLICY, **policy}
    sanitized = sanitize_prompt(prompt, policy)
    risk = semantic_risk_score(prompt, sanitized.text, policy)
    if len(sanitized.redactions) > policy["max_redactions"]:
        return Decision("REVIEW", "too_many_redactions", risk, sanitized.redactions)
    if risk >= policy["min_risk_to_block"]:
        return Decision("BLOCK", "high_risk", risk, sanitized.redactions)
    if risk >= policy["min_risk_to_review"]:
        return Decision("REVIEW", "medium_risk", risk, sanitized.redactions)
    if sanitized.redactions:
        return Decision("REDACTED_ALLOW", "redacted_sensitive", risk, sanitized.redactions)
    return Decision("ALLOW", "ok", risk, sanitized.redactions)

# Audit Logger
class JSONAuditLogger:
    def __init__(self, filename: str = "guard_audit.log"):
        self.filename = filename

    def log(self, prompt: str, decision: Decision, meta: Optional[Dict[str, Any]] = None):
        try:
            with open(self.filename, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "id": str(uuid.uuid4()),
                    "ts": int(time.time()),
                    "action": decision.action,
                    "reason": decision.reason,
                    "risk_score": decision.risk_score,
                    "redactions": len(decision.redactions),
                    "meta": meta or {}
                }) + "\n")
        except Exception:
            print("Warning: Failed to write audit log", file=sys.stderr)

# Post-Response Leakage Check
def inspect_response_for_leakage(response: str, redactions: List[Tuple[str, str]], policy: Dict[str, Any]):
    for orig, _ in redactions:
        if orig and orig in response:
            return True, "response_contains_redacted_content"
    for rx in policy.get("sensitive_regexes", []):
        try:
            if re.search(rx, response):
                return True, f"regex_leak:{rx}"
        except Exception:
            pass
    if detect_high_entropy_tokens(response):
        return True, "high_entropy_leak"
    return False, None

# Guarded LLM Call Wrapper
def guarded_llm_call(prompt: str, llm_func, policy: Optional[Dict[str, Any]] = None,
                     audit_logger: Optional[JSONAuditLogger] = None, metadata: Optional[Dict[str, Any]] = None):
    policy = DEFAULT_POLICY if policy is None else {**DEFAULT_POLICY, **policy}
    decision = evaluate_prompt(prompt, policy)
    if audit_logger:
        audit_logger.log(prompt, decision, metadata)

    if decision.action in ("BLOCK", "REVIEW"):
        return {"status": decision.action.lower(), "reason": decision.reason, "risk_score": decision.risk_score}

    call_prompt = sanitize_prompt(prompt, policy).text if decision.action == "REDACTED_ALLOW" else prompt
    try:
        response = llm_func(call_prompt)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    leaked, leak_reason = inspect_response_for_leakage(response, decision.redactions, policy)
    if leaked:
        if audit_logger:
            audit_logger.log(f"[LEAK]{response}", Decision("BLOCK", "response_leak", 100, decision.redactions))
        return {"status": "leak_detected", "reason": leak_reason, "risk_score": 100}
    return {"status": "ok", "response": response, "risk_score": decision.risk_score}

# Liability Detection 
def detect_liabilities(prompt: str, policy: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    policy = DEFAULT_POLICY if policy is None else {**DEFAULT_POLICY, **policy}
    findings = []
    for i, rx in enumerate(policy.get("sensitive_regexes", [])):
        try:
            pattern = re.compile(rx)
            for m in pattern.finditer(prompt):
                findings.append({
                    "type": f"REGEX_{i}",
                    "match": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "confidence": 0.9,
                    "recommended_action": "redact"
                })
        except re.error:
            continue
    return findings

# Improved Prompt
def produce_improved_prompt(prompt: str, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = DEFAULT_POLICY if policy is None else {**DEFAULT_POLICY, **policy}
    findings = detect_liabilities(prompt, policy)
    sanitized = sanitize_prompt(prompt, policy)
    if not findings and sanitized.redactions:
        for orig, reason in sanitized.redactions:
            findings.append({
                "type": "SANITIZED",
                "match": orig,
                "start": -1,
                "end": -1,
                "confidence": 0.8,
                "recommended_action": "redact"
            })
    improved = sanitized.text
    remediation = "No explicit sensitive items found." if not findings else "Please review and redact:\n" + "\n".join(
        f"- {f['type']}: {f['match'][:50]}" for f in findings
    )
    return {
        "original_prompt": prompt,
        "improved_prompt": improved,
        "liabilities": findings,
        "remediation": remediation
    }

# Demo Response 
def _fake_llm_response(prompt: str) -> str:
    snippet = prompt[:400]
    return f"[DEMO MODEL] Response to sanitized prompt:\n\n{snippet}\n\n(Truncated demo output)"
