from __future__ import annotations

import re
import unicodedata


_TOKEN_RE = re.compile(r"[a-z0-9%/\-]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?;])\s+")
_CONTACT_RE = re.compile(r"(?:https?://|www\.|[\w.\-]+@[\w.\-]+\.\w+|\+\d[\d\s().-]{5,}|\b(?:tel|phone|mail|email|website)\b)", re.IGNORECASE)
_VTR_SIGNATURE_RE = re.compile(
    r"vtr\s*&?\s*beyond.*?(?:technical\s+data\s+sheet)",
    flags=re.IGNORECASE | re.DOTALL,
)
_TECH_SHEET_RE = re.compile(r"technical\s+data\s+sheet", flags=re.IGNORECASE)


def strip_boilerplate_text(
    text: str,
    mode: str = "hybrid",
    min_remaining_tokens: int = 10,
) -> tuple[str, dict[str, int | bool]]:
    if not text.strip():
        return text, {"boilerplate_removed": False, "boilerplate_removed_chars": 0}

    working = text
    if mode in {"signature", "hybrid"}:
        working = _strip_known_signature(working)

    lines = [ln.strip() for ln in working.splitlines() if ln.strip()]
    if not lines:
        return text, {"boilerplate_removed": False, "boilerplate_removed_chars": 0}

    lead = 0
    while lead < len(lines) and _is_boilerplate_line(lines[lead], mode):
        lead += 1

    tail = len(lines)
    while tail > lead and _is_boilerplate_line(lines[tail - 1], mode):
        tail -= 1

    cleaned_lines = lines[lead:tail]
    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        return text, {"boilerplate_removed": False, "boilerplate_removed_chars": 0}

    if _token_count(cleaned) < max(1, min_remaining_tokens):
        return text, {"boilerplate_removed": False, "boilerplate_removed_chars": 0}

    removed_chars = max(0, len(text) - len(cleaned))
    if removed_chars <= 0:
        return text, {"boilerplate_removed": False, "boilerplate_removed_chars": 0}

    return cleaned, {"boilerplate_removed": True, "boilerplate_removed_chars": removed_chars}


def build_presentable_answer(
    question: str,
    text: str,
    mode: str = "hybrid",
    min_remaining_tokens: int = 10,
    max_chars: int | None = None,
) -> str:
    cleaned, _ = strip_boilerplate_text(
        text=text,
        mode=mode,
        min_remaining_tokens=min_remaining_tokens,
    )
    candidate = cleaned if cleaned.strip() else text
    if _looks_boilerplate_heavy(candidate):
        span = _best_span(question=question, text=text)
        return span
    return _clean_answer_text(candidate)


def _strip_known_signature(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        block = match.group(0)
        tech = _TECH_SHEET_RE.search(block)
        if not tech:
            return ""
        return block[tech.start() : tech.end()]

    return _VTR_SIGNATURE_RE.sub(_replace, text)


def _is_boilerplate_line(line: str, mode: str) -> bool:
    line_norm = _normalize_token(line)
    if not line_norm:
        return False

    signature_hits = 0
    signature_tokens = {
        "vtr",
        "vtrbeyond",
        "pingbei",
        "stresemann",
        "zhuhai",
        "guangdong",
        "berlin",
        "technical",
        "sheet",
    }
    for tok in signature_tokens:
        if f" {tok} " in f" {line_norm} ":
            signature_hits += 1

    generic_hits = len(_CONTACT_RE.findall(line))
    comma_count = line.count(",")

    if mode in {"signature", "hybrid"} and (signature_hits >= 3 and generic_hits >= 1):
        return True
    if mode in {"generic", "hybrid"}:
        if generic_hits >= 2:
            return True
        if generic_hits >= 1 and comma_count >= 4 and len(_tokenize(line_norm)) >= 8:
            return True
    return False


def _looks_boilerplate_heavy(text: str) -> bool:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(" ".join(text.split())) if p.strip()]
    if not parts:
        return False
    noisy = sum(1 for p in parts if _is_boilerplate_line(p, mode="hybrid"))
    return noisy / max(1, len(parts)) >= 0.5


def _best_span(question: str, text: str) -> str:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(" ".join(text.split())) if p.strip()]
    if not parts:
        return _clean_answer_text(text)

    q_tokens = _tokenize(question)
    best = ""
    best_score = -1.0
    for part in parts:
        if _is_boilerplate_line(part, mode="hybrid"):
            continue
        p_tokens = _tokenize(part)
        overlap = len(q_tokens.intersection(p_tokens))
        coverage = overlap / max(1, len(q_tokens))
        has_units = float(any(u in p_tokens for u in {"ppm", "mg/kg", "months", "temperature"}))
        score = coverage + (0.2 * has_units)
        if score > best_score:
            best_score = score
            best = part

    if not best:
        best = parts[0]
    return _clean_answer_text(best)


def _clean_answer_text(text: str) -> str:
    cleaned = _strip_table_artifacts(" ".join(text.split()))
    # Never keep trailing synthetic ellipsis in production answers.
    return re.sub(r"\s*\.\.\.$", "", cleaned).strip()


def _strip_table_artifacts(text: str) -> str:
    # OCR/table extraction often yields noisy separators like "| | | |".
    if "|" not in text:
        return text
    out = text
    out = re.sub(r"(?:\s*\|\s*){2,}", " | ", out)
    out = re.sub(r"\|\s*\|\s*\|+", "|", out)
    out = re.sub(r"\s*\|\s*", "; ", out)
    out = re.sub(r"(?:;\s*){2,}", "; ", out)
    out = re.sub(r"^\s*;\s*|\s*;\s*$", "", out)
    return re.sub(r"\s+", " ", out).strip()


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall(_normalize_token(text)) if tok}


def _token_count(text: str) -> int:
    return len([tok for tok in _TOKEN_RE.findall(_normalize_token(text)) if tok])


def _normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9%/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
