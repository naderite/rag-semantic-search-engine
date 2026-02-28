from __future__ import annotations

import re
import unicodedata

from .answer_cleaning import build_presentable_answer


def normalize_unit_string(text: str) -> str:
    out = text
    out = out.replace("–", "-").replace("—", "-")
    out = re.sub(r"\bper\s*cent\b", "%", out, flags=re.IGNORECASE)
    out = re.sub(r"\bpercent(age)?\b", "%", out, flags=re.IGNORECASE)
    out = re.sub(r"\bp\.?\s*p\.?\s*m\.?\b", "ppm", out, flags=re.IGNORECASE)
    out = re.sub(r"\bparts?\s+per\s+million\b", "ppm", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmg\s*/?\s*kg\b", "mg/kg", out, flags=re.IGNORECASE)
    out = re.sub(r"\bg\s*/?\s*kg\b", "g/kg", out, flags=re.IGNORECASE)
    out = re.sub(r"\bkg\s*/?\s*t(?:on(?:ne)?)?\b", "kg/t", out, flags=re.IGNORECASE)
    out = re.sub(r"\bg\s*/?\s*100\s*kg\b", "g/100kg", out, flags=re.IGNORECASE)
    out = re.sub(r"\bi\.?\s*u\.?\s*/?\s*g\b", "u/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\bu\s*/?\s*g\b", "u/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\bskb\s*/?\s*g\b", "skb/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\bfau\s*/?\s*g\b", "fau/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnmau\s*/?\s*g\b", "nmau/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\bxylh\s*/?\s*g\b", "xylh/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\bagi\s*/?\s*g\b", "agi/g", out, flags=re.IGNORECASE)
    out = re.sub(r"\brelative\s+humidity\b", "humidity", out, flags=re.IGNORECASE)
    out = re.sub(r"\brh\b", "humidity", out, flags=re.IGNORECASE)
    out = re.sub(r"\bshelf[\s-]*life\b", "shelf life", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnot\s+more\s+than\b", "<=", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnot\s+less\s+than\b", ">=", out, flags=re.IGNORECASE)
    out = re.sub(r"\bat\s+or\s+below\b", "<=", out, flags=re.IGNORECASE)
    out = re.sub(r"\bat\s+or\s+above\b", ">=", out, flags=re.IGNORECASE)
    out = re.sub(r"(\d)\s+to\s+(\d)", r"\1-\2", out, flags=re.IGNORECASE)
    out = re.sub(r"(\d)\s*(?:~|−)\s*(\d)", r"\1-\2", out, flags=re.IGNORECASE)
    out = re.sub(r"(?<=\d)\s*°\s*c\b", " c", out, flags=re.IGNORECASE)
    out = re.sub(r"(?<=\d)\s*c\b", " c", out, flags=re.IGNORECASE)
    out = re.sub(r"\bcelsius\b", "c", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_answer_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = text.replace("’", "'")
    text = normalize_unit_string(text)
    text = re.sub(r"[^a-z0-9%/\-.\s]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_query_intent(question: str) -> str:
    q = normalize_answer_text(question)
    if any(k in q for k in ("dosage", "dose", "ppm", "recommended", "suggested", "application rate")):
        return "dosage"
    if any(k in q for k in ("activity", "enzyme unit", "u/g", "fau/g", "skb/g", "agi/g", "xylh/g", "nmau/g")):
        return "activity"
    if any(k in q for k in ("cadmium", "lead", "mercury", "arsenic", "heavy metal")):
        return "metals"
    if "allergen" in q or "gluten" in q:
        return "allergen"
    if any(k in q for k in ("storage", "shelf life", "humidity", "temperature", "durability", "expiry")):
        return "storage"
    if any(k in q for k in ("regulation", "directive", "codex", "innorpi", "gmo", "labeling", "compliance")):
        return "regulatory"
    return "generic"


def _extract_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", compact) if s.strip()]


def _parse_allergen_terms(text: str) -> list[str]:
    norm = normalize_answer_text(text)
    aliases = {
        "gluten": ("gluten", "wheat"),
        "soy": ("soy", "soya", "soybean"),
        "milk": ("milk", "lactose", "dairy"),
        "egg": ("egg", "eggs"),
        "nuts": ("nuts", "nut", "peanut", "almond", "hazelnut", "walnut", "cashew"),
    }
    found: list[str] = []
    for canonical, terms in aliases.items():
        if any(re.search(rf"\b{re.escape(term)}\b", norm) for term in terms):
            found.append(canonical)
    return sorted(set(found))


def _extract_metal_limits(text: str) -> list[str]:
    body = normalize_unit_string(text)
    body = re.sub(r",", ".", body)
    matches: list[str] = []
    for metal in ("cadmium", "lead", "mercury", "arsenic"):
        for match in re.finditer(
            rf"{metal}[^.\n;]{{0,80}}?(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*mg/kg",
            body,
            flags=re.IGNORECASE,
        ):
            sign = match.group(1) or ""
            value = match.group(2)
            atom = f"{metal} {sign}{value} mg/kg".strip()
            if atom not in matches:
                matches.append(atom)
    return matches


def _prettify_table_like_text(text: str) -> str:
    compact = " ".join(text.split())
    if "|" not in compact:
        return compact
    raw_parts = [part.strip(" :-") for part in compact.split("|")]
    parts: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        if re.fullmatch(r"[|:/\\-]+", part):
            continue
        if len(part) == 1:
            continue
        parts.append(part)
    return "; ".join(parts) if parts else compact


def _extract_structured_answer_core(question: str, text: str, max_chars: int | None = None) -> tuple[str, float]:
    intent = detect_query_intent(question)
    body = " ".join(text.split())
    body_norm = normalize_answer_text(body)
    sentences = _extract_sentences(body)

    if intent == "dosage":
        match = re.search(
            r"(?:dosage|dose|recommended|suggested|optimum)[^.\n]{0,120}?(\d+(?:[.,]\d+)?)\s*(?:-|to|–)\s*(\d+(?:[.,]\d+)?)\s*ppm",
            body_norm,
        )
        if not match:
            match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:-|to|–)\s*(\d+(?:[.,]\d+)?)\s*ppm", body_norm)
        if not match:
            match = re.search(
                r"(?:dosage|dose|recommended|suggested|optimum|application rate|addition level)[^.\n]{0,120}?(\d+(?:[.,]\d+)?)\s*(g/100kg|kg/t|%)",
                body_norm,
            )
            if match:
                return normalize_unit_string(f"dosage: {match.group(1)} {match.group(2)}"), 0.95
        if match:
            return normalize_unit_string(f"dosage: {match.group(1)}-{match.group(2)} ppm"), 0.95

    if intent == "activity":
        match = re.search(
            r"activity[^.\n]{0,80}?(\d+(?:[.,]\d+)?)\s*(u/g|skb/g|fau/g|nmau/g|xylh/g|agi/g)",
            normalize_unit_string(body_norm),
        )
        if not match:
            match = re.search(
                r"(?:enzyme\s+activity|declared\s+activity|units?)[^.\n]{0,80}?(\d+(?:[.,]\d+)?)\s*(u/g|skb/g|fau/g|nmau/g|xylh/g|agi/g)",
                normalize_unit_string(body_norm),
            )
        if match:
            return normalize_unit_string(f"activity: {match.group(1)} {match.group(2)}"), 0.95

    if intent == "metals":
        limits = _extract_metal_limits(body_norm)
        if limits:
            return "; ".join(limits), 0.95

    if intent == "allergen":
        allergens = _parse_allergen_terms(body_norm)
        if allergens:
            state = "contains"
            if re.search(r"\bfree from\b|\ballergen[-\s]*free\b|\bdoes not contain\b", body_norm):
                state = "free_from"
            return f"allergens {state}: {', '.join(allergens)}", 0.9

    if intent == "storage":
        parts: list[str] = []
        shelf = re.search(r"(\d+(?:\s*-\s*\d+)?)\s*(months?|years?)", body_norm)
        temp = re.search(r"(?:below|under|<|<=|between)\s*(\d+(?:\s*-\s*\d+)?)\s*c", body_norm)
        if not temp:
            temp = re.search(r"(\d+(?:\s*-\s*\d+)?)\s*c", body_norm)
        humidity = re.search(r"humidity[^.\n]{0,30}?(?:<|<=|below|under)?\s*(\d+)\s*%", body_norm)
        if shelf:
            parts.append(f"shelf life: {re.sub(r'\s+', '', shelf.group(1))} {shelf.group(2)}")
        if temp:
            parts.append(f"temperature: {re.sub(r'\s+', '', temp.group(1))} c")
        if humidity:
            parts.append(f"humidity: <{humidity.group(1)}%")
        if parts:
            return normalize_unit_string("; ".join(parts)), 0.9

    if intent == "regulatory":
        for sentence in sentences:
            norm = normalize_answer_text(sentence)
            if any(
                token in norm
                for token in ("regulation", "directive", "codex", "innorpi", "1829/2003", "1830/2003", "1169/2011", "gmo")
            ):
                return normalize_unit_string(sentence), 0.85

    return (
        build_presentable_answer(
            question=question,
            text=text,
            mode="hybrid",
            min_remaining_tokens=10,
            max_chars=max_chars,
        ),
        0.6,
    )


def extract_structured_answer(question: str, text: str, max_chars: int | None = None) -> str:
    answer, _ = _extract_structured_answer_core(
        question=question,
        text=_prettify_table_like_text(text),
        max_chars=max_chars,
    )
    return answer


def extract_structured_answer_with_score(question: str, text: str, max_chars: int | None = None) -> tuple[str, float]:
    return _extract_structured_answer_core(
        question=question,
        text=_prettify_table_like_text(text),
        max_chars=max_chars,
    )
