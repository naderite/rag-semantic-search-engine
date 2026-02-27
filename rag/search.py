from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from .embeddings import SentenceTransformerEmbedder
from .types import SearchConfig, SearchResponse, SearchResult
from .vector_store import QdrantVectorStore


def search_top_k(question: str, k: int, cfg: SearchConfig) -> SearchResponse:
    embedder = cfg.embedder or SentenceTransformerEmbedder(cfg.embedding_model)
    vector_store = cfg.vector_store or QdrantVectorStore(url=cfg.qdrant_url, collection_name=cfg.collection_name)

    query_vec = embedder.embed([question])[0]
    fetch_k = max(k, k * max(1, cfg.fetch_multiplier), cfg.candidate_pool)
    dense_hits = vector_store.search(vector=query_vec, limit=fetch_k)
    ranked = _rank_hits(question=question, hits=dense_hits, cfg=cfg)
    results = ranked[:k]

    if cfg.context_window > 0 and cfg.chunks_jsonl_path:
        results = _expand_with_context(results=results, cfg=cfg)

    for idx, item in enumerate(results, start=1):
        item.rank = idx

    return SearchResponse(question=question, results=results)


def _rank_hits(question: str, hits: list[dict[str, Any]], cfg: SearchConfig) -> list[SearchResult]:
    query_tokens = _tokenize(question)
    query_norm = _normalize_text(question)
    query_intent = _query_intent(query_tokens)
    profile = _query_profile(query_norm, query_tokens)
    dense_w, lex_w, field_w = _dynamic_weights(cfg, profile)

    scored: list[SearchResult] = []
    for hit in hits:
        payload = hit.get("payload") or {}
        text = str(payload.get("text", ""))
        dense_score = float(hit.get("score", 0.0))
        lex_score, matched_terms = _lexical_score(query_tokens, text)
        field_score = _field_intent_score(query_intent=query_intent, payload=payload, text=text)
        doc_specific_score = _doc_specific_score(
            query_norm=query_norm,
            query_tokens=query_tokens,
            payload=payload,
            text=text,
        )
        structured_score = _structured_field_score(profile=profile, payload=payload)

        if cfg.hybrid_enabled:
            final = (
                dense_w * dense_score
                + lex_w * lex_score
                + field_w * field_score
                + doc_specific_score
                + structured_score
            )
        else:
            final = dense_score + doc_specific_score + structured_score

        scored.append(
            SearchResult(
                rank=0,
                chunk_id=str(payload.get("chunk_id", hit.get("id", ""))),
                text=text,
                score=final,
                doc_id=str(payload.get("doc_id", "")),
                page=int(payload.get("page", 0)),
                dense_score=dense_score,
                lex_score=lex_score,
                field_score=field_score,
                final_score=final,
                matched_terms=matched_terms,
                is_context=False,
            )
        )

    scored.sort(key=lambda x: x.final_score, reverse=True)
    scored = _dedup_by_chunk(scored)
    if cfg.diversity_enabled:
        scored = _apply_diversity(scored, cfg)
    return sorted(scored, key=lambda x: x.final_score, reverse=True)


def _dedup_by_chunk(items: list[SearchResult]) -> list[SearchResult]:
    out: list[SearchResult] = []
    seen: set[str] = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        out.append(item)
    return out


def _apply_diversity(items: list[SearchResult], cfg: SearchConfig) -> list[SearchResult]:
    seen_group: dict[tuple[str, int], int] = {}
    for item in items:
        group = (item.doc_id, item.page)
        repeats = seen_group.get(group, 0)
        if repeats > 0:
            penalty = max(0.0, 1.0 - cfg.diversity_penalty * repeats)
            item.final_score *= penalty
            item.score = item.final_score
        seen_group[group] = repeats + 1
    return items


def _query_intent(tokens: set[str]) -> dict[str, bool]:
    return {
        "bread": any(t in tokens for t in {"pain", "panification", "bread", "mie"}),
        "dosage": any(t in tokens for t in {"dosage", "dose", "ppm", "recommande"}),
        "regulatory": any(t in tokens for t in {"max", "maximum", "autorise", "reglement"}),
    }


def _field_intent_score(query_intent: dict[str, bool], payload: dict[str, Any], text: str) -> float:
    metadata = payload.get("metadata") or {}
    product = str(metadata.get("product_type_canonical", "")).lower()
    lowered = _normalize_text(text)

    score = 0.0
    if query_intent["bread"]:
        if product == "bread":
            score += 1.0
        elif "pain" in lowered or "panification" in lowered or "bread" in lowered:
            score += 0.8
        elif product in {"biscuit", "viennoiserie"}:
            score -= 0.3

    if query_intent["dosage"] and ("dosage" in lowered or "ppm" in lowered):
        score += 0.5

    if query_intent["regulatory"] and ("maximum" in lowered or "autorise" in lowered):
        score += 0.5

    return max(0.0, min(1.5, score))


def _query_profile(query_norm: str, query_tokens: set[str]) -> dict[str, Any]:
    codes = _extract_model_codes(query_norm, query_tokens)
    families = _extract_families(query_tokens)
    asks_standardization = "standardization" in query_tokens
    asks_bread_improvement = "improvement" in query_tokens and "bread" in query_tokens
    asks_optimum = any(t in query_tokens for t in {"optimum", "optimal", "suggested"})
    asks_activity = any(t in query_tokens for t in {"activity", "skb", "fau", "xylh", "agi"})
    asks_safety = any(t in query_tokens for t in {"allergen", "allergens", "heavy", "metals", "cadmium", "lead"})
    asks_storage = any(t in query_tokens for t in {"storage", "shelf", "durability", "months", "temperature"})
    asks_regulatory = any(t in query_tokens for t in {"legal", "regulatory", "directive", "codex", "autorise", "maximum"})
    broad_mode = len(codes) == 0

    return {
        "codes": codes,
        "families": families,
        "broad_mode": broad_mode,
        "asks_standardization": asks_standardization,
        "asks_bread_improvement": asks_bread_improvement,
        "asks_optimum": asks_optimum,
        "asks_activity": asks_activity,
        "asks_safety": asks_safety,
        "asks_storage": asks_storage,
        "asks_regulatory": asks_regulatory,
    }


def _dynamic_weights(cfg: SearchConfig, profile: dict[str, Any]) -> tuple[float, float, float]:
    dense_w = cfg.dense_weight
    lex_w = cfg.lexical_weight
    field_w = cfg.field_weight

    # Precision-first default: keep base weights for broad queries.
    if profile["broad_mode"]:
        dense_w = cfg.dense_weight
        lex_w = cfg.lexical_weight
        field_w = cfg.field_weight

    # For exact model/code queries, slightly favor dense to preserve top-rank precision.
    if profile["codes"]:
        dense_w = min(0.74, dense_w + 0.05)
        lex_w = max(0.20, lex_w - 0.03)
        field_w = max(0.08, field_w - 0.02)

    total = dense_w + lex_w + field_w
    if total <= 0:
        return cfg.dense_weight, cfg.lexical_weight, cfg.field_weight
    return dense_w / total, lex_w / total, field_w / total


def _structured_field_score(profile: dict[str, Any], payload: dict[str, Any]) -> float:
    md = payload.get("metadata") or {}
    score = 0.0

    doc_family = str(md.get("product_family", "")).lower()
    query_families = set(profile.get("families", set()))
    if query_families and doc_family and doc_family != "other":
        if doc_family in query_families:
            score += 0.10
        else:
            score -= 0.16

    if profile.get("asks_standardization") and md.get("dosage_standardization"):
        score += 0.18
    if profile.get("asks_bread_improvement") and md.get("dosage_bread_improvement"):
        score += 0.18
    if profile.get("asks_optimum") and (md.get("dosage_suggested_optimum") or md.get("dosage_suggested")):
        score += 0.18
    if profile.get("asks_regulatory") and (
        md.get("dosage_max_authorized") or str(md.get("section_canonical", "")) == "regulatory"
    ):
        score += 0.16
    if profile.get("asks_storage") and str(md.get("section_canonical", "")) == "storage":
        score += 0.12
    if profile.get("asks_safety") and str(md.get("section_canonical", "")) == "food_safety":
        score += 0.12
    if profile.get("asks_activity") and "activity" in _normalize_text(str(payload.get("text", ""))[:180]):
        score += 0.12

    return max(-0.20, min(0.30, score))


def _doc_specific_score(query_norm: str, query_tokens: set[str], payload: dict[str, Any], text: str) -> float:
    doc_id = str(payload.get("doc_id", ""))
    file_name = str(payload.get("file_name", ""))
    # Use short text prefix to keep this fast while still capturing model codes near headers.
    doc_tokens = _tokenize(" ".join([doc_id, file_name, text[:320]]))

    query_codes = _extract_model_codes(query_norm, query_tokens)
    doc_codes = _extract_model_codes(_normalize_text(" ".join([doc_id, file_name])), doc_tokens)

    score = 0.0
    if query_codes and doc_codes:
        overlap = query_codes.intersection(doc_codes)
        if overlap:
            score += 0.45 + (0.20 * min(2, len(overlap)))

    query_families = _extract_families(query_tokens)
    doc_families = _extract_families(doc_tokens)
    if query_families and doc_families and query_families.intersection(doc_families):
        score += 0.25

    query_is_ascorbic = any(t in query_tokens for t in {"ascorbique", "e300", "vitamine"})
    doc_is_ascorbic = "acide ascorbique" in _normalize_text(" ".join([doc_id, file_name]))
    if query_is_ascorbic and doc_is_ascorbic:
        score += 0.20
    if query_is_ascorbic and not doc_is_ascorbic:
        score -= 0.20
    if (not query_is_ascorbic) and doc_is_ascorbic and query_families:
        score -= 0.45

    return max(-0.60, min(0.90, score))


def _extract_families(tokens: set[str]) -> set[str]:
    families: set[str] = set()
    for family in {"hcf", "hcb", "af", "amg", "tg", "gox", "go", "fresh", "soft"}:
        if family in tokens:
            families.add(family)
    if "lipase" in tokens:
        families.add("lipase")
    return families


def _extract_model_codes(text_norm: str, tokens: set[str]) -> set[str]:
    codes: set[str] = set()

    # Compact alphanumeric product markers, e.g. hcf600, hcb708, af110, amg880, tg883, gox110, fresh101, soft305.
    for token in tokens:
        if re.fullmatch(r"(?:hcf|hcb|af|amg|tg|gox|go|fresh|soft)\d{2,4}", token):
            codes.add(token)
        if re.fullmatch(r"max(?:63|64|65)", token):
            codes.add(token)
        if re.fullmatch(r"l(?:55|65)", token):
            codes.add(token)

    # Spaced variants in queries/docs, e.g. "hcf max x", "hcf max63", "l max64", "go max 63".
    for match in re.findall(r"\b(hcf|hcb|tg|go|l)\s*max\s*(x|63|64|65)\b", text_norm):
        codes.add(f"{match[0]}max{match[1]}")

    return codes


def _lexical_score(query_tokens: set[str], text: str) -> tuple[float, list[str]]:
    if not query_tokens:
        return 0.0, []
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0, []
    matched = sorted(query_tokens.intersection(text_tokens))
    if not matched:
        return 0.0, []
    coverage = len(matched) / max(1, len(query_tokens))
    density = len(matched) / max(1, len(text_tokens))
    score = min(1.0, (0.75 * coverage) + (0.25 * density * 4))
    return score, matched


def _expand_with_context(results: list[SearchResult], cfg: SearchConfig) -> list[SearchResult]:
    positions, records = _load_chunk_positions(cfg.chunks_jsonl_path)
    if not positions:
        return results

    expanded: list[SearchResult] = list(results)
    seen = {r.chunk_id for r in expanded}

    for anchor in results:
        key = (anchor.doc_id, anchor.page)
        sequence = positions.get(key, [])
        if not sequence:
            continue
        try:
            idx = sequence.index(anchor.chunk_id)
        except ValueError:
            continue

        start = max(0, idx - cfg.context_window)
        end = min(len(sequence), idx + cfg.context_window + 1)
        for pos in range(start, end):
            neighbor_id = sequence[pos]
            if neighbor_id in seen:
                continue
            raw = records.get(neighbor_id)
            if not raw:
                continue
            seen.add(neighbor_id)
            expanded.append(
                SearchResult(
                    rank=0,
                    chunk_id=neighbor_id,
                    text=str(raw.get("text", "")),
                    score=anchor.score,
                    doc_id=str(raw.get("doc_id", "")),
                    page=int(raw.get("page", 0)),
                    dense_score=anchor.dense_score,
                    lex_score=anchor.lex_score,
                    field_score=anchor.field_score,
                    final_score=anchor.final_score,
                    matched_terms=anchor.matched_terms,
                    is_context=True,
                )
            )

    return expanded


@lru_cache(maxsize=4)
def _load_chunk_positions(chunks_jsonl_path: str | None) -> tuple[dict[tuple[str, int], list[str]], dict[str, dict]]:
    if not chunks_jsonl_path:
        return {}, {}
    path = Path(chunks_jsonl_path)
    if not path.exists():
        return {}, {}

    by_page: dict[tuple[str, int], list[str]] = {}
    by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id", ""))
            doc_id = str(row.get("doc_id", ""))
            page = int(row.get("page", 0))
            by_page.setdefault((doc_id, page), []).append(chunk_id)
            by_id[chunk_id] = row
    return by_page, by_id


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    parts = [p for p in re.split(r"\s+", normalized) if p and len(p) > 1]
    return set(parts)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9%/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
