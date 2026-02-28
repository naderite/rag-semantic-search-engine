from __future__ import annotations

import logging
import math
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from rag.answer_structure import extract_structured_answer_with_score
from rag.embeddings import SentenceTransformerEmbedder
from rag import SearchConfig, search_top_k
from rag.vector_store import QdrantVectorStore

logger = logging.getLogger("semantic_atlas_api")
RETRIEVAL_SCORE_WEIGHT = 0.35
ANSWER_SCORE_WEIGHT = 0.65
RETRIEVAL_CANDIDATES = 10

app = FastAPI(title="Semantic Atlas API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://localhost:5173",
        "https://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty.")
        return cleaned


class SearchResultItem(BaseModel):
    text: str
    score: float
    retrieval_score: float
    answer_score: float
    rank: int


class SearchMeta(BaseModel):
    k: int
    time_ms: int


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResultItem]
    meta: SearchMeta


@lru_cache(maxsize=1)
def get_search_config() -> SearchConfig:
    return SearchConfig(
        qdrant_url="http://localhost:6333",
        collection_name="rag_chunks",
        context_window=0,
        chunks_jsonl_path="artifacts/chunks.jsonl",
        answer_span_max_chars=None,
        embedder=SentenceTransformerEmbedder("models/paraphrase-multilingual-MiniLM-L12-v2"),
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection_name="rag_chunks"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    logger.info("search.start question_len=%s", len(payload.question))

    try:
        cfg = get_search_config()
        rag_response = search_top_k(
            question=payload.question,
            k=RETRIEVAL_CANDIDATES,
            cfg=cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("search.failed")
        raise HTTPException(
            status_code=500,
            detail="Search failed. Ensure Qdrant is running and the collection is indexed.",
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    scored_rows: list[SearchResultItem] = []
    for idx, item in enumerate(rag_response.results[:RETRIEVAL_CANDIDATES], start=1):
        text = item.text
        answer_score = 0.0
        if cfg.answer_span_fallback_enabled:
            text, answer_score = extract_structured_answer_with_score(
                question=payload.question,
                text=item.text,
                max_chars=None,
            )
        raw_retrieval = float(item.final_score if item.final_score else item.score)
        retrieval_score = 1.0 / (1.0 + math.exp(-raw_retrieval))
        combined_score = (RETRIEVAL_SCORE_WEIGHT * retrieval_score) + (
            ANSWER_SCORE_WEIGHT * float(answer_score)
        )
        scored_rows.append(
            SearchResultItem(
                text=text,
                score=combined_score,
                retrieval_score=retrieval_score,
                answer_score=float(answer_score),
                rank=idx,
            )
        )
    scored_rows.sort(key=lambda row: row.score, reverse=True)
    results = [
        SearchResultItem(**{**row.model_dump(), "rank": idx})
        for idx, row in enumerate(scored_rows[:3], start=1)
    ]

    response = SearchResponse(
        question=payload.question,
        results=results,
        meta=SearchMeta(k=3, time_ms=elapsed_ms),
    )

    logger.info(
        "search.end question_len=%s returned=%s time_ms=%s",
        len(payload.question),
        len(response.results),
        elapsed_ms,
    )

    return response
