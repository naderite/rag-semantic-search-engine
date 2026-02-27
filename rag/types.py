from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        ...

    def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        ...


@dataclass
class PrepConfig:
    chunk_size_tokens: int = 420
    chunk_overlap_tokens: int = 60
    min_chunk_tokens: int = 12
    min_text_chars_for_page: int = 20
    enable_ocr_fallback: bool = True
    ocr_lang: str = "eng+fra"
    include_list_items: bool = False
    dedup_across_docs: bool = True


@dataclass
class VectorConfig:
    collection_name: str = "rag_chunks"
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder: Embedder | None = None
    vector_store: VectorStore | None = None


@dataclass
class SearchConfig:
    collection_name: str = "rag_chunks"
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder: Embedder | None = None
    vector_store: VectorStore | None = None
    context_window: int = 1
    chunks_jsonl_path: str | None = "artifacts/chunks.jsonl"
    fetch_multiplier: int = 2
    hybrid_enabled: bool = True
    candidate_pool: int = 30
    dense_weight: float = 0.65
    lexical_weight: float = 0.25
    field_weight: float = 0.10
    diversity_enabled: bool = True
    diversity_penalty: float = 0.10


@dataclass
class RawPageBlock:
    doc_id: str
    file_name: str
    page: int
    block_type: str
    text: str
    rows: list[list[str]] | None = None


@dataclass
class NormalizedBlock:
    doc_id: str
    file_name: str
    page: int
    section: str
    chunk_type: str
    text: str
    text_raw: str
    lang: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    file_name: str
    page: int
    section: str
    chunk_type: str
    text: str
    text_raw: str
    lang: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrepReport:
    documents_processed: int = 0
    pages_processed: int = 0
    pages_with_ocr: int = 0
    chunks_generated: int = 0
    chunks_after_dedup: int = 0
    extraction_errors: list[str] = field(default_factory=list)


@dataclass
class IndexReport:
    collection_name: str
    vectors_upserted: int


@dataclass
class SearchResult:
    rank: int
    chunk_id: str
    text: str
    score: float
    doc_id: str
    page: int
    dense_score: float = 0.0
    lex_score: float = 0.0
    field_score: float = 0.0
    final_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    is_context: bool = False


@dataclass
class SearchResponse:
    question: str
    results: list[SearchResult]


@dataclass
class QualityEvalReport:
    total_queries: int
    hit_at_k: float
    mrr: float
