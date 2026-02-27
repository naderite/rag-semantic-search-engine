from __future__ import annotations

from dataclasses import asdict

from .embeddings import SentenceTransformerEmbedder
from .types import ChunkRecord, IndexReport, VectorConfig
from .vector_store import QdrantVectorStore


def build_embeddings(chunks: list[ChunkRecord], cfg: VectorConfig) -> tuple[list[str], list[list[float]], list[dict]]:
    if not chunks:
        return [], [], []

    embedder = cfg.embedder or SentenceTransformerEmbedder(cfg.embedding_model)
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed(texts)
    ids = [chunk.chunk_id for chunk in chunks]
    payloads = [asdict(chunk) for chunk in chunks]
    return ids, vectors, payloads


def index_to_qdrant(ids: list[str], vectors: list[list[float]], payloads: list[dict], cfg: VectorConfig) -> IndexReport:
    vector_store = cfg.vector_store or QdrantVectorStore(url=cfg.qdrant_url, collection_name=cfg.collection_name)
    vector_store.upsert(ids=ids, vectors=vectors, payloads=payloads)
    return IndexReport(collection_name=cfg.collection_name, vectors_upserted=len(ids))


def build_vector_index(chunks: list[ChunkRecord], cfg: VectorConfig) -> IndexReport:
    ids, vectors, payloads = build_embeddings(chunks=chunks, cfg=cfg)
    if not ids:
        return IndexReport(collection_name=cfg.collection_name, vectors_upserted=0)
    return index_to_qdrant(ids=ids, vectors=vectors, payloads=payloads, cfg=cfg)
