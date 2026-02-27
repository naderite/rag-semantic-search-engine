from __future__ import annotations

import math
import uuid
from typing import Any


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        self._records = [r for r in self._records if r["id"] not in set(ids)]
        for idx, vec, payload in zip(ids, vectors, payloads, strict=True):
            self._records.append({"id": idx, "vector": vec, "payload": payload})

    def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        scored = []
        for record in self._records:
            score = _cosine_similarity(vector, record["vector"])
            scored.append({"id": record["id"], "score": score, "payload": record["payload"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, vector_size: int | None = None) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http.models import Distance, VectorParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for Qdrant integration") from exc

        self._QdrantClient = QdrantClient
        self._Distance = Distance
        self._VectorParams = VectorParams

        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size

    def _ensure_collection(self, vector_size: int) -> None:
        self._vector_size = self._vector_size or vector_size
        collections = self._client.get_collections().collections
        if any(col.name == self._collection_name for col in collections):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=self._VectorParams(size=self._vector_size, distance=self._Distance.COSINE),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        from qdrant_client.http.models import PointStruct  # type: ignore

        if not vectors:
            return

        self._ensure_collection(len(vectors[0]))
        points = []
        for idx, vec, payload in zip(ids, vectors, payloads, strict=True):
            # Qdrant accepts only integer or UUID point IDs.
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, idx))
            points.append(PointStruct(id=qdrant_id, vector=vec, payload=payload))
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        # qdrant-client API differs across versions:
        # - older versions expose `search(...)`
        # - newer versions expose `query_points(...)`
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            hits = getattr(response, "points", response)
        else:
            hits = self._client.search(collection_name=self._collection_name, query_vector=vector, limit=limit)

        return [{"id": str(hit.id), "score": float(hit.score), "payload": hit.payload or {}} for hit in hits]
