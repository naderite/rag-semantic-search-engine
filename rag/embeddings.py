from __future__ import annotations

import os


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for embeddings") from exc

        local_only = os.getenv("RAG_EMBEDDING_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
        try:
            self._model = SentenceTransformer(model_name, local_files_only=local_only)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Ensure a warm local cache/path. Set RAG_EMBEDDING_LOCAL_ONLY=0 for one-time download."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc

        local_only = os.getenv("RAG_EMBEDDING_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
        device = os.getenv("RAG_RERANKER_DEVICE", "cpu")
        try:
            self._model = CrossEncoder(model_name, local_files_only=local_only, device=device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load reranker model '{model_name}'. "
                "Ensure a warm local cache/path. Set RAG_EMBEDDING_LOCAL_ONLY=0 for one-time download."
            ) from exc

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(question, text) for text in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
