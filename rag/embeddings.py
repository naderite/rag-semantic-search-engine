from __future__ import annotations


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for embeddings") from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Ensure internet access for first download or a warm local cache."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]
