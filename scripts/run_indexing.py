from __future__ import annotations

import os
import time
from urllib.error import URLError
from urllib.request import urlopen

from rag import PrepConfig, VectorConfig, index_pdfs_directly


def _wait_for_qdrant(qdrant_url: str, timeout_sec: int = 120) -> None:
    health_url = f"{qdrant_url.rstrip('/')}/healthz"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Qdrant was not ready at {health_url} within {timeout_sec}s")


def main() -> None:
    input_dir = os.getenv("RAG_INPUT_DIR", "data")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    _wait_for_qdrant(qdrant_url=qdrant_url)

    prep_report, index_report = index_pdfs_directly(
        input_dir=input_dir,
        prep_config=PrepConfig(),
        vector_config=VectorConfig(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_model=model_name,
        ),
    )

    print("Preparation report:", prep_report)
    print("Index report:", index_report)


if __name__ == "__main__":
    main()
