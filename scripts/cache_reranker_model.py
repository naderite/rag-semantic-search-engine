from __future__ import annotations

from pathlib import Path


RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "models" / "ms-marco-MiniLM-L-6-v2"


def main() -> None:
    from sentence_transformers import CrossEncoder  # type: ignore

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(RERANKER_ID)
    model.save(str(LOCAL_DIR))
    print(f"saved_reranker: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
