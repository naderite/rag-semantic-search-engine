from __future__ import annotations

from pathlib import Path


BEST_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    from sentence_transformers import SentenceTransformer  # type: ignore

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(BEST_MODEL_ID)
    model.save(str(LOCAL_MODEL_DIR))
    print(f"saved_local_model: {LOCAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
