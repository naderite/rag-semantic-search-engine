from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag.embeddings import SentenceTransformerEmbedder
from rag.index import build_vector_index
from rag.pipeline import run_quality_eval
from rag.prepare import direct_chunks_from_pdfs
from rag.types import PrepConfig, SearchConfig, VectorConfig
from rag.vector_store import InMemoryVectorStore


def load_cases(path: Path) -> list[tuple[str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[tuple[str, set[str]]] = []
    for item in payload:
        expected = {
            Path(candidate["source_file"]).stem
            for candidate in item.get("expected_top3", [])
            if candidate.get("source_file")
        }
        cases.append((item["query"], expected))
    return cases


def config_grid() -> list[dict]:
    weight_sets = [
        {"dense_weight": 0.60, "lexical_weight": 0.30, "field_weight": 0.10},
        {"dense_weight": 0.58, "lexical_weight": 0.32, "field_weight": 0.10},
        {"dense_weight": 0.55, "lexical_weight": 0.30, "field_weight": 0.15},
        {"dense_weight": 0.52, "lexical_weight": 0.33, "field_weight": 0.15},
    ]
    candidate_pools = [40, 60, 80]
    fetch_multipliers = [2, 3]
    diversity_options = [False, True]
    code_mismatch_penalties = [0.10, 0.15]
    section_mismatch_penalties = [0.08, 0.12]

    out: list[dict] = []
    for w in weight_sets:
        for cp in candidate_pools:
            for fm in fetch_multipliers:
                for dv in diversity_options:
                    for cmp in code_mismatch_penalties:
                        for smp in section_mismatch_penalties:
                            out.append(
                                {
                                    **w,
                                    "candidate_pool": cp,
                                    "fetch_multiplier": fm,
                                    "diversity_enabled": dv,
                                    "code_mismatch_penalty": cmp,
                                    "section_mismatch_penalty": smp,
                                }
                            )
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for retrieval on eval dataset.")
    parser.add_argument("--eval-path", default=str(root / "tests" / "pdf_eval_queries_human.json"))
    parser.add_argument("--output", default=str(root / "artifacts" / "eval_hparam_sweep_human.json"))
    parser.add_argument("--model", default="models/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("RAG_EMBEDDING_LOCAL_ONLY", "1")

    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = (root / eval_path).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()

    cases = load_cases(eval_path)
    chunks, prep_report = direct_chunks_from_pdfs(str(root / "data"), PrepConfig())

    embedder = SentenceTransformerEmbedder(args.model)
    store = InMemoryVectorStore()
    index_report = build_vector_index(
        chunks=chunks,
        cfg=VectorConfig(embedder=embedder, vector_store=store, embedding_model=args.model),
    )

    baseline_cfg = SearchConfig(
        embedder=embedder,
        vector_store=store,
        embedding_model=args.model,
        context_window=0,
    )
    baseline = run_quality_eval(cases=cases, search_config=baseline_cfg, top_k=args.top_k)

    runs: list[dict] = []
    for i, hp in enumerate(config_grid(), start=1):
        cfg = SearchConfig(
            embedder=embedder,
            vector_store=store,
            embedding_model=args.model,
            context_window=0,
            dense_weight=hp["dense_weight"],
            lexical_weight=hp["lexical_weight"],
            field_weight=hp["field_weight"],
            candidate_pool=hp["candidate_pool"],
            fetch_multiplier=hp["fetch_multiplier"],
            diversity_enabled=hp["diversity_enabled"],
            code_mismatch_penalty=hp["code_mismatch_penalty"],
            section_mismatch_penalty=hp["section_mismatch_penalty"],
        )
        report = run_quality_eval(cases=cases, search_config=cfg, top_k=args.top_k)
        row = {
            "rank_hint": i,
            "config": hp,
            "queries": report.total_queries,
            "hit_at_3": report.hit_at_k,
            "mrr_at_3": report.mrr,
            "delta_hit_at_3_vs_baseline": report.hit_at_k - baseline.hit_at_k,
            "delta_mrr_at_3_vs_baseline": report.mrr - baseline.mrr,
        }
        runs.append(row)
        print(
            f"[{i:03d}] hit@3={row['hit_at_3']:.4f} mrr@3={row['mrr_at_3']:.4f} "
            f"cfg={hp}"
        )

    runs.sort(
        key=lambda r: (
            r["hit_at_3"],
            r["mrr_at_3"],
            r["config"]["candidate_pool"],
            r["config"]["fetch_multiplier"],
        ),
        reverse=True,
    )

    best = runs[0] if runs else None
    payload = {
        "model": args.model,
        "eval_set": str(eval_path.relative_to(root)) if eval_path.is_relative_to(root) else str(eval_path),
        "prep_report": {
            "documents_processed": prep_report.documents_processed,
            "chunks_after_dedup": prep_report.chunks_after_dedup,
            "vectors_upserted": index_report.vectors_upserted,
        },
        "baseline": {
            "queries": baseline.total_queries,
            "hit_at_3": baseline.hit_at_k,
            "mrr_at_3": baseline.mrr,
            "config": {
                "dense_weight": baseline_cfg.dense_weight,
                "lexical_weight": baseline_cfg.lexical_weight,
                "field_weight": baseline_cfg.field_weight,
                "candidate_pool": baseline_cfg.candidate_pool,
                "fetch_multiplier": baseline_cfg.fetch_multiplier,
                "diversity_enabled": baseline_cfg.diversity_enabled,
                "code_mismatch_penalty": baseline_cfg.code_mismatch_penalty,
                "section_mismatch_penalty": baseline_cfg.section_mismatch_penalty,
            },
        },
        "best": best,
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {output_path}")
    if best:
        print(f"best_hit@3={best['hit_at_3']:.4f} best_mrr@3={best['mrr_at_3']:.4f}")
        print(f"best_config={best['config']}")


if __name__ == "__main__":
    main()
