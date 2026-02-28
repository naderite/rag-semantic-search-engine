from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag.embeddings import SentenceTransformerEmbedder
from rag.index import build_vector_index
from rag.prepare import direct_chunks_from_pdfs
from rag.search import _normalize_text, _query_profile, _tokenize, search_top_k
from rag.types import PrepConfig, SearchConfig, VectorConfig
from rag.vector_store import InMemoryVectorStore


def load_eval_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_name_for_query(query: str) -> str:
    norm = _normalize_text(query)
    tokens = _tokenize(query)
    profile = _query_profile(norm, tokens)
    if profile.get("codes"):
        return "code_specific"
    if profile.get("asks_storage") or profile.get("asks_safety") or profile.get("asks_regulatory"):
        return "compliance_storage_safety"
    if profile.get("asks_activity"):
        return "activity"
    return "broad_semantic"


def evaluate_global(items: list[dict], cfg: SearchConfig, top_k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    failed_ids: list[str] = []
    for item in items:
        expected = {Path(c["source_file"]).stem for c in item.get("expected_top3", []) if c.get("source_file")}
        ranked = [r.doc_id for r in search_top_k(item["query"], top_k, cfg).results]
        ranks = [i + 1 for i, doc in enumerate(ranked) if doc in expected]
        if ranks:
            hits += 1
            rr_sum += 1.0 / min(ranks)
        else:
            failed_ids.append(item["id"])
    total = len(items)
    return {
        "queries": total,
        "hit_at_3": hits / total if total else 0.0,
        "mrr_at_3": rr_sum / total if total else 0.0,
        "failed_ids": failed_ids,
    }


def evaluate_routed(items: list[dict], base_cfg: SearchConfig, route_cfgs: dict[str, dict], top_k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    failed_ids: list[str] = []
    for item in items:
        route = route_name_for_query(item["query"])
        hp = route_cfgs[route]
        cfg = SearchConfig(
            collection_name=base_cfg.collection_name,
            qdrant_url=base_cfg.qdrant_url,
            embedding_model=base_cfg.embedding_model,
            embedder=base_cfg.embedder,
            vector_store=base_cfg.vector_store,
            context_window=base_cfg.context_window,
            chunks_jsonl_path=base_cfg.chunks_jsonl_path,
            fetch_multiplier=hp["fetch_multiplier"],
            hybrid_enabled=base_cfg.hybrid_enabled,
            candidate_pool=hp["candidate_pool"],
            dense_weight=hp["dense_weight"],
            lexical_weight=hp["lexical_weight"],
            field_weight=hp["field_weight"],
            diversity_enabled=hp["diversity_enabled"],
            diversity_penalty=base_cfg.diversity_penalty,
            use_query_constraints=base_cfg.use_query_constraints,
            code_mismatch_penalty=hp["code_mismatch_penalty"],
            section_mismatch_penalty=hp["section_mismatch_penalty"],
            doc_fusion_enabled=base_cfg.doc_fusion_enabled,
            adaptive_fetch_enabled=base_cfg.adaptive_fetch_enabled,
            route_overrides_enabled=False,
            reranker_enabled=base_cfg.reranker_enabled,
            reranker_model=base_cfg.reranker_model,
            reranker_top_n=base_cfg.reranker_top_n,
            reranker_weight=base_cfg.reranker_weight,
        )
        expected = {Path(c["source_file"]).stem for c in item.get("expected_top3", []) if c.get("source_file")}
        ranked = [r.doc_id for r in search_top_k(item["query"], top_k, cfg).results]
        ranks = [i + 1 for i, doc in enumerate(ranked) if doc in expected]
        if ranks:
            hits += 1
            rr_sum += 1.0 / min(ranks)
        else:
            failed_ids.append(item["id"])
    total = len(items)
    return {
        "queries": total,
        "hit_at_3": hits / total if total else 0.0,
        "mrr_at_3": rr_sum / total if total else 0.0,
        "failed_ids": failed_ids,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Benchmark reranker/route-feature variants.")
    parser.add_argument("--eval-path", default=str(root / "tests" / "pdf_eval_queries_human.json"))
    parser.add_argument("--routing-report", default=str(root / "artifacts" / "eval_routing_sweep_human.json"))
    parser.add_argument("--output", default=str(root / "artifacts" / "eval_feature_benchmarks_human.json"))
    parser.add_argument("--model", default="models/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("RAG_EMBEDDING_LOCAL_ONLY", "1")
    os.environ.setdefault("RAG_RERANKER_DEVICE", "cpu")

    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = (root / eval_path).resolve()
    routing_path = Path(args.routing_report)
    if not routing_path.is_absolute():
        routing_path = (root / routing_path).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()

    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    route_cfgs = {route: row["config"] for route, row in routing["best_per_route"].items()}
    items = load_eval_items(eval_path)

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
        route_overrides_enabled=False,
        reranker_enabled=False,
    )
    reranker_only_cfg = SearchConfig(
        embedder=embedder,
        vector_store=store,
        embedding_model=args.model,
        context_window=0,
        route_overrides_enabled=False,
        reranker_enabled=True,
    )
    both_cfg = SearchConfig(
        embedder=embedder,
        vector_store=store,
        embedding_model=args.model,
        context_window=0,
        route_overrides_enabled=False,
        reranker_enabled=True,
    )
    route_only_cfg = SearchConfig(
        embedder=embedder,
        vector_store=store,
        embedding_model=args.model,
        context_window=0,
        route_overrides_enabled=False,
        reranker_enabled=False,
    )

    baseline = evaluate_global(items, baseline_cfg, args.top_k)
    reranker_only = evaluate_global(items, reranker_only_cfg, args.top_k)
    route_only = evaluate_routed(items, route_only_cfg, route_cfgs, args.top_k)
    both = evaluate_routed(items, both_cfg, route_cfgs, args.top_k)

    payload = {
        "model": args.model,
        "eval_set": str(eval_path.relative_to(root)) if eval_path.is_relative_to(root) else str(eval_path),
        "routing_report": str(routing_path),
        "prep_report": {
            "documents_processed": prep_report.documents_processed,
            "chunks_after_dedup": prep_report.chunks_after_dedup,
            "vectors_upserted": index_report.vectors_upserted,
        },
        "variants": {
            "baseline": baseline,
            "reranker_only": reranker_only,
            "route_only": route_only,
            "both": both,
        },
        "delta_vs_baseline": {
            "reranker_only": {
                "hit_at_3": reranker_only["hit_at_3"] - baseline["hit_at_3"],
                "mrr_at_3": reranker_only["mrr_at_3"] - baseline["mrr_at_3"],
            },
            "route_only": {
                "hit_at_3": route_only["hit_at_3"] - baseline["hit_at_3"],
                "mrr_at_3": route_only["mrr_at_3"] - baseline["mrr_at_3"],
            },
            "both": {
                "hit_at_3": both["hit_at_3"] - baseline["hit_at_3"],
                "mrr_at_3": both["mrr_at_3"] - baseline["mrr_at_3"],
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {output_path}")
    print(json.dumps(payload["delta_vs_baseline"], indent=2))


if __name__ == "__main__":
    main()
