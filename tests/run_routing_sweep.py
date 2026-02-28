from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
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
    query_norm = _normalize_text(query)
    query_tokens = _tokenize(query)
    profile = _query_profile(query_norm, query_tokens)

    if profile.get("codes"):
        return "code_specific"
    if profile.get("asks_storage") or profile.get("asks_safety") or profile.get("asks_regulatory"):
        return "compliance_storage_safety"
    if profile.get("asks_activity"):
        return "activity"
    return "broad_semantic"


def default_config_grid() -> list[dict]:
    # Full discrete sweep grid (reranker disabled in this script to isolate route config impact).
    weights = [
        (0.65, 0.25, 0.10),
        (0.62, 0.28, 0.10),
        (0.60, 0.30, 0.10),
        (0.58, 0.32, 0.10),
        (0.55, 0.30, 0.15),
        (0.52, 0.33, 0.15),
    ]
    pools = [30, 40, 50, 60, 80, 100]
    fetch = [1, 2, 3, 4]
    diversity = [False, True]
    code_pen = [0.05, 0.10, 0.15]
    sec_pen = [0.04, 0.08, 0.12]

    out: list[dict] = []
    for dw, lw, fw in weights:
        for cp in pools:
            for fm in fetch:
                for dv in diversity:
                    for cpen in code_pen:
                        for spen in sec_pen:
                            out.append(
                                {
                                    "dense_weight": dw,
                                    "lexical_weight": lw,
                                    "field_weight": fw,
                                    "candidate_pool": cp,
                                    "fetch_multiplier": fm,
                                    "diversity_enabled": dv,
                                    "code_mismatch_penalty": cpen,
                                    "section_mismatch_penalty": spen,
                                }
                            )
    return out


def evaluate_items(items: list[dict], cfg: SearchConfig, top_k: int) -> dict:
    if not items:
        return {"queries": 0, "hit_at_3": 0.0, "mrr_at_3": 0.0}

    hits = 0
    rr_sum = 0.0
    for item in items:
        expected = {
            Path(c["source_file"]).stem
            for c in item.get("expected_top3", [])
            if c.get("source_file")
        }
        resp = search_top_k(item["query"], top_k, cfg)
        ranked = [r.doc_id for r in resp.results]
        ranks = [i + 1 for i, doc_id in enumerate(ranked) if doc_id in expected]
        if ranks:
            hits += 1
            rr_sum += 1.0 / min(ranks)

    total = len(items)
    return {
        "queries": total,
        "hit_at_3": hits / total,
        "mrr_at_3": rr_sum / total,
    }


def make_cfg(base: SearchConfig, hp: dict) -> SearchConfig:
    return SearchConfig(
        collection_name=base.collection_name,
        qdrant_url=base.qdrant_url,
        embedding_model=base.embedding_model,
        embedder=base.embedder,
        vector_store=base.vector_store,
        context_window=base.context_window,
        chunks_jsonl_path=base.chunks_jsonl_path,
        hybrid_enabled=base.hybrid_enabled,
        use_query_constraints=base.use_query_constraints,
        doc_fusion_enabled=base.doc_fusion_enabled,
        adaptive_fetch_enabled=base.adaptive_fetch_enabled,
        diversity_penalty=base.diversity_penalty,
        route_overrides_enabled=base.route_overrides_enabled,
        reranker_enabled=base.reranker_enabled,
        reranker_model=base.reranker_model,
        reranker_top_n=base.reranker_top_n,
        reranker_weight=base.reranker_weight,
        dense_weight=hp["dense_weight"],
        lexical_weight=hp["lexical_weight"],
        field_weight=hp["field_weight"],
        candidate_pool=hp["candidate_pool"],
        fetch_multiplier=hp["fetch_multiplier"],
        diversity_enabled=hp["diversity_enabled"],
        code_mismatch_penalty=hp["code_mismatch_penalty"],
        section_mismatch_penalty=hp["section_mismatch_penalty"],
    )


def evaluate_routed(items: list[dict], base_cfg: SearchConfig, route_cfgs: dict[str, dict], top_k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    per_route = defaultdict(lambda: {"queries": 0, "hits": 0, "rr_sum": 0.0})

    for item in items:
        route = route_name_for_query(item["query"])
        hp = route_cfgs[route]
        cfg = make_cfg(base_cfg, hp)
        expected = {
            Path(c["source_file"]).stem
            for c in item.get("expected_top3", [])
            if c.get("source_file")
        }
        resp = search_top_k(item["query"], top_k, cfg)
        ranked = [r.doc_id for r in resp.results]
        ranks = [i + 1 for i, doc_id in enumerate(ranked) if doc_id in expected]

        per_route[route]["queries"] += 1
        if ranks:
            rr = 1.0 / min(ranks)
            hits += 1
            rr_sum += rr
            per_route[route]["hits"] += 1
            per_route[route]["rr_sum"] += rr

    total = len(items)
    per_route_out = {}
    for route, row in per_route.items():
        q = row["queries"]
        per_route_out[route] = {
            "queries": q,
            "hit_at_3": (row["hits"] / q) if q else 0.0,
            "mrr_at_3": (row["rr_sum"] / q) if q else 0.0,
        }

    return {
        "queries": total,
        "hit_at_3": (hits / total) if total else 0.0,
        "mrr_at_3": (rr_sum / total) if total else 0.0,
        "per_route": per_route_out,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Route-aware config search for retrieval.")
    parser.add_argument("--eval-path", default=str(root / "tests" / "pdf_eval_queries_human.json"))
    parser.add_argument("--output", default=str(root / "artifacts" / "eval_routing_sweep_human.json"))
    parser.add_argument("--model", default="models/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("RAG_EMBEDDING_LOCAL_ONLY", "1")

    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = (root / eval_path).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()

    items = load_eval_items(eval_path)
    route_items: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        route_items[route_name_for_query(item["query"])].append(item)

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
    baseline = evaluate_items(items, baseline_cfg, top_k=args.top_k)

    grid = default_config_grid()
    best_per_route: dict[str, dict] = {}
    route_search_log: dict[str, list[dict]] = {}

    for route, subset in sorted(route_items.items()):
        candidates: list[dict] = []
        for hp in grid:
            cfg = make_cfg(baseline_cfg, hp)
            score = evaluate_items(subset, cfg, top_k=args.top_k)
            row = {
                "config": hp,
                "queries": score["queries"],
                "hit_at_3": score["hit_at_3"],
                "mrr_at_3": score["mrr_at_3"],
            }
            candidates.append(row)

        candidates.sort(
            key=lambda r: (r["hit_at_3"], r["mrr_at_3"], r["config"]["candidate_pool"], r["config"]["fetch_multiplier"]),
            reverse=True,
        )
        best_per_route[route] = candidates[0]
        route_search_log[route] = candidates[:10]
        print(
            f"[{route}] queries={len(subset)} best_hit@3={candidates[0]['hit_at_3']:.4f} "
            f"best_mrr@3={candidates[0]['mrr_at_3']:.4f}"
        )

    routed_cfgs = {route: row["config"] for route, row in best_per_route.items()}
    routed = evaluate_routed(items, baseline_cfg, routed_cfgs, top_k=args.top_k)

    payload = {
        "model": args.model,
        "eval_set": str(eval_path.relative_to(root)) if eval_path.is_relative_to(root) else str(eval_path),
        "prep_report": {
            "documents_processed": prep_report.documents_processed,
            "chunks_after_dedup": prep_report.chunks_after_dedup,
            "vectors_upserted": index_report.vectors_upserted,
        },
        "route_distribution": {route: len(subset) for route, subset in sorted(route_items.items())},
        "baseline_global_config": {
            "dense_weight": baseline_cfg.dense_weight,
            "lexical_weight": baseline_cfg.lexical_weight,
            "field_weight": baseline_cfg.field_weight,
            "candidate_pool": baseline_cfg.candidate_pool,
            "fetch_multiplier": baseline_cfg.fetch_multiplier,
            "diversity_enabled": baseline_cfg.diversity_enabled,
            "code_mismatch_penalty": baseline_cfg.code_mismatch_penalty,
            "section_mismatch_penalty": baseline_cfg.section_mismatch_penalty,
        },
        "baseline": baseline,
        "best_per_route": best_per_route,
        "routed": routed,
        "delta_vs_baseline": {
            "hit_at_3": routed["hit_at_3"] - baseline["hit_at_3"],
            "mrr_at_3": routed["mrr_at_3"] - baseline["mrr_at_3"],
        },
        "route_search_top10": route_search_log,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved: {out_path}")
    print(
        f"baseline hit@3={baseline['hit_at_3']:.4f} mrr@3={baseline['mrr_at_3']:.4f} | "
        f"routed hit@3={routed['hit_at_3']:.4f} mrr@3={routed['mrr_at_3']:.4f}"
    )


if __name__ == "__main__":
    main()
