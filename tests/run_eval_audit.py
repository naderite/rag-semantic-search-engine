from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rag.index import build_vector_index
from rag.prepare import direct_chunks_from_pdfs
from rag.search import search_top_k
from rag.types import PrepConfig, SearchConfig, VectorConfig
from rag.vector_store import InMemoryVectorStore


class OfflineEvalEmbedder:
    def _tokenize(self, text: str) -> set[str]:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
        text = re.sub(r"[^a-z0-9%/\-\s]", " ", text)
        return {tok for tok in text.split() if tok}

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            t = self._tokenize(text)
            has = lambda *keys: float(any(k in t for k in keys))
            ppm = float("ppm" in t or "%" in text)
            vectors.append(
                [
                    has("dosage", "dose", "recommande", "recommended", "optimum", "max"),
                    has("bread", "pain", "panification", "mie", "bakery"),
                    has("hcf", "hcb", "xylanase"),
                    has("ascorbique", "e300", "vitamine", "acide"),
                    has("allergens", "gluten"),
                    has("heavy", "metals", "cadmium", "lead", "mg/kg"),
                    has("storage", "stockage", "durability", "months", "mois", "temperature", "humidite"),
                    ppm,
                    has("gox", "glucose", "oxidase", "go"),
                    has("tg", "transglutaminase"),
                    has("af110", "af220", "af330", "amylase", "fau", "skb"),
                    has("amg", "amyloglucosidase", "agi"),
                    has("fresh", "soft", "nmau"),
                    has("lipase", "lmax", "l55", "l65"),
                    has("regulation", "autorise", "directive", "codex", "innorpi"),
                    has("conversion", "kg", "grammes", "g", "ph"),
                ]
            )
        return vectors


def doc_id_from_source_file(source_file: str) -> str:
    return Path(source_file).stem


def load_eval_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def answer_excerpt(text: str, max_chars: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def compute_eval_score(hit_at_3: float, mrr_at_3: float, hit_at_10: float) -> float:
    # Precision-first aggregate score for quick model iteration tracking.
    return (0.50 * hit_at_3) + (0.40 * mrr_at_3) + (0.10 * hit_at_10)


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Support both JSONL and JSON-array history formats.
    if text.startswith("["):
        payload = json.loads(text)
        return payload if isinstance(payload, list) else []

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def select_previous_run(history: list[dict], eval_set: str, model_name: str, run_name: str) -> dict | None:
    for row in reversed(history):
        if row.get("eval_set") != eval_set:
            continue
        if row.get("model_name") != model_name:
            continue
        if row.get("run_name") != run_name:
            continue
        return row
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run offline retrieval audit on eval query JSON.")
    parser.add_argument("--eval-path", default=str(root / "tests" / "pdf_eval_queries_top3.json"))
    parser.add_argument("--output", default=str(root / "artifacts" / "eval_audit_report.json"))
    parser.add_argument("--history-path", default=str(root / "artifacts" / "eval_score_history.jsonl"))
    parser.add_argument("--model-name", default="offline_eval_embedder")
    parser.add_argument("--run-name", default="default")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = (root / eval_path).resolve()
    data_dir = root / "data"

    items = load_eval_items(eval_path)

    store = InMemoryVectorStore()
    embedder = OfflineEvalEmbedder()

    chunks, prep_report = direct_chunks_from_pdfs(str(data_dir), PrepConfig())
    index_report = build_vector_index(chunks, VectorConfig(vector_store=store, embedder=embedder))
    cfg = SearchConfig(vector_store=store, embedder=embedder, context_window=0)

    hit_at_3 = 0
    rr_sum = 0.0
    hit_at_10 = 0
    category_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "hit3": 0.0, "rr": 0.0, "hit10": 0.0})
    false_top1 = Counter()
    retrieval_bottlenecks = 0
    rerank_bottlenecks = 0

    detailed = []

    for item in items:
        question = item["query"]
        category = item.get("category", "uncategorized")
        expected_rows = item.get("expected_top3", [])
        expected = {
            doc_id_from_source_file(c["source_file"])
            for c in expected_rows
            if c.get("source_file")
        }

        resp3 = search_top_k(question, 3, cfg)
        resp10 = search_top_k(question, 10, cfg)

        ranked3 = [r.doc_id for r in resp3.results]
        ranked10 = [r.doc_id for r in resp10.results]

        ranks3 = [i + 1 for i, d in enumerate(ranked3) if d in expected]
        ranks10 = [i + 1 for i, d in enumerate(ranked10) if d in expected]

        hit3 = bool(ranks3)
        hit10 = bool(ranks10)
        rr = (1.0 / min(ranks3)) if ranks3 else 0.0

        hit_at_3 += int(hit3)
        hit_at_10 += int(hit10)
        rr_sum += rr

        category_stats[category]["n"] += 1
        category_stats[category]["hit3"] += int(hit3)
        category_stats[category]["hit10"] += int(hit10)
        category_stats[category]["rr"] += rr

        if not hit3:
            if hit10:
                rerank_bottlenecks += 1
            else:
                retrieval_bottlenecks += 1
            if resp3.results:
                false_top1[resp3.results[0].doc_id] += 1

        retrieved_top3 = [
            {
                "rank": r.rank,
                "doc_id": r.doc_id,
                "page": r.page,
                "final_score": r.final_score,
                "dense_score": r.dense_score,
                "lex_score": r.lex_score,
                "field_score": r.field_score,
                "matched_terms": r.matched_terms,
                "answer_excerpt": answer_excerpt(r.text),
            }
            for r in resp3.results
        ]
        expected_supporting = [
            {
                "source_file": row.get("source_file"),
                "doc_id": doc_id_from_source_file(row["source_file"]) if row.get("source_file") else None,
                "page": row.get("page"),
                "candidate_answer": row.get("candidate_answer"),
            }
            for row in expected_rows
        ]

        detailed.append(
            {
                "id": item["id"],
                "category": category,
                "query": question,
                "expected": sorted(expected),
                "top3": ranked3,
                "expected_supporting": expected_supporting,
                "retrieved_top3": retrieved_top3,
                "predicted_answer": retrieved_top3[0]["answer_excerpt"] if retrieved_top3 else None,
                "hit3": hit3,
                "hit10": hit10,
                "rr": rr,
            }
        )

    total = len(items)

    hit3_value = hit_at_3 / total
    hit10_value = hit_at_10 / total
    mrr3_value = rr_sum / total
    score_value = compute_eval_score(hit3_value, mrr3_value, hit10_value)

    history_path = Path(args.history_path)
    if not history_path.is_absolute():
        history_path = (root / history_path).resolve()

    eval_set_key = str(eval_path.relative_to(root)) if eval_path.is_relative_to(root) else str(eval_path)
    history = load_history(history_path)
    previous = select_previous_run(
        history=history,
        eval_set=eval_set_key,
        model_name=args.model_name,
        run_name=args.run_name,
    )
    delta = None
    if previous is not None:
        delta = {
            "score_delta": score_value - float(previous.get("score", 0.0)),
            "hit_at_3_delta": hit3_value - float(previous.get("hit_at_3", 0.0)),
            "hit_at_10_delta": hit10_value - float(previous.get("hit_at_10", 0.0)),
            "mrr_at_3_delta": mrr3_value - float(previous.get("mrr_at_3", 0.0)),
            "previous_run_at": previous.get("run_at"),
        }

    print("=== EVAL AUDIT (offline embedder) ===")
    print(f"chunks_indexed: {len(chunks)}")
    print(f"vectors_upserted: {index_report.vectors_upserted}")
    print(f"queries: {total}")
    print(f"hit@3: {hit3_value:.4f}")
    print(f"hit@10: {hit10_value:.4f}")
    print(f"MRR@3: {mrr3_value:.4f}")
    print(f"score: {score_value:.4f} (0.50*hit@3 + 0.40*mrr@3 + 0.10*hit@10)")
    if delta is not None:
        print(
            "delta_vs_previous: "
            f"score={delta['score_delta']:+.4f} "
            f"hit@3={delta['hit_at_3_delta']:+.4f} "
            f"hit@10={delta['hit_at_10_delta']:+.4f} "
            f"mrr@3={delta['mrr_at_3_delta']:+.4f}"
        )
    else:
        print("delta_vs_previous: n/a (no matching baseline yet)")
    print()

    print("=== Per-category ===")
    for category in sorted(category_stats):
        s = category_stats[category]
        n = int(s["n"])
        print(
            f"{category}: n={n} hit@3={s['hit3']/n:.3f} hit@10={s['hit10']/n:.3f} mrr={s['rr']/n:.3f}"
        )
    print()

    print("=== Bottleneck split (on failed @3) ===")
    failed = total - hit_at_3
    print(f"failed@3: {failed}")
    print(f"retrieval_bottlenecks_not_in_top10: {retrieval_bottlenecks}")
    print(f"rerank_bottlenecks_in_top10_not_top3: {rerank_bottlenecks}")
    print()

    print("=== Most common wrong top-1 docs (failed queries) ===")
    for doc_id, count in false_top1.most_common(10):
        print(f"{doc_id}: {count}")
    print()

    print("=== Failed queries ===")
    for row in detailed:
        if row["hit3"]:
            continue
        print(f"{row['id']} [{row['category']}] hit10={row['hit10']} rr={row['rr']:.3f}")
        print(f"Q: {row['query']}")
        print(f"expected: {row['expected']}")
        print(f"top3: {row['top3']}")
        if row["retrieved_top3"]:
            print("retrieved answers:")
            for cand in row["retrieved_top3"][:3]:
                print(
                    f"  - rank={cand['rank']} doc={cand['doc_id']} page={cand['page']} "
                    f"score={cand['final_score']:.4f} ans={cand['answer_excerpt']}"
                )
        print()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": {
                    "queries": total,
                    "hit_at_3": hit3_value,
                    "hit_at_10": hit10_value,
                    "mrr_at_3": mrr3_value,
                    "score": score_value,
                    "score_formula": "0.50*hit_at_3 + 0.40*mrr_at_3 + 0.10*hit_at_10",
                    "delta_vs_previous": delta,
                    "failed_at_3": failed,
                    "retrieval_bottlenecks": retrieval_bottlenecks,
                    "rerank_bottlenecks": rerank_bottlenecks,
                },
                "tracking": {
                    "eval_set": eval_set_key,
                    "model_name": args.model_name,
                    "run_name": args.run_name,
                    "history_path": str(history_path),
                },
                "by_category": category_stats,
                "false_top1": false_top1,
                "details": detailed,
            },
            indent=2,
            ensure_ascii=False,
            default=dict,
        ),
        encoding="utf-8",
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_row = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "eval_set": eval_set_key,
        "model_name": args.model_name,
        "run_name": args.run_name,
        "queries": total,
        "score": score_value,
        "hit_at_3": hit3_value,
        "hit_at_10": hit10_value,
        "mrr_at_3": mrr3_value,
        "output_report": str(out_path),
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_row, ensure_ascii=False) + "\n")

    print(f"saved_report: {out_path}")
    print(f"saved_history: {history_path}")


if __name__ == "__main__":
    main()
