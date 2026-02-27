from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    eval_path = root / "tests" / "pdf_eval_queries_top3.json"
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
        expected = {
            doc_id_from_source_file(c["source_file"])
            for c in item.get("expected_top3", [])
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

        detailed.append(
            {
                "id": item["id"],
                "category": category,
                "query": question,
                "expected": sorted(expected),
                "top3": ranked3,
                "hit3": hit3,
                "hit10": hit10,
                "rr": rr,
            }
        )

    total = len(items)

    print("=== EVAL AUDIT (offline embedder) ===")
    print(f"chunks_indexed: {len(chunks)}")
    print(f"vectors_upserted: {index_report.vectors_upserted}")
    print(f"queries: {total}")
    print(f"hit@3: {hit_at_3 / total:.4f}")
    print(f"hit@10: {hit_at_10 / total:.4f}")
    print(f"MRR@3: {rr_sum / total:.4f}")
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
        print()

    out_path = root / "artifacts" / "eval_audit_report.json"
    out_path.write_text(
        json.dumps(
            {
                "summary": {
                    "queries": total,
                    "hit_at_3": hit_at_3 / total,
                    "hit_at_10": hit_at_10 / total,
                    "mrr_at_3": rr_sum / total,
                    "failed_at_3": failed,
                    "retrieval_bottlenecks": retrieval_bottlenecks,
                    "rerank_bottlenecks": rerank_bottlenecks,
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
    print(f"saved_report: {out_path}")


if __name__ == "__main__":
    main()
