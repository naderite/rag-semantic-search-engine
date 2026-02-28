from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rag.answer_structure import (
    detect_query_intent,
    extract_structured_answer as extract_structured_answer_runtime,
    normalize_answer_text,
)
from rag.embeddings import SentenceTransformerEmbedder
from rag.index import build_vector_index
from rag.prepare import direct_chunks_from_pdfs
from rag.search import search_top_k
from rag.types import PrepConfig, SearchConfig, VectorConfig
from rag.vector_store import InMemoryVectorStore

ANSWER_TOKEN_F1_WEIGHT = 0.70
ANSWER_EXACT_WEIGHT = 0.10
ANSWER_NUMERIC_WEIGHT = 0.20

ANSWER_SCORE_TOP1_WEIGHT = 0.70
ANSWER_SCORE_TOP3_WEIGHT = 0.30


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


def extract_structured_answer(question: str, text: str, max_chars: int | None = None) -> str:
    return extract_structured_answer_runtime(question=question, text=text, max_chars=max_chars)


def canonicalize_answer_by_intent(question: str, text: str) -> str:
    if not text:
        return ""
    intent = detect_query_intent(question)
    structured = extract_structured_answer_runtime(question=question, text=text, max_chars=None)
    norm = normalize_answer_text(structured)
    if intent == "regulatory":
        refs = re.findall(r"\b\d{3,4}/\d{4}\b", norm)
        if refs:
            return "regulations " + " ".join(sorted(set(refs)))
    return norm


def compute_eval_score(hit_at_3: float, mrr_at_3: float, hit_at_10: float) -> float:
    # Precision-first aggregate score for quick model iteration tracking.
    return (0.50 * hit_at_3) + (0.40 * mrr_at_3) + (0.10 * hit_at_10)


def _answer_token_counter(text: str) -> Counter[str]:
    norm = normalize_answer_text(text)
    return Counter(tok for tok in norm.split() if tok)


def answer_exact_match(predicted: str, expected: str) -> float:
    if not predicted or not expected:
        return 0.0
    return 1.0 if normalize_answer_text(predicted) == normalize_answer_text(expected) else 0.0


def answer_token_f1(predicted: str, expected: str) -> float:
    pred = _answer_token_counter(predicted)
    exp = _answer_token_counter(expected)
    pred_n = sum(pred.values())
    exp_n = sum(exp.values())
    if pred_n == 0 or exp_n == 0:
        return 0.0
    overlap = sum((pred & exp).values())
    if overlap == 0:
        return 0.0
    precision = overlap / pred_n
    recall = overlap / exp_n
    return (2.0 * precision * recall) / (precision + recall)


def _extract_numeric_atoms(text: str) -> set[str]:
    norm = normalize_answer_text(text)
    out: set[str] = set()
    for raw in re.findall(r"(?<![a-z0-9])\d[\d\s]*(?:[.,]\d+)?(?![a-z0-9])", norm):
        compact = raw.replace(" ", "").replace(",", ".")
        try:
            value = float(compact)
        except ValueError:
            continue
        # Normalize numeric text so 10, 10.0, and 10.00 match.
        out.add(f"{value:.6f}".rstrip("0").rstrip("."))
    return out


def answer_numeric_f1(predicted: str, expected: str) -> float:
    pred = _extract_numeric_atoms(predicted)
    exp = _extract_numeric_atoms(expected)
    if not exp:
        return 1.0
    if not pred:
        return 0.0
    overlap = len(pred & exp)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(exp)
    return (2.0 * precision * recall) / (precision + recall)


def answer_pair_score(predicted: str, expected: str, question: str = "") -> dict[str, float]:
    exact_raw = answer_exact_match(predicted, expected)
    token_raw = answer_token_f1(predicted, expected)
    numeric_raw = answer_numeric_f1(predicted, expected)

    if question:
        predicted_canon = canonicalize_answer_by_intent(question, predicted)
        expected_canon = canonicalize_answer_by_intent(question, expected)
        exact = max(exact_raw, answer_exact_match(predicted_canon, expected_canon))
        token = max(token_raw, answer_token_f1(predicted_canon, expected_canon))
        numeric = max(numeric_raw, answer_numeric_f1(predicted_canon, expected_canon))
    else:
        exact = exact_raw
        token = token_raw
        numeric = numeric_raw

    score = (
        (ANSWER_TOKEN_F1_WEIGHT * token)
        + (ANSWER_EXACT_WEIGHT * exact)
        + (ANSWER_NUMERIC_WEIGHT * numeric)
    )
    return {
        "score": score,
        "token_f1": token,
        "exact_match": exact,
        "numeric_f1": numeric,
    }


def best_answer_match(predicted: str | None, expected_answers: list[str], question: str = "") -> dict:
    if not predicted or not expected_answers:
        return {
            "score": 0.0,
            "token_f1": 0.0,
            "exact_match": 0.0,
            "numeric_f1": 0.0,
            "matched_expected_answer": None,
            "predicted_answer": predicted,
        }

    best = {
        "score": -1.0,
        "token_f1": 0.0,
        "exact_match": 0.0,
        "numeric_f1": 0.0,
        "matched_expected_answer": None,
        "predicted_answer": predicted,
    }
    for expected in expected_answers:
        current = answer_pair_score(predicted, expected, question=question)
        if current["score"] > best["score"]:
            best = {
                **current,
                "matched_expected_answer": expected,
                "predicted_answer": predicted,
            }

    return best


def best_of_topk_answer_match(predicted_answers: list[str], expected_answers: list[str], question: str = "") -> dict:
    if not predicted_answers:
        return {
            "score": 0.0,
            "token_f1": 0.0,
            "exact_match": 0.0,
            "numeric_f1": 0.0,
            "matched_expected_answer": None,
            "predicted_answer": None,
        }

    best = {
        "score": -1.0,
        "token_f1": 0.0,
        "exact_match": 0.0,
        "numeric_f1": 0.0,
        "matched_expected_answer": None,
        "predicted_answer": None,
    }
    for predicted in predicted_answers:
        current = best_answer_match(predicted=predicted, expected_answers=expected_answers, question=question)
        if current["score"] > best["score"]:
            best = current
    return best


def compute_answer_score(answer_score_top1_mean: float, answer_score_top3_mean: float) -> float:
    return (ANSWER_SCORE_TOP1_WEIGHT * answer_score_top1_mean) + (ANSWER_SCORE_TOP3_WEIGHT * answer_score_top3_mean)


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
    parser = argparse.ArgumentParser(description="Run retrieval audit on eval query JSON.")
    parser.add_argument("--eval-path", default=str(root / "tests" / "pdf_eval_queries_top3.json"))
    parser.add_argument("--output", default=str(root / "artifacts" / "eval_audit_report.json"))
    parser.add_argument("--history-path", default=str(root / "artifacts" / "eval_score_history.jsonl"))
    parser.add_argument("--embedding-model", default="models/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-name", default="default")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.is_absolute():
        eval_path = (root / eval_path).resolve()
    data_dir = root / "data"

    items = load_eval_items(eval_path)

    os.environ.setdefault("RAG_EMBEDDING_LOCAL_ONLY", "1")
    store = InMemoryVectorStore()
    embedder = SentenceTransformerEmbedder(args.embedding_model)
    tracking_model_name = args.model_name or args.embedding_model

    chunks, prep_report = direct_chunks_from_pdfs(str(data_dir), PrepConfig())
    index_report = build_vector_index(
        chunks,
        VectorConfig(vector_store=store, embedder=embedder, embedding_model=args.embedding_model),
    )
    cfg = SearchConfig(vector_store=store, embedder=embedder, embedding_model=args.embedding_model, context_window=0)

    hit_at_3 = 0
    rr_sum = 0.0
    hit_at_10 = 0
    category_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "hit3": 0.0, "rr": 0.0, "hit10": 0.0})
    false_top1 = Counter()
    retrieval_bottlenecks = 0
    rerank_bottlenecks = 0
    answer_score_top1_sum = 0.0
    answer_score_top3_sum = 0.0
    answer_exact_top1_count = 0
    answer_numeric_top1_count = 0

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
                "answer_excerpt": extract_structured_answer(question=question, text=r.text),
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
        expected_answers = [row["candidate_answer"] for row in expected_rows if isinstance(row.get("candidate_answer"), str)]
        predicted_answer = retrieved_top3[0]["answer_excerpt"] if retrieved_top3 else None
        predicted_top3_answers = [row["answer_excerpt"] for row in retrieved_top3 if row.get("answer_excerpt")]
        answer_eval_top1 = best_answer_match(
            predicted=predicted_answer,
            expected_answers=expected_answers,
            question=question,
        )
        answer_eval_top3 = best_of_topk_answer_match(
            predicted_answers=predicted_top3_answers,
            expected_answers=expected_answers,
            question=question,
        )

        answer_score_top1_sum += float(answer_eval_top1["score"])
        answer_score_top3_sum += float(answer_eval_top3["score"])
        answer_exact_top1_count += int(answer_eval_top1["exact_match"] >= 1.0)
        answer_numeric_top1_count += int(answer_eval_top1["numeric_f1"] >= 1.0)

        detailed.append(
            {
                "id": item["id"],
                "category": category,
                "query": question,
                "expected": sorted(expected),
                "top3": ranked3,
                "expected_supporting": expected_supporting,
                "retrieved_top3": retrieved_top3,
                "predicted_answer": predicted_answer,
                "answer_eval": {
                    "top1": answer_eval_top1,
                    "top3_best": answer_eval_top3,
                },
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
    answer_score_top1_mean = answer_score_top1_sum / total
    answer_score_top3_mean = answer_score_top3_sum / total
    answer_exact_match_top1_rate = answer_exact_top1_count / total
    answer_numeric_match_top1_rate = answer_numeric_top1_count / total
    answer_score_value = compute_answer_score(answer_score_top1_mean, answer_score_top3_mean)

    history_path = Path(args.history_path)
    if not history_path.is_absolute():
        history_path = (root / history_path).resolve()

    eval_set_key = str(eval_path.relative_to(root)) if eval_path.is_relative_to(root) else str(eval_path)
    history = load_history(history_path)
    previous = select_previous_run(
        history=history,
        eval_set=eval_set_key,
        model_name=tracking_model_name,
        run_name=args.run_name,
    )
    delta = None
    if previous is not None:
        delta = {
            "score_delta": score_value - float(previous.get("score", 0.0)),
            "hit_at_3_delta": hit3_value - float(previous.get("hit_at_3", 0.0)),
            "hit_at_10_delta": hit10_value - float(previous.get("hit_at_10", 0.0)),
            "mrr_at_3_delta": mrr3_value - float(previous.get("mrr_at_3", 0.0)),
            "answer_score_delta": (
                answer_score_value - float(previous["answer_score"])
                if previous.get("answer_score") is not None
                else None
            ),
            "answer_score_top1_mean_delta": (
                answer_score_top1_mean - float(previous["answer_score_top1_mean"])
                if previous.get("answer_score_top1_mean") is not None
                else None
            ),
            "answer_score_top3_mean_delta": (
                answer_score_top3_mean - float(previous["answer_score_top3_mean"])
                if previous.get("answer_score_top3_mean") is not None
                else None
            ),
            "answer_exact_match_top1_rate_delta": (
                answer_exact_match_top1_rate - float(previous["answer_exact_match_top1_rate"])
                if previous.get("answer_exact_match_top1_rate") is not None
                else None
            ),
            "answer_numeric_match_top1_rate_delta": (
                answer_numeric_match_top1_rate - float(previous["answer_numeric_match_top1_rate"])
                if previous.get("answer_numeric_match_top1_rate") is not None
                else None
            ),
            "previous_run_at": previous.get("run_at"),
        }

    print(f"=== EVAL AUDIT ({tracking_model_name}) ===")
    print(f"chunks_indexed: {len(chunks)}")
    print(f"vectors_upserted: {index_report.vectors_upserted}")
    print(f"queries: {total}")
    print(f"hit@3: {hit3_value:.4f}")
    print(f"hit@10: {hit10_value:.4f}")
    print(f"MRR@3: {mrr3_value:.4f}")
    print(f"score: {score_value:.4f} (0.50*hit@3 + 0.40*mrr@3 + 0.10*hit@10)")
    print(f"answer_score_top1_mean: {answer_score_top1_mean:.4f}")
    print(f"answer_score_top3_mean: {answer_score_top3_mean:.4f}")
    print(f"answer_exact_match_top1_rate: {answer_exact_match_top1_rate:.4f}")
    print(f"answer_numeric_match_top1_rate: {answer_numeric_match_top1_rate:.4f}")
    print(
        f"answer_score: {answer_score_value:.4f} "
        f"({ANSWER_SCORE_TOP1_WEIGHT:.2f}*top1 + {ANSWER_SCORE_TOP3_WEIGHT:.2f}*top3)"
    )
    if delta is not None:
        answer_score_delta_str = "n/a"
        if delta["answer_score_delta"] is not None:
            answer_score_delta_str = f"{delta['answer_score_delta']:+.4f}"
        print(
            "delta_vs_previous: "
            f"score={delta['score_delta']:+.4f} "
            f"hit@3={delta['hit_at_3_delta']:+.4f} "
            f"hit@10={delta['hit_at_10_delta']:+.4f} "
            f"mrr@3={delta['mrr_at_3_delta']:+.4f} "
            f"answer_score={answer_score_delta_str}"
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
                    "answer_score_top1_mean": answer_score_top1_mean,
                    "answer_score_top3_mean": answer_score_top3_mean,
                    "answer_exact_match_top1_rate": answer_exact_match_top1_rate,
                    "answer_numeric_match_top1_rate": answer_numeric_match_top1_rate,
                    "answer_score": answer_score_value,
                    "answer_score_formula": (
                        f"{ANSWER_SCORE_TOP1_WEIGHT:.2f}*answer_score_top1_mean + "
                        f"{ANSWER_SCORE_TOP3_WEIGHT:.2f}*answer_score_top3_mean"
                    ),
                    "delta_vs_previous": delta,
                    "failed_at_3": failed,
                    "retrieval_bottlenecks": retrieval_bottlenecks,
                    "rerank_bottlenecks": rerank_bottlenecks,
                },
                "tracking": {
                    "eval_set": eval_set_key,
                    "model_name": tracking_model_name,
                    "embedding_model": args.embedding_model,
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
        "model_name": tracking_model_name,
        "embedding_model": args.embedding_model,
        "run_name": args.run_name,
        "queries": total,
        "score": score_value,
        "hit_at_3": hit3_value,
        "hit_at_10": hit10_value,
        "mrr_at_3": mrr3_value,
        "answer_score_top1_mean": answer_score_top1_mean,
        "answer_score_top3_mean": answer_score_top3_mean,
        "answer_exact_match_top1_rate": answer_exact_match_top1_rate,
        "answer_numeric_match_top1_rate": answer_numeric_match_top1_rate,
        "answer_score": answer_score_value,
        "output_report": str(out_path),
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_row, ensure_ascii=False) + "\n")

    print(f"saved_report: {out_path}")
    print(f"saved_history: {history_path}")


if __name__ == "__main__":
    main()
