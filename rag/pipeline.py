from __future__ import annotations

from .index import build_vector_index
from .prepare import direct_chunks_from_pdfs, load_chunks_jsonl, prepare_documents
from .search import search_top_k
from .types import PrepConfig, QualityEvalReport, SearchConfig, VectorConfig


def prepare_and_index(input_dir: str, output_dir: str, prep_config: PrepConfig, vector_config: VectorConfig):
    prep_report = prepare_documents(input_dir=input_dir, output_dir=output_dir, config=prep_config)
    chunks = load_chunks_jsonl(f"{output_dir}/chunks.jsonl")
    index_report = build_vector_index(chunks=chunks, cfg=vector_config)
    return prep_report, index_report


def index_pdfs_directly(input_dir: str, prep_config: PrepConfig, vector_config: VectorConfig):
    chunks, prep_report = direct_chunks_from_pdfs(input_dir=input_dir, config=prep_config)
    index_report = build_vector_index(chunks=chunks, cfg=vector_config)
    return prep_report, index_report


def answer_question(question: str, search_config: SearchConfig, top_k: int = 6):
    return search_top_k(question=question, k=top_k, cfg=search_config)


def run_quality_eval(
    cases: list[tuple[str, set[str]]],
    search_config: SearchConfig,
    top_k: int = 3,
) -> QualityEvalReport:
    if not cases:
        return QualityEvalReport(total_queries=0, hit_at_k=0.0, mrr=0.0)

    hits = 0
    rr_sum = 0.0
    for question, expected_doc_ids in cases:
        response = search_top_k(question=question, k=top_k, cfg=search_config)
        ranks = [idx + 1 for idx, item in enumerate(response.results) if item.doc_id in expected_doc_ids]
        if ranks:
            hits += 1
            rr_sum += 1.0 / min(ranks)

    total = len(cases)
    return QualityEvalReport(total_queries=total, hit_at_k=hits / total, mrr=rr_sum / total)
