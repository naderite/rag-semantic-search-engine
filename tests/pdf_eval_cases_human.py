from __future__ import annotations

import json
from pathlib import Path


def _doc_id_from_source_file(source_file: str) -> str:
    return Path(source_file).stem


def load_cases_from_human(
    json_path: str | Path = Path(__file__).with_name("pdf_eval_queries_human.json"),
) -> list[tuple[str, set[str]]]:
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    cases: list[tuple[str, set[str]]] = []
    for item in payload:
        question = item["query"]
        expected_doc_ids = {
            _doc_id_from_source_file(candidate["source_file"])
            for candidate in item.get("expected_top3", [])
            if candidate.get("source_file")
        }
        cases.append((question, expected_doc_ids))

    return cases


CASES_HUMAN: list[tuple[str, set[str]]] = load_cases_from_human()

