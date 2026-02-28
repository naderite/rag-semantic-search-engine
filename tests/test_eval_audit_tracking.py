from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS_DIR))

from run_eval_audit import compute_eval_score, load_history, select_previous_run


class EvalAuditTrackingTests(unittest.TestCase):
    def test_compute_eval_score_weighted(self) -> None:
        score = compute_eval_score(hit_at_3=0.8, mrr_at_3=0.7, hit_at_10=0.9)
        self.assertAlmostEqual(score, 0.77, places=8)

    def test_load_history_supports_jsonl(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"run_name": "a", "score": 0.70}),
                        json.dumps({"run_name": "b", "score": 0.75}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_history(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["run_name"], "b")

    def test_select_previous_run_uses_latest_matching_row(self) -> None:
        history = [
            {"eval_set": "tests/pdf_eval_queries_human.json", "model_name": "m1", "run_name": "default", "score": 0.65},
            {"eval_set": "tests/pdf_eval_queries_top3.json", "model_name": "m1", "run_name": "default", "score": 0.60},
            {"eval_set": "tests/pdf_eval_queries_human.json", "model_name": "m2", "run_name": "default", "score": 0.80},
            {"eval_set": "tests/pdf_eval_queries_human.json", "model_name": "m1", "run_name": "default", "score": 0.72},
        ]
        prev = select_previous_run(
            history=history,
            eval_set="tests/pdf_eval_queries_human.json",
            model_name="m1",
            run_name="default",
        )
        self.assertIsNotNone(prev)
        assert prev is not None
        self.assertEqual(prev["score"], 0.72)


if __name__ == "__main__":
    unittest.main()
