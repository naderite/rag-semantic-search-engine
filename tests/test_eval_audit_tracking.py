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
from run_eval_audit import (
    answer_exact_match,
    answer_numeric_f1,
    answer_pair_score,
    answer_token_f1,
    best_answer_match,
    best_of_topk_answer_match,
    compute_answer_score,
    extract_structured_answer,
    normalize_answer_text,
)


class EvalAuditTrackingTests(unittest.TestCase):
    def test_compute_eval_score_weighted(self) -> None:
        score = compute_eval_score(hit_at_3=0.8, mrr_at_3=0.7, hit_at_10=0.9)
        self.assertAlmostEqual(score, 0.77, places=8)

    def test_compute_answer_score_weighted(self) -> None:
        score = compute_answer_score(answer_score_top1_mean=0.8, answer_score_top3_mean=0.6)
        self.assertAlmostEqual(score, 0.74, places=8)

    def test_normalize_answer_text_strips_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_answer_text("BVZyme GOX 110®: 5-40 ppm."), "bvzyme gox 110 5-40 ppm")

    def test_answer_exact_match_uses_normalized_text(self) -> None:
        self.assertEqual(answer_exact_match("5-40 ppm.", "5-40 PPM"), 1.0)

    def test_answer_token_f1_partial_overlap(self) -> None:
        f1 = answer_token_f1("dosage 5-40 ppm", "recommended dosage 5-40 ppm")
        self.assertGreater(f1, 0.8)
        self.assertLess(f1, 1.0)

    def test_answer_numeric_f1_matches_numbers(self) -> None:
        self.assertEqual(answer_numeric_f1("5-40 ppm", "5 to 40 ppm"), 1.0)
        self.assertEqual(answer_numeric_f1("5-30 ppm", "5 to 40 ppm"), 0.5)

    def test_answer_pair_score_combines_components(self) -> None:
        row = answer_pair_score("5-40 ppm", "5 to 40 ppm")
        self.assertIn("score", row)
        self.assertIn("token_f1", row)
        self.assertIn("exact_match", row)
        self.assertIn("numeric_f1", row)
        self.assertGreater(row["score"], 0.0)

    def test_answer_pair_score_uses_intent_canonicalization(self) -> None:
        row = answer_pair_score(
            "recommended dose is 5 to 40 parts per million",
            "dosage: 5-40 ppm",
            question="what dosage range is recommended?",
        )
        self.assertGreaterEqual(row["numeric_f1"], 1.0)
        self.assertGreater(row["score"], 0.9)

    def test_best_answer_match_picks_highest_scoring_expected_answer(self) -> None:
        row = best_answer_match(
            predicted="activity 10000 u/g",
            expected_answers=["dosage 5-40 ppm", "activity 10000 u g"],
        )
        self.assertEqual(row["matched_expected_answer"], "activity 10000 u g")
        self.assertGreater(row["score"], 0.45)

    def test_best_of_topk_answer_match_picks_best_candidate(self) -> None:
        row = best_of_topk_answer_match(
            predicted_answers=["dosage 5-30 ppm", "dosage 5-40 ppm"],
            expected_answers=["dosage 5-40 ppm"],
        )
        self.assertEqual(row["predicted_answer"], "dosage 5-40 ppm")
        self.assertEqual(row["exact_match"], 1.0)

    def test_extract_structured_answer_dosage(self) -> None:
        text = "Product info. Dosage 5-40 ppm. Additional notes."
        out = extract_structured_answer("what dosage range is recommended?", text)
        self.assertEqual(out, "dosage: 5-40 ppm")

    def test_extract_structured_answer_activity(self) -> None:
        text = "Technical data. Activity 10000U/g. Application bakery."
        out = extract_structured_answer("can you confirm activity value?", text)
        self.assertEqual(out, "activity: 10000 u/g")

    def test_extract_structured_answer_metals(self) -> None:
        text = "Heavy metals: Cadmium: <0.5 mg/kg. Lead: <5 mg/kg."
        out = extract_structured_answer("what are cadmium and lead limits?", text)
        self.assertIn("cadmium 0.5 mg/kg", out)
        self.assertIn("lead 5 mg/kg", out)

    def test_extract_structured_answer_storage_variants(self) -> None:
        text = "Shelf-life 12 months. Store at or below 25 Celsius. Relative humidity below 65%."
        out = extract_structured_answer("storage and shelf life?", text)
        self.assertIn("shelf life: 12 months", out)
        self.assertIn("temperature: 25 c", out)
        self.assertIn("humidity: <65%", out)

    def test_extract_structured_answer_does_not_truncate_fallback(self) -> None:
        long_text = " ".join(["value"] * 120)
        out = extract_structured_answer("give full answer", long_text)
        self.assertTrue(out.endswith("value"))
        self.assertGreater(len(out), 280)

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

    def test_previous_run_without_answer_fields_defaults_to_none_deltas(self) -> None:
        previous = {
            "score": 0.70,
            "hit_at_3": 0.60,
            "hit_at_10": 0.90,
            "mrr_at_3": 0.55,
        }
        self.assertIsNone(previous.get("answer_score"))


if __name__ == "__main__":
    unittest.main()
