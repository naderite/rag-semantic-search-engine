from __future__ import annotations

import json
import math
import re
import unittest
from collections import Counter
from pathlib import Path


class EvalDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.top3_path = self.root / "tests" / "pdf_eval_queries_top3.json"
        self.human_path = self.root / "tests" / "pdf_eval_queries_human.json"
        self.top3 = json.loads(self.top3_path.read_text(encoding="utf-8"))
        self.human = json.loads(self.human_path.read_text(encoding="utf-8"))

    def test_human_dataset_has_same_ids_as_top3(self) -> None:
        top_ids = {row["id"] for row in self.top3}
        human_ids = {row["id"] for row in self.human}
        self.assertEqual(top_ids, human_ids)

    def test_human_dataset_preserves_expected_sources(self) -> None:
        by_id_top = {row["id"]: row for row in self.top3}
        by_id_human = {row["id"]: row for row in self.human}
        for item_id, top_row in by_id_top.items():
            human_row = by_id_human[item_id]
            top_sources = [c.get("source_file") for c in top_row.get("expected_top3", [])]
            human_sources = [c.get("source_file") for c in human_row.get("expected_top3", [])]
            self.assertEqual(top_sources, human_sources)

    def test_human_queries_look_natural(self) -> None:
        for row in self.human:
            query = row["query"].strip()
            self.assertGreaterEqual(len(query.split()), 8)
            self.assertTrue(query.endswith("?"))
            self.assertNotIn("  ", query)
            # Human phrasing should avoid overly templated "what are the dosage values" style.
            self.assertNotIn("candidate_answer", query.lower())

    def test_human_queries_are_rephrased_not_copied(self) -> None:
        top_by_id = {row["id"]: row["query"].strip() for row in self.top3}
        for row in self.human:
            human_query = row["query"].strip()
            top_query = top_by_id[row["id"]]
            self.assertNotEqual(human_query, top_query)

    def test_human_query_corpus_has_conversational_diversity(self) -> None:
        opening_words = Counter()
        formal_prefix = re.compile(r"^(what|which|for|in|among|between|within)\b", re.IGNORECASE)
        conversational_marker = re.compile(r"\b(i|my|we|our|can you|could you|should i|do they|if i|i am|i'm)\b", re.IGNORECASE)

        formal_count = 0
        conversational_count = 0
        total = len(self.human)

        for row in self.human:
            query = row["query"].strip()
            first_word = re.sub(r"[^a-z0-9]+", "", query.split()[0].lower())
            if first_word:
                opening_words[first_word] += 1
            if formal_prefix.match(query):
                formal_count += 1
            if conversational_marker.search(query):
                conversational_count += 1

        # Avoid a single repeated opening pattern ("What...", "For...", etc.) dominating the dataset.
        self.assertLessEqual(max(opening_words.values()), math.ceil(total * 0.30))
        # Keep some formal questions, but not as the dominant style.
        self.assertLessEqual(formal_count, int(total * 0.65))
        # Ensure a meaningful share of real user-like phrasing.
        self.assertGreaterEqual(conversational_count, math.ceil(total * 0.30))


if __name__ == "__main__":
    unittest.main()
