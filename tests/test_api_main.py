from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from rag.types import SearchResponse as RagSearchResponse
from rag.types import SearchResult


class ApiMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_search_returns_structured_answer_and_scores(self) -> None:
        fake = RagSearchResponse(
            question="what dosage range is recommended?",
            results=[
                SearchResult(
                    rank=1,
                    chunk_id="c1",
                    text="Product info. Dosage 5-40 ppm. Additional notes.",
                    score=0.9,
                    doc_id="doc1",
                    page=1,
                    final_score=0.9,
                )
            ],
        )
        with patch("backend.main.search_top_k", return_value=fake):
            resp = self.client.post("/search", json={"question": "what dosage range is recommended?"})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["results"][0]["text"], "dosage: 5-40 ppm")
        self.assertIn("score", payload["results"][0])
        self.assertIn("retrieval_score", payload["results"][0])
        self.assertIn("answer_score", payload["results"][0])
        self.assertGreater(payload["results"][0]["answer_score"], 0.9)
        self.assertGreater(payload["results"][0]["score"], 0.0)
        self.assertLessEqual(payload["results"][0]["score"], 1.0)

    def test_search_no_truncation_for_long_fallback_answer(self) -> None:
        long_text = " ".join(["value"] * 400)
        fake = RagSearchResponse(
            question="tell me about this product",
            results=[
                SearchResult(
                    rank=1,
                    chunk_id="c2",
                    text=long_text,
                    score=0.1,
                    doc_id="doc2",
                    page=1,
                    final_score=0.1,
                )
            ],
        )
        with patch("backend.main.search_top_k", return_value=fake):
            resp = self.client.post("/search", json={"question": "tell me about this product"})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        returned = payload["results"][0]["text"]
        self.assertGreater(len(returned), 280)
        self.assertTrue(returned.endswith("value"))

    def test_combined_score_matches_formula(self) -> None:
        fake = RagSearchResponse(
            question="regulatory status?",
            results=[
                SearchResult(
                    rank=1,
                    chunk_id="c3",
                    text="According to European Regulations 1829/2003 and 1830/2003, no specific labeling is required.",
                    score=1.2,
                    doc_id="doc3",
                    page=1,
                    final_score=1.2,
                )
            ],
        )
        with patch("backend.main.search_top_k", return_value=fake):
            resp = self.client.post("/search", json={"question": "regulatory status?"})
        self.assertEqual(resp.status_code, 200)
        row = resp.json()["results"][0]
        expected = (0.35 * row["retrieval_score"]) + (0.65 * row["answer_score"])
        self.assertAlmostEqual(row["score"], expected, places=8)


if __name__ == "__main__":
    unittest.main()
