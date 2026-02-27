from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rag.index import build_vector_index
from rag.pipeline import index_pdfs_directly
from rag.prepare import chunk_blocks, direct_chunks_from_blocks, normalize_blocks
from rag.search import search_top_k
from rag.types import NormalizedBlock, PrepConfig, RawPageBlock, SearchConfig, VectorConfig
from rag.vector_store import InMemoryVectorStore


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            t = text.lower()
            vectors.append(
                [
                    float("dosage" in t or "ppm" in t),
                    float("storage" in t or "stockage" in t),
                    float("bread" in t or "pain" in t),
                ]
            )
        return vectors


class RagModuleTests(unittest.TestCase):
    def test_table_normalization_to_kv(self) -> None:
        blocks = [
            RawPageBlock(
                doc_id="doc1",
                file_name="doc1.pdf",
                page=2,
                block_type="table",
                text="",
                rows=[
                    ["Category", "Parameter", "Limit"],
                    ["Heavy metals", "Cadmium", "< 0,5 mg/kg"],
                    ["Heavy metals", "Lead", "< 5 mg/kg"],
                ],
            )
        ]

        normalized = normalize_blocks(blocks, config=PrepConfig())
        table_rows = [b for b in normalized if b.chunk_type == "table_kv"]
        self.assertEqual(len(table_rows), 2)
        self.assertTrue(all(b.chunk_type == "table_kv" for b in table_rows))
        self.assertIn("champ « Heavy metals »", table_rows[0].text)
        self.assertIn("mg/kg", normalized[0].text)

    def test_chunk_blocks_keeps_table_atomic(self) -> None:
        blocks = [
            NormalizedBlock(
                doc_id="doc1",
                file_name="doc1.pdf",
                page=1,
                section="Food Safety Data",
                chunk_type="table_kv",
                text="Section: Food Safety Data; Field: Lead; Value: < 5 mg/kg",
                text_raw="x",
                lang="en",
                metadata={},
            )
        ]

        chunks = chunk_blocks(blocks, config=PrepConfig())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "table_kv")

    def test_index_and_search_top_k_sorted(self) -> None:
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        chunks = [
            NormalizedBlock(
                doc_id="a",
                file_name="a.pdf",
                page=1,
                section="General",
                chunk_type="paragraph",
                text="Dosage recommended 15-35 ppm for bread improvement",
                text_raw="",
                lang="en",
                metadata={},
            ),
            NormalizedBlock(
                doc_id="b",
                file_name="b.pdf",
                page=1,
                section="General",
                chunk_type="paragraph",
                text="Storage below 20C in a dry place",
                text_raw="",
                lang="en",
                metadata={},
            ),
        ]
        from rag.prepare import _to_chunk

        records = [_to_chunk(c) for c in chunks]
        build_vector_index(records, VectorConfig(embedder=embedder, vector_store=store))
        response = search_top_k(
            "What is the dosage in ppm for bread?",
            2,
            SearchConfig(embedder=embedder, vector_store=store),
        )

        self.assertEqual(len(response.results), 2)
        self.assertGreaterEqual(response.results[0].score, response.results[1].score)
        self.assertIn("Dosage", response.results[0].text)
        self.assertGreaterEqual(response.results[0].final_score, response.results[1].final_score)
        self.assertIsInstance(response.results[0].matched_terms, list)

    def test_hybrid_reranking_prefers_bread_for_pain_query(self) -> None:
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        chunks = [
            NormalizedBlock(
                doc_id="dosage_biscuit",
                file_name="a.pdf",
                page=1,
                section="Dosages Recommandes",
                chunk_type="table_kv",
                text="Section: Dosages Recommandes; Produit: Biscuits/Crackers. Dosage recommande: 30-50 ppm.",
                text_raw="",
                lang="fr",
                metadata={"product_type_canonical": "biscuit"},
            ),
            NormalizedBlock(
                doc_id="dosage_pain",
                file_name="b.pdf",
                page=1,
                section="Dosages Recommandes",
                chunk_type="table_kv",
                text="Section: Dosages Recommandes; Produit: Pain de mie (CBP). Dosage recommande: 75 ppm.",
                text_raw="",
                lang="fr",
                metadata={"product_type_canonical": "bread"},
            ),
        ]
        from rag.prepare import _to_chunk

        records = [_to_chunk(c) for c in chunks]
        build_vector_index(records, VectorConfig(embedder=embedder, vector_store=store))
        response = search_top_k(
            "Quel dosage recommandé pour le pain ?",
            2,
            SearchConfig(embedder=embedder, vector_store=store, context_window=0),
        )

        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.results[0].doc_id, "dosage_pain")

    def test_direct_chunks_from_blocks_splits_pdf_text(self) -> None:
        blocks = [
            RawPageBlock(
                doc_id="doc1",
                file_name="doc1.pdf",
                page=1,
                block_type="paragraph",
                text=" ".join(["dosage"] * 30),
            )
        ]
        cfg = PrepConfig(chunk_size_tokens=10, chunk_overlap_tokens=2, min_chunk_tokens=5, dedup_across_docs=False)
        chunks = direct_chunks_from_blocks(blocks=blocks, config=cfg)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(c.chunk_type == "paragraph" for c in chunks))
        self.assertTrue(all(c.metadata.get("source_block_type") == "paragraph" for c in chunks))

    def test_index_pdfs_directly_without_prepare_step(self) -> None:
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        fake_blocks = [
            RawPageBlock(
                doc_id="demo",
                file_name="demo.pdf",
                page=1,
                block_type="paragraph",
                text="Recommended dosage is 40 ppm for bread products",
            )
        ]

        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
            with patch("rag.prepare.extract_pdf_document", return_value=(fake_blocks, 0)):
                prep_report, index_report = index_pdfs_directly(
                    input_dir=tmpdir,
                    prep_config=PrepConfig(chunk_size_tokens=50, min_chunk_tokens=2),
                    vector_config=VectorConfig(embedder=embedder, vector_store=store),
                )

        self.assertEqual(prep_report.documents_processed, 1)
        self.assertGreater(index_report.vectors_upserted, 0)


if __name__ == "__main__":
    unittest.main()
