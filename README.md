# Semantic PDF RAG Pipeline

Dans un contexte où une base documentaire contient un grand volume d'informations
(rapports, procédures, recommandations, cas d'usage, etc.), les utilisateurs
rencontrent des difficultés à identifier rapidement les passages réellement pertinents
pour répondre à leur question.

L'objectif de ce projet est de développer un module intelligent capable d'assister
l'utilisateur en retrouvant automatiquement les fragments les plus pertinents à partir
d'une question formulée en langage naturel.

## 1. What This Solves

Le module doit:

- Recevoir une question utilisateur
- Générer son embedding sémantique
- Comparer cette représentation aux embeddings des fragments stockés en base via similarité cosinus
- Classer les résultats par ordre décroissant de pertinence
- Retourner les trois fragments les plus pertinents
- Afficher pour chacun:
  - le texte du fragment
  - le score de similarité

Ce module vise ainsi à améliorer la recherche d'information en privilégiant la
proximité sémantique plutôt qu'une simple correspondance lexicale.

En pratique, ce dépôt fournit une pipeline orientée production pour:
- préparer des PDF bruités (dont tableaux) pour l'indexation vectorielle
- exécuter une indexation directe PDF -> embeddings
- indexer les chunks dans Qdrant avec la similarité cosinus
- récupérer les fragments les plus pertinents pour une question en langage naturel

## 2. Architecture

### Processing flow

1. PDF extraction (`pdfplumber`)
2. OCR fallback on text-empty pages (`pytesseract` + `pdftoppm`)
3. Text cleanup and table normalization to key/value semantic lines
4. Chunking with overlap
5. Embedding generation (`sentence-transformers`)
6. Vector upsert to Qdrant (cosine)
7. Hybrid ranking (dense + lexical + intent-field score + metadata constraints)
8. Route-aware retrieval policy (different search configs per query intent)
9. Optional second-stage cross-encoder reranking (currently disabled by default)
10. Query embedding + top-k search

### Main modules

- `rag/prepare.py`: extraction, normalization, chunking, artifact generation
- `rag/index.py`: embedding + vector index build
- `rag/search.py`: hybrid retrieval/reranking (dense + lexical + metadata intent matching)
- `rag/pipeline.py`: orchestration helpers (`prepare_and_index`, `answer_question`)
- `rag/vector_store.py`: Qdrant + in-memory vector stores
- `rag/types.py`: configs, DTOs, reports

### Retrieval decision path

At query time, `search_top_k(...)` does:
1. Build query profile (`codes`, `storage/safety/regulatory`, `activity`, broad)
2. Apply route-specific config overrides (`candidate_pool`, weights, penalties)
3. Dense candidate retrieval from vector store
4. Hybrid scoring + constraints
5. Optional reranker pass (if enabled)
6. Return top-k (plus optional context neighbors)

## 3. Repository Layout

```text
RAG/
  data/                  # input PDFs
  artifacts/             # generated chunks/report
  rag/                   # core package
  scripts/run_indexing.py
  docker-compose.yml
  Dockerfile
  requirements.txt
  tests/
```

## 4. Core Public API

From `rag/__init__.py`:

- `prepare_documents(input_dir, output_dir, config) -> PrepReport`
- `build_embeddings(chunks, cfg) -> tuple[ids, vectors, payloads]`
- `index_to_qdrant(ids, vectors, payloads, cfg) -> IndexReport`
- `build_vector_index(chunks, cfg) -> IndexReport`
- `search_top_k(question, k, cfg) -> SearchResponse`
- `prepare_and_index(input_dir, output_dir, prep_config, vector_config)`
- `index_pdfs_directly(input_dir, prep_config, vector_config)`
- `answer_question(question, search_config, top_k=6)`
- `load_chunks_jsonl(jsonl_path) -> list[ChunkRecord]`
- `run_quality_eval(cases, search_config, top_k=3) -> QualityEvalReport`
- `normalize_unit_string(text) -> str`
- `normalize_answer_text(text) -> str`
- `detect_query_intent(question) -> str`
- `extract_structured_answer(question, text, max_chars=None) -> str`
- `extract_structured_answer_with_score(question, text, max_chars=None) -> tuple[str, float]`

### Important config objects

- `PrepConfig`
  - `chunk_size_tokens` (default `420`)
  - `chunk_overlap_tokens` (default `60`)
  - `min_chunk_tokens` (default `12`)
  - `include_list_items` (default `False`)
  - `dedup_across_docs` (default `True`)
  - `enable_ocr_fallback` (default `True`)
  - `ocr_lang` (default `eng+fra`)
- `VectorConfig`
  - `qdrant_url` (default `http://localhost:6333`)
  - `collection_name` (default `rag_chunks`)
  - `embedding_model` (default `models/paraphrase-multilingual-MiniLM-L12-v2`)
- `SearchConfig`
  - same runtime params as `VectorConfig`
  - `context_window` (default `1`): adds neighboring chunks from same doc/page around each top hit
  - `chunks_jsonl_path` (default `artifacts/chunks.jsonl`): source used to resolve neighbor order
  - `fetch_multiplier` (default `3`): retrieves a larger candidate pool before selecting top hits
  - `hybrid_enabled` (default `True`)
  - `candidate_pool` (default `60`)
  - `dense_weight`, `lexical_weight`, `field_weight` (defaults `0.58`, `0.32`, `0.10`)
  - `diversity_enabled`, `diversity_penalty` (defaults `False`, `0.10`)
  - `route_overrides_enabled` (default `True`)
  - `reranker_enabled` (default `False`)

## 5. Quick Start (Docker, recommended)

### Preconditions

- Docker + Docker Compose installed.
- Run commands from `ai-night/RAG` directory.

### Start services

```bash
docker compose up -d --build
```

This starts:
- `qdrant` on `localhost:6333`
- `rag-indexer` job container that runs direct PDF indexing + embeddings (no prepare artifact step)

### Cache the default embedder locally (one-time)

```bash
python3 scripts/cache_best_embedder.py
```

After this, local runs use `models/paraphrase-multilingual-MiniLM-L12-v2` by default.

### Optional: cache cross-encoder reranker locally (one-time)

```bash
python3 scripts/cache_reranker_model.py
```

Default runtime keeps reranker disabled. Enable only after benchmark verification.

### Follow indexing logs

```bash
docker compose logs -f rag-indexer
```

Successful end state includes:
- `Preparation report: ...`
- `Index report: IndexReport(collection_name='rag_chunks', vectors_upserted=...)`

### Validate collection in Qdrant

```bash
curl -s http://localhost:6333/collections
```

You should see `rag_chunks` in the response.

## 6. Query Indexed Data

Run from `ai-night/RAG`:

```bash
python3 - <<'PY'
from rag import SearchConfig, search_top_k

resp = search_top_k(
    question="Quel dosage recommandé pour le pain ?",
    k=3,
    cfg=SearchConfig(
        qdrant_url="http://localhost:6333",
        collection_name="rag_chunks",
        context_window=1,
        chunks_jsonl_path="artifacts/chunks.jsonl",
        fetch_multiplier=2,
    ),
)

for r in resp.results:
    kind = "context" if r.is_context else "hit"
    print(
        f"[{r.rank}] {kind} final={r.final_score:.4f} "
        f"dense={r.dense_score:.4f} lex={r.lex_score:.4f} "
        f"field={r.field_score:.4f} doc={r.doc_id} page={r.page}"
    )
    print("matched_terms:", ", ".join(r.matched_terms))
    print(r.text[:220], "\n")
PY
```

Note: `search_top_k(..., k=3, ...)` now returns the 3 semantic hits plus optional context neighbors
when `context_window > 0`, so total returned items can be greater than `k`.

## 7. Local Python Run (without Docker indexer)

### Create and activate a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start only Qdrant via Docker

```bash
docker compose up -d qdrant
```

### Run pipeline from Python (classic direct mode)

```bash
python3 - <<'PY'
from rag import PrepConfig, VectorConfig, SearchConfig, index_pdfs_directly, search_top_k

index_pdfs_directly(
    input_dir="data",
    prep_config=PrepConfig(),
    vector_config=VectorConfig(qdrant_url="http://localhost:6333", collection_name="rag_chunks"),
)

resp = search_top_k(
    "What is the recommended dosage for bread?",
    3,
    SearchConfig(qdrant_url="http://localhost:6333", collection_name="rag_chunks"),
)

print(resp.results[0].text if resp.results else "no results")
PY
```

## 8. Generated Artifacts

After preparation:
- `artifacts/chunks.jsonl`
- `artifacts/prep_report.json`

### `chunks.jsonl` schema (per line)

- `chunk_id`
- `doc_id`
- `file_name`
- `page`
- `section`
- `chunk_type` (`paragraph`, `list_item`, `table_kv`)
- `text` (normalized for embedding)
- `text_raw` (original normalized source)
- `lang` (`fr`, `en`, `mixed`)
- `metadata` (units, fields, etc.)

## 9. Testing

Run unit tests from `ai-night/RAG`:

```bash
PYTHONPATH=. pytest -q
```

Current tests cover:
- table normalization to semantic key/value chunks,
- atomic table chunk behavior,
- indexing + top-k ranking behavior.

### Retrieval score report (for regression tracking)

To track whether your model/search changes are improving retrieval over time, run:

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
PYTHONPATH=. python3 tests/run_eval_audit.py \
  --eval-path tests/pdf_eval_queries_human.json \
  --output artifacts/eval_audit_report_human.json \
  --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --run-name human_queries
```

This writes:
- `artifacts/eval_audit_report_human.json` with retrieval metrics (`hit_at_3`, `hit_at_10`, `mrr_at_3`, `score`) and answer-accuracy metrics (`answer_score_top1_mean`, `answer_score_top3_mean`, `answer_exact_match_top1_rate`, `answer_numeric_match_top1_rate`, `answer_score`)
- `artifacts/eval_score_history.jsonl` with one row per run

The report includes `delta_vs_previous` so you can immediately see improvement/regression versus the last matching run (`eval_set` + `model_name` + `run_name`).
For observability, each query row in `details` now includes:
- `expected_supporting`: expected answers with `source_file` + `page`
- `retrieved_top3`: retrieved document hits with scores and `answer_excerpt`
- `predicted_answer`: compact answer text from the top retrieved hit
- `answer_eval`: deterministic top-1 and best-of-top3 answer scoring diagnostics

### Evaluate on the 100-query set

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
RAG_EMBEDDING_LOCAL_ONLY=1 PYTHONPATH=. python3 tests/run_eval_audit.py \
  --eval-path tests/pdf_eval_queries_human_100.json \
  --output artifacts/eval_audit_report_human_100.json \
  --model-name models/paraphrase-multilingual-MiniLM-L12-v2 \
  --run-name human_100
```

The 100-query report now includes the same retrieval + answer-accuracy metrics and per-query `answer_eval` diagnostics.

### Run full route-config sweep

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
RAG_EMBEDDING_LOCAL_ONLY=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python3 tests/run_routing_sweep.py \
  --eval-path tests/pdf_eval_queries_human.json \
  --output artifacts/eval_routing_sweep_human_full.json \
  --model models/paraphrase-multilingual-MiniLM-L12-v2
```

## 10. What Is Unique Here

### 1) Route-aware retrieval defaults (not one global config)
- Query intent routing is built into default retrieval:
  - `code_specific`
  - `compliance_storage_safety`
  - `activity`
  - `broad_semantic`
- Each route uses its own tuned candidate pool/weights/penalties.

### 2) Evidence-first observability
- Per-query report includes expected docs/answers and retrieved top-3 evidence with scores and answer excerpts.
- Makes ranking/debugging auditable instead of "black box".

### 3) Benchmarked quality gains from routing
- On 44-query human set:
  - baseline (single global config): `hit@3=0.7955`, `mrr@3=0.7538`
  - routed default (full sweep best): `hit@3=0.8409`, `mrr@3=0.7576`
  - delta: `+0.0455 hit@3`, `+0.0038 mrr@3`
- On 100-query set with current defaults:
  - `hit@3=0.91`
  - `mrr@3=0.8933`

### 4) Retrieval speed (measured)
- Local benchmark on 100 queries (in-memory vector store, CPU, local embedder):
  - avg `45.2 ms/query`
  - median `44.9 ms`
  - p95 `59.1 ms`
  - p99 `72.5 ms`
- Exact numbers depend on hardware/runtime mode (Qdrant vs in-memory, CPU/GPU), but this provides a reproducible reference.
## 11. Troubleshooting

### `docker compose up` says "no configuration file provided"

You are not in the project directory containing `docker-compose.yml`.

Fix:
- `cd /home/nader/Projects/hackathons/ai-night/RAG`
- or use `docker compose -f /home/nader/Projects/hackathons/ai-night/RAG/docker-compose.yml up -d`

### Qdrant reachable but no results

- Check collection exists: `curl -s http://localhost:6333/collections`
- Re-run indexing job: `docker compose run --rm rag-indexer`
- Inspect logs: `docker compose logs -f rag-indexer`

### OCR not used

OCR triggers only when extracted text is empty or below threshold.
Adjust `PrepConfig(min_text_chars_for_page=..., enable_ocr_fallback=True)`.

## 12. Integration Notes (for API teams)

Recommended service boundaries:
- Startup job or endpoint: call `prepare_and_index(...)`
- Query endpoint: call `search_top_k(question, 3, SearchConfig(...))`

The module is already function-based and dependency-injectable via:
- `VectorConfig(embedder=..., vector_store=...)`
- `SearchConfig(embedder=..., vector_store=...)`

This makes it straightforward to swap embedding providers or vector stores without rewriting business logic.

## 13. Enterprise Retrieval Notes

- Preparation now emits richer table metadata (`product_type`, `product_type_canonical`, `dosage`, `dosage_unit`) to improve intent-aware ranking.
- Cross-document deduplication removes repetitive table rows that otherwise dominate results.
- Retrieval uses a Qdrant-only hybrid score:
  - cosine dense similarity,
  - lexical token overlap,
  - metadata/intent field boosts (example: bread queries favor bread/panification chunks).
- Returned objects contain audit-ready evidence:
  - source (`doc_id`, `page`, `chunk_id`),
  - `dense_score`, `lex_score`, `field_score`, and `final_score`.

## 14. FastAPI + React UI (Semantic Atlas)

This repository now includes:
- `backend/`: a minimal FastAPI wrapper over existing `search_top_k` logic.
- `frontend/`: a Vite + React + Tailwind UI for querying top-3 semantic matches.

### Launch full stack (recommended)

Use the project launcher script to start everything in one command:

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
bash scripts/launch_all.sh
```

By default, this script:
- starts `qdrant` (`docker compose up -d qdrant`)
- runs `rag-indexer` once (`docker compose run --rm rag-indexer`)
- starts backend API on `http://localhost:8000`
- starts frontend on `http://localhost:5173`

Optional overrides:

```bash
START_INDEXER=0 BACKEND_PORT=8001 FRONTEND_PORT=5174 bash scripts/launch_all.sh
```

Press `Ctrl+C` to stop backend/frontend processes. Qdrant keeps running in Docker.

### Backend run

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health
```

### Frontend run

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG/frontend
npm install
npm run dev
```

The frontend reads `VITE_API_BASE_URL` from `frontend/.env` (default `http://localhost:8000`).
Build tooling can generate local files such as `vite.config.js`, `tailwind.config.js`, and `*.tsbuildinfo`; they are intentionally ignored and should not be committed.

### Endpoint contract

`POST /search`

Request:

```json
{
  "question": "Quel dosage recommande pour le pain ?"
}
```

Response:

```json
{
  "question": "Quel dosage recommande pour le pain ?",
  "results": [
    { "text": "...", "score": 0.87, "retrieval_score": 0.73, "answer_score": 0.95, "rank": 1 },
    { "text": "...", "score": 0.82, "retrieval_score": 0.68, "answer_score": 0.91, "rank": 2 },
    { "text": "...", "score": 0.78, "retrieval_score": 0.66, "answer_score": 0.86, "rank": 3 }
  ],
  "meta": {
    "k": 3,
    "time_ms": 12
  }
}
```

`score` is a combined confidence: `0.35 * retrieval_score + 0.65 * answer_score`.

### CORS / port troubleshooting

- If browser calls fail with CORS, confirm backend is running on `http://localhost:8000` and frontend on `http://localhost:5173`.
- Backend currently allows `http://localhost:5173` by default in CORS middleware.
- If you run Vite on another port, add that origin to `allow_origins` in `backend/main.py`.
