# Evaluated Multilingual PDF Retrieval Pipeline

A production-oriented retrieval module for noisy PDF collections. It combines text extraction with OCR fallback, semantic chunking, multilingual embeddings, Qdrant, hybrid dense/lexical/metadata scoring, intent-aware routing, and optional cross-encoder reranking.

## Measured results

The checked-in evaluation suite contains 100 human-authored queries. On that dataset, the selected configuration achieved:

- **91% Hit@3**
- **0.8933 MRR@3**
- **45.2 ms average local retrieval latency**

The repository includes unit and API tests, evaluation datasets, feature benchmarks, hyperparameter sweeps, Docker configuration, a FastAPI backend, and a React interface. Start with the [architecture](#2-architecture), then use the setup and evaluation sections below to reproduce the pipeline.

> The detailed documentation below is currently in French. The code, public interfaces, configuration names, and test commands are language-neutral.

---

# Pipeline RAG Sémantique sur PDF

Dans un contexte où une base documentaire contient un grand volume d'informations
(rapports, procédures, recommandations, cas d'usage, etc.), les utilisateurs
rencontrent des difficultés à identifier rapidement les passages réellement pertinents
pour répondre à leur question.

L'objectif de ce projet est de développer un module intelligent capable d'assister
l'utilisateur en retrouvant automatiquement les fragments les plus pertinents à partir
d'une question formulée en langage naturel.

## 1. Problème Ciblé

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

### Flux de traitement

1. Extraction PDF (`pdfplumber`)
2. Repli OCR sur pages sans texte (`pytesseract` + `pdftoppm`)
3. Nettoyage texte et normalisation des tableaux en lignes sémantiques clé/valeur
4. Découpage en chunks avec overlap
5. Génération d'embeddings (`sentence-transformers`)
6. Upsert vectoriel dans Qdrant (cosinus)
7. Classement hybride (dense + lexical + score d'intention/metadata)
8. Politique de recherche selon la route d'intention (configs différentes par type de requête)
9. Reranking cross-encoder optionnel (désactivé par défaut)
10. Embedding de la question + recherche top-k

### Modules principaux

- `rag/prepare.py`: extraction, normalisation, chunking, génération d'artefacts
- `rag/index.py`: embeddings + construction d'index vectoriel
- `rag/search.py`: retrieval/reranking hybride (dense + lexical + matching metadata)
- `rag/pipeline.py`: orchestration (`prepare_and_index`, `answer_question`)
- `rag/vector_store.py`: stores Qdrant + en mémoire
- `rag/types.py`: configs, DTOs, rapports

### Chemin de décision du retrieval

À l'exécution, `search_top_k(...)`:
1. Construit le profil de requête (`codes`, `storage/safety/regulatory`, `activity`, broad)
2. Applique les overrides de route (pool candidat, poids, pénalités)
3. Fait la récupération dense depuis le vector store
4. Applique le scoring hybride + contraintes
5. Applique le reranker si activé
6. Retourne le top-k (avec voisins de contexte optionnels)

## 3. Structure du dépôt

```text
RAG/
  data/                  # PDFs d'entrée
  artifacts/             # chunks/rapports générés
  rag/                   # package coeur
  scripts/run_indexing.py
  docker-compose.yml
  Dockerfile
  requirements.txt
  tests/
```

## 4. API Publique Principale

Depuis `rag/__init__.py`:

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

### Objets de configuration importants

- `PrepConfig`
  - `chunk_size_tokens` (défaut `420`)
  - `chunk_overlap_tokens` (défaut `60`)
  - `min_chunk_tokens` (défaut `12`)
  - `include_list_items` (défaut `False`)
  - `dedup_across_docs` (défaut `True`)
  - `enable_ocr_fallback` (défaut `True`)
  - `ocr_lang` (défaut `eng+fra`)
- `VectorConfig`
  - `qdrant_url` (défaut `http://localhost:6333`)
  - `collection_name` (défaut `rag_chunks`)
  - `embedding_model` (défaut `models/paraphrase-multilingual-MiniLM-L12-v2`)
- `SearchConfig`
  - mêmes paramètres runtime que `VectorConfig`
  - `context_window` (défaut `1`): ajoute les chunks voisins même doc/page
  - `chunks_jsonl_path` (défaut `artifacts/chunks.jsonl`): source pour résoudre l'ordre des voisins
  - `fetch_multiplier` (défaut `3`): récupère plus de candidats avant sélection finale
  - `hybrid_enabled` (défaut `True`)
  - `candidate_pool` (défaut `60`)
  - `dense_weight`, `lexical_weight`, `field_weight` (défauts `0.58`, `0.32`, `0.10`)
  - `diversity_enabled`, `diversity_penalty` (défauts `False`, `0.10`)
  - `route_overrides_enabled` (défaut `True`)
  - `reranker_enabled` (défaut `False`)

## 5. Démarrage Rapide (Docker recommandé)

### Prérequis

- Docker + Docker Compose installés.
- Exécuter les commandes depuis `ai-night/RAG`.

### Démarrer les services

```bash
docker compose up -d --build
```

Cela démarre:
- `qdrant` sur `localhost:6333`
- le job `rag-indexer` qui exécute l'indexation PDF + embeddings

### Mettre en cache l'embedder par défaut (une fois)

```bash
python3 scripts/cache_best_embedder.py
```

Ensuite, les runs locaux utilisent `models/paraphrase-multilingual-MiniLM-L12-v2` par défaut.

### Mettre en cache le reranker cross-encoder (optionnel, une fois)

```bash
python3 scripts/cache_reranker_model.py
```

Le reranker reste désactivé par défaut.

### Suivre les logs d'indexation

```bash
docker compose logs -f rag-indexer
```

Résultat attendu:
- `Preparation report: ...`
- `Index report: IndexReport(collection_name='rag_chunks', vectors_upserted=...)`

### Vérifier la collection dans Qdrant

```bash
curl -s http://localhost:6333/collections
```

Vous devez voir `rag_chunks` dans la réponse.

## 6. Interroger les Données Indexées

Depuis `ai-night/RAG`:

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

Note: `search_top_k(..., k=3, ...)` peut retourner plus de 3 éléments si `context_window > 0`.

## 7. Exécution Locale Python (sans indexer Docker)

### Créer et activer un venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Démarrer uniquement Qdrant

```bash
docker compose up -d qdrant
```

### Exécuter la pipeline en Python

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

## 8. Artefacts Générés

Après préparation:
- `artifacts/chunks.jsonl`
- `artifacts/prep_report.json`

### Schéma `chunks.jsonl` (par ligne)

- `chunk_id`
- `doc_id`
- `file_name`
- `page`
- `section`
- `chunk_type` (`paragraph`, `list_item`, `table_kv`)
- `text` (normalisé pour embedding)
- `text_raw` (source normalisée d'origine)
- `lang` (`fr`, `en`, `mixed`)
- `metadata` (unités, champs, etc.)

## 9. Tests

Exécuter les tests unitaires depuis `ai-night/RAG`:

```bash
PYTHONPATH=. pytest -q
```

Couverture actuelle:
- normalisation des tableaux en chunks clé/valeur sémantiques,
- comportement atomique des chunks de table,
- indexation + classement top-k.

### Rapport de score retrieval (suivi régression)

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
PYTHONPATH=. python3 tests/run_eval_audit.py \
  --eval-path tests/pdf_eval_queries_human.json \
  --output artifacts/eval_audit_report_human.json \
  --embedding-model models/paraphrase-multilingual-MiniLM-L12-v2 \
  --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --run-name human_queries
```

Ce rapport écrit:
- `artifacts/eval_audit_report_human.json` avec métriques retrieval (`hit_at_3`, `hit_at_10`, `mrr_at_3`, `score`) et métriques answer (`answer_score_top1_mean`, `answer_score_top3_mean`, `answer_exact_match_top1_rate`, `answer_numeric_match_top1_rate`, `answer_score`)
- `artifacts/eval_score_history.jsonl` avec une ligne par run

Le rapport inclut `delta_vs_previous` pour visualiser immédiatement les régressions/progrès.

### Évaluer sur le jeu 100 requêtes

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
RAG_EMBEDDING_LOCAL_ONLY=1 PYTHONPATH=. python3 tests/run_eval_audit.py \
  --eval-path tests/pdf_eval_queries_human_100.json \
  --output artifacts/eval_audit_report_human_100.json \
  --embedding-model models/paraphrase-multilingual-MiniLM-L12-v2 \
  --model-name models/paraphrase-multilingual-MiniLM-L12-v2 \
  --run-name human_100
```

### Exécuter un sweep complet des routes

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
RAG_EMBEDDING_LOCAL_ONLY=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python3 tests/run_routing_sweep.py \
  --eval-path tests/pdf_eval_queries_human.json \
  --output artifacts/eval_routing_sweep_human_full.json \
  --model models/paraphrase-multilingual-MiniLM-L12-v2
```

## 10. Points Différenciants

### 1) Retrieval orienté routes (pas une config globale unique)
- Routage par intention:
  - `code_specific`
  - `compliance_storage_safety`
  - `activity`
  - `broad_semantic`
- Chaque route a ses poids/pénalités/pools dédiés.

### 2) Observabilité orientée preuves
- Rapport par requête avec docs/réponses attendues et top-3 récupéré avec scores et extraits.

### 3) Gains de qualité mesurés
- Sur 44 requêtes humaines:
  - baseline: `hit@3=0.7955`, `mrr@3=0.7538`
  - default routé: `hit@3=0.8409`, `mrr@3=0.7576`
- Sur 100 requêtes:
  - `hit@3=0.91`
  - `mrr@3=0.8933`

### 4) Vitesse de retrieval (mesurée)
- Benchmark local 100 requêtes (store mémoire, CPU):
  - avg `45.2 ms/query`
  - median `44.9 ms`
  - p95 `59.1 ms`
  - p99 `72.5 ms`

## 11. Dépannage

### `docker compose up` dit "no configuration file provided"

Vous n'êtes pas dans le dossier contenant `docker-compose.yml`.

Correctif:
- `cd /home/nader/Projects/hackathons/ai-night/RAG`
- ou `docker compose -f /home/nader/Projects/hackathons/ai-night/RAG/docker-compose.yml up -d`

### Qdrant joignable mais aucun résultat

- Vérifier la collection: `curl -s http://localhost:6333/collections`
- Relancer l'indexation: `docker compose run --rm rag-indexer`
- Inspecter les logs: `docker compose logs -f rag-indexer`

### OCR non utilisé

L'OCR se déclenche seulement si le texte extrait est vide/trop court.

## 12. Notes d'Intégration (équipes API)

Frontières de service recommandées:
- endpoint/job de startup: `prepare_and_index(...)`
- endpoint de requête: `search_top_k(question, 3, SearchConfig(...))`

Injection de dépendances disponible via:
- `VectorConfig(embedder=..., vector_store=...)`
- `SearchConfig(embedder=..., vector_store=...)`

## 13. Notes Retrieval Enterprise

- La préparation émet des metadata enrichies (`product_type`, `product_type_canonical`, `dosage`, `dosage_unit`).
- La dédup inter-doc réduit les lignes répétitives.
- Le scoring hybride combine:
  - similarité dense cosinus,
  - overlap lexical,
  - boosts metadata/intention.
- Les objets retournés incluent:
  - source (`doc_id`, `page`, `chunk_id`),
  - `dense_score`, `lex_score`, `field_score`, `final_score`.

## 14. FastAPI + React UI (Semantic Atlas)

Ce dépôt inclut:
- `backend/`: wrapper FastAPI autour de `search_top_k`
- `frontend/`: UI Vite + React + Tailwind

### Lancer la stack complète (recommandé)

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
bash scripts/launch_all.sh
```

Par défaut:
- démarre `qdrant`
- exécute `rag-indexer` une fois
- démarre l'API backend sur `http://localhost:8000`
- démarre le frontend sur `http://localhost:5173`

Overrides optionnels:

```bash
START_INDEXER=0 BACKEND_PORT=8001 FRONTEND_PORT=5174 bash scripts/launch_all.sh
```

`Ctrl+C` arrête backend/frontend; Qdrant reste actif dans Docker.

### Lancer le backend

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Vérification de santé:

```bash
curl -s http://localhost:8000/health
```

### Lancer le frontend

```bash
cd /home/nader/Projects/hackathons/ai-night/RAG/frontend
npm install
npm run dev
```

Le frontend lit `VITE_API_BASE_URL` depuis `frontend/.env` (défaut `http://localhost:8000`).

### Contrat d'API

`POST /search`

Requête:

```json
{
  "question": "Quel dosage recommande pour le pain ?"
}
```

Réponse:

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

`score` est un score combiné: `0.35 * retrieval_score + 0.65 * answer_score`.

### CORS / ports

- Si appels navigateur en échec (CORS), vérifier backend `http://localhost:8000` et frontend `http://localhost:5173`.
- Si frontend sur un autre port, ajouter l'origine dans `allow_origins` de `backend/main.py`.
