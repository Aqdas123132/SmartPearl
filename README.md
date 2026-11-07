# SupportPearlz — LangChain RAG customer-support agent (AAI-407 / PRAC-03)

A retrieval-augmented support assistant for the fictional **Pearlz Home Systems**. It ingests 12 company documents (PDF, DOCX, Markdown, CSV) into a persistent Chroma index,
answers customer questions only from retrieved text, cites verified sources, refuses when the documents do not cover the question, and remembers the conversation.
Interfaces: CLI, FastAPI service (JWT auth, customer/admin roles) and a Streamlit app.

> **Status, honestly:** built and tested in a sandbox with **no LLM or embedding API access**. Everything runs offline using `LLM_PROVIDER=extractive` (a quoting baseline, no LLM)
> and `EMBEDDING_PROVIDER=hash` (lexical embeddings). The OpenAI/Ollama/Hugging Face paths are implemented and the chain logic is unit-tested with a stub model, but they have **not been run
> against a real model**. See `docs/technical_report.md` for exactly what was and was not measured.

## What is included
| Area | Where |
|---|---|
| Config + masked settings, central logging | `src/config.py`, `src/utils/logging_setup.py` |
| Loaders, normalisation, dedupe, per-file errors, ingestion report | `src/ingestion/loaders.py` |
| 3 chunking strategies (recursive, fixed, structure-aware) | `src/ingestion/chunking.py`, `evaluation/chunk_experiment.py` |
| Cached/batched/retrying embeddings; persistent Chroma; load-or-build; delete-by-source + reinsert | `src/retrieval/embeddings.py`, `vector_store.py` |
| Similarity / MMR / threshold / hybrid (BM25+dense) / rerank / metadata filter / relevance gate | `src/retrieval/retriever.py` |
| LCEL chain, versioned prompts, Pydantic output, citation validation, memory | `src/chains/` |
| Auth, roles, conversations, feedback, admin, upload | `src/db.py`, `src/services.py`, `src/api.py`, `app.py` |
| Evaluation (22 cases), retrieval experiments, prompt experiment | `evaluation/` |
| Knowledge base + planted gaps/seams | `data/knowledge_base/`, `data/PLANTED_GAPS_AND_SEAMS.md` |
| Bonus challenges implemented | C1 hybrid, C2 reranking (lexical; cross-encoder optional), C3 metadata filtering, C10 Docker. **Not** done: C4 streaming, C5 query expansion, C6 multi-KB, C7 tools, C8 LangSmith judge, C9 semantic cache |

## Prerequisites
Python 3.10-3.12 (built on 3.12.3), git. Optional: an OpenAI key, or Ollama running locally.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then edit .env (never commit it)
python -m src.config                                     # prints config with secrets masked; a bad/missing required value gives one clear error
```
In `.env` set a strong `SECRET_KEY`, and `ADMIN_EMAIL` / `ADMIN_PASSWORD` to create the first admin at start-up.

### Using a real LLM (recommended)
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...        # paste ONLY in your local .env
```
Changing the embedding model requires a rebuild (the app refuses to load an incompatible index), then **recalibrate `SCORE_THRESHOLD`**:
```bash
python -m src.ingestion.build_index --rebuild
python -m evaluation.retrieval_experiments        # see threshold_calibration.recommended_threshold in evaluation/results/retrieval_experiments.json
```

## Build the index (once)
```bash
python -m src.ingestion.build_index                 # load-or-build, then incremental sync of changed files
python -m src.ingestion.build_index --rebuild       # wipe and rebuild
python -m src.ingestion.build_index --search "how long is the warranty" -k 5   # similarity debug tool: score, source, section, chunk id
```
The second run logs `Loaded existing index from disk ... No re-embedding performed.` If a document changes, the sync deletes that source's old chunks and reinserts the new ones.

## Run
```bash
python -m src.app_cli                 # CLI: /reset  /debug  /quit
uvicorn src.api:app --port 8000       # API
streamlit run app.py                  # UI (customer chat; admin console for admins)
```
API example:
```bash
curl -s -X POST localhost:8000/auth/signup -H 'Content-Type: application/json' \
  -d '{"name":"Ali","email":"ali@example.com","password":"Passw0rd1","confirm_password":"Passw0rd1"}'
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"ali@example.com","password":"Passw0rd1"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"How long is the AquaPearl 500 Pro warranty?"}'
curl -s localhost:8000/health
```
Admin-only routes (`/admin/*`) return 403 for customers; this is enforced in `src/services.py`, not only in the UI.

## Evaluate and test
```bash
python -m evaluation.run_eval                    # 22 cases -> evaluation/results/evaluation_results.{json,csv} + evaluation_report.md
python -m evaluation.retrieval_experiments       # retriever comparison, hybrid before/after, reranking, filter demo, threshold calibration
python -m evaluation.chunk_experiment            # chunking table
python -m evaluation.embedding_diagnostic        # paraphrase vs unrelated cosine similarity
python -m evaluation.prompt_experiment           # weak vs final prompt (needs a real LLM)
python -m pytest -q                              # 27 tests
```

## Docker
```bash
cp .env.example .env && edit .env
docker compose build
docker compose run --rm api python -m src.ingestion.build_index --rebuild   # build the index INTO the volume, once
docker compose up -d                                                         # API :8000, UI :8501
```
The index lives in the `pearlz_data` volume (not baked into the image), so image rebuilds do not re-embed and a reindex does not need a redeploy. **The Docker files were written but not built in the sandbox (no Docker daemon).**

## Known limitations
See section 8 of `docs/technical_report.md`. Highlights: no real-LLM measurements; hash embeddings are not semantic; small self-authored test set and knowledge base (~5,400 words);
gate threshold tuned on the same questions it is scored on; SQLite and in-memory rate limiting suit a prototype only; screenshots and the demo video must be recorded by you.

## Remaining deliverables for you
Run the system with your key and record: six screenshots (ingestion, index build, cited answer, refusal, follow-up with rewritten query, handled error), the 5-8 minute demo, and re-run `run_eval` and `prompt_experiment`; then update the report numbers.
