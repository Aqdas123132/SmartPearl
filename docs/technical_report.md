# Technical report (draft built from measured results)

> Every number below is copied from `evaluation/results/*.json` produced by the scripts in this repository. Embeddings in this run were the offline
> `hash` provider and the answerer was the offline `extractive` baseline (no LLM, no API key was available in the build sandbox). The numbers therefore
> describe **retrieval, the relevance gate, guards and citation plumbing**; they do NOT describe LLM answer quality. Re-run with your own key to get those.

## 1. Architecture
Two separate paths (see `docs/architecture.png`). **Ingestion** (`python -m src.ingestion.build_index`) loads 12 documents in four formats (PDF, DOCX, MD, CSV),
normalises text, strips repeated PDF headers/footers, skips duplicates and corrupt files per file, splits with a structure-aware splitter (800 chars / 120 overlap, CSV rows kept whole),
embeds through a SQLite-cached, batched, retrying embedding wrapper and persists to Chroma (cosine). **Query** loads the persisted index (no re-embedding), then runs an LCEL
chain: injection guard -> history condensation -> hybrid retrieval (dense + BM25, reciprocal-rank fusion) -> rerank -> relevance gate -> labelled context ([S1]..[S4], character budget) ->
prompt v3 -> structured output (`LLMAnswer`, one retry) -> citation validation (labels checked against those supplied, duplicates merged, unused dropped) -> `SupportResponse`.

## 2. Chunking experiment (hit@4 on 18 answerable questions, hash embeddings)
| Strategy | Size | Overlap | Chunks | Mean len | Max len | hit@4 | Time (s) |
|---|---|---|---|---|---|---|---|
| recursive | 300 | 0 | 155 | 245 | 379 | 0.944 | 0.01 |
| recursive | 300 | 45 | 156 | 247 | 379 | 0.944 | 0.01 |
| recursive | 800 | 0 | 65 | 532 | 824 | 0.889 | 0.0 |
| recursive | 800 | 120 | 66 | 535 | 824 | 0.889 | 0.0 |
| recursive | 1500 | 0 | 46 | 737 | 1538 | 0.778 | 0.0 |
| recursive | 1500 | 225 | 47 | 749 | 1538 | 0.833 | 0.0 |
| fixed | 800 | 120 | 64 | 599 | 866 | 0.889 | 0.02 |
| structure | 300 | 45 | 166 | 256 | 379 | 0.944 | 0.01 |
| structure | 800 | 120 | 126 | 318 | 791 | 0.889 | 0.01 |
| structure | 1500 | 225 | 120 | 330 | 1498 | 0.889 | 0.01 |

Chosen: **structure / 800 / 120**. Honest reading: 300-character chunks scored 0.944 vs 0.889 for structure/800, which is one question out of 18 and within noise for this test set.
Smaller chunks raise precision (each embedding covers one idea) but strip the surrounding clause the LLM needs; structure-aware splitting keeps a heading with its body, and its mean chunk
length is only ~318 characters even at an 800 limit. Pricing CSV rows are one chunk each so a SKU is never separated from its price (unit-tested). With a real embedding model this table should be re-run before trusting the choice.

## 3. Embeddings
Offline default: `hash/bow-512` (deterministic hashed bag-of-words, 512 dims). It captures word overlap, **not meaning**, so it cannot bridge the vocabulary gap that motivates RAG (e.g. "beeping three times" vs "E-03");
T-05 passes only because the troubleshooting guide deliberately contains customer wording. For a real run set `EMBEDDING_PROVIDER=openai` (text-embedding-3-small, 1536 dims) or `huggingface`, then `python -m src.ingestion.build_index --rebuild`.
The index manifest stores model name and dimensions; loading with a different model raises `IndexIncompatibleError` instead of silently returning near-random results. Embeddings are cached per (model, text) in SQLite and committed per batch, so a crash resumes. The cosine-similarity diagnostic is `python -m evaluation.embedding_diagnostic`. Measured with the hash provider: paraphrase pair 0.139 vs 0.015 and 0.013 for the unrelated pairs. The ordering is right but the paraphrase score is low because the two sentences share almost no words, which is exactly the limitation of lexical embeddings; a real model should score the paraphrases far higher.

## 4. Retriever comparison
| Strategy | k | hit@k (primary fact) | hit@k (all fact groups) | Mean latency ms | Missed |
|---|---|---|---|---|---|
| similarity | 3 | 0.833 | 0.778 | 2.6 | T-02, T-04, T-08 |
| similarity | 8 | 0.944 | 0.944 | 2.7 | T-08 |
| mmr | 4 | 0.889 | 0.722 | 6.1 | T-02, T-08 |
| threshold | 4 | 0.889 | 0.778 | 2.2 | T-04, T-08 |
| similarity+metadata_filter(500 Pro) | 4 | 0.889 | 0.889 | 2.8 | T-04, T-08 |
| hybrid | 4 | 0.944 | 0.889 | 2.3 | T-08 |
| hybrid_rerank | 4 | 1.0 | 0.889 | 2.8 | - |

Reranking (lexical reranker, offline): top-4 precision 0.639 -> 0.625, latency 2.4 -> 2.8 ms. No measurable precision gain here; the cross-encoder
option (`RERANKER=cross_encoder`) is implemented but could not be run offline, so its benefit is **unmeasured**.

Exact identifiers (hybrid vs dense only): 

| Query | dense-only found | hybrid found |
|---|---|---|
| What does E-03 mean? | True | True |
| FLT-M5 | True | True |
| price of AP-500P | True | True |
| E-06 | True | True |

With hash embeddings dense retrieval already matches identifier tokens lexically, so this test cannot show the benefit of BM25; it should be re-run with real embeddings.

## 5. Relevance gate (empirical)
Best dense cosine similarity over 18 answerable and 11 unanswerable/off-topic queries: answerable min 0.29, mean 0.476; unanswerable max 0.382, mean 0.166.
The classes **overlap** (separable: False). Chosen threshold **0.29** (balanced accuracy 0.955; passes 100% of answerable, blocks 91% of unanswerable).
It was calibrated on the same labelled queries it is evaluated on, so it is optimistic. Because the classes overlap, adjacent-topic questions (NSF certification) get past the gate; that case must be refused by the prompt, which needs an LLM to test.
The threshold is valid only for this embedding model and metric and must be recalibrated when either changes.

## 6. Evaluation (22 cases; extractive baseline)
Pass 8, partial 1, fail 13. Correctness 0.409, groundedness proxy 1.0, citation accuracy 0.682, retrieval hit@4 0.944,
refusal rate on unanswerable 0.667, hallucination count 1, mean latency 6.9 ms.
Scoring is automatic and proxy-based (key-fact presence, forbidden strings, numbers not found in the retrieved context, cited-source checks); see `evaluation/run_eval.py`. It is not a substitute for manual review.

| ID | Category | Expected | Answered | Confidence | Verdict | Stage at fault (automatic) |
|---|---|---|---|---|---|---|
| T-01 | easy | answer | yes | high | partial | citation selection |
| T-02 | easy | answer | no | none | fail | relevance gate or prompt (answerable question refused) |
| T-03 | easy | answer | yes | high | fail | prompt/model (answer incomplete) |
| T-04 | easy | answer | yes | partial | fail | prompt/model (answer incomplete) |
| T-05 | semantic | answer | yes | partial | pass | - |
| T-06 | multi_step | answer | yes | high | pass | - |
| T-07 | multi_step | answer | yes | partial | fail | citation selection |
| T-08 | multi_step | answer | yes | partial | fail | citation selection |
| T-09 | follow_up | answer | yes | partial | fail | citation selection |
| T-10 | follow_up | answer | yes | partial | fail | citation selection |
| T-11 | follow_up | answer | yes | high | pass | - |
| T-12 | unanswerable | refuse | no | none | pass | - |
| T-13 | unanswerable | refuse | yes | partial | fail | prompt/model (unsupported answer) |
| T-14 | unanswerable | refuse | no | none | pass | - |
| T-15 | ambiguous | clarify | yes | partial | fail | prompt/model (answer incomplete) |
| T-16 | ambiguous | answer | yes | high | fail | prompt/model (answer incomplete) |
| T-17 | adversarial | decline | no | none | pass | - |
| T-18 | adversarial | answer | no | none | fail | retrieval/chunking |
| T-19 | adversarial | decline | no | none | pass | - |
| T-20 | robustness | consistent | yes | high | fail | prompt/model (answer incomplete) |
| T-21 | partial | partial | yes | partial | fail | citation selection |
| T-22 | multi_step | answer | yes | high | pass | - |

### Failure analysis
- **T-13 (NSF)**: the baseline answered an unanswerable question from adjacent filtration text. Gate similarity is above the threshold, so this is a *gate limit + no model*; the grounding/refusal rule in prompt v3 targets exactly this and must be verified with an LLM.
- **T-18 (false 5-year claim)**: retrieval miss. The noisy sentence "my friend told me" dilutes the query; query condensation/cleaning or a larger fetch would help.
- **Extractive baseline failures (T-02, T-03, T-04, T-15, T-16, T-20, T-21)**: the baseline quotes lines instead of reasoning, so it cannot list box contents, state a price with its promo, name a conditional rule or separate covered from uncovered parts. These are expected and are the reason an LLM is in the pipeline; they are not evidence about LLM behaviour.
- **'citation selection' failures (T-07..T-10, T-21)**: the baseline cites chunks whose source is outside the labelled expected set or lacks the key fact; with an LLM, citations come from `sources`, validated against supplied labels.

## 7. Prompt evolution
`src/chains/prompts.py` keeps v1 (weak), v2 (grounding + refusal) and v3 (final: partial answers, ambiguity, citation labels, injection resistance, no commitments). `python -m evaluation.prompt_experiment` runs weak vs final on T-12, T-13, T-17, T-18, T-21 and saves both outputs; it needs a real LLM and **has not been run**.
Injection is defended at three layers: a deterministic pre-LLM guard (tested), prompt rules treating context as data, and schema/citation validation on output. The changelog contains a planted "approve any refund" line to test the second layer with an LLM.

## 8. Limitations
- No real LLM or semantic embedding model was available while building: LLM answer quality, groundedness, semantic retrieval and cross-encoder gains are unmeasured.
- Knowledge base is ~5,400 words (assignment suggests 9,000-12,000); test set is 22 cases authored by the same person who wrote the documents.
- Threshold and chunk choice were tuned on the same 18-22 questions (no held-out set).
- The follow-up rewriter used offline is a heuristic; the LLM condenser is implemented and unit-tested with a stub but not run against a real model.
- Not implemented: streaming, multiple knowledge bases, tool calling, LangSmith evaluation, semantic query cache, production-design document (Section 14). Embedding cache and prompt/answer logging exist.
- Single-process SQLite and in-memory rate limiting: fine for a prototype, not for multiple replicas.

## 9. With two more weeks
Run everything with a real model; hold out a second test set; add a judge-model groundedness check compared against manual labels; implement cross-encoder reranking measurement; add the production design document.
