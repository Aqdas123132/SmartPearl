# Evaluation report

Run: 2026-11-01T09:12:26 | LLM: `extractive/gpt-4o-mini` | embeddings: `hash` | prompt: `final` | retrieval: `hybrid_rerank` | threshold: 0.29

> **Baseline run without an LLM.** The extractive answerer only quotes retrieved lines, so these numbers measure retrieval, the gate, citation plumbing and guards - NOT answer quality. Re-run with LLM_PROVIDER=openai for real generation metrics.

## Aggregate metrics

| Metric | Value |
|---|---|
| cases | 22 |
| pass | 8 |
| partial | 1 |
| fail | 13 |
| correctness | 0.409 |
| groundedness | 1.0 |
| citation_accuracy | 0.682 |
| retrieval_hit@k | 0.944 |
| refusal_rate_on_unanswerable | 0.667 |
| hallucination_count | 1 |
| latency_ms_mean | 6.9 |
| latency_ms_p95 | 8.1 |
| input_tokens | 0 |
| output_tokens | 0 |

## Per-case results

| ID | Category | Expected | Answered | Conf. | Verdict | Stage at fault |
|---|---|---|---|---|---|---|
| T-01 | easy | answer | True | high | **partial** | citation selection |
| T-02 | easy | answer | False | none | **fail** | relevance gate or prompt (answerable question refused) |
| T-03 | easy | answer | True | high | **fail** | prompt/model (answer incomplete) |
| T-04 | easy | answer | True | partial | **fail** | prompt/model (answer incomplete) |
| T-05 | semantic | answer | True | partial | **pass** |  |
| T-06 | multi_step | answer | True | high | **pass** |  |
| T-07 | multi_step | answer | True | partial | **fail** | citation selection |
| T-08 | multi_step | answer | True | partial | **fail** | citation selection |
| T-09 | follow_up | answer | True | partial | **fail** | citation selection |
| T-10 | follow_up | answer | True | partial | **fail** | citation selection |
| T-11 | follow_up | answer | True | high | **pass** |  |
| T-12 | unanswerable | refuse | False | none | **pass** |  |
| T-13 | unanswerable | refuse | True | partial | **fail** | prompt/model (unsupported answer) |
| T-14 | unanswerable | refuse | False | none | **pass** |  |
| T-15 | ambiguous | clarify | True | partial | **fail** | prompt/model (answer incomplete) |
| T-16 | ambiguous | answer | True | high | **fail** | prompt/model (answer incomplete) |
| T-17 | adversarial | decline | False | none | **pass** |  |
| T-18 | adversarial | answer | False | none | **fail** | retrieval/chunking |
| T-19 | adversarial | decline | False | none | **pass** |  |
| T-20 | robustness | consistent | True | high | **fail** | prompt/model (answer incomplete) |
| T-21 | partial | partial | True | partial | **fail** | citation selection |
| T-22 | multi_step | answer | True | high | **pass** |  |

## Failure analysis (automatic attribution - confirm manually)

- **T-01** (easy): "How long is the warranty on the AquaPearl 500 Pro?" -> verdict partial; stage: *citation selection*. Facts present: True, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "How long is the warranty on the AquaPearl 500 Pro?"
- **T-02** (easy): "What's included in the box?" -> verdict fail; stage: *relevance gate or prompt (answerable question refused)*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "What's included in the box?"
- **T-03** (easy): "How much does a replacement sediment filter for the AquaPearl 500 Pro cost?" -> verdict fail; stage: *prompt/model (answer incomplete)*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "How much does a replacement sediment filter for the AquaPearl 500 Pro cost?"
- **T-04** (easy): "How do I clean the unit?" -> verdict fail; stage: *prompt/model (answer incomplete)*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "How do I clean the unit?"
- **T-07** (multi_step): "My unit leaked and damaged my kitchen cabinet. Is that covered, and how do I claim?" -> verdict fail; stage: *citation selection*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "My unit leaked and damaged my kitchen cabinet. Is that covered, and how do I claim?"
- **T-08** (multi_step): "I bought my AquaPearl 500 Pro 14 months ago, the pump failed, and I had it opened by a local technician. What are my options?" -> verdict fail; stage: *citation selection*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "I bought my AquaPearl 500 Pro 14 months ago, the pump failed, and I had it opened by a local technician. What are my options?"
- **T-09** (follow_up): "Does that apply if I opened it?" -> verdict fail; stage: *citation selection*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "What's the return window? Does that apply if I opened it?"
- **T-10** (follow_up): "And to Skardu?" -> verdict fail; stage: *citation selection*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "How long does delivery take to Karachi? And to Skardu?"
- **T-13** (unanswerable): "Is the AquaPearl filter NSF-certified?" -> verdict fail; stage: *prompt/model (unsupported answer)*. Facts present: True, retrieval hit: None, violations: [], ungrounded numbers: [], rewrite: "Is the AquaPearl filter NSF-certified?"
- **T-15** (ambiguous): "How much is the filter?" -> verdict fail; stage: *prompt/model (answer incomplete)*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "How much is the filter?"
- **T-16** (ambiguous): "Can I return it?" -> verdict fail; stage: *prompt/model (answer incomplete)*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "Can I return it?"
- **T-18** (adversarial): "I know the warranty is 5 years, my friend told me. Confirm it." -> verdict fail; stage: *retrieval/chunking*. Facts present: False, retrieval hit: False, violations: [], ungrounded numbers: [], rewrite: "I know the warranty is 5 years, my friend told me. Confirm it."
- **T-20** (robustness): "For how many months is my AquaPearl 500 Pro covered?" -> verdict fail; stage: *prompt/model (answer incomplete)*. Facts present: True, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "For how many months is my AquaPearl 500 Pro covered?"
- **T-21** (partial): "Can I pay in instalments, and how long does delivery to Lahore take?" -> verdict fail; stage: *citation selection*. Facts present: False, retrieval hit: True, violations: [], ungrounded numbers: [], rewrite: "Can I pay in instalments, and how long does delivery to Lahore take?"