"""Automated evaluation: python -m evaluation.run_eval [--prompt final|weak|v2]
Writes JSON, CSV and a readable Markdown report to evaluation/results/. Scores are automatic PROXIES (see docs); groundedness and
correctness for free-text answers should also be spot-checked manually."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from src.chains.rag_chain import SupportAgent
from src.config import get_settings
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import load_or_build
from src.utils.logging_setup import setup_logging

HERE = Path(__file__).parent
GAP_PHRASES = ("not cover", "couldn't find", "could not find", "does not", "doesn't", "no information", "not stated", "not in", "can't confirm", "cannot confirm")


def groups_ok(text: str, groups: list[list[str]]) -> bool:
    t = text.lower()
    return all(any(s.lower() in t for s in g) for g in groups)


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def run_case(agent: SupportAgent, case: dict, chunk_text: dict[str, str]) -> dict:
    history, outs = [], []
    for turn in case["turns"]:
        r = agent.ask(turn, history, session_id=case["id"])
        history += [("user", turn), ("assistant", r.answer)]
        outs.append(r)
    r, kf, beh = outs[-1], case["key_facts"], case["expected_behavior"]
    ctx = " ".join(chunk_text.get(c["chunk_id"], "") for c in r.retrieved)
    low = r.answer.lower()
    violations = [m for m in case["must_not"] if m.lower() in low]
    ungrounded = sorted(n for n in numbers(r.answer) - numbers(ctx) if len(n) > 1) if r.answered else []
    hit = any(any(s.lower() in chunk_text.get(c["chunk_id"], "").lower() for s in kf[0]) for c in r.retrieved) if kf else None
    facts = groups_ok(r.answer, kf) if kf else True
    if beh == "answer":
        behavior_ok = r.answered and facts
    elif beh == "refuse":
        behavior_ok = not r.answered
    elif beh == "clarify":
        behavior_ok = r.answered and ("?" in r.answer or sum(any(s.lower() in low for s in g) for g in kf for s in g) >= 2)
    elif beh == "decline":
        behavior_ok = (not r.answered) if not kf else (facts or not r.answered)
    elif beh == "partial":
        behavior_ok = r.answered and facts and any(p in low for p in GAP_PHRASES)
    else:  # consistent
        behavior_ok = all(o.answered and groups_ok(o.answer, kf) for o in outs) and len({tuple(sorted(c.source for c in o.sources)) for o in outs}) == 1
    rw_ok = all(w in r.rewritten_query.lower() for w in case["rewrite_must_have"]) and not any(w in r.rewritten_query.lower() for w in case["rewrite_must_not"])
    exp_src = set(case["source_docs"])
    if r.answered:
        cite_ok = bool(r.sources) and all(c.source in exp_src or any(s.lower() in chunk_text.get(c.chunk_id, "").lower() for g in kf for s in g) for c in r.sources)
    else:
        cite_ok = not r.sources
    grounded = not violations and not ungrounded
    verdict = "pass" if (behavior_ok and grounded and cite_ok and rw_ok) else ("partial" if behavior_ok and not violations else "fail")
    stage = ""
    if verdict != "pass":
        if kf and hit is False:
            stage = "retrieval/chunking"
        elif not rw_ok:
            stage = "query rewrite"
        elif beh in ("answer", "clarify", "partial", "consistent") and not r.answered:
            stage = "relevance gate or prompt (answerable question refused)"
        elif violations or (beh == "refuse" and r.answered):
            stage = "prompt/model (unsupported answer)"
        elif not cite_ok:
            stage = "citation selection"
        else:
            stage = "prompt/model (answer incomplete)"
    return {"id": case["id"], "category": case["category"], "question": case["turns"][-1], "expected": beh, "answer": r.answer,
            "answered": r.answered, "confidence": r.confidence.value, "escalation": r.escalation_required, "rewritten_query": r.rewritten_query,
            "citations": [f"{c.source} — {c.location}" for c in r.sources], "chunk_ids": [c["chunk_id"] for c in r.retrieved],
            "retrieval_hit@k": hit, "behavior_ok": bool(behavior_ok), "facts_present": facts, "citation_ok": cite_ok, "rewrite_ok": rw_ok,
            "must_not_violations": violations, "ungrounded_numbers": ungrounded, "flags": r.flags, "latency_ms": r.latency_ms,
            "input_tokens": sum(o.input_tokens for o in outs), "output_tokens": sum(o.output_tokens for o in outs), "verdict": verdict, "failure_stage": stage}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    unans = [r for r in rows if r["expected"] == "refuse"]
    answered = [r for r in rows if r["answered"]]
    kf_rows = [r for r in rows if r["retrieval_hit@k"] is not None]
    lat = sorted(r["latency_ms"] for r in rows)
    halluc = sum(len(r["must_not_violations"]) + len(r["ungrounded_numbers"]) + (1 if r["expected"] == "refuse" and r["answered"] else 0) for r in rows)
    return {"cases": n, "pass": sum(r["verdict"] == "pass" for r in rows), "partial": sum(r["verdict"] == "partial" for r in rows),
            "fail": sum(r["verdict"] == "fail" for r in rows),
            "correctness": round(sum(r["behavior_ok"] for r in rows) / n, 3),
            "groundedness": round(sum(not r["must_not_violations"] and not r["ungrounded_numbers"] for r in answered) / (len(answered) or 1), 3),
            "citation_accuracy": round(sum(r["citation_ok"] for r in rows) / n, 3),
            "retrieval_hit@k": round(sum(r["retrieval_hit@k"] for r in kf_rows) / (len(kf_rows) or 1), 3),
            "refusal_rate_on_unanswerable": round(sum(not r["answered"] for r in unans) / (len(unans) or 1), 3),
            "hallucination_count": halluc, "latency_ms_mean": round(statistics.mean(lat), 1), "latency_ms_p95": lat[int(0.95 * (n - 1))],
            "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows)}


def write_outputs(meta: dict, rows: list[dict], agg: dict) -> None:
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation_results.json").write_text(json.dumps({"meta": meta, "aggregate": agg, "cases": rows}, indent=2, ensure_ascii=False))
    with (out / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    md = [f"# Evaluation report\n", f"Run: {meta['run_at']} | LLM: `{meta['llm_provider']}/{meta['llm_model']}` | embeddings: `{meta['embedding']}` | prompt: `{meta['prompt_version']}` | "
          f"retrieval: `{meta['retrieval_strategy']}` | threshold: {meta['score_threshold']}\n"]
    if meta["llm_provider"] == "extractive":
        md.append("> **Baseline run without an LLM.** The extractive answerer only quotes retrieved lines, so these numbers measure retrieval, the gate, "
                  "citation plumbing and guards - NOT answer quality. Re-run with LLM_PROVIDER=openai for real generation metrics.\n")
    md.append("## Aggregate metrics\n\n| Metric | Value |\n|---|---|")
    md += [f"| {k} | {v} |" for k, v in agg.items()]
    md.append("\n## Per-case results\n\n| ID | Category | Expected | Answered | Conf. | Verdict | Stage at fault |\n|---|---|---|---|---|---|---|")
    md += [f"| {r['id']} | {r['category']} | {r['expected']} | {r['answered']} | {r['confidence']} | **{r['verdict']}** | {r['failure_stage']} |" for r in rows]
    md.append("\n## Failure analysis (automatic attribution - confirm manually)\n")
    for r in rows:
        if r["verdict"] != "pass":
            md.append(f"- **{r['id']}** ({r['category']}): \"{r['question']}\" -> verdict {r['verdict']}; stage: *{r['failure_stage']}*. "
                      f"Facts present: {r['facts_present']}, retrieval hit: {r['retrieval_hit@k']}, violations: {r['must_not_violations']}, "
                      f"ungrounded numbers: {r['ungrounded_numbers']}, rewrite: \"{r['rewritten_query']}\"")
    (out / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="final")
    args = ap.parse_args()
    setup_logging()
    s = get_settings()
    cases = json.loads((HERE / "test_questions.json").read_text())
    retriever = Retriever(load_or_build())
    agent = SupportAgent(retriever, prompt_version=args.prompt)
    chunk_text = {d.metadata["chunk_id"]: d.page_content for d in retriever.docs}
    rows = [run_case(agent, c, chunk_text) for c in cases]
    agg = aggregate(rows)
    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "llm_provider": s.llm_provider, "llm_model": s.llm_model, "embedding": s.embedding_provider,
            "prompt_version": args.prompt, "retrieval_strategy": s.retrieval_strategy, "score_threshold": s.score_threshold, "chunk": f"{s.chunk_strategy}/{s.chunk_size}/{s.chunk_overlap}"}
    write_outputs(meta, rows, agg)
    print(json.dumps(agg, indent=2))
    for r in rows:
        print(f"{r['id']} {r['category']:13} {r['verdict']:8} {r['failure_stage']}")


if __name__ == "__main__":
    main()
