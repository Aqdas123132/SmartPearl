"""Retriever comparison (hit@k), hybrid before/after, reranking, metadata filter demo, threshold calibration.
Measures retrieval only: no LLM is involved, so this runs offline."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from src.config import get_settings
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import load_or_build
from src.utils.logging_setup import setup_logging

HERE = Path(__file__).parent
CASES = json.loads((HERE / "test_questions.json").read_text())
ANSWERABLE = [c for c in CASES if c["expected_behavior"] in ("answer", "clarify", "consistent", "partial", "decline") and c["key_facts"]]
UNANSWERABLE = [c for c in CASES if c["expected_behavior"] == "refuse"]
OFF_TOPIC = ["what is the capital of france", "how do I bake sourdough bread", "who won the cricket world cup", "best laptop to buy", "student discount", "is it NSF certified",
             "can I pay in instalments", "tell me a joke"]


def hit(result, case, all_groups: bool = False) -> bool:
    """hit@k: a retrieved chunk from an expected source contains the primary key fact (or, strict mode, every key-fact group
    appears somewhere in the retrieved set)."""
    chunks = result.hits
    if all_groups:
        text = " ".join(h.doc.page_content for h in chunks).lower()
        return all(any(s.lower() in text for s in grp) for grp in case["key_facts"])
    return any(any(s.lower() in h.doc.page_content.lower() for s in case["key_facts"][0]) for h in chunks)


def relevant(h, case) -> bool:
    """A chunk counts as relevant if it comes from an expected source or contains any key fact."""
    txt = h.doc.page_content.lower()
    return h.doc.metadata["source"] in case["source_docs"] or any(s.lower() in txt for g in case["key_facts"] for s in g)


def evaluate(r: Retriever, strategy: str, k: int, **kw) -> dict:
    hits, hits_all, lat = 0, 0, []
    misses = []
    for c in ANSWERABLE:
        res = r.retrieve(c["standalone"], strategy=strategy, k=k, **kw)
        lat.append(res.latency_ms)
        h = hit(res, c)
        hits += h
        hits_all += hit(res, c, all_groups=True)
        if not h:
            misses.append(c["id"])
    n = len(ANSWERABLE)
    return {"strategy": strategy, "k": k, "hit@k": round(hits / n, 3), "hit_all_facts@k": round(hits_all / n, 3),
            "mean_latency_ms": round(statistics.mean(lat), 1), "misses": misses, "n": n}


def calibrate(r: Retriever) -> dict:
    pos = [r.retrieve(c["standalone"], strategy="similarity", k=4).best_dense for c in ANSWERABLE]
    neg = [r.retrieve(c["standalone"], strategy="similarity", k=4).best_dense for c in UNANSWERABLE] + \
          [r.retrieve(q, strategy="similarity", k=4).best_dense for q in OFF_TOPIC]
    lo_pos, hi_neg = min(pos), max(neg)
    candidates = sorted(set(round(x, 3) for x in pos + neg))
    best = None
    for t in candidates:  # maximise balanced accuracy over the labelled queries
        tpr = sum(p >= t for p in pos) / len(pos)
        tnr = sum(n < t for n in neg) / len(neg)
        score = (tpr + tnr) / 2
        if best is None or score > best[0] or (score == best[0] and t > best[1]):
            best = (score, t, tpr, tnr)
    return {"answerable_min": round(lo_pos, 3), "answerable_mean": round(statistics.mean(pos), 3), "unanswerable_max": round(hi_neg, 3),
            "unanswerable_mean": round(statistics.mean(neg), 3), "separable": lo_pos > hi_neg,
            "recommended_threshold": best[1], "balanced_accuracy": round(best[0], 3), "answerable_pass_rate": round(best[2], 3),
            "unanswerable_block_rate": round(best[3], 3), "n_pos": len(pos), "n_neg": len(neg)}


def main() -> None:
    setup_logging()
    r = Retriever(load_or_build())
    out: dict = {"embedding": get_settings().embedding_provider}
    rows = [evaluate(r, "similarity", 3), evaluate(r, "similarity", 8), evaluate(r, "mmr", 4), evaluate(r, "threshold", 4),
            evaluate(r, "similarity", 4, filters={"product_line": "AquaPearl 500 Pro"}),
            evaluate(r, "hybrid", 4), evaluate(r, "hybrid_rerank", 4)]
    rows[4]["strategy"] = "similarity+metadata_filter(500 Pro)"
    out["comparison"] = rows
    # C1: exact identifiers, before vs after hybrid
    ident = [("What does E-03 mean?", "E-03", "troubleshooting_guide.md"), ("FLT-M5", "FLT-M5", "pricing_guide.csv"),
             ("price of AP-500P", "AP-500P", "pricing_guide.csv"), ("E-06", "E-06", "product_manual_aquapearl500pro.pdf")]
    id_rows = []
    for q, tok, src in ident:
        row = {"query": q}
        for strat in ("similarity", "hybrid"):
            res = r.retrieve(q, strategy=strat, k=4)
            row[strat] = any(tok.lower() in h.doc.page_content.lower() for h in res.hits)
        id_rows.append(row)
    out["identifier_queries"] = id_rows
    # C2: top-4 precision before/after reranking (precision = share of top-4 chunks from an expected source with the primary fact)
    def prec(strategy, rr):
        vals, lat = [], []
        for c in ANSWERABLE:
            res = r.retrieve(c["standalone"], strategy=strategy, k=4)
            vals.append(sum(relevant(h, c) for h in res.hits) / 4)
            lat.append(res.latency_ms)
        return round(statistics.mean(vals), 3), round(statistics.mean(lat), 1)
    p_before, l_before = prec("hybrid", False)
    p_after, l_after = prec("hybrid_rerank", True)
    out["reranking"] = {"top4_precision_before": p_before, "top4_precision_after": p_after, "latency_ms_before": l_before,
                        "latency_ms_after": l_after, "reranker": get_settings().reranker}
    # C3: metadata filter prevents wrong-product retrieval
    q = "How much is the replacement filter?"
    unf = r.retrieve(q, strategy="similarity", k=4)
    fil = r.retrieve(q, strategy="similarity", k=4, filters={"product_line": "AeroPearl Mini", "doc_type": "pricing"})
    out["metadata_filter_demo"] = {"query": q, "unfiltered_top4": [(h.doc.metadata["source"], h.doc.metadata["product_line"], h.doc.metadata["section"]) for h in unf.hits],
                                   "filtered_AeroPearl_Mini_pricing": [(h.doc.metadata["source"], h.doc.metadata["product_line"], h.doc.metadata["section"]) for h in fil.hits]}
    out["threshold_calibration"] = calibrate(r)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "retrieval_experiments.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
