"""Chunking experiment: python -m evaluation.chunk_experiment
Compares 3 strategies x sizes/overlaps. hit@4 is measured over the answerable test questions with an in-memory index."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore

from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_directory
from src.retrieval.embeddings import get_embeddings

HERE = Path(__file__).parent
CONFIGS = [("recursive", 300, 0), ("recursive", 300, 45), ("recursive", 800, 0), ("recursive", 800, 120), ("recursive", 1500, 0),
           ("recursive", 1500, 225), ("fixed", 800, 120), ("structure", 300, 45), ("structure", 800, 120), ("structure", 1500, 225)]


def main() -> None:
    docs, _ = load_directory()
    cases = [c for c in json.loads((HERE / "test_questions.json").read_text()) if c["key_facts"] and c["expected_behavior"] != "refuse"]
    emb, rows = get_embeddings(), []
    for strat, size, ov in CONFIGS:
        t0 = time.perf_counter()
        chunks = split_documents([d.model_copy(deep=True) for d in docs], strat, size, ov)
        store = InMemoryVectorStore(emb)
        store.add_documents(chunks)
        elapsed = time.perf_counter() - t0
        lens = [len(c.page_content) for c in chunks]
        hits = sum(any(any(s.lower() in d.page_content.lower() for s in c["key_facts"][0]) for d in store.similarity_search(c["standalone"], k=4)) for c in cases)
        rows.append({"strategy": strat, "size": size, "overlap": ov, "chunks": len(chunks), "mean_len": round(statistics.mean(lens)), "max_len": max(lens),
                     "hit@4": round(hits / len(cases), 3), "seconds": round(elapsed, 2)})
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "chunking_experiment.json").write_text(json.dumps(rows, indent=2))
    md = ["| Strategy | Size | Overlap | Chunks | Mean len | Max len | hit@4 | Time (s) |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['strategy']} | {r['size']} | {r['overlap']} | {r['chunks']} | {r['mean_len']} | {r['max_len']} | {r['hit@4']} | {r['seconds']} |" for r in rows]
    (HERE / "results" / "chunking_experiment.md").write_text("\n".join(md))
    print("\n".join(md), f"\n(n={len(cases)} questions, embeddings={emb.tag})")


if __name__ == "__main__":
    main()
