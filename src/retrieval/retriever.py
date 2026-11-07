"""Retriever policies layered over the vector store: similarity, MMR, threshold, hybrid (BM25 + dense), reranking,
metadata filters and the relevance gate."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import Settings, get_settings
from src.retrieval.embeddings import STOP, _stem

log = logging.getLogger(__name__)
ID_PATTERN = re.compile(r"\b[a-z]{1,4}-\d+[a-z0-9]*\b", re.I)  # E-03, FLT-S5 style identifiers
RRF_K = 60


def tokenize(text: str) -> list[str]:
    toks = []
    for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        if "-" in t:
            toks.append(t)
            toks.extend(t.split("-"))
        elif t not in STOP:
            toks.append(_stem(t))
    return toks


@dataclass
class Hit:
    doc: Document
    dense: float | None = None
    bm25: float | None = None
    fused: float = 0.0
    rerank: float | None = None

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.fused


@dataclass
class RetrievalResult:
    hits: list[Hit]
    best_dense: float
    passed: bool
    strategy: str
    latency_ms: float
    reason: str = ""
    timings: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, store: Chroma, settings: Settings | None = None) -> None:
        self.store, self.s = store, settings or get_settings()
        self._cross = None
        self.refresh()

    # ------------------------------------------------------------------ index views
    def refresh(self) -> None:
        got = self.store.get(include=["documents", "metadatas"])
        self.docs = [Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs]) if self.docs else None

    @staticmethod
    def _chroma_filter(filters: dict | None) -> dict | None:
        if not filters:
            return None
        clauses = []
        if filters.get("product_line"):
            clauses.append({"product_line": {"$in": [filters["product_line"], "all"]}})
        if filters.get("doc_type"):
            dt = filters["doc_type"]
            clauses.append({"doc_type": {"$in": dt if isinstance(dt, list) else [dt]}})
        if filters.get("source"):
            clauses.append({"source": filters["source"]})
        return clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)

    @staticmethod
    def _match(meta: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        if filters.get("product_line") and meta.get("product_line") not in (filters["product_line"], "all"):
            return False
        dt = filters.get("doc_type")
        if dt and meta.get("doc_type") not in (dt if isinstance(dt, list) else [dt]):
            return False
        return not (filters.get("source") and meta.get("source") != filters["source"])

    # ------------------------------------------------------------------ primitives
    def dense(self, query: str, n: int, filters: dict | None = None) -> list[Hit]:
        res = self.store.similarity_search_with_score(query, k=n, filter=self._chroma_filter(filters))
        return [Hit(doc=d, dense=1.0 - dist) for d, dist in res]

    def lexical(self, query: str, n: int, filters: dict | None = None) -> list[Hit]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = [Hit(doc=self.docs[i], bm25=float(scores[i])) for i in order if scores[i] > 0 and self._match(self.docs[i].metadata, filters)]
        return out[:n]

    def _fuse(self, dense: list[Hit], lexical: list[Hit]) -> list[Hit]:
        w = self.s.hybrid_weight_dense
        merged: dict[str, Hit] = {}
        for rank, h in enumerate(dense):
            m = merged.setdefault(h.doc.metadata["chunk_id"], Hit(doc=h.doc))
            m.dense, m.fused = h.dense, m.fused + w / (RRF_K + rank + 1)
        for rank, h in enumerate(lexical):
            m = merged.setdefault(h.doc.metadata["chunk_id"], Hit(doc=h.doc))
            m.bm25, m.fused = h.bm25, m.fused + (1 - w) / (RRF_K + rank + 1)
        return sorted(merged.values(), key=lambda h: -h.fused)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if self.s.reranker == "none" or not hits:
            return hits
        if self.s.reranker == "cross_encoder":
            if self._cross is None:
                try:
                    from sentence_transformers import CrossEncoder
                    self._cross = CrossEncoder(self.s.cross_encoder_model)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("RERANKER=cross_encoder needs `pip install sentence-transformers` and a downloadable model "
                                       f"({self.s.cross_encoder_model}). Use RERANKER=lexical offline.") from exc
            scores = self._cross.predict([(query, h.doc.page_content) for h in hits])
            for h, sc in zip(hits, scores):
                h.rerank = float(sc)
        else:  # lexical reranker: query-term coverage + bigram overlap, tie-broken by dense similarity
            q = [t for t in tokenize(query) if t not in STOP]
            qset, qbi = set(q), set(zip(q, q[1:]))
            for h in hits:
                toks = tokenize(h.doc.page_content)
                tset, tbi = set(toks), set(zip(toks, toks[1:]))
                cover = len(qset & tset) / (len(qset) or 1)
                bigram = len(qbi & tbi) / (len(qbi) or 1)
                h.rerank = 0.6 * cover + 0.3 * bigram + 0.1 * (h.dense or 0.0)
        return sorted(hits, key=lambda h: -(h.rerank or 0.0))

    # ------------------------------------------------------------------ public API
    def retrieve(self, query: str, strategy: str | None = None, k: int | None = None, filters: dict | None = None,
                 rerank: bool | None = None) -> RetrievalResult:
        strategy, k = strategy or self.s.retrieval_strategy, k or self.s.retrieval_k
        start, timings = time.perf_counter(), {}
        n = self.s.fetch_k
        dense_all = self.dense(query, n, filters)
        best_dense = max((h.dense for h in dense_all), default=0.0)
        if strategy == "similarity":
            hits = dense_all[:k]
        elif strategy == "threshold":
            hits = [h for h in dense_all if h.dense >= self.s.score_threshold][:k]
        elif strategy == "mmr":
            docs = self.store.max_marginal_relevance_search(query, k=k, fetch_k=n, filter=self._chroma_filter(filters))
            by_id = {h.doc.metadata["chunk_id"]: h for h in dense_all}
            hits = [by_id.get(d.metadata["chunk_id"], Hit(doc=d)) for d in docs]
        elif strategy in ("hybrid", "hybrid_rerank"):
            fused = self._fuse(dense_all, self.lexical(query, n, filters))
            if strategy == "hybrid_rerank" and (rerank is not False):
                t0 = time.perf_counter()
                fused = self.rerank(query, fused[:n])
                timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            hits = fused[:k]
        else:
            raise ValueError(f"unknown retrieval strategy {strategy}")
        passed, reason = self.gate(query, best_dense, hits)
        latency = (time.perf_counter() - start) * 1000
        log.info("retrieve strategy=%s k=%d best_dense=%.3f passed=%s chunks=%s", strategy, k, best_dense, passed,
                 [(h.doc.metadata["chunk_id"], round(h.dense or 0, 3)) for h in hits])
        return RetrievalResult(hits=hits, best_dense=best_dense, passed=passed, strategy=strategy, latency_ms=latency, reason=reason, timings=timings)

    def gate(self, query: str, best_dense: float, hits: list[Hit]) -> tuple[bool, str]:
        """Relevance gate. Retrieval has failed if nothing came back, or if the best dense cosine similarity is below
        the empirically calibrated SCORE_THRESHOLD (see evaluation/calibrate_threshold.py). An exact identifier match
        (e.g. E-03, FLT-S5) found by BM25 also passes, because short identifier queries score low on dense similarity."""
        if not hits:
            return False, "empty result set"
        if best_dense >= self.s.score_threshold:
            return True, f"best dense similarity {best_dense:.3f} >= {self.s.score_threshold}"
        ids = {m.lower() for m in ID_PATTERN.findall(query)}
        if ids and any(i in h.doc.page_content.lower() for h in hits for i in ids):
            return True, f"exact identifier match {sorted(ids)}"
        return False, f"best dense similarity {best_dense:.3f} < {self.s.score_threshold}"
