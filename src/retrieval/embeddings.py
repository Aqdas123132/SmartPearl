"""Embedding providers, a persistent embedding cache (resumable runs) and batching with backoff."""
from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import sqlite3
import time
import zlib
from array import array

from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings

log = logging.getLogger(__name__)
STOP = {"the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for", "do", "does", "my", "i", "it", "you", "your", "can", "how", "what", "me"}


def _stem(w: str) -> str:
    for suf in ("ing", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings. OFFLINE DEV/TEST ONLY: it captures lexical overlap,
    not meaning, so it cannot bridge a vocabulary gap the way a real embedding model does."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]
        feats = [(w, 1.0) for w in words] + [(f"{a}_{b}", 0.5) for a, b in zip(words, words[1:])]
        for w in words:
            feats += [(f"#{w[i:i + 3]}", 0.25) for i in range(max(len(w) - 2, 0))]
        for f, wt in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += wt if (h >> 20) & 1 else -wt
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class CachedEmbeddings(Embeddings):
    """Wraps any Embeddings: SQLite cache keyed by (provider, model, text), batching and retry with backoff.
    A crash part-way through an embedding run loses nothing already cached."""

    def __init__(self, inner: Embeddings, tag: str, cache_path, batch_size: int = 64, max_retries: int = 4) -> None:
        self.inner, self.tag, self.batch_size, self.max_retries = inner, tag, batch_size, max_retries
        self.cache_path = str(cache_path)
        self.calls = {"cache_hits": 0, "embedded": 0}
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def _db(self):
        return sqlite3.connect(self.cache_path, timeout=30)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.tag}|{text}".encode()).hexdigest()

    def _retry(self, fn, *args):
        for attempt in range(self.max_retries):
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001 - transient provider errors
                if attempt == self.max_retries - 1:
                    raise
                delay = 2 ** attempt + random.random()
                log.warning("Embedding call failed (%s); retry %d in %.1fs", type(exc).__name__, attempt + 1, delay)
                time.sleep(delay)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        out: dict[str, list[float]] = {}
        with self._db() as db:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                rows = db.execute(f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})", part).fetchall()
                out.update({k: list(array("f", v)) for k, v in rows})
        todo = [(k, t) for k, t in zip(keys, texts) if k not in out]
        self.calls["cache_hits"] += len(texts) - len(todo)
        start = time.perf_counter()
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            vecs = self._retry(self.inner.embed_documents, [t for _, t in batch])
            with self._db() as db:  # commit per batch -> resumable
                db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", [(k, array("f", v).tobytes()) for (k, _), v in zip(batch, vecs)])
            out.update({k: v for (k, _), v in zip(batch, vecs)})
        if todo:
            self.calls["embedded"] += len(todo)
            log.info("Embedded %d texts with %s in %.2fs (%d served from cache)", len(todo), self.tag,
                     time.perf_counter() - start, len(texts) - len(todo))
        return [out[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _inner(s: Settings) -> tuple[Embeddings, str]:
    if s.embedding_provider == "hash":
        return HashEmbeddings(), "hash/bow-512"
    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key, timeout=s.llm_timeout_s), f"openai/{s.embedding_model}"
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=s.embedding_model), f"huggingface/{s.embedding_model}"


_cached: CachedEmbeddings | None = None


def get_embeddings() -> CachedEmbeddings:
    global _cached
    if _cached is None:
        s = get_settings()
        s.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        inner, tag = _inner(s)
        _cached = CachedEmbeddings(inner, tag, s.embedding_cache_path)
    return _cached


def embedding_info() -> dict:
    e = get_embeddings()
    return {"tag": e.tag, "dimensions": len(e.embed_query("dimension probe"))}
