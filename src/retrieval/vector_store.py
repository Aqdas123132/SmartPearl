"""Persistent Chroma index: load_or_build, rebuild, incremental sync (delete-by-source + reinsert)."""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_settings
from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_directory
from src.retrieval.embeddings import embedding_info, get_embeddings

log = logging.getLogger(__name__)
WRITE_LOCK = threading.RLock()  # Chroma writes are serialised; reads need no lock


class IndexIncompatibleError(RuntimeError):
    """The persisted index was built with a different embedding model or chunking config."""


def _manifest_path():
    return get_settings().vector_store_path / "index_manifest.json"


def read_manifest() -> dict | None:
    p = _manifest_path()
    return json.loads(p.read_text()) if p.exists() else None


def _write_manifest(store: Chroma, extra: dict | None = None) -> None:
    s, info = get_settings(), embedding_info()
    got = store.get(include=["metadatas"])
    sources: dict[str, dict] = {}
    for m in got["metadatas"]:
        sources.setdefault(m["source"], {"content_hash": m["content_hash"], "version": m.get("version"), "chunks": 0})["chunks"] += 1
    data = {"embedding": info, "chunk_strategy": s.chunk_strategy, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
            "collection": s.collection_name, "chunks": len(got["ids"]), "sources": sources,
            "updated_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    s.vector_store_path.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps(data, indent=2))


def _open() -> Chroma:
    s = get_settings()
    s.vector_store_path.mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name=s.collection_name, embedding_function=get_embeddings(), persist_directory=str(s.vector_store_path),
                  collection_configuration={"hnsw": {"space": "cosine"}})


def _check_compatible(manifest: dict) -> None:
    s, info = get_settings(), embedding_info()
    if manifest["embedding"]["tag"] != info["tag"] or manifest["embedding"]["dimensions"] != info["dimensions"]:
        raise IndexIncompatibleError(
            f"Index was built with {manifest['embedding']['tag']} ({manifest['embedding']['dimensions']}d) but the configured model is "
            f"{info['tag']} ({info['dimensions']}d). Run `python -m src.ingestion.build_index --rebuild`.")


def build_index(rebuild: bool = False) -> tuple[Chroma, dict]:
    """Full build. With rebuild=True the existing collection is wiped first."""
    s = get_settings()
    with WRITE_LOCK:
        if rebuild and s.vector_store_path.exists():
            shutil.rmtree(s.vector_store_path)
        docs, report = load_directory()
        if not docs:
            raise RuntimeError("Knowledge base is empty: no documents could be loaded.")
        chunks = split_documents(docs, s.chunk_strategy, s.chunk_size, s.chunk_overlap)
        store = _open()
        start = time.perf_counter()
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        log.info("Built index: %d chunks embedded and persisted to %s in %.2fs", len(chunks), s.vector_store_path, time.perf_counter() - start)
        _write_manifest(store, {"ingestion": report.summary()})
    return store, report.to_json()


def load_or_build(rebuild: bool = False) -> Chroma:
    manifest = read_manifest()
    if rebuild or manifest is None:
        return build_index(rebuild=True)[0]
    _check_compatible(manifest)
    store = _open()
    log.info("Loaded existing index from disk (%d chunks, %s). No re-embedding performed.", manifest["chunks"], manifest["embedding"]["tag"])
    return store


def delete_source(store: Chroma, source: str) -> int:
    with WRITE_LOCK:
        ids = store.get(where={"source": source})["ids"]
        if ids:
            store.delete(ids=ids)
        _write_manifest(store)
    return len(ids)


def sync_index(store: Chroma) -> dict:
    """Incremental update: re-ingest the knowledge base, then delete-by-source and reinsert only what changed."""
    s = get_settings()
    with WRITE_LOCK:
        docs, report = load_directory()
        chunks = split_documents(docs, s.chunk_strategy, s.chunk_size, s.chunk_overlap)
        by_source: dict[str, list[Document]] = defaultdict(list)
        for c in chunks:
            by_source[c.metadata["source"]].append(c)
        got = store.get(include=["metadatas"])
        indexed: dict[str, str] = {m["source"]: m["content_hash"] for m in got["metadatas"]}
        added, updated, removed = [], [], []
        for src in set(indexed) - set(by_source):
            store.delete(ids=store.get(where={"source": src})["ids"])
            removed.append(src)
        for src, cs in by_source.items():
            h = cs[0].metadata["content_hash"]
            if indexed.get(src) == h:
                continue
            if src in indexed:
                store.delete(ids=store.get(where={"source": src})["ids"])  # drop superseded chunks first
                updated.append(src)
            else:
                added.append(src)
            store.add_documents(cs, ids=[c.metadata["chunk_id"] for c in cs])
        _write_manifest(store, {"ingestion": report.summary()})
    result = {"added": sorted(added), "updated": sorted(updated), "removed": sorted(removed), "report": report.to_json()}
    log.info("Index sync: +%d new, ~%d updated, -%d removed", len(added), len(updated), len(removed))
    return result
