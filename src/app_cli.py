"""Interactive CLI: python -m src.app_cli   (commands: /reset, /debug, /quit)"""
from __future__ import annotations

import uuid

from src.chains.memory import SessionMemory
from src.chains.rag_chain import SupportAgent
from src.config import ConfigError, get_settings
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import IndexIncompatibleError, load_or_build, read_manifest
from src.utils.logging_setup import setup_logging


def main() -> None:
    try:
        s = get_settings()
    except ConfigError as e:
        raise SystemExit(f"Configuration error: {e}")
    setup_logging()
    if read_manifest() is None:
        print("No index found. Building it once now (use `python -m src.ingestion.build_index` to do this explicitly)...")
    try:
        store = load_or_build()
    except (IndexIncompatibleError, RuntimeError) as e:
        raise SystemExit(f"Index problem: {e}")
    agent = SupportAgent(Retriever(store))
    mem, sid, debug = SessionMemory(s.history_window_turns), uuid.uuid4().hex[:8], False
    m = read_manifest() or {}
    print(f"[session {sid} | llm={s.llm_provider} | embeddings={m.get('embedding', {}).get('tag')} | {m.get('chunks')} chunks | {len(m.get('sources', {}))} documents]")
    print("Ask a question. Commands: /reset  /debug  /quit")
    while True:
        try:
            q = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            mem.reset()
            print("Session history cleared.")
            continue
        if q == "/debug":
            debug = not debug
            print(f"debug={'on' if debug else 'off'}")
            continue
        r = agent.ask(q, mem.window(), sid)
        mem.add("user", q)
        mem.add("assistant", r.answer)
        if r.rewritten_query != q:
            print(f"  (rewritten query: {r.rewritten_query})")
        print(f"Bot > {r.answer}")
        if r.sources:
            print("Sources:")
            for i, c in enumerate(r.sources, 1):
                print(f"  [{i}] {c.source} — {c.location} (v{c.version})")
        else:
            print("Sources: none found")
        print(f"Confidence: {r.confidence.value}" + ("  | escalate to human support" if r.escalation_required else ""))
        if debug:
            print("  retrieved:", r.retrieved, "flags:", r.flags, f"{r.latency_ms}ms")


if __name__ == "__main__":
    main()
