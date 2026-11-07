"""CLI: python -m src.ingestion.build_index [--rebuild] [--debug-search "query"]"""
from __future__ import annotations

import argparse
import json

from src.config import get_settings
from src.retrieval.vector_store import load_or_build, sync_index, build_index
from src.utils.logging_setup import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Build, load or sync the SupportPearlz vector index")
    ap.add_argument("--rebuild", action="store_true", help="wipe and rebuild the whole index")
    ap.add_argument("--search", metavar="QUERY", help="debug: show raw similarity hits for a query")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    setup_logging()
    s = get_settings()
    if args.rebuild:
        store, report = build_index(rebuild=True)
        print(json.dumps(report["summary"], indent=2))
    else:
        store = load_or_build()
        result = sync_index(store)
        print(json.dumps({k: result[k] for k in ("added", "updated", "removed")}), json.dumps(result["report"]["summary"]))
    (s.log_dir).mkdir(exist_ok=True)
    if args.search:
        for doc, dist in store.similarity_search_with_score(args.search, k=args.k):
            m = doc.metadata
            print(f"sim={1 - dist:.3f} {m['source']} [{m['location']}] id={m['chunk_id']}\n   {doc.page_content[:200]!r}")


if __name__ == "__main__":
    main()
