"""Central logging: console + file handlers, plus a JSON-lines per-turn log."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

from src.config import get_settings

_configured = False
_turn_lock = threading.Lock()


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    s = get_settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(s.log_level.upper())
    for h in (logging.StreamHandler(), logging.FileHandler(s.log_dir / "app.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _configured = True


def log_turn(record: dict) -> None:
    """Append one structured per-turn record (query, rewrite, chunk ids and scores, latency, refusal, error)."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    path = get_settings().log_dir / "turns.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _turn_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
