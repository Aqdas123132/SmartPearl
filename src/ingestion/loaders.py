"""File-type dispatch -> LangChain loaders -> normalised, metadata-carrying Documents."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from docx import Document as DocxFile
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from src.config import get_settings

log = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".docx", ".md", ".txt", ".csv"}
MANIFEST_NAME = "manifest.json"


class DocxStructuredLoader(BaseLoader):
    """DOCX loader that keeps Word heading styles as markdown '#' markers so structure-aware splitting works."""

    def __init__(self, path: str) -> None:
        self.path = path

    def lazy_load(self) -> Iterator[Document]:
        lines = []
        for p in DocxFile(self.path).paragraphs:
            if not p.text.strip():
                continue
            style = (p.style.name or "").lower()
            if style == "title":
                lines.append(f"# {p.text}")
            elif style.startswith("heading"):
                level = int(re.sub(r"\D", "", style) or 1) + 1
                lines.append(f"{'#' * min(level, 4)} {p.text}")
            else:
                lines.append(p.text)
        yield Document(page_content="\n".join(lines), metadata={"source": self.path})


@dataclass
class FileReport:
    file: str
    status: str  # loaded | skipped | failed
    detail: str = ""
    doc_type: str = ""
    chars: int = 0
    sections: int = 0
    metadata_keys: list[str] = field(default_factory=list)


@dataclass
class IngestionReport:
    files: list[FileReport] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for f in self.files if f.status == status)

    def summary(self) -> dict:
        return {"files_found": len(self.files), "loaded": self.count("loaded"), "skipped": self.count("skipped"),
                "failed": self.count("failed"), "total_chars": sum(f.chars for f in self.files),
                "total_sections": sum(f.sections for f in self.files)}

    def render(self) -> str:
        rows = [f"{'file':42} {'status':8} {'type':9} {'chars':>7} {'sects':>5}  detail"]
        for f in self.files:
            rows.append(f"{f.file:42} {f.status:8} {f.doc_type:9} {f.chars:>7} {f.sections:>5}  {f.detail}")
        rows.append(json.dumps(self.summary()))
        return "\n".join(rows)

    def to_json(self) -> dict:
        return {"summary": self.summary(), "files": [asdict(f) for f in self.files]}


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_repeated_lines(pages: list[Document]) -> tuple[list[Document], list[str]]:
    """Remove short lines that repeat on more than half of the pages (running headers and footers)."""
    if len(pages) < 3:
        return pages, []
    counts: Counter = Counter()
    for p in pages:
        for line in {re.sub(r"\d+", "#", ln.strip()) for ln in p.page_content.split("\n") if 0 < len(ln.strip()) < 90}:
            counts[line] += 1
    repeated = {ln for ln, c in counts.items() if c > len(pages) / 2}
    for p in pages:
        p.page_content = "\n".join(ln for ln in p.page_content.split("\n") if re.sub(r"\d+", "#", ln.strip()) not in repeated)
    return pages, sorted(repeated)


def _manifest(kb_dir: Path) -> dict:
    path = kb_dir / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_one(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".docx":
        return DocxStructuredLoader(str(path)).load()
    if ext in (".md", ".txt"):
        return TextLoader(str(path), encoding="utf-8").load()
    if ext == ".csv":
        return CSVLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"unsupported file type {ext}")


def doc_id_for(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:10]


def load_directory(kb_dir: Path | None = None) -> tuple[list[Document], IngestionReport]:
    """Recursively load every supported file. Errors are handled per file, never per run."""
    kb_dir = Path(kb_dir or get_settings().knowledge_base_dir)
    manifest = _manifest(kb_dir)
    report, docs, seen_hashes = IngestionReport(), [], {}
    for path in sorted(p for p in kb_dir.rglob("*") if p.is_file() and p.name != MANIFEST_NAME):
        rel = path.relative_to(kb_dir).as_posix()
        fr = FileReport(file=rel, status="loaded")
        report.files.append(fr)
        if path.suffix.lower() not in SUPPORTED:
            fr.status, fr.detail = "skipped", "unsupported file type"
            log.warning("Skipping %s: unsupported file type", rel)
            continue
        try:
            raw = _load_one(path)
            if path.suffix.lower() == ".pdf":
                raw, removed = strip_repeated_lines(raw)
                fr.detail = f"stripped {len(removed)} repeated header/footer lines" if removed else ""
            meta_in = manifest.get(rel, {})
            for d in raw:
                d.page_content = normalise(d.page_content)
            raw = [d for d in raw if d.page_content]
            if not raw:
                fr.status, fr.detail = "skipped", "empty document"
                log.warning("Skipping %s: empty document", rel)
                continue
            digest = hashlib.sha256("".join(d.page_content for d in raw).encode()).hexdigest()
            if digest in seen_hashes:
                fr.status, fr.detail = "skipped", f"duplicate of {seen_hashes[digest]}"
                log.warning("Skipping %s: duplicate content of %s", rel, seen_hashes[digest])
                continue
            seen_hashes[digest] = rel
            for i, d in enumerate(raw):
                meta = {
                    "source": rel, "title": meta_in.get("title", path.stem), "doc_type": meta_in.get("doc_type", "general"),
                    "product_line": meta_in.get("product_line", "all"), "version": meta_in.get("version", "unknown"),
                    "last_updated": meta_in.get("last_updated", "unknown"), "doc_id": doc_id_for(rel),
                    "content_hash": digest[:16], "file_type": path.suffix.lower().lstrip("."),
                }
                if path.suffix.lower() == ".pdf":
                    meta["page"] = int(d.metadata.get("page", i)) + 1
                if path.suffix.lower() == ".csv":
                    first = d.page_content.split("\n")[0]
                    sku = first.split(":", 1)[-1].strip() if first.startswith("sku") else str(i + 1)
                    meta["section"] = f"SKU {sku}"
                    m = re.search(r"product_line: (.+)", d.page_content)
                    meta["product_line"] = m.group(1).strip() if m else meta["product_line"]
                d.metadata = meta
            fr.doc_type, fr.chars, fr.sections = meta["doc_type"], sum(len(d.page_content) for d in raw), len(raw)
            fr.metadata_keys = sorted(raw[0].metadata)
            docs.extend(raw)
        except Exception as exc:  # noqa: BLE001 - one bad file must never stop the run
            fr.status, fr.detail = "failed", f"{type(exc).__name__}: {exc}"[:200]
            log.warning("Failed to load %s: %s", rel, fr.detail)
    return docs, report
