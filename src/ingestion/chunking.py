"""Three chunking strategies. Metadata is carried to every chunk and enriched with section and chunk_id."""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import (CharacterTextSplitter, MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
HEADING_LINE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,70})$", re.M)


def _structure_split(docs: list[Document], size: int, overlap: int) -> list[Document]:
    md = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    rec = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    out: list[Document] = []
    for d in docs:
        if d.metadata.get("file_type") == "csv":  # one row = one chunk, so a SKU never loses its price
            out.append(d)
            continue
        if d.metadata.get("file_type") in ("md", "docx"):
            for piece in md.split_text(d.page_content):
                piece.metadata = {**d.metadata, **piece.metadata}
                out.extend(rec.split_documents([piece]))
        else:
            out.extend(rec.split_documents([d]))
    return out


def split_documents(docs: list[Document], strategy: str, size: int, overlap: int) -> list[Document]:
    if strategy == "recursive":
        chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)
    elif strategy == "fixed":
        chunks = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=overlap).split_documents(docs)
    elif strategy == "structure":
        chunks = _structure_split(docs, size, overlap)
    else:
        raise ValueError(f"unknown chunk strategy {strategy}")
    return _enrich(chunks)


def _section_of(chunk: Document) -> str | None:
    heads = [chunk.metadata[k] for k in ("h1", "h2", "h3", "h4") if chunk.metadata.get(k)]
    if heads:
        return " > ".join(heads[1:] or heads)
    found = HEADING_LINE.findall(chunk.page_content)
    return found[-1].strip() if found else None


def _enrich(chunks: list[Document]) -> list[Document]:
    counters: dict[str, int] = {}
    last_section: dict[str, str] = {}
    for c in chunks:
        src = c.metadata["source"]
        sec = c.metadata.get("section") or _section_of(c) or last_section.get(src, "")
        last_section[src] = sec
        idx = counters.get(src, 0)
        counters[src] = idx + 1
        for k in ("h1", "h2", "h3", "h4"):
            c.metadata.pop(k, None)
        c.metadata["section"] = sec
        c.metadata["chunk_id"] = f"{c.metadata['doc_id']}-{idx:03d}"
        where = f"p.{c.metadata['page']}" if "page" in c.metadata else ""
        c.metadata["location"] = " · ".join(x for x in (where, sec) if x) or "document"
        header = f"[{c.metadata['title']} | {c.metadata['location']}]"
        if not c.page_content.startswith("["):
            c.page_content = f"{header}\n{c.page_content}"
    return chunks
