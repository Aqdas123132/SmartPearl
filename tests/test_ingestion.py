from src.ingestion.loaders import load_directory, normalise
from src.ingestion.chunking import split_documents


def test_corrupt_file_is_skipped_and_others_load(tmp_path):
    (tmp_path / "ok.md").write_text("# Policy\n\nReturns within 14 days.")
    (tmp_path / "bad.pdf").write_bytes(b"this is not a pdf")
    (tmp_path / "notes.exe").write_bytes(b"x")
    (tmp_path / "dup.md").write_text("# Policy\n\nReturns within 14 days.")
    docs, rep = load_directory(tmp_path)
    status = {f.file: f.status for f in rep.files}
    assert status["bad.pdf"] == "failed" and status["notes.exe"] == "skipped"
    assert sorted([status["ok.md"], status["dup.md"]]) == ["loaded", "skipped"]  # identical content is indexed once
    assert all(d.metadata["source"] and d.metadata["doc_type"] for d in docs)


def test_real_kb_loads_all_formats_and_metadata():
    docs, rep = load_directory()
    assert rep.summary()["failed"] == 0 and {f.file.rsplit(".", 1)[-1] for f in rep.files if f.status == "loaded"} >= {"pdf", "docx", "md", "csv"}
    for key in ("source", "doc_type", "product_line", "version", "last_updated", "doc_id"):
        assert all(d.metadata.get(key) for d in docs)


def test_metadata_survives_splitting_and_csv_rows_stay_whole():
    docs, _ = load_directory()
    for strat in ("recursive", "fixed", "structure"):
        chunks = split_documents(docs, strat, 800, 120)
        assert all(c.metadata.get("source") and c.metadata.get("chunk_id") and c.metadata.get("location") for c in chunks)
    rows = [c for c in split_documents(docs, "structure", 300, 0) if c.metadata["source"] == "pricing_guide.csv"]
    assert all("sku:" in c.page_content and "list_price_pkr:" in c.page_content for c in rows)  # SKU never split from its price


def test_normalise_collapses_whitespace():
    assert normalise("a   b\t\tc\n\n\n\nd") == "a b c\n\nd"
