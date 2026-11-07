def test_index_reload_does_not_reembed(retriever):
    from src.retrieval.embeddings import get_embeddings
    from src.retrieval.vector_store import load_or_build
    before = get_embeddings().calls["embedded"]
    load_or_build()
    assert get_embeddings().calls["embedded"] == before


def test_warranty_query_returns_warranty_chunks(retriever):
    res = retriever.retrieve("how long is the warranty", strategy="similarity", k=5)
    assert any(h.doc.metadata["source"] == "warranty_policy.md" for h in res.hits) and res.passed


def test_gate_blocks_off_topic(retriever):
    assert not retriever.retrieve("what is the capital of france", k=4).passed


def test_exact_identifier_hybrid(retriever):
    res = retriever.retrieve("E-03", strategy="hybrid", k=4)
    assert any("E-03" in h.doc.page_content for h in res.hits)


def test_metadata_filter_restricts_product_line(retriever):
    res = retriever.retrieve("replacement filter price", strategy="similarity", k=4, filters={"product_line": "AeroPearl Mini", "doc_type": "pricing"})
    assert res.hits and all(h.doc.metadata["product_line"] in ("AeroPearl Mini", "all") and h.doc.metadata["doc_type"] == "pricing" for h in res.hits)


def test_incremental_sync_replaces_superseded_chunks(retriever, tmp_path, monkeypatch):
    from src.retrieval import vector_store as vs
    n_before = len(retriever.store.get()["ids"])
    assert vs.delete_source(retriever.store, "shipping_policy.md") > 0
    res = vs.sync_index(retriever.store)
    assert res["added"] == ["shipping_policy.md"] and len(retriever.store.get()["ids"]) == n_before  # restored, no duplicates
    retriever.refresh()
