from fastapi.testclient import TestClient

import app as app_module
import ingest as ingest_module
import retrieval as retrieval_module
from app import app
from retrieval import retrieve_top_k, RetrievalError
from vector_store import VectorStore


client = TestClient(app)


def test_retrieve_top_k_orders_by_similarity():
    store = VectorStore(store_path="./data/test_vector_store_retrieval_internal.json", dimension=4)
    store.clear()

    docs = [
        {"id": "d1", "text": "doc one", "meta": {}},
        {"id": "d2", "text": "doc two", "meta": {}},
        {"id": "d3", "text": "doc three", "meta": {}},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
    ]
    store.add_documents(docs, embeddings)

    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = retrieve_top_k(query_embedding, store, k=2)

    assert len(results) == 2
    assert results[0]["id"] == "d1"
    assert results[0]["score"] >= results[1]["score"]


def test_query_to_embedding_empty_raises():
    try:
        retrieval_module.query_to_embedding("")
        assert False, "Expected RetrievalError for empty query"
    except RetrievalError:
        assert True


def test_query_endpoint_returns_ranked_results(monkeypatch):
    def fake_load_hf_dataset(*args, **kwargs):
        return [
            {"id": "hf-1", "text": "Python docs", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "More Python docs", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.05] * 3072 for _ in texts]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)

    # ingest first so query endpoint is enabled
    ingest_resp = client.post("/ingest", params={"max_docs": 2, "reingest": True})
    assert ingest_resp.status_code == 200

    def fake_query_to_embedding(query: str):
        assert query == "Python"
        return [0.1] * 3072

    def fake_retrieve_top_k(_query_embedding, _store, k: int):
        return [
            {"id": "doc-1", "text": "Python is great", "meta": {}, "score": 0.95},
            {"id": "doc-2", "text": "Snakes are reptiles", "meta": {}, "score": 0.73},
        ][:k]

    monkeypatch.setattr(app_module, "query_to_embedding", fake_query_to_embedding)
    monkeypatch.setattr(app_module, "retrieve_top_k", fake_retrieve_top_k)

    response = client.post("/query", json={"query": "Python", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Python"
    assert len(body["top_docs"]) == 2
    assert body["top_docs"][0]["score"] >= body["top_docs"][1]["score"]
    assert body["scores"] == [0.95, 0.73]


def test_query_endpoint_before_ingest_rejects():
    # clear persisted + in-memory pipeline state
    clear_resp = client.delete("/ingest")
    assert clear_resp.status_code == 200
    if hasattr(app.state, "ingest_pipeline"):
        delattr(app.state, "ingest_pipeline")

    response = client.post("/query", json={"query": "Python", "k": 1})
    assert response.status_code == 400
    assert "Ingest data first" in response.json()["detail"]
