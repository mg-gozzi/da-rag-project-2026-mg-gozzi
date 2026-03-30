from fastapi.testclient import TestClient

import app as app_module
import ingest as ingest_module
from app import app


client = TestClient(app)


def _setup_hf_ingest_mocks(monkeypatch):
    def fake_load_hf_dataset(*args, **kwargs):
        return [
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.1] * 3072 for _ in texts]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)


def test_answer_endpoint_returns_llm_response(monkeypatch):
    _setup_hf_ingest_mocks(monkeypatch)

    def fake_query_to_embedding(_query: str):
        return [0.2] * 3072

    def fake_retrieve_top_k(_query_embedding, _store, k: int):
        docs = [
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}, "score": 0.97},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}, "score": 0.51},
        ]
        return docs[:k]

    def fake_generate_answer(question: str, _docs):
        assert question == "What is the capital of France?"
        return "Paris"

    monkeypatch.setattr(app_module, "query_to_embedding", fake_query_to_embedding)
    monkeypatch.setattr(app_module, "retrieve_top_k", fake_retrieve_top_k)
    monkeypatch.setattr(app_module, "generate_rag_answer", fake_generate_answer)

    ingest_resp = client.post("/ingest", params={"max_docs": 2, "reingest": True})
    assert ingest_resp.status_code == 200

    response = client.post("/answer", json={"question": "What is the capital of France?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris"
    assert len(body["top_docs"]) == 2
    assert body["scores"] == [0.97, 0.51]


def test_evaluate_endpoint_returns_metrics(monkeypatch):
    _setup_hf_ingest_mocks(monkeypatch)

    def fake_query_to_embedding(_query: str):
        return [0.3] * 3072

    def fake_retrieve_top_k(_query_embedding, _store, k: int):
        docs = [
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}, "score": 0.99},
        ]
        return docs[:k]

    def fake_generate_answer(question: str, _docs):
        if "capital of France" in question:
            return "Paris"
        return "Unknown"

    def fake_test_data(limit: int):
        return [
            {"id": "qa-1", "question": "What is the capital of France?", "answer": "Paris"},
            {"id": "qa-2", "question": "What is 2+2?", "answer": "4"},
        ][:limit]

    monkeypatch.setattr(app_module, "query_to_embedding", fake_query_to_embedding)
    monkeypatch.setattr(app_module, "retrieve_top_k", fake_retrieve_top_k)
    monkeypatch.setattr(app_module, "generate_rag_answer", fake_generate_answer)
    monkeypatch.setattr(app_module, "get_sample_test_data", fake_test_data)

    ingest_resp = client.post("/ingest", params={"max_docs": 2, "reingest": True})
    assert ingest_resp.status_code == 200

    response = client.post("/evaluate", json={"limit": 2, "k": 1})
    assert response.status_code == 200
    body = response.json()

    assert body["evaluated"] == 2
    assert "exact_match_rate" in body
    assert "average_f1" in body
    assert len(body["results"]) == 2


def test_answer_requires_ingest():
    clear_resp = client.delete("/ingest")
    assert clear_resp.status_code == 200

    if hasattr(app.state, "ingest_pipeline"):
        delattr(app.state, "ingest_pipeline")

    response = client.post("/answer", json={"question": "Test", "k": 1})
    assert response.status_code == 400
    assert "Ingest data first" in response.json().get("detail", "")
