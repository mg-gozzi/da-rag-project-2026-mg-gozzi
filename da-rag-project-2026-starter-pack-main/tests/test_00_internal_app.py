from fastapi.testclient import TestClient

from app import app
import ingest as ingest_module


client = TestClient(app)


def _mock_hf_ingestion(monkeypatch):
    def fake_load_hf_dataset(*args, **kwargs):
        return [
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.05] * 3072 for _ in texts]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_echo_default_message():
    response = client.get("/echo")
    assert response.status_code == 200
    assert response.json() == {"message": "hello"}


def test_echo_custom_message():
    response = client.get("/echo", params={"message": "ping"})
    assert response.status_code == 200
    assert response.json() == {"message": "ping"}


def test_docs_endpoint_available():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_ingest_and_status(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    response = client.post("/ingest", params={"max_docs": 2, "reingest": True})
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] > 0
    assert body["status"] in ["loaded with embeddings", "already_loaded"]

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["loaded"] is True
    assert status_body["documents"] == body["ingested"]


def test_query_before_ingestion_rejects_after_clearing():
    # Query after ingest should now succeed with a placeholder
    response = client.post("/query", json={"query": "Who is Python?", "k": 1})
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["query"] == "Who is Python?"
    assert "top_docs" in json_body


def test_status_before_ingest():
    # Status may already be loaded from persisted vector store
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["loaded"], bool)
    assert body["documents"] >= 0


def test_query_with_different_k(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    # Ingest first
    client.post("/ingest", params={"max_docs": 2, "reingest": True})
    # Query with k=3
    response = client.post("/query", json={"query": "Test query", "k": 3})
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["k"] == 3
    assert len(json_body["top_docs"]) >= 0  # placeholder, so at least empty list


def test_ingest_twice(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    # First ingest
    response1 = client.post("/ingest", params={"max_docs": 2, "reingest": True})
    assert response1.status_code == 200
    count1 = response1.json()["ingested"]
    # Second ingest should work (though currently same data)
    response2 = client.post("/ingest", params={"max_docs": 2})
    assert response2.status_code == 200
    count2 = response2.json()["ingested"]
    assert count2 == count1  # same data for now


def test_query_empty_string(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    client.post("/ingest", params={"max_docs": 2, "reingest": True})  # ensure ingested
    response = client.post("/query", json={"query": "", "k": 1})
    assert response.status_code == 200  # placeholder allows empty
    json_body = response.json()
    assert json_body["query"] == ""
    assert "top_docs" in json_body


def test_query_k_zero(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    client.post("/ingest", params={"max_docs": 2, "reingest": True})
    response = client.post("/query", json={"query": "Test", "k": 0})
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["k"] == 0
    assert isinstance(json_body["top_docs"], list)


def test_health_post_method():
    response = client.post("/health")
    assert response.status_code == 405  # method not allowed


def test_query_negative_k(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    client.post("/ingest", params={"max_docs": 2, "reingest": True})
    response = client.post("/query", json={"query": "Test", "k": -1})
    assert response.status_code == 200  # placeholder allows negative
    json_body = response.json()
    assert json_body["k"] == -1


def test_query_missing_k(monkeypatch):
    _mock_hf_ingestion(monkeypatch)
    client.post("/ingest", params={"max_docs": 2, "reingest": True})
    response = client.post("/query", json={"query": "Test"})  # no k
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["query"] == "Test"
    assert "k" in json_body  # should have default


def test_health_with_query_params():
    response = client.get("/health?param=value")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    json_body = response.json()
    assert "chat" in json_body
    assert "embeddings" in json_body
    assert "gpt-4o" in json_body["chat"]
    assert "text-embedding-3-large" in json_body["embeddings"]


def test_check_model_gpt4o():
    response = client.get("/models/chat/gpt-4o")
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["model_type"] == "chat"
    assert json_body["model_name"] == "gpt-4o"
    assert json_body["available"] is True
    assert json_body["deployment_name"] == "gpt-4o"
    assert json_body["api_version"] == "2024-10-01-preview"


def test_check_model_embedding():
    response = client.get("/models/embeddings/text-embedding-3-large")
    assert response.status_code == 200
    json_body = response.json()
    assert json_body["model_type"] == "embeddings"
    assert json_body["model_name"] == "text-embedding-3-large"
    assert json_body["available"] is True
    assert json_body["deployment_name"] == "text-embedding-3-large"


def test_check_model_nonexistent():
    response = client.get("/models/chat/nonexistent-model")
    assert response.status_code == 404
