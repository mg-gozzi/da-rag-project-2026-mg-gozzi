from fastapi.testclient import TestClient

import app as app_module
from app import app


client = TestClient(app)


def test_embed_endpoint_success(monkeypatch):
    def fake_create_embedding(text: str):
        assert text == "hello world"
        return [0.1] * 3072

    monkeypatch.setattr(app_module, "create_embedding", fake_create_embedding)

    response = client.post("/embed", json={"text": "hello world"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["dimension"] == 3072
    assert len(body["embedding"]) == 3072


def test_embed_endpoint_empty_text_rejected(monkeypatch):
    def fake_create_embedding(_text: str):
        raise app_module.EmbeddingError("Cannot create embedding for empty text")

    monkeypatch.setattr(app_module, "create_embedding", fake_create_embedding)

    response = client.post("/embed", json={"text": ""})
    assert response.status_code == 400
    assert "Cannot create embedding" in response.json()["detail"]
