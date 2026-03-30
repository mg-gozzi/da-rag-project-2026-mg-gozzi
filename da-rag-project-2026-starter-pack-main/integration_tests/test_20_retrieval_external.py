import socket
import threading
import time
from collections.abc import Generator

import pytest
import requests
import uvicorn

import app as app_module
import ingest as ingest_module
from app import app


SERVER_HOST = "127.0.0.1"
REQUEST_TIMEOUT_SECONDS = 10


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((SERVER_HOST, 0))
        return sock.getsockname()[1]


def start_server(port: int) -> tuple[uvicorn.Server, threading.Thread, str]:
    if hasattr(app.state, "ingest_pipeline"):
        delattr(app.state, "ingest_pipeline")

    base_url = f"http://{SERVER_HOST}:{port}"
    config = uvicorn.Config(app, host=SERVER_HOST, port=port, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    for _ in range(20):
        try:
            requests.get(f"{base_url}/health", timeout=0.5)
            break
        except requests.RequestException:
            time.sleep(0.1)

    return server, server_thread, base_url


@pytest.fixture()
def base_url(monkeypatch) -> Generator[str, None, None]:
    def fake_load_hf_dataset(*args, **kwargs):
        return [
            {"id": "hf-1", "text": "Python language", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "Monty Python", "meta": {"source": "huggingface"}},
            {"id": "hf-3", "text": "Other text", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.03] * 3072 for _ in texts]

    def fake_query_to_embedding(_query: str):
        return [0.2] * 3072

    def fake_retrieve_top_k(_query_embedding, _store, k: int):
        items = [
            {"id": "doc-1", "text": "Python language", "meta": {}, "score": 0.91},
            {"id": "doc-2", "text": "Monty Python", "meta": {}, "score": 0.77},
            {"id": "doc-3", "text": "Other text", "meta": {}, "score": 0.44},
        ]
        return items[:k]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)
    monkeypatch.setattr(app_module, "query_to_embedding", fake_query_to_embedding)
    monkeypatch.setattr(app_module, "retrieve_top_k", fake_retrieve_top_k)

    port = _get_free_port()
    server, server_thread, base_url_value = start_server(port)
    try:
        yield base_url_value
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def test_query_retrieval_end_to_end(base_url: str):
    ingest_resp = requests.post(
        f"{base_url}/ingest?max_docs=3&reingest=true",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert ingest_resp.status_code == 200

    query_resp = requests.post(
        f"{base_url}/query",
        json={"query": "Python", "k": 2},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert query_resp.status_code == 200
    body = query_resp.json()

    assert body["query"] == "Python"
    assert body["k"] == 2
    assert len(body["top_docs"]) == 2
    assert len(body["scores"]) == 2
    assert body["scores"][0] >= body["scores"][1]


def test_query_retrieval_requires_ingest(base_url: str):
    clear_resp = requests.delete(f"{base_url}/ingest", timeout=REQUEST_TIMEOUT_SECONDS)
    assert clear_resp.status_code == 200

    query_resp = requests.post(
        f"{base_url}/query",
        json={"query": "Python", "k": 2},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert query_resp.status_code == 400
    assert "Ingest data first" in query_resp.json().get("detail", "")
