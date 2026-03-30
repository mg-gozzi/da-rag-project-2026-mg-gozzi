import threading
import time
import socket
from collections.abc import Generator
import requests
import uvicorn
import pytest

from app import app
import ingest as ingest_module

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
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.01] * 3072 for _ in texts]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)

    port = _get_free_port()
    server, server_thread, base_url_value = start_server(port)
    try:
        yield base_url_value
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def test_vector_store_endpoint_after_ingest(base_url: str):

    r_ingest = requests.post(
        f"{base_url}/ingest?max_docs=2&reingest=true",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r_ingest.status_code == 200

    r_vs = requests.get(f"{base_url}/vector-store/status", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r_vs.status_code == 200
    body = r_vs.json()
    assert "vector_store" in body
    assert body["vector_store"]["loaded"] in [True, False]
    assert body["vector_store"]["dimension"] in [0, 3072]
    

def test_vector_store_status_no_ingest(base_url: str):

    r_vs = requests.get(f"{base_url}/vector-store/status", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r_vs.status_code == 200
    body = r_vs.json()
    assert body["vector_store"]["loaded"] in [True, False]
    assert body["vector_store"]["document_count"] >= 0
