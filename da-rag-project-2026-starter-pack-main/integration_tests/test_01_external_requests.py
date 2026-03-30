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


def start_server(port: int) -> tuple[uvicorn.Server, threading.Thread]:
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

    return server, server_thread


@pytest.fixture()
def base_url(monkeypatch) -> Generator[str, None, None]:
    def fake_load_hf_dataset(*args, **kwargs):
        return [
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.05] * 3072 for _ in texts]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)

    port = _get_free_port()
    base_url = f"http://{SERVER_HOST}:{port}"
    server, server_thread = start_server(port)
    try:
        yield base_url
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def test_integration_health_and_ingest(base_url: str):

    r = requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r_ingest = requests.post(
        f"{base_url}/ingest?max_docs=2&reingest=true",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r_ingest.status_code == 200
    ingested = r_ingest.json()["ingested"]
    assert ingested > 0

    r_status = requests.get(f"{base_url}/status", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r_status.status_code == 200
    status_obj = r_status.json()
    assert status_obj["loaded"] is True
    assert status_obj["documents"] == ingested

    r_query = requests.post(
        f"{base_url}/query",
        json={"query": "Python", "k": 2},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r_query.status_code == 200
    json_q = r_query.json()
    assert json_q["query"] == "Python"
    assert isinstance(json_q["top_docs"], list)


def test_integration_echo_default(base_url: str):
    r = requests.get(f"{base_url}/echo", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    assert r.json() == {"message": "hello"}


def test_integration_echo_custom(base_url: str):
    r = requests.get(
        f"{base_url}/echo",
        params={"message": "ping"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r.status_code == 200
    assert r.json() == {"message": "ping"}


def test_integration_docs_available(base_url: str):
    r = requests.get(f"{base_url}/docs", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_integration_status_before_ingest(base_url: str):

    r_status = requests.get(f"{base_url}/status", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r_status.status_code == 200
    status_obj = r_status.json()
    assert isinstance(status_obj["loaded"], bool)
    assert status_obj["documents"] >= 0


def test_integration_query_with_large_k(base_url: str):

    # Ingest
    requests.post(f"{base_url}/ingest?max_docs=2&reingest=true", timeout=REQUEST_TIMEOUT_SECONDS)

    # Query with k=5
    r_query = requests.post(
        f"{base_url}/query",
        json={"query": "Large query", "k": 5},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r_query.status_code == 200
    json_q = r_query.json()
    assert json_q["k"] == 5
    assert "top_docs" in json_q


def test_integration_query_empty_string(base_url: str):

    requests.post(f"{base_url}/ingest?max_docs=2&reingest=true", timeout=REQUEST_TIMEOUT_SECONDS)

    r_query = requests.post(
        f"{base_url}/query",
        json={"query": "", "k": 1},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r_query.status_code == 200
    json_q = r_query.json()
    assert json_q["query"] == ""
    assert "top_docs" in json_q


def test_integration_health_post(base_url: str):

    r = requests.post(f"{base_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 405  # method not allowed


def test_integration_query_negative_k(base_url: str):

    requests.post(f"{base_url}/ingest?max_docs=2&reingest=true", timeout=REQUEST_TIMEOUT_SECONDS)

    r_query = requests.post(
        f"{base_url}/query",
        json={"query": "Test", "k": -1},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r_query.status_code == 200
    json_q = r_query.json()
    assert json_q["k"] == -1


def test_integration_health_with_params(base_url: str):

    r = requests.get(f"{base_url}/health?test=123", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
