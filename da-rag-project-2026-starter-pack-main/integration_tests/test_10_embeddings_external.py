import socket
import threading
import time
from collections.abc import Generator

import pytest
import requests
import uvicorn

import app as app_module
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
    def fake_create_embedding(_text: str):
        return [0.2] * 3072

    monkeypatch.setattr(app_module, "create_embedding", fake_create_embedding)

    port = _get_free_port()
    server, server_thread, base_url_value = start_server(port)
    try:
        yield base_url_value
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def test_embed_external_success(base_url: str):
    r = requests.post(
        f"{base_url}/embed",
        json={"text": "External test"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "External test"
    assert body["dimension"] == 3072
    assert len(body["embedding"]) == 3072


def test_embed_external_validation_failure(base_url: str, monkeypatch):
    def fake_create_embedding(_text: str):
        raise app_module.EmbeddingError("Cannot create embedding for empty text")

    monkeypatch.setattr(app_module, "create_embedding", fake_create_embedding)

    r = requests.post(
        f"{base_url}/embed",
        json={"text": ""},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r.status_code == 400
    assert "Cannot create embedding" in r.json()["detail"]
