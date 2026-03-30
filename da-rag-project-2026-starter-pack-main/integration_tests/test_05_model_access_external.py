import threading
import time
from collections.abc import Generator
import requests
import uvicorn
import pytest

from app import app


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8002
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
REQUEST_TIMEOUT_SECONDS = 10


def start_server() -> tuple[uvicorn.Server, threading.Thread]:
    if hasattr(app.state, "ingest_pipeline"):
        delattr(app.state, "ingest_pipeline")

    config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    for _ in range(20):
        try:
            requests.get(f"{BASE_URL}/health", timeout=0.5)
            break
        except requests.RequestException:
            time.sleep(0.1)

    return server, server_thread


@pytest.fixture()
def base_url() -> Generator[str, None, None]:
    server, server_thread = start_server()
    try:
        yield BASE_URL
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def test_integration_model_registry(base_url: str):

    r = requests.get(f"{base_url}/models", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    json_body = r.json()
    assert "chat" in json_body
    assert "embeddings" in json_body
    assert "gpt-4o" in json_body["chat"]
    assert "text-embedding-3-large" in json_body["embeddings"]


def test_integration_check_gpt4o_model(base_url: str):

    r = requests.get(f"{base_url}/models/chat/gpt-4o", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    json_body = r.json()
    assert json_body["model_type"] == "chat"
    assert json_body["model_name"] == "gpt-4o"
    assert json_body["available"] is True
    assert json_body["api_version"] == "2024-10-01-preview"


def test_integration_check_embedding_model(base_url: str):

    r = requests.get(
        f"{base_url}/models/embeddings/text-embedding-3-large",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r.status_code == 200
    json_body = r.json()
    assert json_body["model_type"] == "embeddings"
    assert json_body["model_name"] == "text-embedding-3-large"
    assert json_body["available"] is True


def test_integration_check_nonexistent_model(base_url: str):

    r = requests.get(f"{base_url}/models/chat/fake-model", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 404


def test_integration_check_invalid_model_type(base_url: str):

    r = requests.get(
        f"{base_url}/models/invalid_type/some-model",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r.status_code == 404
