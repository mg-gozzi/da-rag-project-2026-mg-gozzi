import threading
import time
from collections.abc import Generator
import requests
import uvicorn
import pytest

from app import app


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8003
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


def test_integration_get_test_data(base_url: str):

    r = requests.get(f"{base_url}/test-data?limit=3", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    json_body = r.json()
    assert "count" in json_body
    assert "examples" in json_body
    assert json_body["count"] == 3
    assert len(json_body["examples"]) == 3


def test_integration_test_data_has_qa(base_url: str):

    r = requests.get(f"{base_url}/test-data?limit=1", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    json_body = r.json()
    
    examples = json_body["examples"]
    assert len(examples) > 0
    
    example = examples[0]
    assert "question" in example
    assert "answer" in example
    assert "id" in example
    assert len(example["question"]) > 0
    assert len(example["answer"]) > 0


def test_integration_get_specific_test_qa(base_url: str):

    # First get all data to find an ID
    r = requests.get(f"{base_url}/test-data?limit=10", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    examples = r.json()["examples"]
    assert len(examples) > 0
    
    test_id = examples[0]["id"]
    
    # Now get that specific Q&A
    r_specific = requests.get(f"{base_url}/test-data/{test_id}", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r_specific.status_code == 200
    json_body = r_specific.json()
    assert json_body["id"] == test_id
    assert "question" in json_body
    assert "answer" in json_body


def test_integration_test_qa_not_found(base_url: str):

    r = requests.get(
        f"{base_url}/test-data/nonexistent-qa-id",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert r.status_code == 404


def test_integration_test_data_default_limit(base_url: str):

    r = requests.get(f"{base_url}/test-data", timeout=REQUEST_TIMEOUT_SECONDS)
    assert r.status_code == 200
    json_body = r.json()
    # Default limit is 5
    assert json_body["count"] <= 5
