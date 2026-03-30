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
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}},
        ]

    def fake_create_embeddings_batch(texts, *args, **kwargs):
        return [[0.1] * 3072 for _ in texts]

    def fake_query_to_embedding(_query: str):
        return [0.2] * 3072

    def fake_retrieve_top_k(_query_embedding, _store, k: int):
        items = [
            {"id": "hf-1", "text": "Paris is the capital of France.", "meta": {"source": "huggingface"}, "score": 0.97},
            {"id": "hf-2", "text": "Python is a programming language.", "meta": {"source": "huggingface"}, "score": 0.52},
        ]
        return items[:k]

    def fake_generate_answer(question: str, _docs):
        if "capital of France" in question:
            return "Paris"
        return "I do not have enough information from the provided passages."

    def fake_test_data(limit: int):
        return [
            {"id": "qa-1", "question": "What is the capital of France?", "answer": "Paris"},
            {"id": "qa-2", "question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
        ][:limit]

    monkeypatch.setattr(ingest_module, "load_huggingface_dataset", fake_load_hf_dataset)
    monkeypatch.setattr(ingest_module, "create_embeddings_batch", fake_create_embeddings_batch)
    monkeypatch.setattr(app_module, "query_to_embedding", fake_query_to_embedding)
    monkeypatch.setattr(app_module, "retrieve_top_k", fake_retrieve_top_k)
    monkeypatch.setattr(app_module, "generate_rag_answer", fake_generate_answer)
    monkeypatch.setattr(app_module, "get_sample_test_data", fake_test_data)

    port = _get_free_port()
    server, server_thread, base_url_value = start_server(port)
    try:
        yield base_url_value
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


def test_rag_answer_end_to_end(base_url: str):
    ingest_resp = requests.post(
        f"{base_url}/ingest?max_docs=15&reingest=true",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert ingest_resp.status_code == 200

    answer_resp = requests.post(
        f"{base_url}/answer",
        json={"question": "What is the capital of France?", "k": 2},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert answer_resp.status_code == 200
    body = answer_resp.json()
    assert body["answer"] == "Paris"
    assert len(body["top_docs"]) == 2


def test_rag_evaluate_end_to_end(base_url: str):
    ingest_resp = requests.post(
        f"{base_url}/ingest?max_docs=15&reingest=true",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert ingest_resp.status_code == 200

    eval_resp = requests.post(
        f"{base_url}/evaluate",
        json={"limit": 2, "k": 2},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    assert eval_resp.status_code == 200
    body = eval_resp.json()
    assert body["evaluated"] == 2
    assert "average_f1" in body
    assert "exact_match_rate" in body
