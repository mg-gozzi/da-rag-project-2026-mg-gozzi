import tempfile
import os
from vector_store import VectorStore, VectorStoreError


def test_vector_store_add_and_stats():
    temp_path = tempfile.mktemp(suffix=".json")
    store = VectorStore(store_path=temp_path, dimension=4)

    docs = [
        {"id": "d1", "text": "First", "meta": {}},
        {"id": "d2", "text": "Second", "meta": {}},
    ]
    embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

    store.add_documents(docs, embeddings)

    stats = store.get_stats()
    assert stats["loaded"] is True
    assert stats["document_count"] == 2
    assert stats["dimension"] == 4
    assert os.path.exists(temp_path)

    # Validate persistence
    reloaded = VectorStore(store_path=temp_path, dimension=4)
    stats_reloaded = reloaded.get_stats()
    assert stats_reloaded["loaded"] is True
    assert stats_reloaded["document_count"] == 2


def test_vector_store_rejects_bad_embedding():
    temp_path = tempfile.mktemp(suffix=".json")
    store = VectorStore(store_path=temp_path, dimension=3)
    docs = [{"id": "x", "text": "x", "meta": {}}]
    embeddings = [[0.1, 0.2]]

    try:
        store.add_documents(docs, embeddings)
        assert False, "Expected VectorStoreError for invalid embedding dimension"
    except VectorStoreError:
        assert True
