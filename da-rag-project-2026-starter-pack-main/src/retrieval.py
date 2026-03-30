"""Query embedding and vector similarity retrieval functions."""

from typing import Dict, List

from embeddings import create_embedding, EmbeddingError
from vector_store import VectorStore, VectorStoreError


class RetrievalError(Exception):
    """Raised when query retrieval fails."""


def query_to_embedding(query: str) -> List[float]:
    if not query or not query.strip():
        raise RetrievalError("Query text cannot be empty")

    try:
        return create_embedding(query)
    except EmbeddingError as err:
        raise RetrievalError(f"Failed to embed query: {str(err)}") from err


def retrieve_top_k(query_embedding: List[float], vector_store: VectorStore, k: int = 5) -> List[Dict]:
    if k <= 0:
        return []

    try:
        return vector_store.search_by_embedding(query_embedding, top_k=k)
    except VectorStoreError as err:
        raise RetrievalError(f"Vector search failed: {str(err)}") from err
