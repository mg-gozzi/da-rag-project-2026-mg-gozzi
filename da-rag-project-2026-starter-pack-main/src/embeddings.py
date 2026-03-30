"""Embedding pipeline for RAG system.

This module provides functions to create embeddings for text documents and queries
using Azure OpenAI's text-embedding-3-large model.
"""
import re
import time
from typing import List, Dict, Any, Callable
from llamaindex_models import get_embedding_model, ModelAccessError


class EmbeddingError(Exception):
    """Raised when embedding creation fails."""
    pass


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def _extract_retry_after_seconds(error: Exception) -> float | None:
    message = str(error)
    match = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s+seconds", message, flags=re.IGNORECASE)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _embed_with_retry(
    embedding_model,
    text: str,
    max_retries: int,
    initial_backoff_seconds: float,
) -> List[float]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            embedding = embedding_model.get_text_embedding(text)
            if not embedding or len(embedding) != 3072:
                raise EmbeddingError("Embedding model returned invalid dimensions")
            return embedding
        except Exception as err:
            last_error = err
            is_rate_limited = _is_rate_limit_error(err)

            if attempt >= max_retries:
                if is_rate_limited:
                    raise EmbeddingError(
                        f"Rate limited by embedding API after {max_retries + 1} attempts"
                    ) from err
                raise EmbeddingError(
                    f"Failed to create embedding after {max_retries + 1} attempts: {str(err)}"
                ) from err

            if is_rate_limited:
                sleep_seconds = initial_backoff_seconds * (2 ** attempt)
                retry_after_seconds = _extract_retry_after_seconds(err)
                if retry_after_seconds is not None:
                    sleep_seconds = max(sleep_seconds, retry_after_seconds)
            else:
                sleep_seconds = min(1.0, initial_backoff_seconds * (attempt + 1))
            time.sleep(sleep_seconds)

    raise EmbeddingError(f"Failed to create embedding: {str(last_error)}")


def create_embedding(
    text: str,
    model_name: str = "text-embedding-3-large",
    max_retries: int = 3,
    initial_backoff_seconds: float = 0.5,
) -> List[float]:
    """Create an embedding vector for the given text.

    Args:
        text: The text to embed.
        model_name: Name of the embedding model to use.

    Returns:
        List of float values representing the embedding vector.

    Raises:
        EmbeddingError: If embedding creation fails.
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot create embedding for empty text")

    try:
        # Get the configured embedding model
        embedding_model = get_embedding_model(model_name)

        # Create the embedding with retry for transient/rate-limit failures.
        embedding_result = _embed_with_retry(
            embedding_model,
            text,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
        )

        # Validate the embedding
        if not embedding_result:
            raise EmbeddingError("Embedding model returned empty result")

        if not isinstance(embedding_result, list):
            raise EmbeddingError(f"Expected list, got {type(embedding_result)}")

        # text-embedding-3-large should return 3072-dimensional vectors
        expected_dim = 3072
        if len(embedding_result) != expected_dim:
            raise EmbeddingError(
                f"Expected embedding dimension {expected_dim}, got {len(embedding_result)}"
            )

        return embedding_result

    except ModelAccessError as e:
        raise EmbeddingError(f"Model access error: {str(e)}")
    except Exception as e:
        raise EmbeddingError(f"Failed to create embedding: {str(e)}")


def create_embeddings_batch(
    texts: List[str],
    model_name: str = "text-embedding-3-large",
    batch_size: int = 10,
    delay_between_batches: float = 0.1,
    max_retries: int = 3,
    initial_backoff_seconds: float = 0.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> List[List[float]]:
    """Create embeddings for multiple texts in batches.

    Args:
        texts: List of texts to embed.
        model_name: Name of the embedding model to use.
        batch_size: Number of texts to process in each batch.
        delay_between_batches: Delay in seconds between batches to avoid rate limits.

    Returns:
        List of embedding vectors, one for each input text.

    Raises:
        EmbeddingError: If any embedding creation fails.
    """
    if not texts:
        return []

    embeddings = []
    total_texts = len(texts)

    try:
        # Get the configured embedding model once
        embedding_model = get_embedding_model(model_name)

        # Process in batches
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]

            # Create embeddings for this batch
            for batch_offset, text in enumerate(batch_texts):
                if not text or not text.strip():
                    # For empty texts, create zero vector of correct dimension
                    embeddings.append([0.0] * 3072)
                    if progress_callback:
                        progress_callback(i + batch_offset + 1, total_texts)
                    continue

                try:
                    embedding = _embed_with_retry(
                        embedding_model,
                        text,
                        max_retries=max_retries,
                        initial_backoff_seconds=initial_backoff_seconds,
                    )

                    embeddings.append(embedding)
                    if progress_callback:
                        progress_callback(i + batch_offset + 1, total_texts)

                except Exception as e:
                    raise EmbeddingError(f"Failed to embed text '{text[:50]}...': {str(e)}")

            # Small delay between batches to be respectful to the API
            if i + batch_size < total_texts:
                time.sleep(delay_between_batches)

    except ModelAccessError as e:
        raise EmbeddingError(f"Model access error: {str(e)}")
    except Exception as e:
        raise EmbeddingError(f"Batch embedding failed: {str(e)}")

    if len(embeddings) != total_texts:
        raise EmbeddingError(f"Expected {total_texts} embeddings, got {len(embeddings)}")

    return embeddings


def validate_embedding_dimensions(embedding: List[float], expected_dim: int = 3072) -> bool:
    """Validate that an embedding has the expected dimensions.

    Args:
        embedding: The embedding vector to validate.
        expected_dim: Expected dimension (default 3072 for text-embedding-3-large).

    Returns:
        True if valid, False otherwise.
    """
    return (
        isinstance(embedding, list) and
        len(embedding) == expected_dim and
        all(isinstance(x, (int, float)) for x in embedding)
    )


def get_embedding_info() -> Dict[str, Any]:
    """Get information about the current embedding configuration.

    Returns:
        Dictionary with embedding model information.
    """
    return {
        "model_name": "text-embedding-3-large",
        "dimension": 3072,
        "provider": "Azure OpenAI",
        "description": "Large text embedding model for semantic search and RAG"
    }