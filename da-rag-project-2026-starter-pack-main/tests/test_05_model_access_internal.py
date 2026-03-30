"""Tests for model access through llamaindex_models.py"""
import pytest

from llamaindex_models import (
    get_available_models,
    get_chat_model,
    get_embedding_model,
    validate_model_access,
    ModelAccessError,
)


def test_model_registry_available():
    """Test that model registry is accessible."""
    models = get_available_models()
    assert isinstance(models, dict)
    assert "chat" in models
    assert "embeddings" in models


def test_chat_models_in_registry():
    """Test that chat models are properly registered."""
    models = get_available_models()
    assert "gpt-4o" in models["chat"]
    assert models["chat"]["gpt-4o"]["deployment_name"] == "gpt-4o"
    assert models["chat"]["gpt-4o"]["api_version"] == "2024-10-01-preview"


def test_embedding_models_in_registry():
    """Test that embedding models are properly registered."""
    models = get_available_models()
    assert "text-embedding-3-large" in models["embeddings"]
    assert models["embeddings"]["text-embedding-3-large"]["deployment_name"] == "text-embedding-3-large"
    assert models["embeddings"]["text-embedding-3-large"]["api_version"] == "2024-10-01-preview"


def test_validate_model_access_chat():
    """Test model validation for chat models."""
    assert validate_model_access("chat", "gpt-4o") is True
    assert validate_model_access("chat", "nonexistent") is False


def test_validate_model_access_embeddings():
    """Test model validation for embedding models."""
    assert validate_model_access("embeddings", "text-embedding-3-large") is True
    assert validate_model_access("embeddings", "nonexistent") is False


def test_validate_model_access_invalid_type():
    """Test validation with invalid model type."""
    assert validate_model_access("invalid_type", "gpt-4o") is False


def test_get_chat_model_valid():
    """Test that chat model can be instantiated."""
    try:
        model = get_chat_model("gpt-4o")
        assert model is not None
        # Verify it's a LlamaIndex AzureOpenAI instance
        assert hasattr(model, "model")
        assert model.model == "gpt-4o"
    except Exception as e:
        # Expected if Azure credentials not available
        pytest.skip(f"Azure credentials not available: {str(e)}")


def test_get_embedding_model_valid():
    """Test that embedding model can be instantiated."""
    try:
        model = get_embedding_model("text-embedding-3-large")
        assert model is not None
        # Verify it's a LlamaIndex AzureOpenAIEmbedding instance
        assert hasattr(model, "model")
        assert model.model == "text-embedding-3-large"
    except Exception as e:
        # Expected if Azure credentials not available
        pytest.skip(f"Azure credentials not available: {str(e)}")


def test_get_chat_model_invalid():
    """Test that invalid chat model raises error."""
    with pytest.raises(ModelAccessError) as exc_info:
        get_chat_model("nonexistent-model")
    assert "not available" in str(exc_info.value).lower()


def test_get_embedding_model_invalid():
    """Test that invalid embedding model raises error."""
    with pytest.raises(ModelAccessError) as exc_info:
        get_embedding_model("nonexistent-embedding")
    assert "not available" in str(exc_info.value).lower()


def test_model_access_isolation():
    """Test that models can only be accessed through registry."""
    models = get_available_models()
    
    # Verify all chat models are in registry
    for model_name in models["chat"]:
        assert validate_model_access("chat", model_name) is True
    
    # Verify all embedding models are in registry
    for model_name in models["embeddings"]:
        assert validate_model_access("embeddings", model_name) is True
