"""Tests using HuggingFace test dataset Q&A pairs."""
import pytest
import sys
sys.path.insert(0, 'src')

from test_data import get_sample_test_data, load_all_cached_test_data


def test_load_sample_test_data():
    """Test that we can load sample test data."""
    examples = get_sample_test_data(3)
    assert len(examples) > 0
    assert all("question" in ex for ex in examples)
    assert all("answer" in ex for ex in examples)


def test_sample_test_data_has_questions_answers():
    """Test that sample data has valid Q&A pairs."""
    examples = get_sample_test_data(5)
    
    for ex in examples:
        assert len(ex.get("question", "")) > 0, "Question should not be empty"
        assert len(ex.get("answer", "")) > 0, "Answer should not be empty"
        assert ex.get("id"), "Example should have an ID"


def test_load_all_cached_test_data():
    """Test that all cached test data can be loaded."""
    all_data = load_all_cached_test_data()
    assert len(all_data) >= 5, "Should have at least 5 cached examples"


def test_specific_qa_pairs():
    """Test specific known Q&A pairs from cache."""
    examples = load_all_cached_test_data()
    
    # Verify we have at least some data
    assert len(examples) > 0
    
    # Check first example structure
    first = examples[0]
    assert "question" in first
    assert "answer" in first
    assert "id" in first
    
    # Verify data is strings
    assert isinstance(first["question"], str)
    assert isinstance(first["answer"], str)
    assert isinstance(first["id"], str)
