"""Utility to load and manage test data from HuggingFace."""
import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset


TEST_DATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "test_data.json"


def load_test_dataset() -> List[Dict]:
    """Load test questions and answers from HuggingFace.
    
    Returns:
        List of test examples with 'question' and 'answer' fields
    """
    try:
        dataset = load_dataset(
            "rag-datasets/rag-mini-wikipedia",
            "question-answer",
            split="test"
        )
        
        test_examples = []
        for item in dataset:
            test_examples.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "id": item.get("id", ""),
            })
        
        return test_examples
    except Exception as e:
        raise Exception(f"Failed to load test dataset: {str(e)}")


def get_sample_test_data(sample_size: int = 5) -> List[Dict]:
    """Get sample test Q&A pairs, with caching.
    
    Args:
        sample_size: Number of sample Q&A pairs to return
    
    Returns:
        List of sample test examples
    """
    # Try to load from cache first
    if TEST_DATA_CACHE.exists():
        with open(TEST_DATA_CACHE, 'r') as f:
            cached_data = json.load(f)
            return cached_data[:sample_size]
    
    # Load from HuggingFace and cache
    try:
        test_data = load_test_dataset()
        
        # Cache the data
        TEST_DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_DATA_CACHE, 'w') as f:
            json.dump(test_data[:10], f, indent=2)  # Cache first 10 for speed
        
        return test_data[:sample_size]
    except Exception as e:
        # Fallback to hardcoded examples if HF load fails
        return get_hardcoded_test_data()[:sample_size]


def get_hardcoded_test_data() -> List[Dict]:
    """Fallback hardcoded test data for when HuggingFace is unavailable."""
    return [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "id": "sample-qa-1",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "id": "sample-qa-2",
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter",
            "id": "sample-qa-3",
        },
        {
            "question": "In what year did World War II end?",
            "answer": "1945",
            "id": "sample-qa-4",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "answer": "Au",
            "id": "sample-qa-5",
        },
    ]


def load_all_cached_test_data() -> List[Dict]:
    """Load all cached test data if available."""
    if TEST_DATA_CACHE.exists():
        with open(TEST_DATA_CACHE, 'r') as f:
            return json.load(f)
    return []
