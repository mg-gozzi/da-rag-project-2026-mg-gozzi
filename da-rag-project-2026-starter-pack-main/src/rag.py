from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from llamaindex_models import get_chat_model, ModelAccessError


class RagError(Exception):
    pass


def build_rag_prompt(question: str, retrieved_docs: List[Dict], max_chars_per_doc: int = 1200) -> str:
    if not question or not question.strip():
        raise RagError("Question cannot be empty")

    if not retrieved_docs:
        raise RagError("No retrieved documents available for RAG prompt")

    context_blocks: List[str] = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        text = str(doc.get("text", "")).strip()
        if not text:
            continue
        trimmed = text[:max_chars_per_doc]
        context_blocks.append(f"[{idx}] {trimmed}")

    if not context_blocks:
        raise RagError("Retrieved documents did not contain usable text")

    context = "\n\n".join(context_blocks)
    return (
        "You are a careful question-answering assistant.\n"
        "Use only the provided context passages to answer.\n"
        "If the answer is not in the context, say: I do not have enough information from the provided passages.\n\n"
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_rag_answer(question: str, retrieved_docs: List[Dict], model_name: str = "gpt-4o") -> str:
    prompt = build_rag_prompt(question, retrieved_docs)

    try:
        llm = get_chat_model(model_name=model_name, temperature=0.0, max_tokens=300)
        response = llm.complete(prompt)
    except ModelAccessError as ex:
        raise RagError(f"Model access error: {str(ex)}") from ex
    except Exception as ex:
        raise RagError(f"Failed to generate answer: {str(ex)}") from ex

    text = getattr(response, "text", "")
    answer = str(text).strip() if text is not None else ""
    if not answer:
        raise RagError("LLM returned an empty answer")
    return answer


def _normalize_text(value: str) -> str:
    lowered = value.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, reference: str) -> Dict[str, float | bool]:
    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)
    exact_match = ref_norm in pred_norm if ref_norm else False
    f1 = _token_f1(prediction, reference)
    return {
        "exact_match": exact_match,
        "f1": f1,
    }
