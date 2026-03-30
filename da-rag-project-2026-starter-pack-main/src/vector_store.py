"""Simple local vector store for document embeddings."""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(self, store_path: str = "./data/vector_store.json", dimension: int = 3072):
        self.store_path = Path(store_path)
        self.dimension = dimension
        self.documents: List[Dict] = []
        self.loaded: bool = False

        self._ensure_store_dir()
        if self.store_path.exists():
            self.load()

    def _ensure_store_dir(self):
        if not self.store_path.parent.exists():
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: List[Dict[str, str]], embeddings: List[List[float]]):
        if len(documents) != len(embeddings):
            raise VectorStoreError("Document count and embedding count must match")

        for doc, emb in zip(documents, embeddings):
            if not isinstance(emb, list) or len(emb) != self.dimension:
                raise VectorStoreError(
                    f"Embedding for doc {doc.get('id')} has invalid dimension {len(emb) if isinstance(emb, list) else 'n/a'}"
                )

            entry = {
                "id": doc.get("id"),
                "text": doc.get("text"),
                "meta": doc.get("meta", {}),
                "embedding": emb,
            }
            self.documents.append(entry)

        self.loaded = True
        self.save()

    def save(self):
        self._ensure_store_dir()
        payload = {
            "dimension": self.dimension,
            "document_count": len(self.documents),
            "documents": self.documents,
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self):
        if not self.store_path.exists():
            self.loaded = False
            self.documents = []
            return

        with open(self.store_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.dimension = payload.get("dimension", self.dimension)
        self.documents = payload.get("documents", [])
        self.loaded = True

    def clear(self):
        self.documents = []
        self.loaded = False
        if self.store_path.exists():
            self.store_path.unlink()

    def get_stats(self) -> Dict[str, object]:
        return {
            "loaded": self.loaded,
            "document_count": len(self.documents),
            "dimension": self.dimension,
            "store_path": str(self.store_path),
        }

    def get_document(self, doc_id: str) -> Optional[Dict]:
        for doc in self.documents:
            if doc.get("id") == doc_id:
                return doc
        return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if top_k <= 0:
            return []

        if not isinstance(query_embedding, list) or len(query_embedding) != self.dimension:
            raise VectorStoreError(
                f"Query embedding must be a list with dimension {self.dimension}"
            )

        scored: List[Dict] = []
        for doc in self.documents:
            embedding = doc.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != self.dimension:
                continue

            score = self._cosine_similarity(query_embedding, embedding)
            scored.append(
                {
                    "id": doc.get("id"),
                    "text": doc.get("text"),
                    "meta": doc.get("meta", {}),
                    "score": score,
                }
            )

        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored[:top_k]

    def search(self, top_k: int = 5):
        # Legacy helper retained for compatibility with existing tests/callers.
        return self.documents[:top_k]
