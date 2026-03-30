from typing import Dict, List
from datasets import load_dataset
from embeddings import create_embeddings_batch, EmbeddingError
from vector_store import VectorStore, VectorStoreError


class IngestionError(Exception):
    pass


def load_huggingface_dataset(
    dataset_name: str = "rag-datasets/rag-mini-wikipedia",
    config: str = "text-corpus",
    split: str = "passages",
    max_docs: int | None = None,
) -> List[Dict[str, str]]:
    """Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        config: Dataset configuration (e.g., 'text-corpus', 'question-answer')
        split: Dataset split to load
        max_docs: Optional maximum number of passages to load
    
    Returns:
        List of document dictionaries with 'text' and 'meta' fields
        
    Raises:
        IngestionError: If dataset loading fails
    """
    try:
        dataset = load_dataset(dataset_name, config, split=split)
        
        # Convert Arrow Table to list of dicts
        documents = []
        for item in dataset:
            # Handle different field names in the dataset
            text = item.get("passage", item.get("text", ""))
            doc_id = item.get("id", f"doc-{len(documents)}")
            
            documents.append({
                "text": text,
                "id": doc_id,
                "meta": {"source": "huggingface", "dataset": dataset_name},
            })

            if max_docs is not None and len(documents) >= max_docs:
                break
        
        return documents
    except Exception as e:
        raise IngestionError(f"Failed to load dataset: {str(e)}")


def get_sample_data() -> List[Dict[str, str]]:
    """Return sample passage data as fallback."""
    return [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "id": "sample-1",
            "meta": {"source": "local-sample"},
        },
        {
            "text": "Python is a programming language that emphasizes readability.",
            "id": "sample-2",
            "meta": {"source": "local-sample"},
        },
    ]


class IngestPipeline:
    def __init__(self):
        self._loaded = False
        self._count = 0
        self._index_path = "./data/index"
        self._documents = []
        self._embeddings_created = False
        self._embeddings = []
        self._embedding_progress = 0
        self._embedding_total = 0
        self._embedding_status = "idle"
        self._vector_store = VectorStore()
        self._hydrate_from_vector_store()

    def _source_counts_from_docs(self, docs: List[Dict[str, object]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for doc in docs:
            meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
            source = "unknown"
            if isinstance(meta, dict):
                source = str(meta.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return counts

    def _hydrate_from_vector_store(self) -> None:
        stats = self._vector_store.get_stats() if self._vector_store else {}
        doc_count = int(stats.get("document_count", 0) or 0)
        loaded = bool(stats.get("loaded", False)) and doc_count > 0

        if not loaded:
            self._loaded = False
            self._count = 0
            self._documents = []
            self._embeddings_created = False
            self._embeddings = []
            self._embedding_progress = 0
            self._embedding_total = 0
            self._embedding_status = "idle"
            return

        self._loaded = True
        self._count = doc_count
        self._documents = [
            {
                "id": doc.get("id"),
                "text": doc.get("text"),
                "meta": doc.get("meta", {}),
            }
            for doc in self._vector_store.documents
        ]
        self._embeddings_created = True
        self._embeddings = []
        self._embedding_progress = doc_count
        self._embedding_total = doc_count
        self._embedding_status = "completed"

    def clear(self) -> None:
        self._vector_store.clear()
        self._hydrate_from_vector_store()

    @property
    def status(self) -> Dict[str, object]:
        vector_stats = self._vector_store.get_stats() if self._vector_store else {}
        source_counts = self._source_counts_from_docs(self._vector_store.documents)
        return {
            "loaded": self._loaded,
            "documents": self._count,
            "embeddings_created": self._embeddings_created,
            "embeddings_count": len(self._embeddings) if self._embeddings else 0,
            "embedding_progress": self._embedding_progress,
            "embedding_total": self._embedding_total,
            "embedding_status": self._embedding_status,
            "index_path": self._index_path,
            "vector_store_loaded": vector_stats.get("loaded", False),
            "vector_store_count": vector_stats.get("document_count", 0),
            "vector_store_path": vector_stats.get("store_path", ""),
            "vector_store_dimension": vector_stats.get("dimension", 0),
            "source_counts": source_counts,
            "has_sample_data": source_counts.get("local-sample", 0) > 0,
        }

    def ingest(
        self,
        data: List[Dict[str, str]] = None,
        create_embeddings: bool = True,
        max_docs: int | None = None,
        reingest: bool = False,
        checkpoint_size: int = 50,
    ) -> Dict[str, object]:
        """Ingest documents into the pipeline.
        
        Args:
            data: Optional pre-loaded data. If None, load from HuggingFace.
            create_embeddings: If True, create embeddings for all documents after loading.
            max_docs: Optional cap on number of documents ingested.
            reingest: If True, replace existing vector store data with a fresh ingestion.
            checkpoint_size: Number of docs to persist per checkpoint while embedding.
        
        Returns:
            Ingestion results with count and status
            
        Raises:
            IngestionError: If no data available
        """
        if max_docs is not None and max_docs <= 0:
            raise IngestionError("max_docs must be greater than 0")
        if checkpoint_size <= 0:
            raise IngestionError("checkpoint_size must be greater than 0")

        source_counts = self._source_counts_from_docs(self._vector_store.documents)
        has_existing = self._vector_store.get_stats().get("document_count", 0) > 0
        has_sample_data = source_counts.get("local-sample", 0) > 0

        if has_existing and not reingest and not has_sample_data:
            self._hydrate_from_vector_store()
            return {
                "ingested": self._count,
                "embeddings_created": True,
                "status": "already_loaded",
                "index_path": self._index_path,
                "source_counts": source_counts,
            }

        if data is None:
            try:
                data = load_huggingface_dataset(max_docs=max_docs)
            except IngestionError:
                raise

        if max_docs is not None and data is not None:
            data = data[:max_docs]

        if not isinstance(data, list) or not data:
            raise IngestionError("No data to ingest")

        self._documents = data
        self._count = len(data)
        self._loaded = True

        # Create embeddings if requested
        if create_embeddings:
            try:
                print(f"Creating embeddings for {self._count} documents...")
                texts = [doc["text"] for doc in self._documents]
                self._embeddings = []
                self._embeddings_created = False
                self._embedding_status = "running"
                self._embedding_progress = 0
                self._embedding_total = len(texts)

                # Reset store at the start of a re-ingest and persist incrementally.
                self._vector_store.clear()

                for start in range(0, len(texts), checkpoint_size):
                    end = min(start + checkpoint_size, len(texts))
                    chunk_texts = texts[start:end]
                    chunk_docs = self._documents[start:end]

                    def _update_progress(done: int, total: int) -> None:
                        self._embedding_progress = start + done
                        self._embedding_total = len(texts)

                    chunk_embeddings = create_embeddings_batch(
                        chunk_texts,
                        batch_size=10,
                        max_retries=3,
                        initial_backoff_seconds=0.5,
                        progress_callback=_update_progress,
                    )

                    self._embeddings.extend(chunk_embeddings)
                    self._vector_store.add_documents(chunk_docs, chunk_embeddings)
                    self._embedding_progress = end
                    print(
                        f"Checkpoint persisted: {end}/{len(texts)} documents"
                    )

                self._embeddings_created = True
                self._embedding_status = "completed"
                print(f"Successfully created {len(self._embeddings)} embeddings")

            except EmbeddingError as e:
                print(f"Warning: Failed to create embeddings: {str(e)}")
                self._embeddings_created = False
                self._embeddings = []
                self._embedding_status = "failed"
            except Exception as e:
                print(f"Warning: Unexpected error during embedding creation: {str(e)}")
                self._embeddings_created = False
                self._embeddings = []
                self._embedding_status = "failed"
        else:
            # preserve existing vector store state when embeddings are not created
            self._vector_store.load()
            self._embedding_status = "skipped"

        source_counts = self._source_counts_from_docs(self._vector_store.documents)

        return {
            "ingested": self._count,
            "embeddings_created": self._embeddings_created,
            "status": "loaded" + (" with embeddings" if self._embeddings_created else ""),
            "index_path": self._index_path,
            "source_counts": source_counts,
        }
