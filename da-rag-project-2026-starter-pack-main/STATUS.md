# Project Status & Implementation Plan

## Phase 1: Foundation & Testing (COMPLETED ✅)

### 1.1 Project Bootstrap
- [x] FastAPI app scaffold (`src/app.py`)
- [x] Core package structure (src/, tests/, integration_tests/)
- [x] Editable package installation (`pyproject.toml` with build-system config)
- [x] Pyrightconfig for IDE/Pylance support
- [x] Hot-reload server runner (`run_server.py`)

### 1.2 Testing Infrastructure (48 tests, all passing)
- [x] 24 internal tests (TestClient, `tests/` directory)
  - `test_00_internal_app.py` - Core API endpoint tests
  - `test_05_model_access_internal.py` - Model validation tests
  - `test_06_test_data_internal.py` - Test data loading tests
- [x] 24 external tests (requests/HTTP, `integration_tests/` directory)
  - `test_01_external_requests.py` - HTTP endpoint validation
  - `test_05_model_access_external.py` - External model access tests
  - `test_06_test_data_external.py` - External test data tests
- [x] Test coverage:
  - Health checks and method validation
  - Ingest/status flow with real HuggingFace data
  - Query with various parameters (k, empty string, negative k)
  - Model access validation (Azure OpenAI chat and embedding models)
  - Test data loading and caching
  - Before/after ingestion states
  - Query parameter edge cases

### 1.3 Data Integration
- [x] HuggingFace dataset loader (`load_huggingface_dataset()`)
- [x] Real dataset: `rag-datasets/rag-mini-wikipedia` (3,200 passages + 918 Q&A pairs)
- [x] Sample data fallback for testing
- [x] `IngestPipeline` class with status tracking
- [x] Dependencies added: `datasets>=2.14.0`, `pyarrow>=15.0.0`
- [x] `POST /ingest?use_sample=true` endpoint supports both sample and real data

### 1.4 Model Access Layer
- [x] `src/llamaindex_models.py` - Model registry and access functions
- [x] Azure OpenAI integration with `DefaultAzureCredential`
- [x] Support for GPT-4o chat model and text-embedding-3-large
- [x] Model validation functions (`validate_model_access()`)
- [x] Error handling for authentication and API issues

### 1.5 Test Data Preparation
- [x] `src/test_data.py` - Load and cache Q&A pairs from HuggingFace
- [x] `data/test_data.json` - Cached test data (10 Q&A pairs)
- [x] Real Q&A data from dataset (Abraham Lincoln questions, etc.)
- [x] Fallback to sample data when HuggingFace unavailable

### 1.6 API Endpoints (Phase 1)
- [x] `GET /health` - Health check
- [x] `GET /echo` - Baseline request/response smoke endpoint
- [x] `GET /status` - Pipeline status (loaded count, path)
- [x] `POST /ingest` - Load data from HuggingFace or samples
- [x] `POST /query` - Placeholder query endpoint
- [x] `GET /models` - Model access validation and available models
- [x] `GET /test-data` - Test Q&A data serving

---

## Phase 2: Embeddings & Vector Store (COMPLETED ✅)

### 2.1 Embedding Pipeline Implementation
- [x] Create `src/embeddings.py`:
  - [x] `create_embedding(text: str) -> List[float]` using the configured embedding model
  - [x] Batch embedding processing for multiple documents
  - [x] Explicit retry logic for API failures
  - [x] Embedding dimension validation (3072 for text-embedding-3-large)
- Partial: Update `src/ingest.py`:
  - [x] Integrate embedding creation into `IngestPipeline.ingest()`
  - [x] Add progress tracking for embedding generation
  - [x] Update status to include embedding completion state
  - [x] `use_sample=true` now forces local sample ingestion before any HuggingFace call
- [x] Tests:
  - [x] `tests/test_10_embeddings_internal.py` - Unit tests for embedding functions
  - [x] `integration_tests/test_10_embeddings_external.py` - API endpoint tests
- [x] New API Endpoint:
  - [x] `POST /embed` - Create embedding for query text
    - Request: `{"text": "What is the capital of France?"}`
    - Response: `{"text": "...", "embedding": [0.1, 0.2, ...], "dimension": 3072}`

### 2.2 Vector Store Integration
- [x] Create `src/vector_store.py`:
  - [x] Initialize local JSON-backed vector store
  - [x] `add_documents(documents: List[Dict], embeddings: List[List[float]])`
  - [x] Persistence to disk (`./data/vector_store.json`)
  - [x] Load existing store on startup via `VectorStore.load()`
  - [x] `get_stats()` for metadata (doc count, dimension, persistence path)
- [x] Integrate with `IngestPipeline`:
  - [x] After creating embeddings, automatically store in vector store
  - [x] Update pipeline status to include vector store state
- [x] Tests:
  - [x] `tests/test_15_vector_store_internal.py` - Vector store operations
  - [x] `integration_tests/test_15_vector_store_external.py` - API tests
  - [x] Verify persistence/reload functionality
  - [x] Verify document count matches ingestion count
- [x] New API Endpoint:
  - [x] `GET /vector-store/status` - Return vector store metadata
    - Response: `{"document_count": 3200, "dimension": 3072, "persisted": true, "path": "./data/vector_store.json"}`

### 2.3 Observability Notebook
- [x] Create `notebooks/01_ingest_and_embeddings.ipynb`:
  - [x] Step 1: Call `/ingest` to load documents
  - [x] Step 2: Show document count and sample texts from the loaded data
  - [x] Step 3: Call `/vector-store/status` to show indexing results
  - [x] Step 4: Visualize embedding dimensions and sample embedding vectors
  - [x] Step 5: Show timing and performance metrics for ingestion + embedding
  - [x] Include markdown explanations of each step
  - [x] Run notebook end-to-end against live API with sample ingest

### 2.4 Phase 2 Notes
- [x] Integration tests were stabilized with managed server lifecycle, request timeouts, and unique test ports
- [x] Sample ingest path was fixed to avoid accidental HuggingFace loading when `use_sample=true`
- [x] Azure embedding rate limits now use explicit retry/backoff handling in embedding calls

---

## Phase 3: Query & Retrieval (NEXT)

### 3.1 Query Embedding & Similarity Search
- [x] Implement `src/retrieval.py`:
  - [x] `query_to_embedding(query: str) -> List[float]` using same embedding model
  - [x] `retrieve_top_k(query_embedding: List[float], k: int) -> List[Dict]`
    - Return: `[{"id": "...", "text": "...", "score": 0.95, ...}, ...]`
- [x] Tests:
  - [x] `tests/test_20_retrieval_internal.py`
  - [x] `integration_tests/test_20_retrieval_external.py`
  - [x] Test with sample queries from HuggingFace test set
  - [x] Verify top-k ordering by relevance score
  - [x] Verify non-empty results
- [x] Endpoint:
  - [x] `POST /query` - Updated to return real top-k results
    - Request: `{"query": "...", "k": 5}`
    - Response: `{"query": "...", "k": 5, "top_docs": [...], "scores": [0.95, 0.88, ...]}`

### 3.2 Observability Notebook
- [ ] Create `notebooks/02_query_and_retrieval.ipynb`:
  - [ ] Sample queries from test set
  - [ ] Show embedding similarity scores
  - [ ] Visualize top-k results
  - [ ] Show relevance ranking

---

## Phase 4: RAG Generation & Evaluation (PLANNED)

### 4.1 Prompt Augmentation & LLM Response
- [ ] Implement `src/rag.py`:
  - [ ] `build_rag_prompt(query: str, top_docs: List[Dict]) -> str`
    - Format: "Context: {docs}\nQuestion: {query}\nAnswer:"
  - [ ] `generate_answer(prompt: str) -> str` using `llamaindex_models.get_gpt4o()`
- [ ] Tests:
  - [ ] `tests/test_30_rag_internal.py`
  - [ ] `integration_tests/test_30_rag_external.py`
  - [ ] Test with HuggingFace test Q&A pairs
  - [ ] Verify answer contains expected keywords/entities
  - [ ] Measure answer quality (basic metrics)
- [ ] Endpoint:
  - [ ] `POST /answer` - Full RAG pipeline
    - Request: `{"query": "...", "k": 5}`
    - Response: `{"query": "...", "answer": "...", "top_docs": [...], "prompt": "...", "model": "gpt-4o"}`

### 4.2 Evaluation & Metrics
- [ ] Load test set from HuggingFace: `hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet`
- [ ] Implement `src/evaluation.py`:
  - [ ] Run full RAG pipeline on test questions
  - [ ] Compare answers with gold standard
  - [ ] Compute metrics: BLEU, ROUGE, semantic similarity, answer presence
- [ ] Tests:
  - [ ] `tests/test_40_evaluation_internal.py`
  - [ ] Sample 10 test cases and verify answer quality

### 4.3 Observability Notebook
- [ ] Create `notebooks/03_rag_generation.ipynb`:
  - [ ] Full query-to-answer pipeline trace
  - [ ] Show augmented prompt
  - [ ] Show model output step-by-step
  - [ ] Evaluation metrics on test set
  - [ ] Example Q&A pairs with explanations

---

## Current Status Summary

| Component | Status | Tests | Endpoints |
|-----------|--------|-------|-----------|
| **Bootstrap** | ✅ Done | 20 passing | 4 live |
| **Data Loading** | ✅ Done | 20 passing | 1 live |
| **Embeddings** | ✅ Done | Internal + external tests present | 1/1 live |
| **Vector Store** | ✅ Done | Internal + external tests present | 1/1 live |
| **Retrieval** | ✅ Core done | Internal + external tests present | 1/1 |
| **RAG Generation** | ⏳ Planned | 0/3 | 0/1 |
| **Evaluation** | ⏳ Planned | 0/2 | 0/0 |
| **Notebooks** | 🚧 Partial | 1/3 created and executed | - |

---

## Testing Strategy (All Phases)

### Prerequisites (always met)
- Create FastAPI endpoints for each feature
- Internal tests using TestClient (in `tests/`)
- External tests using requests library (in `integration_tests/`)
- Run tests before proceeding to next phase
- Observability notebooks for each major step

### Test Organization
- `test_00_*` - Smoke/bootstrap tests (20 tests passing)
- `test_10_*` - Embeddings tests
- `test_20_*` - Retrieval tests
- `test_30_*` - RAG generation tests
- `test_40_*` - Evaluation tests

---

## Key Files

- **App & API**: `src/app.py`
- **Data Loading**: `src/ingest.py` (HF + sample data)
- **Model Access**: `src/llamaindex_models.py` (all models via this layer)
- **Azure Auth**: `src/ailab/utils/azure.py`
- **Tests**: `tests/test_00_internal_app.py`, `integration_tests/test_01_external_requests.py`
- **Config**: `pyproject.toml`, `pyrightconfig.json`
- **Server**: `run_server.py`

---

## Prerequisites Summary

✅ **Done** - All foundational work complete
- Python 3.13, FastAPI, LlamaIndex ecosystem installed
- Azure authentication configured
- Package structure properly set up
- Test infrastructure in place (20 tests)
- Data loading pipeline ready
- Hot-reload development server ready

🔧 **Next** - Move into Phase 3: Retrieval
1. Create `notebooks/02_query_and_retrieval.ipynb`
2. Expand retrieval evaluation depth against more HuggingFace test examples
3. Move into Phase 4 answer generation (`/answer`) once retrieval notebook is in place

