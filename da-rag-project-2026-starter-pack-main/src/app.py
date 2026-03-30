from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embeddings import create_embedding, EmbeddingError
from ingest import IngestPipeline, IngestionError
from llamaindex_models import get_available_models, validate_model_access
from rag import generate_rag_answer, score_answer, RagError
from retrieval import query_to_embedding, retrieve_top_k, RetrievalError
from test_data import get_sample_test_data, load_all_cached_test_data


class IngestResponse(BaseModel):
    ingested: int
    status: str
    index_path: str
    source_counts: dict[str, int] | None = None


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class AnswerRequest(BaseModel):
    question: str
    k: int = 5


class EvaluateRequest(BaseModel):
    limit: int = 5
    k: int = 5


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    text: str
    embedding: list[float]
    dimension: int


class ModelStatusResponse(BaseModel):
    model_type: str
    model_name: str
    available: bool
    deployment_name: str = None
    api_version: str = None


class TestQAExample(BaseModel):
    question: str
    answer: str
    id: str


app = FastAPI(title="Mini Wikipedia RAG API")

async def get_ingest_pipeline() -> IngestPipeline:
    if not hasattr(app.state, "ingest_pipeline"):
        app.state.ingest_pipeline = IngestPipeline()
    return app.state.ingest_pipeline


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/echo")
async def echo(message: str = "hello"):
    return {"message": message}


@app.get("/status")
async def status():
    pipeline = await get_ingest_pipeline()
    return pipeline.status


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(max_docs: int = 15, reingest: bool = False):
    pipeline = await get_ingest_pipeline()
    try:
        result = pipeline.ingest(max_docs=max_docs, reingest=reingest)
        return result
    except IngestionError as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@app.delete("/ingest")
async def clear_ingest_data():
    pipeline = await get_ingest_pipeline()
    pipeline.clear()
    return {"status": "cleared"}


@app.post("/query")
async def query_docs(request: QueryRequest):
    pipeline = await get_ingest_pipeline()
    if not pipeline.status["vector_store_loaded"]:
        raise HTTPException(status_code=400, detail="Ingest data first")

    if request.k <= 0:
        return {"query": request.query, "k": request.k, "top_docs": [], "scores": []}

    if not request.query or not request.query.strip():
        return {"query": request.query, "k": request.k, "top_docs": [], "scores": []}

    try:
        query_embedding = query_to_embedding(request.query)
        top_docs = retrieve_top_k(query_embedding, pipeline._vector_store, k=request.k)
    except RetrievalError as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    return {
        "query": request.query,
        "k": request.k,
        "top_docs": top_docs,
        "scores": [doc.get("score", 0.0) for doc in top_docs],
    }


@app.post("/answer")
async def answer_question(request: AnswerRequest):
    pipeline = await get_ingest_pipeline()
    if not pipeline.status["vector_store_loaded"]:
        raise HTTPException(status_code=400, detail="Ingest data first")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")

    try:
        query_embedding = query_to_embedding(request.question)
        top_docs = retrieve_top_k(query_embedding, pipeline._vector_store, k=request.k)
        answer = generate_rag_answer(request.question, top_docs)
    except (RetrievalError, RagError) as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    return {
        "question": request.question,
        "k": request.k,
        "answer": answer,
        "top_docs": top_docs,
        "scores": [doc.get("score", 0.0) for doc in top_docs],
    }


@app.post("/evaluate")
async def evaluate_rag(request: EvaluateRequest):
    pipeline = await get_ingest_pipeline()
    if not pipeline.status["vector_store_loaded"]:
        raise HTTPException(status_code=400, detail="Ingest data first")

    if request.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be greater than 0")
    if request.k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0")

    examples = get_sample_test_data(request.limit)
    if not examples:
        raise HTTPException(status_code=400, detail="No evaluation examples available")

    results = []
    exact_matches = 0
    f1_sum = 0.0

    for item in examples:
        question = item.get("question", "")
        reference_answer = item.get("answer", "")
        qa_id = item.get("id", "")

        try:
            query_embedding = query_to_embedding(question)
            top_docs = retrieve_top_k(query_embedding, pipeline._vector_store, k=request.k)
            prediction = generate_rag_answer(question, top_docs)
            score = score_answer(prediction, reference_answer)
        except (RetrievalError, RagError) as ex:
            prediction = ""
            top_docs = []
            score = {"exact_match": False, "f1": 0.0}
            results.append(
                {
                    "id": qa_id,
                    "question": question,
                    "reference_answer": reference_answer,
                    "prediction": prediction,
                    "exact_match": score["exact_match"],
                    "f1": score["f1"],
                    "error": str(ex),
                    "retrieved_count": len(top_docs),
                }
            )
            continue

        if score["exact_match"]:
            exact_matches += 1
        f1_sum += float(score["f1"])

        results.append(
            {
                "id": qa_id,
                "question": question,
                "reference_answer": reference_answer,
                "prediction": prediction,
                "exact_match": score["exact_match"],
                "f1": score["f1"],
                "retrieved_count": len(top_docs),
            }
        )

    total = len(results)
    return {
        "evaluated": total,
        "exact_match_rate": (exact_matches / total) if total else 0.0,
        "average_f1": (f1_sum / total) if total else 0.0,
        "results": results,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    try:
        embedding = create_embedding(request.text)
        return {
            "text": request.text,
            "embedding": embedding,
            "dimension": len(embedding),
        }
    except EmbeddingError as ex:
        raise HTTPException(status_code=400, detail=str(ex))


@app.get("/models")
async def list_models():
    """Get available models registry."""
    models = get_available_models()
    return {"chat": list(models["chat"].keys()), "embeddings": list(models["embeddings"].keys())}


@app.get("/models/{model_type}/{model_name}")
async def check_model(model_type: str, model_name: str):
    """Check if a specific model is available and accessible."""
    available = validate_model_access(model_type, model_name)
    
    if not available:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found in {model_type}")
    
    models = get_available_models()
    model_config = models[model_type][model_name]
    
    return ModelStatusResponse(
        model_type=model_type,
        model_name=model_name,
        available=True,
        deployment_name=model_config.get("deployment_name"),
        api_version=model_config.get("api_version"),
    )


@app.get("/test-data")
async def get_test_data(limit: int = 5):
    """Get sample test Q&A pairs for development and testing."""
    examples = get_sample_test_data(limit)
    return {"count": len(examples), "examples": examples}


@app.get("/test-data/{qa_id}")
async def get_test_qa(qa_id: str):
    """Get a specific test Q&A pair by ID."""
    all_data = load_all_cached_test_data()
    
    for qa in all_data:
        if qa["id"] == qa_id:
            return TestQAExample(**qa)
    
    raise HTTPException(status_code=404, detail=f"Test Q&A pair {qa_id} not found")


@app.get("/vector-store/status")
async def vector_store_status():
    pipeline = await get_ingest_pipeline()
    return {
        "vector_store": {
            "loaded": pipeline.status.get("vector_store_loaded", False),
            "document_count": pipeline.status.get("vector_store_count", 0),
            "dimension": pipeline.status.get("vector_store_dimension", 0),
            "path": pipeline.status.get("vector_store_path", ""),
        }
    }
