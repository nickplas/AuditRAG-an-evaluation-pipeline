"""
api/app.py — FastAPI production service for the RAG pipeline.

Endpoints
---------
POST /query          — ask a question, get a cited, auditable answer
GET  /health         — liveness probe
GET  /metrics        — latency and usage statistics
POST /index          — (re)index documents from the configured data directory

Design notes
------------
• All responses follow a consistent schema regardless of whether the answer
  was generated, refused, or errored — consumers can always parse the same shape.
• Metrics are kept in-process (no external monitoring required for a v1).
  In production, swap for Prometheus counters.
• The pipeline is initialised at startup and shared across requests.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.config import settings
from src.pipeline import RAGPipeline


# ── Shared state ──────────────────────────────────────────────────────────────

pipeline: RAGPipeline | None = None

# Rolling window of recent latencies (last 200 requests)
_latency_window: Deque[float] = deque(maxlen=200)
_request_counts: dict[str, int] = {"total": 0, "refused": 0, "error": 0}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting up: loading RAG pipeline…")
    pipeline = RAGPipeline(
        in_memory=False,       # Use real Qdrant in production
        use_knowledge_graph=True,
    )
    # Auto-index on startup if data directory exists and has documents
    data_dir = settings.data_dir
    if data_dir.exists() and any(data_dir.rglob("*.txt")) or any(data_dir.rglob("*.pdf") if data_dir.exists() else []):
        logger.info(f"Auto-indexing from {data_dir}…")
        pipeline.index(data_dir)
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Auditable RAG Pipeline",
    description=(
        "Knowledge-grounded QA with inline citations, hallucination detection, "
        "and full source provenance. Built for regulated / high-trust contexts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question.")
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


class Citation(BaseModel):
    index: int
    chunk_id: str
    source: str
    page: int | None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool
    refusal_reason: str
    kg_flags: list[str]
    latency_ms: float
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    collection_size: int
    kg_entities: int | None


class MetricsResponse(BaseModel):
    total_requests: int
    refused_requests: int
    error_requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class IndexResponse(BaseModel):
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question. Returns a cited, auditable answer.

    The `refused` field is True when the corpus does not contain sufficient
    information to answer reliably — prefer this signal over low confidence scores
    when deciding whether to surface the answer to end users.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    _request_counts["total"] += 1
    t0 = time.perf_counter()

    try:
        response = pipeline.query(request.query, top_k=request.top_k)
    except Exception as exc:
        _request_counts["error"] += 1
        logger.exception(f"Pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    latency = (time.perf_counter() - t0) * 1000
    _latency_window.append(latency)

    if response.refused:
        _request_counts["refused"] += 1

    return QueryResponse(
        answer=response.answer,
        citations=[Citation(**c) for c in response.citations],
        confidence=response.confidence,
        refused=response.refused,
        refusal_reason=response.refusal_reason,
        kg_flags=response.kg_flags,
        latency_ms=response.latency_ms,
        sources=response.sources,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness probe."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    kg = pipeline.knowledge_graph
    return HealthResponse(
        status="ok",
        collection_size=pipeline.collection_size,
        kg_entities=kg.graph.number_of_nodes() if kg else None,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Inference latency statistics and request counts."""
    latencies = sorted(_latency_window)
    n = len(latencies)

    def percentile(p: float) -> float:
        if not latencies:
            return 0.0
        idx = int(n * p / 100)
        return latencies[min(idx, n - 1)]

    return MetricsResponse(
        total_requests=_request_counts["total"],
        refused_requests=_request_counts["refused"],
        error_requests=_request_counts["error"],
        p50_latency_ms=round(percentile(50), 1),
        p95_latency_ms=round(percentile(95), 1),
        p99_latency_ms=round(percentile(99), 1),
    )


@app.post("/index", response_model=IndexResponse)
async def index_documents(background_tasks: BackgroundTasks) -> IndexResponse:
    """
    Trigger a (re)index of the configured data directory.
    Runs in the background so the API stays responsive.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    def _do_index():
        try:
            pipeline.index(settings.data_dir)
            logger.info("Background indexing complete.")
        except Exception as exc:
            logger.error(f"Background indexing failed: {exc}")

    background_tasks.add_task(_do_index)
    return IndexResponse(
        status="accepted",
        message=f"Indexing {settings.data_dir} in the background.",
    )
