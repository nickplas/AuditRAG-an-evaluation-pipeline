"""
retriever.py: 
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed; vector store disabled.")

from src.config import settings
from src.ingestion.ingestor import Chunk

# Data Model
@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with its similarity score."""
    chunk: Chunk
    score: float

    @property
    def is_relevant(self) -> bool:
        return self.score >= settings.score_threshold


# Embedding model
class Embedder:
    """Thin wrapper around SentenceTransformer."""

    def __init__(self, model_name: str = settings.embedding_model):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings of shape (N, dim)."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

# Retrieval    
class VectorStore:
    """
    Qdrant-backed vector store.

    Parameters
    ----------
    in_memory : bool
        If True, spins up a local in-memory Qdrant instance (no server needed).
        Perfect for tests and demos. Set False for production deployments.
    """

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str = settings.qdrant_collection,
        in_memory: bool = False,
    ):
        if not _QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client is required for VectorStore.")

        self.embedder = embedder
        self.collection_name = collection_name

        if in_memory:
            self.client = QdrantClient(":memory:")
            logger.info("VectorStore: using in-memory Qdrant")
        else:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            logger.info(
                f"VectorStore: connected to Qdrant at "
                f"{settings.qdrant_host}:{settings.qdrant_port}"
            )

        self._ensure_collection()

    # Indexing
    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed and upsert chunks into the vector store."""
        logger.info(f"Indexing {len(chunks)} chunks…")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            vectors = self.embedder.embed(texts)
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                    vector=vectors[j].tolist(),
                    payload={
                        "chunk_id": c.id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "source": c.source,
                        "page": c.page,
                        **c.metadata,
                    },
                )
                for j, c in enumerate(batch)
            ]
            self.client.upsert(
                collection_name=self.collection_name, points=points
            )
        logger.info("Indexing complete.")

    def collection_size(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    # Search
    def search(
        self,
        query: str,
        top_k: int = settings.top_k,
        filter_doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the *top_k* most similar chunks for *query*.

        Parameters
        ----------
        query : str
            The user's natural-language question.
        top_k : int
            Maximum number of chunks to return.
        filter_doc_id : str | None
            Restrict search to chunks from a specific document.
        """
        query_vec = self.embedder.embed_query(query).tolist()

        qdrant_filter = None
        if filter_doc_id:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=filter_doc_id),
                )]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        retrieved: list[RetrievedChunk] = []
        for r in results:
            payload = r.payload or {}
            chunk = Chunk(
                text=payload.get("text", ""),
                doc_id=payload.get("doc_id", ""),
                chunk_index=int(payload.get("chunk_id", "0").split("_")[-1]),
                source=payload.get("source", ""),
                page=payload.get("page"),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in {"text", "doc_id", "chunk_id", "source", "page"}
                },
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=r.score))

        logger.debug(
            f"Query '{query[:60]}…' → {len(retrieved)} results "
            f"(top score: {retrieved[0].score:.3f})" if retrieved else
            f"Query '{query[:60]}…' → 0 results"
        )
        return retrieved
