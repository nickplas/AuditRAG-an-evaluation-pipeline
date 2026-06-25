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

from rank_bm25 import BM25Okapi
import re


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
 
 
def _reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Combine any number of ranked lists using Reciprocal Rank Fusion.
 
    RRF score for a chunk = Σ  1 / (k + rank_in_list_i)
 
    k=60 is the standard default from the original paper (Cormack 2009).
    Higher k reduces the influence of top-ranked items; lower k amplifies it.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}
 
    for ranked in ranked_lists:
        for rank, rc in enumerate(ranked, start=1):
            cid = rc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = rc
 
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, rrf_score in fused:
        rc = chunk_map[cid]
        result.append(RetrievedChunk(
            chunk=rc.chunk,
            score=round(rrf_score, 6),
            retrieval_mode=settings.retrieval_mode,
        ))
    return result

# Data Model
@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with its similarity score."""
    chunk: Chunk
    score: float
    retrieval_mode: str = settings.retrieval_mode

    @property
    def is_relevant(self) -> bool:
        if self.retrieval_mode == "hybrid":
            return self.score >= 0.005
        if self.retrieval_mode == "bm25":
            return self.score >= 0.01
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

class BM25Index:
    """
    In-memory BM25 index built from a list of Chunks.
 
    Kept in sync with the vector store by always calling both
    VectorStore.index_chunks() and BM25Index.index_chunks() via
    the unified Retriever.index_chunks() method.
    """
 
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
 
    def index_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} documents.")
 
    def search(self, query: str, top_k: int = settings.top_k) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            raise RuntimeError("BM25 index is empty — call index_chunks() first.")
 
        tokens = _tokenize(query)
        raw_scores = self._bm25.get_scores(tokens)
 
        # Normalise to [0, 1] so scores are comparable across queries
        max_score = raw_scores.max()
        if max_score > 0:
            norm_scores = raw_scores / max_score
        else:
            norm_scores = raw_scores
 
        top_indices = np.argsort(norm_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if norm_scores[idx] > 0:
                results.append(RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(norm_scores[idx]),
                    retrieval_mode=settings.retrieval_mode,
                ))
        return results
 
    @property
    def size(self) -> int:
        return len(self._chunks)


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
            retrieved.append(RetrievedChunk(chunk=chunk, score=r.score, retrieval_mode=""))

        logger.debug(
            f"Query '{query[:60]}…' → {len(retrieved)} results "
            f"(top score: {retrieved[0].score:.3f})" if retrieved else
            f"Query '{query[:60]}…' → 0 results"
        )
        return retrieved
    
class Retriever:
    """
    Single entry point for all retrieval modes.
 
    Usage
    -----
    >>> retriever = Retriever(embedder, in_memory=True)
    >>> retriever.index_chunks(chunks)
    >>> results = retriever.search("What is a transition matrix?", mode="hybrid")
 
    Modes
    -----
    • "semantic"  — dense vector search only
    • "bm25"      — keyword search only
    • "hybrid"    — RRF fusion of both (recommended default)
    """
 
    def __init__(
        self,
        embedder: Embedder,
        in_memory: bool = False,
        collection_name: str = settings.qdrant_collection,
    ):
        self.vector_store = VectorStore(
            embedder=embedder,
            in_memory=in_memory,
            collection_name=collection_name,
        )
        self.bm25_index = BM25Index()
 
    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into both the vector store and the BM25 index."""
        self.vector_store.index_chunks(chunks)
        if self.bm25_index is not None:
            self.bm25_index.index_chunks(chunks)
        else:
            logger.warning("BM25 index skipped (rank-bm25 not installed).")
 
    def search(
        self,
        query: str,
        top_k: int = settings.top_k,
        mode: str = settings.retrieval_mode,
    ) -> list[RetrievedChunk]:
        """
        Search for relevant chunks.
 
        Parameters
        ----------
        query : str
            Natural-language question.
        top_k : int
            Number of results to return.
        mode : str
            "semantic", "bm25", or "hybrid".
        """
        if mode == "semantic":
            return self.vector_store.search(query, top_k=top_k)
 
        if mode == "bm25":
            if self.bm25_index is None:
                raise RuntimeError("BM25 unavailable — install rank-bm25.")
            return self.bm25_index.search(query, top_k=top_k)
 
        if mode == "hybrid":
            # Fetch more candidates from each source before fusing,
            # so RRF has a wider pool to rerank.
            fetch_k = min(top_k * 2, 20)
 
            semantic_results = self.vector_store.search(query, top_k=fetch_k)
 
            if self.bm25_index is not None:
                bm25_results = self.bm25_index.search(query, top_k=fetch_k)
                fused = _reciprocal_rank_fusion(semantic_results, bm25_results)
            else:
                logger.warning("BM25 unavailable — hybrid falling back to semantic.")
                fused = semantic_results
 
            return fused[:top_k]
 
        raise ValueError(f"Unknown retrieval mode: {mode!r}. "
                         f"Choose from 'semantic', 'bm25', 'hybrid'.")
 
    def collection_size(self) -> int:
        return self.vector_store.collection_size()