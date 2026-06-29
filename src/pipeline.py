"""
pipeline.py — Top-level RAG pipeline that wires all components together.

This is the single object you import in production code, tests, and the API.
It exposes a clean .query() method and manages component lifecycle.

Usage
-----
>>> pipeline = RAGPipeline(in_memory=True)
>>> pipeline.index(Path("data/corpus"))
>>> response = pipeline.query("What are the main risks?")
>>> print(response.answer)
>>> print(response.citations)
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.config import settings
from src.ingestion.ingestor import DocumentIngestor, Chunk
from src.retrieval.retriever import Embedder, Retriever 
from src.generation.generator import RAGGenerator, RAGResponse
from src.knowledge_graph.kg_builder import KnowledgeGraphBuilder, KnowledgeGraph


class RAGPipeline:
    """
    Orchestrates ingestion → retrieval → generation.

    Parameters
    ----------
    in_memory : bool
        Use in-memory Qdrant (no server needed). Set False for production.
    use_knowledge_graph : bool
        Enable KG extraction and hallucination cross-check.
    llm_model : str
        HuggingFace model name for generation.
    """

    def __init__(
        self,
        in_memory: bool = False,
        use_knowledge_graph: bool = True,
        llm_model: str = settings.llm_model,
        retrieval_mode: str = "hybrid",
    ):
        logger.info("Initialising RAG pipeline…")

        self.ingestor = DocumentIngestor()
        self.embedder = Embedder()
        self.retriever = Retriever(
            embedder=self.embedder,
            in_memory=in_memory,
        )
        self.retrieval_mode = retrieval_mode

        self._kg_enabled = use_knowledge_graph
        self._kg_builder = KnowledgeGraphBuilder() if use_knowledge_graph else None
        self._kg: KnowledgeGraph | None = None

        # Generator is initialised lazily so the pipeline can be indexed
        # without loading the LLM (useful for indexing jobs)
        self._generator: RAGGenerator | None = None
        self._llm_model = llm_model

        logger.info("Pipeline ready.")

    # Indexing 

    def index(self, source: Path | list[Chunk]) -> None:
        """
        Index documents from a directory path or a pre-built list of Chunks.
        Also builds the knowledge graph if enabled.
        """
        chunks = (
            self.ingestor.ingest_directory(source)
            if isinstance(source, Path)
            else source
        )

        if not chunks:
            logger.warning("No chunks to index.")
            return

        self.retriever.index_chunks(chunks)

        if self._kg_enabled and self._kg_builder is not None:
            logger.info("Building knowledge graph…")
            self._kg = self._kg_builder.build(chunks)
            logger.info(f"KG summary: {self._kg.summary()}")

    def index_texts(self, texts: list[str], source: str = "inline") -> None:
        """Convenience: index raw strings (e.g. for demos and tests)."""
        chunks = self.ingestor.ingest_texts(texts, source=source)
        self.index(chunks)

    # ── Querying ──────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: int = settings.top_k,
        retrieval_mode: str | None = None,
    ) -> RAGResponse:
        """
        Run a query through the full pipeline.

        Returns a RAGResponse with:
          • answer        — generated text with inline citations
          • citations     — source chunks used
          • confidence    — retrieval-based confidence signal
          • refused       — True if the question couldn't be answered safely
          • kg_flags      — entities in the answer not grounded in the KG
          • latency_ms    — wall-clock time
          • sources       — deduplicated list of source documents
        """
        query = query.strip()
        if not query:
            return RAGResponse(
                answer="",
                refused=True,
                refusal_reason="Query is empty.",
            )

        mode = retrieval_mode or self.retrieval_mode
        retrieved = self.retriever.search(query, top_k=top_k, mode=mode)
        return self._get_generator().generate(query, retrieved, top_k=top_k)

    # ── Persistence helpers ───────────────────────────────────────────────

    def save_kg(self, path: Path) -> None:
        if self._kg is None:
            raise RuntimeError("No knowledge graph built yet.")
        self._kg.save(path)

    def load_kg(self, path: Path) -> None:
        self._kg = KnowledgeGraph.load(path)
        # Inject into generator if already initialised
        if self._generator is not None:
            self._generator.kg = self._kg

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def collection_size(self) -> int:
        return self.retriever.collection_size()

    @property
    def knowledge_graph(self) -> KnowledgeGraph | None:
        return self._kg

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_generator(self) -> RAGGenerator:
        """Lazy-load the LLM on first query."""
        if self._generator is None:
            self._generator = RAGGenerator(
                model_name=self._llm_model,
                knowledge_graph=self._kg,
            )
        return self._generator
