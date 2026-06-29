"""
tests/test_pipeline.py — Unit and integration tests for the RAG pipeline.

Run with:
    pytest tests/ -v

These tests use in-memory Qdrant and a tiny synthetic corpus so they run
without a GPU, a Qdrant server, or any external API keys.
"""

from __future__ import annotations

import pytest

from src.ingestion.ingestor import DocumentIngestor
from src.retrieval.retriever import Embedder, VectorStore
from src.pipeline import RAGPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

CORPUS = [
   "Bread is a staple food prepared from a dough of flour and water, usually by baking. "
   "It has been a prominent food in large parts of the world and is considered one of the oldest human-made foods, dating back thousands of years."
    "The most common ingredients in bread making are wheat flour, water, yeast, and salt. "
    "Yeast is a microscopic fungus that plays a crucial role in the baking process. "
    "It consumes sugars present in the flour and releases carbon dioxide gas. "
    "This gas gets trapped in the stretchy dough, causing it to expand and rise, which ultimately gives the baked bread its soft, airy texture."
    "There are countless varieties of bread across different cultures. "
    "Sourdough, for example, relies on a fermented starter of naturally occurring yeast and bacteria, giving it a distinct tangy flavor. "
    "Meanwhile, baguettes are long, thin loaves originating from France that are famous for their very crispy crust.",
]


@pytest.fixture(scope="module")
def ingestor():
    return DocumentIngestor(chunk_size=200, chunk_overlap=40)


@pytest.fixture(scope="module")
def chunks(ingestor):
    return ingestor.ingest_texts(CORPUS, source="test_corpus")


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def vector_store(embedder, chunks):
    vs = VectorStore(embedder=embedder, in_memory=True)
    vs.index_chunks(chunks)
    return vs


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestIngestor:
    def test_produces_chunks(self, chunks):
        assert len(chunks) > 0

    def test_chunk_has_text(self, chunks):
        for c in chunks:
            assert c.text.strip()

    def test_chunk_has_provenance(self, chunks):
        for c in chunks:
            assert c.doc_id
            assert c.source

    def test_chunk_id_unique(self, chunks):
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_empty_text_produces_no_chunks(self, ingestor):
        result = ingestor.ingest_texts(["", "   ", "\n"])
        assert result == []


# ── Retrieval tests ───────────────────────────────────────────────────────────

class TestRetriever:
    def test_returns_results(self, vector_store):
        results = vector_store.search("What is bread?", top_k=3)
        assert len(results) > 0

    def test_results_have_scores(self, vector_store):
        results = vector_store.search("Sourdough and baguettes", top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_relevant_chunk_ranked_first(self, vector_store):
        results = vector_store.search("Yeast and carbon dioxide", top_k=3)
        assert "Yeast" in results[0].chunk.text

    def test_top_k_respected(self, vector_store):
        results = vector_store.search("French bread", top_k=2)
        assert len(results) <= 2

    def test_is_relevant_flag(self, vector_store):
        # Highly specific query should return at least one relevant result
        results = vector_store.search("Microscopic fungus fermentation", top_k=5)
        assert any(r.is_relevant for r in results)

    def test_out_of_domain_low_score(self, vector_store):
        # Query completely unrelated to corpus
        results = vector_store.search(
            "xyzzy plugh frobnicate completely unrelated nonsense", top_k=5
        )
        # All scores should be below a reasonable threshold
        assert all(r.score < 0.7 for r in results)

    def test_collection_size(self, vector_store, chunks):
        assert vector_store.collection_size() == len(chunks)


# ── Pipeline integration tests ────────────────────────────────────────────────

class TestPipeline:
    """
    Full pipeline tests. We skip the LLM to keep CI fast — the generator is
    tested separately. These tests verify the ingestion → retrieval path.
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        p = RAGPipeline(
            in_memory=True,
            use_knowledge_graph=False,  # spaCy may not be installed in CI
        )
        p.index_texts(CORPUS, source="test")
        return p

    def test_collection_populated(self, pipeline):
        assert pipeline.collection_size > 0

    def test_retrieval_returns_results(self, pipeline):
        # Access vector store directly (bypass LLM for speed)
        results = pipeline.retriever.search("Bread ingredients", top_k=3)
        assert len(results) > 0

    def test_empty_query_handled(self, pipeline):
        response = pipeline.query("")
        assert response.refused is True
        assert "empty" in response.refusal_reason.lower()

    def test_response_schema_complete(self, pipeline):
        """Every field of RAGResponse should be present and typed correctly."""
        # We mock the generator to avoid loading the LLM
        from unittest.mock import patch, MagicMock
        from src.generation.generator import RAGResponse as RR

        mock_response = RR(
            answer="Bread rises because yeast consumes sugars in the flour and releases carbon dioxide gas, which gets trapped in the dough and causes it to expand [1].",
            citations=[{
                "index": 1, "chunk_id": "test__chunk_0",
                "source": "test", "page": None,
                "score": 0.9, "text": "Bread rises because…",
            }],
            confidence=0.9,
            refused=False,
            refusal_reason="",
            kg_flags=[],
            latency_ms=120.0,
            sources=["test"],
        )

        with patch.object(pipeline, "_get_generator") as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = mock_response
            mock_gen_fn.return_value = mock_gen

            response = pipeline.query("Why does bread rise?")

        assert response.answer
        assert isinstance(response.citations, list)
        assert 0.0 <= response.confidence <= 1.0
        assert isinstance(response.refused, bool)
        assert isinstance(response.sources, list)
