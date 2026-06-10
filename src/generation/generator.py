"""
generation/generator.py 
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import settings
from src.retrieval.retriever import RetrievedChunk
from src.knowledge_graph.kg_builder import KnowledgeGraph


# ── Response model ────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    The complete, auditable output of a RAG query.

    Every field here is intentional:
    • answer         — the generated text
    • citations      — maps [N] → chunk so every claim can be verified
    • confidence     — aggregate retrieval signal (not the LLM's own estimate)
    • refused        — True when context was too weak to answer safely
    • kg_flags       — entities in the answer not grounded in the KG (if available)
    • latency_ms     — for monitoring
    • sources        — deduplicated list of source documents
    """
    answer: str
    citations: list[dict] = field(default_factory=list)   # [{index, chunk_id, source, score, text}]
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: str = ""
    kg_flags: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "confidence": round(self.confidence, 4),
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "kg_flags": self.kg_flags,
            "latency_ms": round(self.latency_ms, 1),
            "sources": self.sources,
        }


# ── Generator ─────────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Wraps a HuggingFace seq2seq model with retrieval-augmented prompting,
    citation injection, and optional KG cross-checking.

    Parameters
    ----------
    model_name : str
        Any HuggingFace seq2seq model (default: flan-t5-base).
    knowledge_graph : KnowledgeGraph | None
        If provided, answers are cross-checked against structured facts.
    device : str | None
        "cuda", "mps", or "cpu". Auto-detected if None.
    """

    # Refusal triggers
    _REFUSAL_PHRASES = {
        "i don't know", "i do not know", "cannot answer",
        "no information", "not found in", "no relevant",
    }

    def __init__(
        self,
        model_name: str = settings.llm_model,
        knowledge_graph: KnowledgeGraph | None = None,
        device: str | None = None,
    ):
        self.kg = knowledge_graph

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        logger.info(f"Generator using device: {device}")

        # Load model
        logger.info(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        logger.info("LLM ready.")

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        top_k: int = settings.top_k,
    ) -> RAGResponse:
        """
        Generate an answer for *query* given *retrieved* context chunks.

        Steps
        -----
        1. Check if any retrieved chunk meets the relevance threshold.
        2. Build a cited context block (each chunk labelled [1], [2] …).
        3. Prompt the model with a strict instruction to cite sources.
        4. Post-process: extract citations, check KG, compute confidence.
        """
        t0 = time.perf_counter()

        # Step 1: Relevance gate 
        relevant = [r for r in retrieved if r.is_relevant][:top_k]

        if not relevant:
            return RAGResponse(
                answer="",
                refused=True,
                refusal_reason=(
                    "No sufficiently relevant context was found in the corpus "
                    "to answer this question reliably."
                ),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Step 2: Build context block with inline citation markers 
        context_lines = []
        citation_map: list[dict] = []
        for i, r in enumerate(relevant, start=1):
            context_lines.append(f"[{i}] {r.chunk.text}")
            citation_map.append({
                "index": i,
                "chunk_id": r.chunk.id,
                "source": r.chunk.source,
                "page": r.chunk.page,
                "score": round(r.score, 4),
                "text": r.chunk.text[:200] + "…" if len(r.chunk.text) > 200 else r.chunk.text,
            })

        context = "\n\n".join(context_lines)

        # Step 3: Prompt
        prompt = self._build_prompt(query, context)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature if settings.llm_temperature > 0 else 1.0,
                do_sample=settings.llm_temperature > 0,
            )

        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Step 4: Detect model self-refusal
        if self._is_self_refusal(raw_output):
            return RAGResponse(
                answer=raw_output,
                citations=citation_map,
                refused=True,
                refusal_reason="Model indicated it could not answer from the provided context.",
                confidence=self._confidence(relevant),
                latency_ms=(time.perf_counter() - t0) * 1000,
                sources=self._dedupe_sources(relevant),
            )

        # Step 5: KG cross-check ────────────────────────────────────────
        kg_flags: list[str] = []
        if self.kg is not None:
            check = self.kg.check_claim(raw_output)
            if check["flag"]:
                kg_flags = check["entities_found"]
                logger.warning(
                    f"KG flag: entities {kg_flags} in answer not grounded in KG."
                )

        latency = (time.perf_counter() - t0) * 1000
        logger.info(f"Generated answer in {latency:.0f} ms")

        return RAGResponse(
            answer=raw_output,
            citations=citation_map,
            confidence=self._confidence(relevant),
            refused=False,
            kg_flags=kg_flags,
            latency_ms=latency,
            sources=self._dedupe_sources(relevant),
        )

    # Prompt template 
    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """
        Instruction-tuned prompt for Flan-T5 / similar models.
        The explicit citation instruction is key for auditability.
        """
        return (
            "You are a precise, factual assistant. Answer the question rephrasing "
            "the information from the original document and add the numbered context passages below. "
            "Cite the passage numbers (e.g. [1], [2]) inline for every claim. "
            "If the context does not contain enough information to answer, "
            "say exactly: 'I cannot answer this from the available context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (with inline citations):"
        )

    # Helpers 
    def _is_self_refusal(self, text: str) -> bool:
        low = text.lower()
        return any(phrase in low for phrase in self._REFUSAL_PHRASES)

    @staticmethod
    def _confidence(relevant: list[RetrievedChunk]) -> float:
        """
        Aggregate confidence signal based on retrieval scores.
        Not the LLM's own probability — deliberately retrieval-grounded.
        """
        if not relevant:
            return 0.0
        scores = [r.score for r in relevant]
        # Weighted average: top result counts more
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    @staticmethod
    def _dedupe_sources(relevant: list[RetrievedChunk]) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for r in relevant:
            if r.chunk.source not in seen:
                seen.add(r.chunk.source)
                sources.append(r.chunk.source)
        return sources
