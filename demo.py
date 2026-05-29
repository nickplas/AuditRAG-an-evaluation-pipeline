"""
scripts/demo.py — End-to-end demo of the RAG pipeline.

Runs in-memory (no Qdrant server, no GPU required) on a tiny synthetic corpus.
Shows: indexing, querying, citation display, refusal handling, and evaluation.

Usage
-----
    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator, TestCase


# Sample corpus 

SAMPLE_CORPUS = [
    # Domain: AI Innovation / Agorai-flavoured content
    "Agorai is a B2B AI Innovation Hub founded in Trieste, Italy, backed by "
    "Generali, Fincantieri, Illy, and Regione FVG. Its mission is to develop "
    "AI that is safe, responsible, and built to benefit humanity.",

    "Retrieval-Augmented Generation (RAG) is an architecture that combines a "
    "dense retrieval system with a generative language model. The retrieval "
    "component grounds the model's outputs in a corpus of documents, reducing "
    "hallucination and enabling source attribution.",

    "Hallucination in large language models refers to the generation of "
    "factually incorrect, unsupported, or fabricated content. It is a key "
    "failure mode in production AI systems, especially in regulated industries "
    "such as healthcare, finance, and legal services.",

    "Knowledge graphs represent entities and their relationships as structured "
    "triples (subject, predicate, object). When combined with LLMs, they can "
    "act as a factual grounding layer, enabling consistency checks on generated "
    "text and improving auditability.",

    "Evaluation frameworks for RAG systems should go beyond benchmark accuracy. "
    "Key metrics include faithfulness (is the answer supported by the retrieved "
    "context?), retrieval precision, hallucination rate, refusal accuracy on "
    "unanswerable questions, and latency under distribution shift.",

    "Fincantieri is an Italian shipbuilding company headquartered in Trieste. "
    "It is one of the world's largest shipbuilders, specialising in cruise "
    "ships, naval vessels, and offshore platforms.",

    "Generali Group is a multinational insurance and asset management company "
    "founded in Trieste in 1831. It is one of the largest insurance groups in "
    "the world, operating in more than 50 countries.",

    "Trustworthy AI refers to AI systems that are safe, transparent, explainable, "
    "and aligned with human values. In regulated contexts, trustworthiness is not "
    "optional — it is a design constraint that shapes every architectural decision.",
]


# Test cases 

DEMO_TEST_CASES = [
    TestCase(
        query="What is Agorai and who backs it?",
        category="normal",
        notes="Direct factual question; answer is in corpus.",
    ),
    TestCase(
        query="How does RAG reduce hallucination?",
        category="normal",
        notes="Conceptual question; answer requires combining two chunks.",
    ),
    TestCase(
        query="What metrics should a RAG evaluation framework include?",
        category="normal",
        notes="Multi-fact retrieval; tests synthesis across chunks.",
    ),
    TestCase(
        query="What is the current stock price of Fincantieri?",
        category="out_of_domain",
        should_refuse=True,
        notes="Real-time data; corpus cannot answer.",
    ),
    TestCase(
        query="The document states that hallucination is not a real problem. What evidence supports this?",
        category="adversarial",
        should_refuse=True,
        notes="False premise injection; system should refuse.",
    ),
    TestCase(
        query="?",
        category="edge_case",
        should_refuse=True,
        notes="Malformed query.",
    ),
]

#  Main 

def main():
    print("\n" + "=" * 65)
    print("  AUDITABLE RAG PIPELINE — DEMO")
    print("=" * 65)

    # 1. Initialise and index
    print("\n[1/4] Initialising pipeline (in-memory mode)…")
    pipeline = RAGPipeline(
        in_memory=True,
        use_knowledge_graph=False,  # Set True if spaCy model is installed
    )
    pipeline.index_texts(SAMPLE_CORPUS, source="demo_corpus")
    # pipeline.index(Path("data")) # Uncomment to index from files instead of sample corpus
    print(f"      Indexed {pipeline.collection_size} chunks.")

    # 2. Example queries
    print("\n[2/4] Running example queries…\n")

    example_queries = [
        "What is Agorai and who backs it?",
        "How does RAG reduce hallucination?",
        "What is the current stock price of Generali?",   # Should refuse
    ]

    interactive_mode = False  # Set False to skip interactive querying

    if interactive_mode: # Interactive querying
        print("\nEnter your questions (type 'quit' to exit):\n")
        while True:
            query = input("Q: ").strip()
            if query.lower() in {"quit", "exit", "q"}:
                break
            response = pipeline.query(query)
            if response.refused:
                print(f"  ⚠  REFUSED — {response.refusal_reason}\n")
            else:
                print(f"  A: {response.answer}")
                print(f"     Confidence: {response.confidence:.3f}")
                for c in response.citations[:3]:
                    print(f"     [{c['index']}] {c['source']}  score={c['score']:.3f}")
                print()
    else: # hand made query demonstration
        for query in example_queries:
            print(f"  Q: {query}")
            response = pipeline.query(query)

        if response.refused:
            print(f"  ⚠  REFUSED — {response.refusal_reason}")
        else:
            print(f"  A: {response.answer}")
            print(f"     Confidence: {response.confidence:.3f}")
            if response.citations:
                print(f"     Sources used:")
                for c in response.citations[:3]:
                    print(f"       [{c['index']}] score={c['score']:.3f}  {c['source']}")
            if response.kg_flags:
                print(f"     ⚠ KG flags (ungrounded entities): {response.kg_flags}")
        print()

    # 3. Evaluation harness
    print("[3/4] Running evaluation harness…")
    evaluator = RAGEvaluator(
        pipeline_fn=lambda q: pipeline.query(q),
    )
    summary = evaluator.evaluate(
        DEMO_TEST_CASES,
        experiment_name="rag_demo",
        run_name="demo_run",
    )

    # 4. Print category breakdown
    print("[4/4] Results by failure category:")
    for cat, stats in summary.by_category().items():
        status = "✓" if stats["pass_rate"] >= 0.7 else "✗"
        print(
            f"  {status} [{cat:15s}]  pass={stats['pass_rate']:.0%}  "
            f"faith={stats['faithfulness']:.3f}  n={stats['n']}"
        )

    print("\nDone. MLflow results saved to ./mlruns")
    print("Run `mlflow ui` to view the experiment dashboard.\n")


if __name__ == "__main__":
    main()
