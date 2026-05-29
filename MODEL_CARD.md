# Model Card — Auditable RAG Pipeline

> **TL;DR** A production-grade Retrieval-Augmented Generation system that prioritises
> auditability and trustworthiness over raw benchmark performance.

---

## System Overview

| Component | Default | Notes |
|-----------|---------|-------|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, runs on CPU |
| LLM | `google/flan-t5-base` | Seq2seq, no API key needed |
| Vector store | Qdrant | In-memory or server mode |
| Knowledge graph | spaCy + NetworkX | NER + dependency relations |
| Experiment tracking | MLflow | Local by default |

---

## Intended Use

Answering natural-language questions over a private document corpus in contexts
where every answer must be traceable to a source — e.g. regulatory documents,
internal knowledge bases, industrial manuals.

**Primary users:** Engineers and analysts in regulated industries (finance,
insurance, maritime, healthcare).

---

## How Auditability Is Achieved

1. **Inline citations** — Every generated claim is tagged with a `[N]` marker
   linked to the exact retrieved chunk that supports it.

2. **Source provenance** — Every chunk stores its origin document, page number,
   and chunk index. The `citations` field in the API response exposes this.

3. **Confidence signal** — Derived from retrieval scores (not the LLM's own
   probabilities), making it interpretable and stable.

4. **Explicit refusals** — When no retrieved chunk exceeds the similarity
   threshold, the system refuses to answer rather than hallucinate.

5. **KG cross-check** — Named entities in the answer are cross-referenced against
   a structured knowledge graph extracted from the corpus. Mismatches are flagged
   in the `kg_flags` field.

---

## Known Failure Modes

| Failure | Condition | Mitigation |
|---------|-----------|------------|
| **Hallucination** | LLM ignores retrieved context | KG cross-check + faithfulness evaluation |
| **False refusal** | Score threshold too aggressive | Tune `score_threshold` in config |
| **Chunk boundary loss** | Answer spans two chunks | Increase `chunk_overlap` |
| **Out-of-domain leakage** | LLM draws on parametric memory | Strict prompt + refusal gate |
| **NER errors in KG** | spaCy misclassifies entities | Fine-tune NER on domain data |
| **Latency under load** | LLM inference is slow | Quantise model; use vLLM for serving |

---

## When NOT to Use This System

- When answers require real-time data (stock prices, live events)
- When the corpus is not curated — garbage in, garbage out
- As a sole decision-maker in high-stakes clinical or legal contexts without
  human review

---

## Evaluation Results (Demo Corpus)

Run `python scripts/demo.py` to reproduce. Results will vary by corpus size and domain.

| Metric | Target | Notes |
|--------|--------|-------|
| Faithfulness | ≥ 0.6 | Semantic similarity to cited chunks |
| Refusal accuracy | ≥ 0.9 | On out-of-domain queries |
| Hallucination rate | ≤ 0.15 | Flagged by KG + faithfulness |
| P95 latency | ≤ 2000 ms | CPU, base model |

---

## What I Would Do With More Time

1. **Fine-tune the embedding model** on domain-specific sentence pairs (e.g.
   insurance policy Q&A) using `sentence-transformers` training utilities.

2. **Replace heuristic relation extraction** in the KG with a fine-tuned RE model
   (e.g. REBEL) for higher-precision fact triples.

3. **Add reranking** — use a cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`) to
   rerank the top-k chunks before passing to the generator.

4. **Streaming responses** — FastAPI + server-sent events for real-time token
   streaming to the client.

5. **Prometheus + Grafana** dashboard for production monitoring (currently tracked
   in-process).

6. **Adversarial evaluation** — red-team the prompt injection and false-premise
   cases with a larger suite.

---

## Licence

MIT — see `LICENSE`.
