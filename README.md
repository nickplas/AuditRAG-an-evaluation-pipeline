# AuditRAG-an-evaluation-pipeline

### Overview:

A local Retrieval-Augmented Generation (RAG) system that ingests complex documents, extracts structured knowledge, and serves answers via an API. The main principle is: **every answer must be traceable, refusable, and auditable**.

Built with HuggingFace-native components — no proprietary API keys required.

## Architecture

```
Documents
    │
    ▼
┌──────────────┐     ┌───────────────────┐
│  Ingestor    │────▶│  Knowledge Graph  │  (spaCy NER + NetworkX)
│  (chunking)  │     │  (entity triples) │
└──────────────┘     └───────────────────┘
        │                       │
        ▼                       │ cross-check
┌──────────────┐                │
│  VectorStore │     ┌──────────┴────────┐
│  (Qdrant)    │────▶│    Generator      │────▶ RAGResponse
│  +Embedder   │     │  (HF seq2seq LLM) │       ├─ answer + citations
└──────────────┘     └───────────────────┘       ├─ confidence
        ▲                                        ├─ refused flag
        │                                        └─ kg_flags
     Query
        │
┌──────────────┐
│  FastAPI     │  /query  /health  /metrics  /index
│  Service     │
└──────────────┘
        │
┌──────────────┐
│  Evaluator   │  faithfulness, refusal accuracy,
│  + MLflow    │  hallucination rate, latency p95
└──────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Refusal gate** | Refuses to answer when retrieval scores are below threshold — prefers silence over hallucination |
| **Inline citations** | Every claim is tagged `[N]` → source chunk, enabling human verification |
| **KG cross-check** | Entity-level consistency check between answer and structured facts — catches hallucinations the embedding similarity misses |
| **Retrieval-based confidence** | Confidence derived from embedding similarity, not LLM logprobs — more interpretable and stable |
| **Evaluation harness** | Tests four failure modes: normal, out-of-domain, adversarial, edge-case — not just benchmark accuracy |

---

### 1. Install

```bash
git clone <repository-url>
cd AUDITRAG-AN-EVALUATION-PIPELINE

uv sync

source .venv/bin/activate
```

### 2. Run the demo (no GPU, no server needed)

```bash
python scripts/demo.py
```

This runs the full pipeline in-memory on a synthetic corpus, prints cited answers,
shows the refusal mechanism, and runs the evaluation harness.

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Start the API

``` 
streamlit run app_streamlit.py 
```


### Project Structure

```
AuditRAG/
├── src/
│   ├── ingestion/
│   │   └── ingestor.py          # Document loading, cleaning, chunking
│   ├── retrieval/
│   │   └── retriever.py         # Embedder + Qdrant vector store
│   ├── generation/
│   │   └── generator.py         # LLM generation with citation + KG guard
│   ├── knowledge_graph/
│   │   └── kg_builder.py        # spaCy NER + NetworkX relation extraction
│   ├── evaluation/
│   │   └── evaluator.py         # Failure-mode test harness + MLflow logging
│   ├── api/
│   │   └── app.py               # FastAPI service
│   ├── pipeline.py              # Top-level orchestrator
│   └── config.py                # Centralised configuration
├── tests/
│   └── test_pipeline.py
├── demo.py                      # End-to-end demo
├── app_streamlit.py             # Web App
├── MODEL_CARD.md                # Failure modes, limitations, evaluation results
└── uv.lock
```
