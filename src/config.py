"""
config.py: Configuration of the RAG pipeline. Here all tunable parameters are defined.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):

    # Path
    data_dir : Path = Path("data")
    mlflow_tracking_uri: str = "mlruns"

    # Embedding model (change model in order to make it fit in your memory)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # LLM model
    llm_model: str = "google/flan-t5-xl"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1

    # Chunks
    chunk_size: int = 400
    chunk_overlap: int = 80

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.3
    qdrant_host : str = "localhost"
    qdrant_port : int = 6333
    qdrant_collection: str = "rag_corpus"

    # Knowledge graph
    spacy_model: str = "en_core_web_sm"
    kg_confidence_threshold: float = 0.6

    # Evaluation
    hallucination_threshold: float = 0.45
    min_retrieval_precision: float = 0.5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()