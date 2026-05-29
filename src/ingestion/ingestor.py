"""
ingestor.py: Document loading, cleaning and chunking. The ingestor is responsible for taking raw documents, processing them and storing them in the vector database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from loguru import logger

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed; PDF ingestion disabled.")

from src.config import settings

# Define a dataclass for text chunks with provenance information
@dataclass
class Chunk:
    """A single text chunk with full provenance."""
    text: str
    doc_id: str
    chunk_index: int
    source: str                          # file path or URL
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}__chunk_{self.chunk_index}"
    
class DocumentIngestor:
    """
    Loads documents from disk and splits them into overlapping chunks.

    Usage
    -----
    >>> ingestor = DocumentIngestor()
    >>> chunks = ingestor.ingest_directory(Path("data/corpus"))
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # Public API 

    def ingest_directory(self, directory: Path) -> list[Chunk]:
        """Recursively ingest all supported files in *directory*."""
        chunks: list[Chunk] = []
        for path in sorted(directory.rglob("*")):
            if path.suffix in {".txt", ".pdf", ".json"}:
                try:
                    chunks.extend(self.ingest_file(path))
                except Exception as exc:
                    logger.error(f"Failed to ingest {path}: {exc}")
        logger.info(f"Ingested {len(chunks)} chunks from {directory}")
        return chunks

    def ingest_file(self, path: Path) -> list[Chunk]:
        """Dispatch to the right loader based on file extension."""
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._from_text(path)
        elif suffix == ".pdf":
            return self._from_pdf(path)
        elif suffix == ".json":
            return self._from_json(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def ingest_texts(
        self, texts: list[str], source: str = "inline"
    ) -> list[Chunk]:
        """Convenience method for ingesting raw strings (e.g. from tests)."""
        all_chunks: list[Chunk] = []
        for i, text in enumerate(texts):
            doc_id = f"{source}__{i}"
            all_chunks.extend(
                self._split(text, doc_id=doc_id, source=source)
            )
        return all_chunks

    # Loaders 

    def _from_text(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean(text)
        doc_id = path.stem
        return self._split(text, doc_id=doc_id, source=str(path))

    def _from_pdf(self, path: Path) -> list[Chunk]:
        if not _PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF required for PDF ingestion.")
        chunks: list[Chunk] = []
        doc = fitz.open(str(path))
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = self._clean(text)
            if not text.strip():
                continue
            page_chunks = self._split(
                text,
                doc_id=f"{path.stem}_p{page_num}",
                source=str(path),
                base_index=len(chunks),
                extra_metadata={"page": page_num},
            )
            # Stamp page number on each chunk
            for c in page_chunks:
                c.page = page_num
            chunks.extend(page_chunks)
        doc.close()
        return chunks

    def _from_json(self, path: Path) -> list[Chunk]:
        """
        Expected format: a JSON array of objects with at least a "text" key.
        Optional keys: "id", "source", and any other metadata fields.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"JSON file must contain a list: {path}")
        chunks: list[Chunk] = []
        for item in raw:
            text = self._clean(item.pop("text", ""))
            doc_id = item.pop("id", path.stem)
            source = item.pop("source", str(path))
            item_chunks = self._split(
                text,
                doc_id=doc_id,
                source=source,
                base_index=len(chunks),
                extra_metadata=item,
            )
            chunks.extend(item_chunks)
        return chunks

    # Splitter

    def _split(
        self,
        text: str,
        *,
        doc_id: str,
        source: str,
        base_index: int = 0,
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Sentence-aware sliding-window chunker.
        Tries to split on sentence boundaries first; falls back to character
        windows if sentences are too long.
        """
        sentences = self._sentence_split(text)
        chunks: list[Chunk] = []
        window: list[str] = []
        window_len = 0
        chunk_idx = base_index

        for sentence in sentences:
            sent_len = len(sentence)
            if window_len + sent_len > self.chunk_size and window:
                # Emit current window
                chunks.append(
                    Chunk(
                        text=" ".join(window).strip(),
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        source=source,
                        metadata=extra_metadata or {},
                    )
                )
                chunk_idx += 1
                # Overlap: keep last N chars worth of sentences
                overlap_text = " ".join(window)[-self.chunk_overlap:]
                window = [overlap_text]
                window_len = len(overlap_text)
            window.append(sentence)
            window_len += sent_len

        if window:
            chunks.append(
                Chunk(
                    text=" ".join(window).strip(),
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    source=source,
                    metadata=extra_metadata or {},
                )
            )

        return [c for c in chunks if c.text.strip()]

    # Helpers

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and remove control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Lightweight sentence splitter (no heavy deps)."""
        # Split on '.', '!', '?' followed by whitespace or end-of-string
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


    
