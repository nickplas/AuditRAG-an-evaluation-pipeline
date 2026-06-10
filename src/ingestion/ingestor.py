"""
ingestion/ingestor.py — Document loading, conversion to Markdown, and chunking.
 
All file types are normalised to Markdown first via MarkItDown, then chunked.
This gives two advantages:
  1. Consistent text quality across formats — MarkItDown preserves headings,
     tables, and lists rather than dumping raw characters like PyMuPDF does.
  2. Token efficiency — Markdown is ~30-50% more compact than raw PDF text
     because it removes repeated whitespace, page headers/footers, and
     table-of-contents dots.
 
Supported formats (via MarkItDown)
-----------------------------------
  .pdf   .docx  .pptx  .xlsx  .xls
  .html  .htm   .csv   .json  .xml
  .md    .txt   .zip   .epub
 
The raw .txt and .md loaders are kept as fast-path fallbacks for plain text
that doesn't need conversion.
 
Chunking strategy
-----------------
Markdown-aware: splits preferentially on heading boundaries (# / ## / ###)
so each chunk is a semantically coherent section rather than an arbitrary
window of characters. Falls back to sentence-boundary splitting inside
sections that are too large.
"""
 
from __future__ import annotations
 
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
 
from loguru import logger
 
try:
    from markitdown import MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:
    _MARKITDOWN_AVAILABLE = False
    logger.warning(
        "markitdown not installed — PDF/Office/HTML conversion disabled. "
        "Run: pip install 'markitdown[all]'"
    )
 
from src.config import settings
 
 
# ── Supported extensions ──────────────────────────────────────────────────────
 
# Handled natively (no conversion needed)
_PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}
 
# Converted to Markdown via MarkItDown before chunking
_MARKITDOWN_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm",
    ".csv", ".json", ".xml", ".epub", ".zip",
}
 
ALL_SUPPORTED = _PLAIN_TEXT_EXTENSIONS | _MARKITDOWN_EXTENSIONS
 
 
# ── Data model ────────────────────────────────────────────────────────────────
 
@dataclass
class Chunk:
    """A single text chunk with full provenance."""
    text: str
    doc_id: str
    chunk_index: int
    source: str                     # original file path
    page: int | None = None         # best-effort page number
    section: str | None = None      # heading of the section this chunk came from
    metadata: dict = field(default_factory=dict)
 
    @property
    def id(self) -> str:
        return f"{self.doc_id}__chunk_{self.chunk_index}"
 
 
# ── Converter ─────────────────────────────────────────────────────────────────
 
class MarkdownConverter:
    """
    Converts any supported file to a Markdown string using MarkItDown.
    Falls back to plain-text reading for .txt and .md files.
    """
 
    def __init__(self):
        self._md = MarkItDown(enable_plugins=False) if _MARKITDOWN_AVAILABLE else None
 
    def convert(self, path: Path) -> str:
        """Return the Markdown representation of *path*."""
        suffix = path.suffix.lower()
 
        # Fast path: plain text / markdown — just read
        if suffix in _PLAIN_TEXT_EXTENSIONS:
            return path.read_text(encoding="utf-8", errors="replace")
 
        # MarkItDown path
        if suffix in _MARKITDOWN_EXTENSIONS:
            if not _MARKITDOWN_AVAILABLE:
                raise RuntimeError(
                    f"markitdown is required to convert {suffix} files. "
                    f"Run: pip install 'markitdown[all]'"
                )
            result = self._md.convert(str(path))
            return result.text_content or ""
 
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {sorted(ALL_SUPPORTED)}"
        )
 
    def convert_text(self, raw: str) -> str:
        """Pass-through for already-converted text."""
        return raw
 
 
# ── Splitter ──────────────────────────────────────────────────────────────────
 
class MarkdownSplitter:
    """
    Splits a Markdown document into chunks that respect section boundaries.
 
    Strategy
    --------
    1. Split the document on Markdown headings (# / ## / ###).
       Each heading starts a new section.
    2. If a section is larger than chunk_size, subdivide it using
       sentence-boundary splitting with overlap.
    3. If a section is smaller than min_chunk_size, merge it with the next.
 
    This produces chunks that are semantically coherent (one topic per chunk)
    which improves both retrieval precision and LLM answer quality.
    """
 
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        min_chunk_size: int = 80,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
 
    def split(
        self,
        markdown: str,
        doc_id: str,
        source: str,
        base_index: int = 0,
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        markdown = self._clean(markdown)
        sections = self._split_by_headings(markdown)
        chunks: list[Chunk] = []
        chunk_idx = base_index
 
        for section_title, section_text in sections:
            if not section_text.strip():
                continue
 
            if len(section_text) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(Chunk(
                    text=section_text.strip(),
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    source=source,
                    section=section_title,
                    metadata=extra_metadata or {},
                ))
                chunk_idx += 1
            else:
                # Section too large — split by sentences with overlap
                sub_chunks = self._sentence_split(
                    section_text,
                    doc_id=doc_id,
                    source=source,
                    section=section_title,
                    base_index=chunk_idx,
                    extra_metadata=extra_metadata,
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
 
        # Merge tiny trailing chunks into the previous one
        return self._merge_small(chunks)
 
    # ── Heading-based section split ───────────────────────────────────────
 
    def _split_by_headings(self, text: str) -> list[tuple[str | None, str]]:
        """
        Returns [(heading_title, section_text), ...].
        Text before the first heading is returned with title=None.
        """
        sections: list[tuple[str | None, str]] = []
        last_end = 0
        last_title: str | None = None
 
        for match in self._heading_re.finditer(text):
            # Emit the text accumulated since the last heading
            section_text = text[last_end:match.start()].strip()
            if section_text:
                sections.append((last_title, section_text))
 
            last_title = match.group(2).strip()
            last_end = match.end()
 
        # Emit remainder after last heading
        remainder = text[last_end:].strip()
        if remainder:
            sections.append((last_title, remainder))
 
        return sections
 
    # ── Sentence-level split (fallback for large sections) ────────────────
 
    def _sentence_split(
        self,
        text: str,
        *,
        doc_id: str,
        source: str,
        section: str | None,
        base_index: int,
        extra_metadata: dict | None,
    ) -> list[Chunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[Chunk] = []
        window: list[str] = []
        window_len = 0
        chunk_idx = base_index
 
        for sentence in sentences:
            s_len = len(sentence)
            if window_len + s_len > self.chunk_size and window:
                chunks.append(Chunk(
                    text=" ".join(window).strip(),
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    source=source,
                    section=section,
                    metadata=extra_metadata or {},
                ))
                chunk_idx += 1
                # Carry overlap forward
                overlap = " ".join(window)[-self.chunk_overlap:]
                window = [overlap]
                window_len = len(overlap)
            window.append(sentence)
            window_len += s_len
 
        if window:
            chunks.append(Chunk(
                text=" ".join(window).strip(),
                doc_id=doc_id,
                chunk_index=chunk_idx,
                source=source,
                section=section,
                metadata=extra_metadata or {},
            ))
        return chunks
 
    # ── Post-processing ───────────────────────────────────────────────────
 
    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks below min_chunk_size into their predecessor."""
        if not chunks:
            return chunks
        merged: list[Chunk] = [chunks[0]]
        for c in chunks[1:]:
            if len(c.text) < self.min_chunk_size and merged:
                merged[-1] = Chunk(
                    text=merged[-1].text + "\n\n" + c.text,
                    doc_id=merged[-1].doc_id,
                    chunk_index=merged[-1].chunk_index,
                    source=merged[-1].source,
                    section=merged[-1].section,
                    metadata=merged[-1].metadata,
                )
            else:
                merged.append(c)
        return merged
 
    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
 
 
# ── Main ingestor ─────────────────────────────────────────────────────────────
 
class DocumentIngestor:
    """
    Converts documents to Markdown and splits them into chunks.
 
    Usage
    -----
    >>> ingestor = DocumentIngestor()
    >>> chunks = ingestor.ingest_directory(Path("data/"))
    >>> chunks = ingestor.ingest_file(Path("report.pdf"))
    >>> chunks = ingestor.ingest_texts(["some raw text..."], source="inline")
 
    The `section` field on each Chunk carries the Markdown heading it came
    from — useful for display in the audit panel.
    """
 
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.converter = MarkdownConverter()
        self.splitter = MarkdownSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
 
    # ── Public API ────────────────────────────────────────────────────────
 
    def ingest_directory(self, directory: Path) -> list[Chunk]:
        """Recursively ingest all supported files under *directory*."""
        chunks: list[Chunk] = []
        found = sorted(
            p for p in directory.rglob("*")
            if p.suffix.lower() in ALL_SUPPORTED
        )
        logger.info(f"Found {len(found)} files in {directory}")
        for path in found:
            try:
                file_chunks = self.ingest_file(path)
                chunks.extend(file_chunks)
                logger.info(f"  {path.name} → {len(file_chunks)} chunks")
            except Exception as exc:
                logger.error(f"  {path.name} FAILED: {exc}")
        logger.info(f"Total: {len(chunks)} chunks from {len(found)} files")
        return chunks
 
    def ingest_file(self, path: Path) -> list[Chunk]:
        """Convert and chunk a single file."""
        suffix = path.suffix.lower()
        if suffix not in ALL_SUPPORTED:
            raise ValueError(
                f"Unsupported: {suffix}. Supported: {sorted(ALL_SUPPORTED)}"
            )
 
        markdown = self.converter.convert(path)
        if not markdown.strip():
            logger.warning(f"Empty content from {path.name} — skipped.")
            return []
 
        return self.splitter.split(
            markdown,
            doc_id=path.stem,
            source=str(path),
        )
 
    def ingest_texts(
        self, texts: list[str], source: str = "inline"
    ) -> list[Chunk]:
        """Ingest raw strings directly (for tests and demos)."""
        all_chunks: list[Chunk] = []
        for i, text in enumerate(texts):
            doc_id = f"{source}__{i}"
            chunks = self.splitter.split(text, doc_id=doc_id, source=source)
            all_chunks.extend(chunks)
        return [c for c in all_chunks if c.text.strip()]