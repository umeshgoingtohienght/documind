"""PDF loading and text chunking utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import BinaryIO, List

from pypdf import PdfReader


@dataclass
class Chunk:
    """A piece of text with metadata pointing back to its source."""

    text: str
    source: str
    page: int
    metadata: dict = field(default_factory=dict)


def extract_pages(file: BinaryIO, source_name: str) -> List[Chunk]:
    """Extract text page-by-page from a PDF file object."""
    reader = PdfReader(file)
    pages: List[Chunk] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Chunk(text=text, source=source_name, page=idx))
    return pages


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries.

    Args:
        text: Raw text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks (keeps context).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(text.split())  # normalise whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to break at a sentence end, then at a space
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut < chunk_size * 0.5:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: List[Chunk], chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """Turn page-level chunks into smaller retrieval-sized chunks."""
    result: List[Chunk] = []
    for page in pages:
        for i, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            result.append(
                Chunk(text=piece, source=page.source, page=page.page, metadata={"part": i})
            )
    return result
