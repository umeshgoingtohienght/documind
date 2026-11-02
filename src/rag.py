"""Retrieval-Augmented Generation pipeline that ties everything together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Callable, List, Tuple

from .loader import Chunk, chunk_pages, extract_pages
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are DocuMind, a precise assistant that answers questions using ONLY the "
    "provided document excerpts. Cite sources inline like [source, p.N]. If the answer "
    "is not in the excerpts, say you could not find it in the documents. Be concise."
)


@dataclass
class Answer:
    text: str
    sources: List[Tuple[Chunk, float]]


def build_prompt(question: str, contexts: List[Tuple[Chunk, float]], history: List[dict] | None = None) -> str:
    context_block = "\n\n".join(
        f"[{c.source}, p.{c.page}]\n{c.text}" for c, _ in contexts
    ) or "(no relevant excerpts found)"
    history_block = ""
    if history:
        turns = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        history_block = "Previous conversation:\n" + "\n".join(turns) + "\n\n"
    return (
        f"{history_block}Document excerpts:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


class RAGPipeline:
    def __init__(self, embed_fn: Callable, generate_fn: Callable,
                 chunk_size: int = 1000, overlap: int = 200, top_k: int = 4):
        self.store = VectorStore(embed_fn)
        self.generate_fn = generate_fn
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def ingest(self, file: BinaryIO, name: str) -> int:
        """Load a PDF, chunk it and add it to the index. Returns number of chunks."""
        pages = extract_pages(file, name)
        chunks = chunk_pages(pages, self.chunk_size, self.overlap)
        self.store.add(chunks)
        return len(chunks)

    def ask(self, question: str, history: List[dict] | None = None) -> Answer:
        contexts = self.store.search(question, k=self.top_k)
        prompt = build_prompt(question, contexts, history)
        text = self.generate_fn(prompt, system=SYSTEM_PROMPT)
        return Answer(text=text, sources=contexts)
