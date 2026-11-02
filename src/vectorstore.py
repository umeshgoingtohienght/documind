"""A small FAISS-backed vector store with cosine similarity search."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import faiss
import numpy as np

from .loader import Chunk

EmbedFn = Callable[[Sequence[str]], List[List[float]]]


class VectorStore:
    """Stores chunk embeddings in a FAISS index (inner product on L2-normalised vectors)."""

    def __init__(self, embed_fn: EmbedFn):
        self.embed_fn = embed_fn
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        if not chunks:
            return
        embeddings: List[List[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = [c.text for c in chunks[i : i + batch_size]]
            embeddings.extend(self.embed_fn(batch))
        matrix = self._normalise(np.array(embeddings))
        if self.index is None:
            self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        query_vec = self._normalise(np.array(self.embed_fn([query])))
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vec, k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- persistence -------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str | Path, embed_fn: EmbedFn) -> "VectorStore":
        directory = Path(directory)
        store = cls(embed_fn)
        store.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
