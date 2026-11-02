"""Unit tests that run offline (no API key needed) using a fake embedder."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loader import Chunk, split_text  # noqa: E402
from src.rag import RAGPipeline, build_prompt  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402


def fake_embed(texts):
    """Deterministic bag-of-words embedding so similar texts get similar vectors."""
    dim = 64
    out = []
    for t in texts:
        v = np.zeros(dim)
        for word in t.lower().split():
            idx = int(hashlib.md5(word.strip(".,?!").encode()).hexdigest(), 16) % dim
            v[idx] += 1
        out.append(v.tolist())
    return out


def test_split_text_respects_size_and_overlap():
    text = "This is a sentence. " * 200
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_split_text_short_input():
    assert split_text("hello world", 100, 10) == ["hello world"]
    assert split_text("", 100, 10) == []


def test_split_text_invalid_args():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_vectorstore_returns_most_relevant_chunk():
    store = VectorStore(fake_embed)
    store.add([
        Chunk("Python is a popular programming language", "a.pdf", 1),
        Chunk("The Taj Mahal is located in Agra India", "b.pdf", 2),
        Chunk("Bananas are rich in potassium", "c.pdf", 3),
    ])
    results = store.search("Where is the Taj Mahal located?", k=1)
    assert results[0][0].source == "b.pdf"


def test_build_prompt_contains_citations():
    prompt = build_prompt("Q?", [(Chunk("ctx", "doc.pdf", 7), 0.9)])
    assert "[doc.pdf, p.7]" in prompt and "Q?" in prompt


def test_pipeline_ask_uses_generator():
    pipe = RAGPipeline(fake_embed, lambda prompt, system=None: "mock answer")
    pipe.store.add([Chunk("Kota is a city in Rajasthan", "geo.pdf", 1)])
    ans = pipe.ask("Where is Kota?")
    assert ans.text == "mock answer"
    assert ans.sources[0][0].source == "geo.pdf"
