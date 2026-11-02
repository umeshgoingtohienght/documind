"""Thin wrapper around the Google Gemini API (google-genai SDK)."""
from __future__ import annotations

import os
from typing import List, Sequence

from google import genai
from google.genai import types

DEFAULT_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
DEFAULT_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")


class GeminiClient:
    def __init__(self, api_key: str | None = None,
                 chat_model: str = DEFAULT_CHAT_MODEL,
                 embed_model: str = DEFAULT_EMBED_MODEL):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        result = self.client.models.embed_content(model=self.embed_model, contents=list(texts))
        return [e.values for e in result.embeddings]

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
        )
        response = self.client.models.generate_content(
            model=self.chat_model, contents=prompt, config=config
        )
        return (response.text or "").strip()
