"""LLM backend abstraction.

``get_backend(name)`` returns an object with
``complete(prompt, system=None, temperature=0.0) -> str``. Backends that can
embed text also expose ``embed(texts) -> list[list[float]]``.

* ``mock``      - deterministic and offline (default). See ``mock.py``.
* ``ollama``    - HTTP to http://localhost:11434 (OLLAMA_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_HOST).
* ``anthropic`` - Messages API over HTTPS (ANTHROPIC_API_KEY, ANTHROPIC_MODEL).
* ``openai``    - Chat Completions API over HTTPS (OPENAI_API_KEY, OPENAI_MODEL).
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import httpx


class BackendUnavailableError(RuntimeError):
    """Raised when a network backend cannot be reached or is not configured."""


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(
                f"Could not reach Ollama at {self.host} ({exc.__class__.__name__}). "
                "Start it yourself with `ollama serve` and pull the model, or use --backend mock."
            ) from exc
        if resp.status_code != 200:
            raise BackendUnavailableError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        return self._post("/api/generate", payload).get("response", "").strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": self.embed_model, "input": texts})
        if "embeddings" not in data:
            raise BackendUnavailableError(f"Unexpected Ollama embed response: {str(data)[:200]}")
        return data["embeddings"]


class AnthropicBackend:
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, timeout: float = 120.0):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise BackendUnavailableError("ANTHROPIC_API_KEY is not set; export it or use --backend mock.")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.timeout = timeout

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        payload: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        try:
            resp = httpx.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"Could not reach the Anthropic API: {exc}") from exc
        if resp.status_code != 200:
            raise BackendUnavailableError(f"Anthropic API returned HTTP {resp.status_code}: {resp.text[:300]}")
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str | None = None, timeout: float = 120.0):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise BackendUnavailableError("OPENAI_API_KEY is not set; export it or use --backend mock.")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=payload, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"Could not reach the OpenAI API: {exc}") from exc
        if resp.status_code != 200:
            raise BackendUnavailableError(f"OpenAI API returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        data = self._post("/chat/completions", {"model": self.model, "messages": messages, "temperature": temperature})
        return data["choices"][0]["message"]["content"].strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        data = self._post("/embeddings", {"model": model, "input": texts})
        return [row["embedding"] for row in data["data"]]


BACKENDS = ("mock", "ollama", "anthropic", "openai")


def get_backend(name: str = "mock", **kwargs) -> LLMBackend:
    name = (name or "mock").lower()
    if name == "mock":
        from .mock import MockBackend

        return MockBackend(**kwargs)
    if name == "ollama":
        return OllamaBackend(**kwargs)
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    if name == "openai":
        return OpenAIBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
