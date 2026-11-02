"""Retrievers over chunks: BM25, TF-IDF cosine and dense embeddings."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .text import content_terms


class Hit(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    name = "base"

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        self.chunks = chunks

    def scores(self, query: str) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.scores(query)
        # stable sort: ties broken by original chunk order
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1) for r, i in enumerate(order)]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.bm25 = BM25Okapi([content_terms(c.text) or ["<empty>"] for c in chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(content_terms(query)), dtype=float)


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, chunks: list[Chunk], ngram_range: tuple[int, int] = (1, 2)):
        super().__init__(chunks)
        self.vectorizer = TfidfVectorizer(
            analyzer=lambda text: _ngrams(content_terms(text), ngram_range), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        # rows are L2-normalised by default, so the dot product is cosine similarity
        return np.asarray((self.matrix @ q.T).toarray()).ravel()


def _ngrams(terms: list[str], ngram_range: tuple[int, int]) -> list[str]:
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        out.extend(" ".join(terms[i : i + n]) for i in range(len(terms) - n + 1))
    return out


class EmbeddingRetriever(Retriever):
    """Dense retriever. ``embed_fn`` maps a list of texts to a list of vectors."""

    name = "embed"

    def __init__(self, chunks: list[Chunk], embed_fn: Callable[[list[str]], list[list[float]]], batch_size: int = 32):
        super().__init__(chunks)
        self.embed_fn = embed_fn
        vectors = []
        for i in range(0, len(chunks), batch_size):
            vectors.extend(embed_fn([c.text for c in chunks[i : i + batch_size]]))
        self.matrix = _normalise(np.asarray(vectors, dtype=float))

    def scores(self, query: str) -> np.ndarray:
        q = _normalise(np.asarray(self.embed_fn([query]), dtype=float))[0]
        return self.matrix @ q


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


RETRIEVERS = ("bm25", "tfidf", "embed")


def build_retriever(name: str, chunks: list[Chunk], embed_backend=None) -> Retriever:
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "tfidf":
        return TfidfRetriever(chunks)
    if name == "embed":
        if embed_backend is None or not hasattr(embed_backend, "embed"):
            raise ValueError("the embed retriever needs a backend with an embed() method (ollama, openai or mock)")
        return EmbeddingRetriever(chunks, embed_backend.embed)
    raise ValueError(f"unknown retriever {name!r}; choose from {', '.join(RETRIEVERS)}")
