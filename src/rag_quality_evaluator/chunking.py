"""Chunking strategies.

Three strategies are implemented:

* ``fixed``     - sliding window of N words with M words of overlap.
* ``sentence``  - pack whole sentences into chunks of at most N words.
* ``heading``   - one chunk per markdown section (heading + body); long sections
                  fall back to paragraph splits.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from pydantic import BaseModel

from .data import Document
from .text import split_sentences


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    position: int


def _strip_markdown(text: str) -> str:
    return re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)


def _make(doc: Document, pieces: Iterable[str]) -> list[Chunk]:
    out = []
    for piece in pieces:
        piece = piece.strip()
        if piece:
            out.append(Chunk(chunk_id=f"{doc.doc_id}#{len(out)}", doc_id=doc.doc_id, text=piece, position=len(out)))
    return out


def fixed_chunker(doc: Document, size: int = 60, overlap: int = 15) -> list[Chunk]:
    """Sliding window over whitespace tokens."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = _strip_markdown(doc.text).split()
    step = size - overlap
    pieces = []
    for start in range(0, max(len(words), 1), step):
        window = words[start : start + size]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + size >= len(words):
            break
    return _make(doc, pieces)


def sentence_chunker(doc: Document, max_words: int = 50, overlap_sentences: int = 0) -> list[Chunk]:
    """Greedily pack whole sentences; a single long sentence becomes its own chunk."""
    sentences = [s for s in split_sentences(doc.text) if not s.startswith("#")]
    pieces: list[str] = []
    current: list[str] = []
    for sent in sentences:
        n_current = sum(len(s.split()) for s in current)
        if current and n_current + len(sent.split()) > max_words:
            pieces.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)
    if current:
        pieces.append(" ".join(current))
    return _make(doc, pieces)


def heading_chunker(doc: Document, max_words: int = 120) -> list[Chunk]:
    """One chunk per markdown section, prefixed with the doc title and section heading."""
    title = doc.title
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    for line in doc.text.splitlines():
        m = re.match(r"^(#+)\s*(.*)", line)
        if m:
            if body:
                sections.append((heading, "\n".join(body)))
            level, heading_text = len(m.group(1)), m.group(2).strip()
            heading = "" if level == 1 else heading_text
            body = []
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body)))

    pieces = []
    for heading, text in sections:
        if not text.strip():
            continue
        prefix = f"{title} - {heading}: " if heading else f"{title}: "
        paragraphs = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", text) if p.strip()]
        if sum(len(p.split()) for p in paragraphs) <= max_words:
            pieces.append(prefix + " ".join(paragraphs))
        else:
            pieces.extend(prefix + p for p in paragraphs)
    return _make(doc, pieces)


CHUNKERS: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": fixed_chunker,
    "sentence": sentence_chunker,
    "heading": heading_chunker,
}


def chunk_corpus(docs: Iterable[Document], strategy: str, **params) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {sorted(CHUNKERS)}")
    fn = CHUNKERS[strategy]
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(fn(doc, **params))
    return chunks
