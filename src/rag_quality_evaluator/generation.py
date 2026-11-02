"""Answer generation with mandatory [doc_id] citations, plus answer parsing."""

from __future__ import annotations

import re

from pydantic import BaseModel

from .retrieval import Hit
from .text import CITATION_RE, split_sentences, strip_citations

ABSTAIN_TEXT = "I cannot answer this from the provided documents."

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using only the provided documents. "
    "Every sentence of your answer must end with at least one citation of the form [doc_id], "
    "using the ids shown in the context. Do not use outside knowledge."
)

ANSWER_TEMPLATE = """TASK: answer_with_citations
Answer the question using only the context below.
Rules:
- Keep the answer short (one to three sentences).
- Put a citation like [doc_id] at the end of every sentence, using the id of the document that supports it.
- If the context does not contain the answer, reply exactly: "{abstain}"

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"Document [{h.chunk.doc_id}]:\n{h.chunk.text}" for h in hits)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return ANSWER_TEMPLATE.format(abstain=ABSTAIN_TEXT, context=format_context(hits), question=question)


_CONTEXT_BLOCK_RE = re.compile(r"Document \[(?P<doc>[^\]]+)\]:\n(?P<text>.*?)(?=\n\nDocument \[|\n\nQUESTION:)", re.S)


def parse_prompt(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    """Inverse of build_prompt: returns (question, [(doc_id, text), ...]). Used by the mock backend."""
    question = ""
    m = re.search(r"QUESTION:\s*(.*?)\nANSWER:", prompt, re.S)
    if m:
        question = m.group(1).strip()
    blocks = [(b.group("doc"), b.group("text").strip()) for b in _CONTEXT_BLOCK_RE.finditer(prompt)]
    return question, blocks


_ABSTAIN_PATTERNS = re.compile(
    r"cannot answer|can't answer|unable to answer|not (?:contain|mention|provide|include)|"
    r"no information|insufficient information|does not say|don't know|do not know|not enough information",
    re.I,
)


def is_abstention(answer: str) -> bool:
    return not answer.strip() or bool(_ABSTAIN_PATTERNS.search(answer))


class CitedSentence(BaseModel):
    text: str  # sentence with citations removed
    citations: list[str]


def parse_citations(answer: str) -> list[CitedSentence]:
    """Split an answer into sentences and attach the [doc_id] markers that appear in each."""
    out = []
    for sent in split_sentences(answer):
        cites = list(dict.fromkeys(CITATION_RE.findall(sent)))
        clean = strip_citations(sent)
        if clean:
            out.append(CitedSentence(text=clean, citations=cites))
        elif cites and out:
            out[-1].citations.extend(c for c in cites if c not in out[-1].citations)
    return out


def generate_answer(backend, question: str, hits: list[Hit], temperature: float = 0.0) -> str:
    return backend.complete(build_prompt(question, hits), system=SYSTEM_PROMPT, temperature=temperature).strip()
