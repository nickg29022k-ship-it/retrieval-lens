"""Retrieval and answer metrics.

The generic ranking functions (recall/precision/MRR/nDCG) take a ranked list of
ids and a set of relevant ids so they can be checked against hand-computed values.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel

from .chunking import Chunk
from .data import QAItem
from .text import content_terms, normalize_answer, split_sentences

# --------------------------------------------------------------------------- ranking metrics


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for r in ranked[:k] if r in relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    for i, r in enumerate(ranked[:k] if k else ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevances: list[float], k: int, n_relevant_total: int | None = None) -> float:
    """nDCG with gain = relevance and log2(rank + 1) discount.

    ``relevances`` are the graded relevances of the ranked list. The ideal ranking is
    the sorted list itself, or ``n_relevant_total`` items of relevance 1 when given
    (useful when relevant items exist that were not retrieved at all).
    """
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    if n_relevant_total is not None:
        ideal = [1.0] * min(n_relevant_total, k)
    else:
        ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- gold passage matching

EVIDENCE_COVERAGE = 0.7  # a chunk "contains" an evidence span if it has >= 70% of its content terms


def evidence_coverage(chunk_text: str, evidence: str) -> float:
    ev = content_terms(evidence)
    if not ev:
        return 0.0
    chunk_terms = set(content_terms(chunk_text))
    return sum(1 for t in ev if t in chunk_terms) / len(ev)


def chunk_is_gold(chunk: Chunk, item: QAItem) -> bool:
    return chunk.doc_id in item.gold_doc_ids and any(
        evidence_coverage(chunk.text, ev) >= EVIDENCE_COVERAGE for ev in item.gold_evidence
    )


class RetrievalScores(BaseModel):
    recall: float          # fraction of gold evidence spans covered by the top-k chunks
    precision: float       # fraction of top-k chunks that contain gold evidence
    mrr: float
    ndcg: float
    doc_recall: float      # fraction of gold documents among the top-k chunks' documents
    gold_retrieved: bool   # every gold evidence span is covered
    context_relevance: float


def context_relevance(question: str, chunk_texts: list[str], threshold: float = 0.4) -> float:
    """Fraction of retrieved sentences that share >= ``threshold`` of the question's content terms.

    A cheap, reference-free proxy for "how much of the context is on-topic"
    (similar in spirit to RAGAS context relevance).
    """
    q = set(content_terms(question))
    sentences = [s for t in chunk_texts for s in split_sentences(t)]
    if not q or not sentences:
        return 0.0
    hits = sum(1 for s in sentences if len(q & set(content_terms(s))) / len(q) >= threshold)
    return hits / len(sentences)


def score_retrieval(item: QAItem, retrieved: list[Chunk], all_chunks: list[Chunk], k: int) -> RetrievalScores | None:
    if not item.answerable or not item.gold_evidence:
        return None
    top = retrieved[:k]
    rels = [1.0 if chunk_is_gold(c, item) else 0.0 for c in top]
    n_gold_chunks = sum(1 for c in all_chunks if chunk_is_gold(c, item))
    covered = [
        any(c.doc_id in item.gold_doc_ids and evidence_coverage(c.text, ev) >= EVIDENCE_COVERAGE for c in top)
        for ev in item.gold_evidence
    ]
    ranked_ids = [c.chunk_id for c in top]
    gold_ids = {c.chunk_id for c in top if chunk_is_gold(c, item)}
    return RetrievalScores(
        recall=sum(covered) / len(covered),
        precision=precision_at_k(ranked_ids, gold_ids, k),
        mrr=reciprocal_rank(ranked_ids, gold_ids),
        ndcg=ndcg_at_k(rels, k, n_relevant_total=max(n_gold_chunks, 1)),
        doc_recall=recall_at_k([c.doc_id for c in top], set(item.gold_doc_ids), k),
        gold_retrieved=all(covered),
        context_relevance=context_relevance(item.question, [c.text for c in top]),
    )


# --------------------------------------------------------------------------- answer metrics


def exact_match(prediction: str, golds: list[str]) -> bool:
    p = normalize_answer(prediction)
    return any(p == normalize_answer(g) for g in golds)


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_token_recall(prediction: str, gold: str) -> float:
    """Fraction of gold answer tokens that appear in the prediction (order-insensitive)."""
    g = normalize_answer(gold).split()
    if not g:
        return 0.0
    p = set(normalize_answer(prediction).split())
    return sum(1 for t in g if t in p) / len(g)


def contains_gold(prediction: str, golds: list[str]) -> bool:
    p = f" {normalize_answer(prediction)} "
    return any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g))


class AnswerScores(BaseModel):
    exact_match: bool
    f1: float
    contains_gold: bool
    gold_recall: float
    correct: bool                 # heuristic correctness (or LLM judge when enabled)
    judge_score: float | None = None


def score_answer(prediction: str, golds: list[str]) -> AnswerScores:
    em = exact_match(prediction, golds)
    f1 = max((token_f1(prediction, g) for g in golds), default=0.0)
    cont = contains_gold(prediction, golds)
    grec = max((gold_token_recall(prediction, g) for g in golds), default=0.0)
    return AnswerScores(
        exact_match=em, f1=f1, contains_gold=cont, gold_recall=grec,
        correct=em or cont or grec >= 1.0 or f1 >= 0.6,
    )


JUDGE_PROMPT = """TASK: judge_answer
You are grading a question-answering system. Compare the PREDICTED answer with the GOLD answers.
Reply with "SCORE: 1" if the prediction gives the same answer as a gold answer (extra correct detail is fine),
otherwise reply with "SCORE: 0".

QUESTION: {question}
GOLD: {gold}
PREDICTED: {prediction}
"""


def llm_judge(backend, question: str, prediction: str, golds: list[str]) -> float:
    raw = backend.complete(JUDGE_PROMPT.format(question=question, gold=" | ".join(golds), prediction=prediction))
    m = re.search(r"SCORE:\s*([01](?:\.\d+)?)", raw)
    if m:
        return float(m.group(1))
    return 1.0 if raw.strip().startswith("1") else 0.0
