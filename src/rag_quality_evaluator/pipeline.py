"""Run one RAG configuration over the QA set and score every question."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from .attribution import CATEGORIES, Attribution, attribute_failure
from .chunking import Chunk, chunk_corpus
from .citations import CitationResult, score_citations
from .claims import FaithfulnessResult, score_faithfulness
from .data import Document, QAItem
from .generation import generate_answer, is_abstention
from .metrics import AnswerScores, RetrievalScores, llm_judge, score_answer, score_retrieval
from .retrieval import build_retriever


class RunConfig(BaseModel):
    chunker: str
    retriever: str
    k: int
    chunk_params: dict = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.chunker}/{self.retriever}/k={self.k}"


class EvalOptions(BaseModel):
    llm_judge: bool = False      # use the backend as an answer-accuracy judge
    llm_claims: bool = False     # use the backend to split answers into claims
    llm_verify: bool = False     # use the backend to verify claims


class QuestionResult(BaseModel):
    config: str
    qid: str
    question: str
    answerable: bool
    gold_answers: list[str]
    retrieved: list[str]
    retrieved_docs: list[str]
    answer: str
    abstained: bool
    retrieval: RetrievalScores | None
    answer_scores: AnswerScores | None
    faithfulness: FaithfulnessResult
    citations: CitationResult
    attribution: Attribution


def evaluate_question(
    item: QAItem,
    config: RunConfig,
    retriever,
    chunks: list[Chunk],
    corpus: dict[str, Document],
    backend,
    options: EvalOptions,
) -> QuestionResult:
    hits = retriever.retrieve(item.question, config.k)
    retrieved_chunks = [h.chunk for h in hits]
    answer = generate_answer(backend, item.question, hits)
    abstained = is_abstention(answer)

    retrieval = score_retrieval(item, retrieved_chunks, chunks, config.k)
    answer_scores = None
    if item.answerable:
        answer_scores = score_answer(answer, item.answers) if not abstained else score_answer("", item.answers)
        if options.llm_judge:
            judge = 0.0 if abstained else llm_judge(backend, item.question, answer, item.answers)
            answer_scores.judge_score = judge
            answer_scores.correct = judge >= 0.5

    passages = [(c.doc_id, c.text) for c in retrieved_chunks]
    if abstained:
        faith = FaithfulnessResult(claims=[], faithfulness=None, hallucinated=[])
        cites = CitationResult(checks=[], n_sentences=0, n_sentences_supported=0, n_uncited_sentences=0, precision=None, recall=None)
    else:
        faith = score_faithfulness(answer, passages, backend, llm_split=options.llm_claims, llm_verify=options.llm_verify)
        cites = score_citations(answer, corpus, {c.doc_id for c in retrieved_chunks})

    attribution = attribute_failure(
        answerable=item.answerable,
        abstained=abstained,
        correct=bool(answer_scores and answer_scores.correct),
        gold_retrieved=bool(retrieval and retrieval.gold_retrieved),
        n_hallucinated=len(faith.hallucinated),
    )
    return QuestionResult(
        config=config.label,
        qid=item.id,
        question=item.question,
        answerable=item.answerable,
        gold_answers=item.answers,
        retrieved=[c.chunk_id for c in retrieved_chunks],
        retrieved_docs=list(dict.fromkeys(c.doc_id for c in retrieved_chunks)),
        answer=answer,
        abstained=abstained,
        retrieval=retrieval,
        answer_scores=answer_scores,
        faithfulness=faith,
        citations=cites,
        attribution=attribution,
    )


def run_config(
    config: RunConfig,
    corpus: dict[str, Document],
    qa: list[QAItem],
    backend,
    options: EvalOptions | None = None,
    embed_backend=None,
    _cache: dict | None = None,
) -> list[QuestionResult]:
    """Evaluate one configuration. ``_cache`` lets the grid reuse chunkings and indexes across k."""
    options = options or EvalOptions()
    cache = _cache if _cache is not None else {}
    chunk_key = (config.chunker, tuple(sorted(config.chunk_params.items())))
    if chunk_key not in cache:
        cache[chunk_key] = chunk_corpus(corpus.values(), config.chunker, **config.chunk_params)
    chunks = cache[chunk_key]
    index_key = (chunk_key, config.retriever)
    if index_key not in cache:
        cache[index_key] = build_retriever(config.retriever, chunks, embed_backend=embed_backend or backend)
    retriever = cache[index_key]
    return [evaluate_question(item, config, retriever, chunks, corpus, backend, options) for item in qa]


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(config: RunConfig, results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    counts = Counter(r.attribution.category for r in results)
    n_claims = sum(len(r.faithfulness.claims) for r in results)
    n_contradicted = sum(1 for r in results for c in r.faithfulness.claims if c.label == "contradicted")
    n_unsupported = sum(1 for r in results for c in r.faithfulness.claims if c.label == "unsupported")
    row = {
        "config": config.label,
        "chunker": config.chunker,
        "retriever": config.retriever,
        "k": config.k,
        "n_questions": len(results),
        "recall@k": _mean(r.retrieval.recall for r in answerable if r.retrieval),
        "precision@k": _mean(r.retrieval.precision for r in answerable if r.retrieval),
        "mrr": _mean(r.retrieval.mrr for r in answerable if r.retrieval),
        "ndcg@k": _mean(r.retrieval.ndcg for r in answerable if r.retrieval),
        "doc_recall@k": _mean(r.retrieval.doc_recall for r in answerable if r.retrieval),
        "context_relevance": _mean(r.retrieval.context_relevance for r in answerable if r.retrieval),
        "exact_match": _mean(r.answer_scores.exact_match for r in answerable if r.answer_scores),
        "token_f1": _mean(r.answer_scores.f1 for r in answerable if r.answer_scores),
        "accuracy": _mean(r.answer_scores.correct for r in answerable if r.answer_scores),
        "judge_accuracy": _mean(r.answer_scores.judge_score for r in answerable if r.answer_scores),
        "faithfulness": _mean(r.faithfulness.faithfulness for r in results),
        "n_claims": n_claims,
        "n_unsupported_claims": n_unsupported,
        "n_contradicted_claims": n_contradicted,
        "citation_precision": _mean(r.citations.precision for r in results),
        "citation_recall": _mean(r.citations.recall for r in results),
        "abstention_accuracy": _mean((r.attribution.category == "correct_abstention") for r in unanswerable),
    }
    for cat in CATEGORIES:
        row[cat] = counts.get(cat, 0)
    return row
