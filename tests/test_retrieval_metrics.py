import math

import pytest

from rag_quality_evaluator.chunking import Chunk
from rag_quality_evaluator.data import QAItem
from rag_quality_evaluator.metrics import (
    context_relevance,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "x"}  # "x" is relevant but never retrieved


def test_recall_at_k_hand_computed():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_precision_at_k_hand_computed():
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 2)
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    # precision divides by k even when fewer items were returned
    assert precision_at_k(["b"], RELEVANT, 4) == pytest.approx(1 / 4)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(RANKED, {"e"}) == pytest.approx(0.2)
    assert reciprocal_rank(RANKED, {"z"}) == 0.0
    assert reciprocal_rank(RANKED, {"e"}, k=3) == 0.0


def test_ndcg_binary_hand_computed():
    # relevances [0,1,0,0,1]; 3 relevant items exist in total
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k([0, 1, 0, 0, 1], 5, n_relevant_total=3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([0, 1, 0, 0, 1], 5, n_relevant_total=3) == pytest.approx(0.4776, abs=1e-4)


def test_ndcg_graded_textbook_example():
    # The standard worked example: DCG6 = 6.861, IDCG6 = 7.141, nDCG = 0.961
    assert ndcg_at_k([3, 2, 3, 0, 1, 2], 6) == pytest.approx(0.961, abs=1e-3)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k([1, 1, 0], 3, n_relevant_total=2) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], 3, n_relevant_total=2) == 0.0


def test_recall_requires_relevant():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 3)


def _chunk(cid, doc, text):
    return Chunk(chunk_id=cid, doc_id=doc, text=text, position=0)


def test_score_retrieval_against_gold_evidence():
    item = QAItem(
        id="t", question="How many days of leave do staff get?", answers=["12 days"],
        gold_doc_ids=["toy"], gold_evidence=["Staff get 12 days of leave per year."],
    )
    gold = _chunk("toy#0", "toy", "Staff get 12 days of leave per year. Leave must be booked early.")
    other = _chunk("toy#1", "toy", "Laptops are replaced every 4 years.")
    wrong_doc = _chunk("x#0", "x", "Staff get 12 days of leave per year.")  # same text, wrong doc: not gold
    s = score_retrieval(item, [other, gold, wrong_doc], [gold, other, wrong_doc], k=3)
    assert s.recall == 1.0
    assert s.precision == pytest.approx(1 / 3)
    assert s.mrr == pytest.approx(0.5)
    assert s.ndcg == pytest.approx((1 / math.log2(3)) / 1.0)
    assert s.gold_retrieved
    s1 = score_retrieval(item, [other], [gold, other], k=1)
    assert s1.recall == 0.0 and not s1.gold_retrieved and s1.mrr == 0.0


def test_score_retrieval_none_for_unanswerable():
    item = QAItem(id="u", question="?", answerable=False)
    assert score_retrieval(item, [], [], k=3) is None


def test_context_relevance():
    q = "How many days of leave do staff get?"
    # 2 sentences, only the first is on-topic
    assert context_relevance(q, ["Staff get 12 days of leave. Laptops are replaced every 4 years."]) == pytest.approx(0.5)
    assert context_relevance(q, []) == 0.0
