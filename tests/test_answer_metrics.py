import pytest

from rag_quality_evaluator.metrics import contains_gold, exact_match, gold_token_recall, score_answer, token_f1
from rag_quality_evaluator.text import normalize_answer


def test_normalize_answer():
    assert normalize_answer("The 1,500 Euros! [hb-benefits]") == "1500 euros"
    assert normalize_answer("4.5 hours.") == "4.5 hours"


def test_exact_match_normalized():
    assert exact_match("the Bay of Fundy.", ["Bay of Fundy"])
    assert not exact_match("Fundy", ["Bay of Fundy"])


def test_token_f1_hand_computed():
    # pred tokens [cat, sat], gold [cat, sat, on, mat]: P=1, R=0.5 -> F1=2/3
    assert token_f1("the cat sat", "cat sat on the mat") == pytest.approx(2 / 3)
    assert token_f1("dog", "cat") == 0.0
    assert token_f1("", "") == 1.0


def test_contains_and_recall():
    pred = "Employees accrue 18 days of PTO per year [hb-pto]."
    assert contains_gold(pred, ["18 days"])
    assert not contains_gold(pred, ["8 days"])  # token boundary, not substring of "18"
    assert gold_token_recall("In 1973 UNIX was rewritten, by Dennis Ritchie", "1973, Dennis Ritchie") == 1.0


def test_score_answer_correctness():
    assert score_answer("Employees accrue 18 days of PTO per year.", ["18 days"]).correct
    assert not score_answer("Employees accrue 23 days of PTO per year.", ["18 days", "18"]).correct
    s = score_answer("18 days", ["18 days"])
    assert s.exact_match and s.f1 == 1.0
