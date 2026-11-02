import pytest

from rag_quality_evaluator.attribution import attribute_failure


@pytest.mark.parametrize(
    "answerable,abstained,correct,gold_retrieved,n_hall,expected,subtype",
    [
        (False, True, False, False, 0, "correct_abstention", ""),
        (False, False, False, False, 1, "missed_abstention", "hallucinated_answer"),
        (True, True, False, False, 0, "retrieval_failure", "abstained"),
        (True, True, False, True, 0, "incorrect_abstention", ""),
        (True, False, True, True, 0, "correct", ""),
        (True, False, True, False, 0, "correct", ""),          # lucky: right without gold passage
        (True, False, True, True, 1, "generation_failure", "unfaithful"),
        (True, False, False, False, 0, "retrieval_failure", "wrong_answer"),
        (True, False, False, True, 0, "generation_failure", "wrong_answer"),
        (True, False, False, True, 2, "generation_failure", "wrong_answer"),
    ],
)
def test_attribution_table(answerable, abstained, correct, gold_retrieved, n_hall, expected, subtype):
    a = attribute_failure(answerable, abstained, correct, gold_retrieved, n_hall)
    assert a.category == expected
    assert a.subtype == subtype
