from rag_quality_evaluator.citations import score_citations
from rag_quality_evaluator.claims import score_faithfulness, split_claims, verify_claim_heuristic
from rag_quality_evaluator.data import Document
from rag_quality_evaluator.generation import is_abstention, parse_citations

CONTEXT = [
    ("hb-pto", "Full-time employees accrue 18 days of paid time off (PTO) per year. Sick leave is tracked separately from PTO."),
    ("hb-expenses", "Alcohol is not reimbursable unless it is part of an approved client dinner."),
]


def test_split_claims_sentences_and_clauses():
    answer = ("Employees accrue 18 days of PTO per year, and sick leave is tracked separately [hb-pto]. "
              "Alcohol is not reimbursable; client dinners are the exception [hb-expenses].")
    claims = split_claims(answer)
    texts = [c.text for c in claims]
    assert texts == [
        "Employees accrue 18 days of PTO per year.",
        "sick leave is tracked separately.",
        "Alcohol is not reimbursable.",
        "client dinners are the exception.",
    ]
    assert claims[0].citations == ["hb-pto"] and claims[3].citations == ["hb-expenses"]
    assert claims[2].sentence_index == 1


def test_split_claims_keeps_short_fragments_together():
    claims = split_claims("Rain, snow and hail are forms of precipitation.")
    assert len(claims) == 1


def test_verify_supported():
    v = verify_claim_heuristic("Full-time employees accrue 18 days of PTO per year.", CONTEXT)
    assert v.label == "supported" and v.evidence_doc == "hb-pto"


def test_verify_contradicted_number():
    v = verify_claim_heuristic("Full-time employees accrue 23 days of PTO per year.", CONTEXT)
    assert v.label == "contradicted"
    assert "23" in v.reason


def test_verify_contradicted_negation():
    v = verify_claim_heuristic("Alcohol is reimbursable as part of an approved client dinner.", CONTEXT)
    assert v.label == "contradicted"


def test_verify_unsupported():
    v = verify_claim_heuristic("This rule was last revised in 2019 after an internal audit.", CONTEXT)
    assert v.label == "unsupported"
    v2 = verify_claim_heuristic("Full-time employees in Berlin accrue PTO per year.", CONTEXT)
    assert v2.label == "unsupported" and "berlin" in v2.reason


def test_faithfulness_fraction_and_hallucinations():
    answer = ("Full-time employees accrue 18 days of PTO per year [hb-pto]. "
              "This rule was last revised in 2019 after an internal audit [hb-pto].")
    res = score_faithfulness(answer, CONTEXT)
    assert res.faithfulness == 0.5
    assert res.hallucinated == ["This rule was last revised in 2019 after an internal audit."]
    assert score_faithfulness("", CONTEXT).faithfulness is None


def test_parse_citations():
    parsed = parse_citations("A is true [d1][d2]. B is true [d3]. C has no citation.")
    assert [p.citations for p in parsed] == [["d1", "d2"], ["d3"], []]
    assert parsed[0].text == "A is true."


def test_abstention_detection():
    assert is_abstention("I cannot answer this from the provided documents.")
    assert is_abstention("The context does not mention interns.")
    assert not is_abstention("Employees accrue 18 days [hb-pto].")


def test_citation_precision_and_recall():
    corpus = {d: Document(doc_id=d, title=d, text=t) for d, t in CONTEXT}
    answer = (
        "Full-time employees accrue 18 days of PTO per year [hb-pto]. "   # correct citation
        "Sick leave is tracked separately from PTO [hb-expenses]. "      # wrong document cited
        "Alcohol is not reimbursable."                                      # no citation
    )
    res = score_citations(answer, corpus, retrieved_doc_ids={"hb-pto"})
    assert res.n_sentences == 3
    assert res.precision == 0.5
    assert res.recall == 1 / 3
    assert res.n_uncited_sentences == 1
    wrong = [c for c in res.checks if not c.supported][0]
    assert wrong.doc_id == "hb-expenses" and not wrong.in_context


def test_citation_unknown_doc():
    res = score_citations("Something [nope].", {})
    assert res.precision == 0.0 and res.checks[0].reason == "unknown document id"
