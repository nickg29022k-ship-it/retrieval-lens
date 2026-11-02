import pytest

from rag_quality_evaluator.chunking import chunk_corpus
from rag_quality_evaluator.claims import score_faithfulness
from rag_quality_evaluator.generation import ABSTAIN_TEXT, build_prompt, parse_citations, parse_prompt
from rag_quality_evaluator.llm import BackendUnavailableError, get_backend
from rag_quality_evaluator.mock import MockBackend
from rag_quality_evaluator.retrieval import build_retriever


@pytest.fixture(scope="module")
def heading_chunks(corpus):
    return chunk_corpus(corpus.values(), "heading")


@pytest.mark.parametrize("name", ["bm25", "tfidf", "embed"])
def test_retrievers_find_the_right_document(name, heading_chunks):
    r = build_retriever(name, heading_chunks, embed_backend=MockBackend())
    hits = r.retrieve("How many vacuum tubes did ENIAC contain?", k=3)
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "hist-eniac"
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_embed_retriever_requires_embed_backend(heading_chunks):
    with pytest.raises(ValueError):
        build_retriever("embed", heading_chunks, embed_backend=None)
    with pytest.raises(ValueError):
        build_retriever("dense", heading_chunks)


def test_get_backend_mock_and_unknown():
    assert get_backend("mock").name == "mock"
    with pytest.raises(ValueError):
        get_backend("nope")


def test_network_backends_fail_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailableError, match="ANTHROPIC_API_KEY"):
        get_backend("anthropic")
    with pytest.raises(BackendUnavailableError, match="OPENAI_API_KEY"):
        get_backend("openai")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens on the discard port
    backend = get_backend("ollama")
    assert backend.model == "llama3.1:8b" or backend.model
    with pytest.raises(BackendUnavailableError, match="Ollama"):
        backend.complete("hello")


def test_prompt_roundtrip(heading_chunks):
    hits = build_retriever("bm25", heading_chunks).retrieve("When did Mount St. Helens erupt?", k=2)
    prompt = build_prompt("When did Mount St. Helens erupt?", hits)
    assert "[doc_id]" in prompt
    question, blocks = parse_prompt(prompt)
    assert question == "When did Mount St. Helens erupt?"
    assert [b[0] for b in blocks] == [h.chunk.doc_id for h in hits]


def test_mock_answers_from_context_with_citation(heading_chunks):
    clean = MockBackend(contradiction_rate=0, fabrication_rate=0, citation_error_rate=0)
    hits = build_retriever("bm25", heading_chunks).retrieve("How many vacuum tubes did ENIAC contain?", k=3)
    prompt = build_prompt("How many vacuum tubes did ENIAC contain?", hits)
    answer = clean.complete(prompt)
    assert "17,468" in answer
    assert parse_citations(answer)[0].citations == ["hist-eniac"]
    assert clean.complete(prompt) == answer  # deterministic
    res = score_faithfulness(answer, [(h.chunk.doc_id, h.chunk.text) for h in hits])
    assert res.faithfulness == 1.0


def test_mock_abstains_without_support(heading_chunks):
    hits = build_retriever("bm25", heading_chunks).retrieve("How many moons does Jupiter have?", k=1)
    # only a weakly related chunk is in context; ask something it cannot answer
    prompt = build_prompt("What colour is the CEO's car?", hits)
    assert MockBackend().complete(prompt) == ABSTAIN_TEXT


def test_mock_injected_errors_are_detected(heading_chunks):
    q = "How many vacuum tubes did ENIAC contain?"
    hits = build_retriever("bm25", heading_chunks).retrieve(q, k=3)
    passages = [(h.chunk.doc_id, h.chunk.text) for h in hits]
    prompt = build_prompt(q, hits)

    contradicting = MockBackend(contradiction_rate=1.0, fabrication_rate=0, citation_error_rate=0)
    res = score_faithfulness(contradicting.complete(prompt), passages)
    assert any(c.label == "contradicted" for c in res.claims)

    fabricating = MockBackend(contradiction_rate=0, fabrication_rate=1.0, citation_error_rate=0)
    res = score_faithfulness(fabricating.complete(prompt), passages)
    assert any(c.label == "unsupported" for c in res.claims)

    miciting = MockBackend(contradiction_rate=0, fabrication_rate=0, citation_error_rate=1.0)
    cites = parse_citations(miciting.complete(prompt))[0].citations
    assert cites != ["hist-eniac"] and cites[0] in {h.chunk.doc_id for h in hits}


def test_mock_judge_verify_split_tasks():
    from rag_quality_evaluator.claims import split_claims_llm, verify_claim_llm
    from rag_quality_evaluator.metrics import llm_judge

    m = MockBackend()
    assert llm_judge(m, "q?", "It is 18 days [hb-pto].", ["18 days"]) == 1.0
    assert llm_judge(m, "q?", "It is 20 days [hb-pto].", ["18 days"]) == 0.0
    ctx = [("hb-pto", "Full-time employees accrue 18 days of paid time off per year.")]
    assert verify_claim_llm("Full-time employees accrue 18 days of paid time off per year.", ctx, m).label == "supported"
    assert verify_claim_llm("Full-time employees accrue 25 days of paid time off per year.", ctx, m).label == "contradicted"
    claims = split_claims_llm("Staff get 18 days of leave [hb-pto]. Sick leave is separate [hb-pto].", m)
    assert [c.citations for c in claims] == [["hb-pto"], ["hb-pto"]]
