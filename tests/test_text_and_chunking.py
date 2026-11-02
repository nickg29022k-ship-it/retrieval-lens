import pytest

from rag_quality_evaluator.chunking import chunk_corpus, fixed_chunker, heading_chunker, sentence_chunker
from rag_quality_evaluator.text import content_terms, extract_numbers, split_sentences, stem, tokenize


def test_split_sentences_handles_abbreviations_and_citations():
    text = "Mount St. Helens erupted in 1980 [sci-volcanoes]. It was large. [sci-volcanoes] J. Presper Eckert helped."
    sents = split_sentences(text)
    assert sents[0] == "Mount St. Helens erupted in 1980 [sci-volcanoes]."
    assert sents[1] == "It was large. [sci-volcanoes]"
    assert sents[2].startswith("J. Presper")


def test_tokenize_and_numbers():
    assert tokenize("17,468 tubes") == ["17468", "tubes"]
    assert extract_numbers("about 2,000 eggs and two queens, 4.5 hours") == {"2000", "2", "4.5"}
    assert stem("days") == "day" and stem("policies") == "policy"
    assert "the" not in content_terms("the days")


def test_fixed_chunker_windows_and_overlap(small_doc):
    chunks = fixed_chunker(small_doc, size=10, overlap=4)
    words = [c.text.split() for c in chunks]
    assert all(len(w) <= 10 for w in words)
    # consecutive windows share exactly `overlap` words
    for a, b in zip(words, words[1:]):
        assert a[-4:] == b[:4]
    total = len(" ".join(l for l in small_doc.text.splitlines()).replace("#", "").split())
    assert len(chunks) == 1 + -(-(total - 10) // 6)
    assert [c.chunk_id for c in chunks] == [f"toy#{i}" for i in range(len(chunks))]


def test_fixed_chunker_rejects_bad_params(small_doc):
    with pytest.raises(ValueError):
        fixed_chunker(small_doc, size=5, overlap=5)


def test_sentence_chunker_never_splits_sentences(small_doc):
    chunks = sentence_chunker(small_doc, max_words=12)
    sentences = [s for s in split_sentences(small_doc.text) if not s.startswith("#")]
    joined = [c.text for c in chunks]
    for s in sentences:
        assert any(s in c for c in joined)
    assert all(len(c.split()) <= 12 for c in joined)


def test_heading_chunker_one_chunk_per_section(small_doc):
    chunks = heading_chunker(small_doc)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Toy Policy - Leave: Staff get 12 days")
    assert chunks[1].text.startswith("Toy Policy - Laptops:")


def test_chunk_corpus_unknown_strategy(small_doc):
    with pytest.raises(ValueError):
        chunk_corpus([small_doc], "semantic")


def test_all_chunkers_cover_real_corpus(corpus):
    for strategy in ("fixed", "sentence", "heading"):
        chunks = chunk_corpus(corpus.values(), strategy)
        assert {c.doc_id for c in chunks} == set(corpus)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
