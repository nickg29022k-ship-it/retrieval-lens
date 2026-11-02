"""Citation support: does each cited [doc_id] actually support the sentence it is attached to?

Definitions (in the spirit of ALCE, Gao et al. 2023):

* A *citation* is a (sentence, doc_id) pair. It is **supporting** if at least one
  claim of the sentence is verified as ``supported`` against the full text of the
  cited document.
* **Citation precision** = supporting citations / all citations.
* A sentence is **fully supported by its citations** if every claim in it is
  supported by the union of the documents it cites.
* **Citation recall** = fully supported sentences / all answer sentences
  (sentences with no citation count as unsupported).
"""

from __future__ import annotations

from pydantic import BaseModel

from .claims import _split_clauses, verify_claim_heuristic
from .data import Document
from .generation import parse_citations


class CitationCheck(BaseModel):
    sentence: str
    doc_id: str
    supported: bool
    in_context: bool
    reason: str = ""


class CitationResult(BaseModel):
    checks: list[CitationCheck]
    n_sentences: int
    n_sentences_supported: int
    n_uncited_sentences: int
    precision: float | None
    recall: float | None


def score_citations(answer: str, corpus: dict[str, Document], retrieved_doc_ids: set[str] | None = None) -> CitationResult:
    sentences = parse_citations(answer)
    checks: list[CitationCheck] = []
    n_supported_sentences = 0
    n_uncited = 0
    for sent in sentences:
        claims = _split_clauses(sent.text)
        if not sent.citations:
            n_uncited += 1
            continue
        claim_supported = [False] * len(claims)
        for doc_id in sent.citations:
            in_ctx = retrieved_doc_ids is None or doc_id in retrieved_doc_ids
            doc = corpus.get(doc_id)
            if doc is None:
                checks.append(CitationCheck(sentence=sent.text, doc_id=doc_id, supported=False, in_context=in_ctx, reason="unknown document id"))
                continue
            verdicts = [verify_claim_heuristic(c, [(doc_id, doc.text)]) for c in claims]
            ok = [v.label == "supported" for v in verdicts]
            claim_supported = [a or b for a, b in zip(claim_supported, ok)]
            reason = "supports" if any(ok) else (verdicts[0].reason if verdicts else "")
            checks.append(CitationCheck(sentence=sent.text, doc_id=doc_id, supported=any(ok), in_context=in_ctx, reason=reason))
        if claims and all(claim_supported):
            n_supported_sentences += 1

    precision = (sum(c.supported for c in checks) / len(checks)) if checks else None
    recall = (n_supported_sentences / len(sentences)) if sentences else None
    return CitationResult(
        checks=checks,
        n_sentences=len(sentences),
        n_sentences_supported=n_supported_sentences,
        n_uncited_sentences=n_uncited,
        precision=precision,
        recall=recall,
    )
