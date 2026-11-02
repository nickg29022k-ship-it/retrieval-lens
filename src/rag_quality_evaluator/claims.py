"""Claim-level grounding: split answers into atomic claims and verify each one.

Verification labels:

* ``supported``    - the context states the claim.
* ``contradicted`` - the context talks about the same thing but disagrees on a
                     number or on polarity (negation).
* ``unsupported``  - the context does not back the claim up.

Two verifiers are available: a lexical/entailment-style heuristic (default,
offline) and an LLM verifier that asks the backend for a label.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from .text import (
    CITATION_RE,
    content_terms,
    extract_entities,
    extract_numbers,
    has_negation,
    split_sentences,
    strip_citations,
)

Label = Literal["supported", "unsupported", "contradicted"]

SUPPORT_THRESHOLD = 0.6      # fraction of claim content terms found in one evidence window
TOPIC_THRESHOLD = 0.5        # overlap needed before a number/negation mismatch counts as a contradiction
MIN_CLAIM_TOKENS = 3

_CLAUSE_SPLIT_RE = re.compile(r";\s+|,\s+(?=(?:and|but|while|whereas|although|though|yet|so)\s)|\s+(?=(?:but|whereas)\s)")
_AND_SUBJECTS = {"the", "a", "an", "it", "they", "he", "she", "we", "this", "these", "those", "its", "their", "there"}


class Claim(BaseModel):
    text: str
    sentence_index: int
    citations: list[str] = []


class ClaimVerdict(BaseModel):
    claim: str
    label: Label
    score: float
    evidence: str | None = None
    evidence_doc: str | None = None
    reason: str = ""


# --------------------------------------------------------------------------- splitting


def _split_clauses(sentence: str) -> list[str]:
    parts = [p.strip(" ,") for p in _CLAUSE_SPLIT_RE.split(sentence) if p and p.strip(" ,")]
    # strip leading conjunctions left behind by the split
    parts = [re.sub(r"^(and|but|while|whereas|although|though|yet|so)\s+", "", p, flags=re.I) for p in parts]
    refined: list[str] = []
    for part in parts:
        # "X did A and the Y did B" -> two clauses, only when the right side starts a new subject
        pieces = re.split(r"\s+and\s+(?=(?:%s)\s)" % "|".join(_AND_SUBJECTS), part, flags=re.I)
        if len(pieces) > 1 and all(len(p.split()) >= 4 for p in pieces):
            refined.extend(pieces)
        else:
            refined.append(part)
    merged: list[str] = []
    for part in refined:
        if merged and len(part.split()) < MIN_CLAIM_TOKENS:
            merged[-1] = f"{merged[-1]}, {part}"
        else:
            merged.append(part)
    if len(merged) > 1 and len(merged[0].split()) < MIN_CLAIM_TOKENS:
        merged[1] = f"{merged[0]}, {merged[1]}"
        merged.pop(0)
    return [m.rstrip(".") + "." if not m.endswith((".", "?", "!")) else m for m in merged]


def split_claims(answer: str) -> list[Claim]:
    """Heuristic splitter: sentences, then clause boundaries. Citations are inherited from the sentence."""
    claims = []
    for i, sent in enumerate(split_sentences(answer)):
        cites = list(dict.fromkeys(CITATION_RE.findall(sent)))
        clean = strip_citations(sent)
        if not clean:
            continue
        for clause in _split_clauses(clean):
            claims.append(Claim(text=clause, sentence_index=i, citations=cites))
    return claims


SPLIT_PROMPT = """TASK: split_claims
Break the answer below into short, self-contained factual claims. Each claim must make sense on its own
(replace pronouns with what they refer to). Ignore citations. Output one claim per line with no numbering.

ANSWER:
{answer}

CLAIMS:"""


def split_claims_llm(answer: str, backend) -> list[Claim]:
    """LLM-based splitting. Claims inherit the citations of the sentence they overlap most."""
    from .generation import parse_citations

    sentences = parse_citations(answer)
    raw = backend.complete(SPLIT_PROMPT.format(answer=answer))
    claims = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if len(line.split()) < 2:
            continue
        best_i, best = 0, -1.0
        terms = set(content_terms(line))
        for i, s in enumerate(sentences):
            ov = len(terms & set(content_terms(s.text))) / max(len(terms), 1)
            if ov > best:
                best_i, best = i, ov
        cites = sentences[best_i].citations if sentences else []
        claims.append(Claim(text=strip_citations(line), sentence_index=best_i, citations=cites))
    return claims


# --------------------------------------------------------------------------- verification


def _evidence_windows(passages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Single sentences plus adjacent sentence pairs, tagged with doc id."""
    windows = []
    for doc_id, text in passages:
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            windows.append((doc_id, s))
            if i + 1 < len(sents):
                windows.append((doc_id, s + " " + sents[i + 1]))
    return windows


def verify_claim_heuristic(claim: str, passages: list[tuple[str, str]]) -> ClaimVerdict:
    """Lexical entailment heuristic.

    1. Find the evidence window (sentence or sentence pair) covering the most claim content terms.
    2. If coverage >= TOPIC_THRESHOLD and the claim's numbers are not all in that window
       (while the window does contain numbers), or the polarity differs -> contradicted.
    3. Supported if coverage >= SUPPORT_THRESHOLD, all numbers present and every
       proper noun in the claim appears somewhere in the context.
    4. Otherwise unsupported.
    """
    terms = set(content_terms(claim))
    if not terms:
        return ClaimVerdict(claim=claim, label="supported", score=1.0, reason="no content terms")
    windows = _evidence_windows(passages)
    if not windows:
        return ClaimVerdict(claim=claim, label="unsupported", score=0.0, reason="empty context")

    best_doc, best_text, best_cov = None, None, -1.0
    for doc_id, text in windows:
        cov = len(terms & set(content_terms(text))) / len(terms)
        # prefer the shorter window on ties so the evidence shown is tight
        if cov > best_cov + 1e-9:
            best_doc, best_text, best_cov = doc_id, text, cov
    assert best_text is not None

    claim_nums = extract_numbers(claim)
    ev_nums = extract_numbers(best_text)
    all_context = " ".join(t for _, t in passages)
    missing_nums = claim_nums - ev_nums

    if best_cov >= TOPIC_THRESHOLD and missing_nums and ev_nums:
        return ClaimVerdict(
            claim=claim, label="contradicted", score=best_cov, evidence=best_text, evidence_doc=best_doc,
            reason=f"number mismatch: claim has {sorted(missing_nums)}, evidence has {sorted(ev_nums)}",
        )
    if best_cov >= TOPIC_THRESHOLD and has_negation(claim) != has_negation(best_text):
        return ClaimVerdict(
            claim=claim, label="contradicted", score=best_cov, evidence=best_text, evidence_doc=best_doc,
            reason="polarity mismatch (negation)",
        )
    context_terms = set(content_terms(all_context))
    missing_ents = {e for e in extract_entities(claim) if not set(content_terms(e)) <= context_terms}
    if best_cov >= SUPPORT_THRESHOLD and not missing_nums and not missing_ents:
        return ClaimVerdict(
            claim=claim, label="supported", score=best_cov, evidence=best_text, evidence_doc=best_doc,
            reason="lexical coverage",
        )
    reasons = [f"coverage {best_cov:.2f}"]
    if missing_nums:
        reasons.append(f"numbers not in context {sorted(missing_nums)}")
    if missing_ents:
        reasons.append(f"entities not in context {sorted(missing_ents)}")
    return ClaimVerdict(
        claim=claim, label="unsupported", score=best_cov, evidence=best_text, evidence_doc=best_doc,
        reason="; ".join(reasons),
    )


VERIFY_PROMPT = """TASK: verify_claim
Decide whether the CLAIM is supported by the CONTEXT.
Answer with exactly one word: SUPPORTED, CONTRADICTED or UNSUPPORTED.
- SUPPORTED: the context states the claim.
- CONTRADICTED: the context states something incompatible with the claim.
- UNSUPPORTED: the context does not say.

CONTEXT:
{context}

CLAIM: {claim}
LABEL:"""


def verify_claim_llm(claim: str, passages: list[tuple[str, str]], backend) -> ClaimVerdict:
    context = "\n\n".join(f"Document [{d}]:\n{t}" for d, t in passages)
    raw = backend.complete(VERIFY_PROMPT.format(context=context, claim=claim)).upper()
    label: Label = "unsupported"
    for candidate in ("CONTRADICTED", "UNSUPPORTED", "SUPPORTED"):
        if candidate in raw:
            label = candidate.lower()  # type: ignore[assignment]
            break
    heuristic = verify_claim_heuristic(claim, passages)
    return ClaimVerdict(
        claim=claim, label=label, score=1.0 if label == "supported" else 0.0,
        evidence=heuristic.evidence, evidence_doc=heuristic.evidence_doc, reason=f"llm: {raw.strip()[:40]}",
    )


class FaithfulnessResult(BaseModel):
    claims: list[ClaimVerdict]
    faithfulness: float | None  # supported / total, None when the answer has no claims
    hallucinated: list[str]     # unsupported + contradicted claim texts


def score_faithfulness(
    answer: str,
    passages: list[tuple[str, str]],
    backend=None,
    llm_split: bool = False,
    llm_verify: bool = False,
) -> FaithfulnessResult:
    claims = split_claims_llm(answer, backend) if (llm_split and backend) else split_claims(answer)
    verdicts = [
        verify_claim_llm(c.text, passages, backend) if (llm_verify and backend) else verify_claim_heuristic(c.text, passages)
        for c in claims
    ]
    if not verdicts:
        return FaithfulnessResult(claims=[], faithfulness=None, hallucinated=[])
    supported = sum(v.label == "supported" for v in verdicts)
    return FaithfulnessResult(
        claims=verdicts,
        faithfulness=supported / len(verdicts),
        hallucinated=[v.claim for v in verdicts if v.label != "supported"],
    )
