"""Failure attribution: was a bad answer the retriever's fault or the generator's?

Decision procedure (applied in order):

Unanswerable questions (no gold answer in the corpus)
  1. abstained                        -> correct_abstention
  2. answered anyway                  -> missed_abstention   (the model made something up)

Answerable questions
  3. abstained, gold not retrieved    -> retrieval_failure   (abstaining was reasonable given the context)
  4. abstained, gold retrieved        -> incorrect_abstention (generation side: the answer was in the context)
  5. correct and faithful             -> correct
  6. correct but has hallucinated claims -> generation_failure (subtype "unfaithful")
  7. wrong, gold not retrieved        -> retrieval_failure
  8. wrong, gold retrieved            -> generation_failure  (subtype "wrong_answer")

"Gold retrieved" means every gold evidence span is covered by a top-k chunk.
"Faithful" means no claim was labelled unsupported or contradicted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Category = Literal[
    "correct",
    "retrieval_failure",
    "generation_failure",
    "correct_abstention",
    "incorrect_abstention",
    "missed_abstention",
]

CATEGORIES: tuple[str, ...] = (
    "correct",
    "retrieval_failure",
    "generation_failure",
    "correct_abstention",
    "incorrect_abstention",
    "missed_abstention",
)


class Attribution(BaseModel):
    category: Category
    subtype: str = ""
    reason: str


def attribute_failure(
    answerable: bool,
    abstained: bool,
    correct: bool,
    gold_retrieved: bool,
    n_hallucinated: int,
) -> Attribution:
    if not answerable:
        if abstained:
            return Attribution(category="correct_abstention", reason="question is unanswerable and the system abstained")
        return Attribution(category="missed_abstention", subtype="hallucinated_answer",
                           reason="question is unanswerable but the system produced an answer")
    if abstained:
        if not gold_retrieved:
            return Attribution(category="retrieval_failure", subtype="abstained",
                               reason="gold evidence was not retrieved, so the generator had nothing to answer from")
        return Attribution(category="incorrect_abstention",
                           reason="gold evidence was in the context but the system abstained")
    if correct and n_hallucinated == 0:
        return Attribution(category="correct", reason="correct and fully supported")
    if correct:
        return Attribution(category="generation_failure", subtype="unfaithful",
                           reason=f"answer is correct but contains {n_hallucinated} unsupported/contradicted claim(s)")
    if not gold_retrieved:
        return Attribution(category="retrieval_failure", subtype="wrong_answer",
                           reason="answer is wrong and the gold evidence was not retrieved")
    return Attribution(category="generation_failure", subtype="wrong_answer",
                       reason="gold evidence was retrieved but the answer is wrong")
