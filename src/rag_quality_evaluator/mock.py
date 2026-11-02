"""Deterministic offline backend.

The mock backend recognises the task header at the top of each prompt
(``TASK: answer_with_citations``, ``judge_answer``, ``verify_claim``,
``split_claims``) and answers it with simple heuristics, so the full pipeline
runs without a network.

For answering it behaves like a weak extractive reader: it picks the context
sentence(s) with the highest overlap with the question and cites the document
they came from, or abstains when nothing overlaps enough. To make the
faithfulness and citation metrics non-trivial it injects errors at fixed rates,
seeded by a hash of the question so runs are reproducible:

* contradiction  - a number in the answer is changed, or a negation is inserted
* fabrication    - an extra sentence that is not in any document is appended
* wrong citation - the citation is swapped for another document in the context
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .generation import ABSTAIN_TEXT, parse_prompt
from .text import content_terms, extract_entities, extract_numbers, split_sentences

FABRICATIONS = [
    "This rule was last revised in 2019 after an internal audit.",
    "Exceptions can be granted by the regional director in writing.",
    "The figure was confirmed by an independent review in 2015.",
    "The same limit applies to contractors and seasonal staff.",
    "This was widely reported in the national press at the time.",
]

_HEADING_PREFIX_RE = re.compile(r"^[A-Z][^.:]{2,80}: (?=[A-Z0-9])")


def _hash_unit(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{text}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockBackend:
    name = "mock"

    def __init__(
        self,
        contradiction_rate: float = 0.08,
        fabrication_rate: float = 0.08,
        citation_error_rate: float = 0.10,
        abstain_threshold: float = 0.4,
        seed: int = 0,
        embed_dim: int = 512,
    ):
        self.contradiction_rate = contradiction_rate
        self.fabrication_rate = fabrication_rate
        self.citation_error_rate = citation_error_rate
        self.abstain_threshold = abstain_threshold
        self.seed = seed
        self.embed_dim = embed_dim
        self.model = "mock-extractive-v1"

    # ------------------------------------------------------------------ public API

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        task = re.search(r"TASK:\s*(\w+)", prompt)
        task_name = task.group(1) if task else ""
        if task_name == "answer_with_citations":
            return self._answer(prompt)
        if task_name == "judge_answer":
            return self._judge(prompt)
        if task_name == "verify_claim":
            return self._verify(prompt)
        if task_name == "split_claims":
            return self._split(prompt)
        return "MOCK RESPONSE"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Signed feature hashing of stemmed unigrams and bigrams.

        This is a lexical stand-in so the dense-retriever code path can run offline;
        it is not a semantic embedding.
        """
        out = []
        for text in texts:
            vec = np.zeros(self.embed_dim)
            terms = content_terms(text)
            feats = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                vec[h % self.embed_dim] += 1.0 if (h >> 20) & 1 else -1.0
            out.append(vec.tolist())
        return out

    # ------------------------------------------------------------------ answering

    def _score_sentence(self, q_terms: set[str], question: str, sentence: str) -> float:
        s_terms = set(content_terms(sentence))
        if not q_terms:
            return 0.0
        score = len(q_terms & s_terms) / len(q_terms)
        ql = question.lower()
        if re.match(r"(how many|how much|how long|how far|how often|what percentage|what is the (daily|maximum|minimum|annual))", ql) or "year" in ql or "date" in ql:
            score += 0.1 if extract_numbers(sentence) else 0.0
        if ql.startswith(("who", "which", "where")):
            score += 0.05 if extract_entities(sentence) else 0.0
        return score

    def _answer(self, prompt: str) -> str:
        question, blocks = parse_prompt(prompt)
        q_terms = set(content_terms(question))
        candidates = []
        for block_rank, (doc_id, text) in enumerate(blocks):
            for i, sent in enumerate(split_sentences(text)):
                if i == 0:
                    sent = _HEADING_PREFIX_RE.sub("", sent)
                sent = sent.strip()
                if len(sent.split()) < 3:
                    continue
                # small position bias: earlier context blocks win ties, like a real reader
                score = self._score_sentence(q_terms, question, sent) - 0.01 * block_rank
                candidates.append((score, doc_id, sent))
        if not candidates:
            return ABSTAIN_TEXT
        candidates.sort(key=lambda c: -c[0])
        best_score, best_doc, best_sent = candidates[0]
        if best_score < self.abstain_threshold:
            return ABSTAIN_TEXT

        chosen = [(best_doc, best_sent)]
        covered = q_terms & set(content_terms(best_sent))
        for score, doc_id, sent in candidates[1:4]:
            new_terms = (q_terms & set(content_terms(sent))) - covered
            if score >= 0.4 and len(new_terms) >= 2 and sent != best_sent:
                chosen.append((doc_id, sent))
                break

        context_docs = list(dict.fromkeys(d for d, _ in blocks))
        sentences = [[sent.rstrip("."), doc] for doc, sent in chosen]

        if _hash_unit(question, f"contradict{self.seed}") < self.contradiction_rate:
            mutated = self._contradict(sentences[0][0], question)
            if mutated:
                sentences[0][0] = mutated
            else:
                sentences.append([self._fabrication(question).rstrip("."), sentences[0][1]])
        if _hash_unit(question, f"fabricate{self.seed}") < self.fabrication_rate:
            sentences.append([self._fabrication(question).rstrip("."), sentences[0][1]])
        if _hash_unit(question, f"cite{self.seed}") < self.citation_error_rate:
            others = [d for d in context_docs if d != sentences[0][1]]
            if others:
                sentences[0][1] = others[int(_hash_unit(question, "pick") * len(others)) % len(others)]
        return " ".join(f"{text} [{doc}]." for text, doc in sentences)

    def _fabrication(self, question: str) -> str:
        return FABRICATIONS[int(_hash_unit(question, f"fab{self.seed}") * len(FABRICATIONS)) % len(FABRICATIONS)]

    @staticmethod
    def _contradict(sentence: str, question: str) -> str | None:
        m = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", sentence)
        if m:
            raw = m.group(0).replace(",", "")
            value = float(raw) if "." in raw else int(raw)
            bump = 1 + int(_hash_unit(question, "bump") * 4)
            new = value + bump * (10 if value >= 100 else 1)
            new_text = f"{new:g}" if isinstance(new, float) else str(new)
            return sentence[: m.start()] + new_text + sentence[m.end() :]
        for word, repl in ((" is ", " is not "), (" are ", " are not "), (" was ", " was not "), (" can ", " cannot "), (" has ", " does not have ")):
            if word in sentence:
                return sentence.replace(word, repl, 1)
        return None

    # ------------------------------------------------------------------ evaluation helpers

    @staticmethod
    def _judge(prompt: str) -> str:
        from .metrics import score_answer

        q = re.search(r"GOLD:\s*(.*)", prompt)
        p = re.search(r"PREDICTED:\s*(.*)", prompt, re.S)
        golds = [g.strip() for g in (q.group(1) if q else "").split("|") if g.strip()]
        pred = p.group(1).strip() if p else ""
        return f"SCORE: {int(score_answer(pred, golds).correct)}"

    @staticmethod
    def _verify(prompt: str) -> str:
        from .claims import verify_claim_heuristic

        claim_m = re.search(r"CLAIM:\s*(.*?)\nLABEL:", prompt, re.S)
        blocks = re.findall(r"Document \[([^\]]+)\]:\n(.*?)(?=\n\nDocument \[|\n\nCLAIM:)", prompt, re.S)
        verdict = verify_claim_heuristic(claim_m.group(1).strip() if claim_m else "", blocks)
        return verdict.label.upper()

    @staticmethod
    def _split(prompt: str) -> str:
        from .claims import split_claims

        m = re.search(r"ANSWER:\n(.*?)\n\nCLAIMS:", prompt, re.S)
        return "\n".join(c.text for c in split_claims(m.group(1) if m else ""))
