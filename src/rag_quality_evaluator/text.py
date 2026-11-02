"""Small text utilities shared by chunkers, retrievers and metrics.

Everything here is deliberately simple and dependency-free so that the
heuristic metrics are easy to reason about and to test by hand.
"""

from __future__ import annotations

import re
import string

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into onto
    over under about as is are was were be been being am do does did done has
    have had having it its this that these those there here what which who whom
    whose when where why how i you he she we they me him her us them my your his
    our their not no nor so than too very can could will would shall should may
    might must also any all each such only own same some just per up out off
    after before again further once both few more most other s t don now
    many much get gets got via within
    """.split()
)

# Words that flip the polarity of a statement. Used by the claim verifier.
NEGATIONS = frozenset({"not", "no", "never", "none", "cannot", "nobody", "nothing", "neither", "nor", "without"})

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twice": "2",
}

_ABBREVIATIONS = ["e.g.", "i.e.", "etc.", "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Mt.", "J.", "vs.", "U.S.", "approx."]
_ABBR_PLACEHOLDER = "․"  # one-dot leader, never appears in the corpus

CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.\-]*)\]")


def strip_citations(text: str) -> str:
    """Remove [doc_id] markers and tidy the whitespace left behind."""
    out = CITATION_RE.sub("", text)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _join_thousands(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Thousands separators are removed first."""
    text = _join_thousands(text.lower())
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+", text)


def stem(token: str) -> str:
    """A tiny suffix-stripping stemmer (enough to match 'days'/'day', 'accrues'/'accrue')."""
    if token.isdigit() or len(token) <= 3:
        return token
    for suffix, repl, min_len in (("ies", "y", 5), ("sses", "ss", 6), ("ing", "", 6), ("ed", "", 5), ("es", "e", 5), ("s", "", 4)):
        if token.endswith(suffix) and len(token) >= min_len and not token.endswith("ss"):
            return token[: -len(suffix)] + repl
    return token


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed; number words mapped to digits."""
    out = []
    for tok in tokenize(text):
        tok = NUMBER_WORDS.get(tok, tok)
        if tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


def extract_numbers(text: str) -> set[str]:
    """Numbers mentioned in the text, normalised (no thousands separators, digits for small number words)."""
    text = _join_thousands(text)
    nums = set()
    for m in re.findall(r"\d+(?:\.\d+)?", text):
        nums.add(m.rstrip("0").rstrip(".") if "." in m else m)
    for tok in re.findall(r"[A-Za-z]+", text):
        if tok.lower() in NUMBER_WORDS and tok.lower() != "twice":
            nums.add(NUMBER_WORDS[tok.lower()])
    return nums


def extract_entities(text: str) -> set[str]:
    """Capitalised words that are not sentence-initial. A crude proper-noun detector."""
    ents = set()
    for sent in split_sentences(text):
        words = re.findall(r"[A-Za-z][A-Za-z\-]*", strip_citations(sent))
        for w in words[1:]:
            if w[0].isupper() and w.lower() not in STOPWORDS:
                ents.add(w.lower())
    return ents


def has_negation(text: str) -> bool:
    toks = tokenize(text)
    return any(t in NEGATIONS for t in toks) or "n't" in text.lower()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Handles common abbreviations and markdown line breaks.

    A trailing citation like "... per year. [hb-pto]" is attached to the sentence before it.
    """
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _ABBR_PLACEHOLDER))
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=#)|\n(?=[-*] )", protected):
        block = " ".join(line.strip() for line in block.splitlines()).strip()
        if not block:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\[(])", block)
        for part in parts:
            part = part.replace(_ABBR_PLACEHOLDER, ".").strip()
            if not part:
                continue
            # citations placed after the full stop belong to the previous sentence
            lead = re.match(r"((?:\[[^\]]+\]\s*)+\.?)\s*", part)
            if sentences and lead:
                sentences[-1] = sentences[-1] + " " + lead.group(1).strip()
                part = part[lead.end():].strip()
                if not part:
                    continue
            sentences.append(part)
    return sentences


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop citations, punctuation and articles."""
    text = CITATION_RE.sub(" ", text)
    text = _join_thousands(text.lower())
    text = "".join(ch if ch not in set(string.punctuation) - {"."} else " " for ch in text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())
