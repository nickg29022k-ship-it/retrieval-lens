"""Corpus and QA-set loading."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
DEFAULT_QA_PATH = PROJECT_ROOT / "data" / "qa.jsonl"


class Document(BaseModel):
    doc_id: str
    title: str
    text: str


class QAItem(BaseModel):
    id: str
    question: str
    answers: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[str] = Field(default_factory=list)
    answerable: bool = True


def load_corpus(corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> dict[str, Document]:
    """Load every .md/.txt file in a directory. The file stem is the document id."""
    corpus_dir = Path(corpus_dir)
    docs: dict[str, Document] = {}
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        first = text.splitlines()[0] if text else path.stem
        title = first.lstrip("# ").strip() if first.startswith("#") else path.stem
        docs[path.stem] = Document(doc_id=path.stem, title=title, text=text)
    if not docs:
        raise FileNotFoundError(f"No .md or .txt documents found in {corpus_dir}")
    return docs


def load_qa(path: Path | str = DEFAULT_QA_PATH) -> list[QAItem]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(QAItem.model_validate(json.loads(line)))
    return items


def validate_qa(items: list[QAItem], corpus: dict[str, Document]) -> list[str]:
    """Return a list of problems: unknown gold docs or evidence that is not verbatim in its doc."""
    problems = []
    seen = set()
    for item in items:
        if item.id in seen:
            problems.append(f"{item.id}: duplicate id")
        seen.add(item.id)
        if item.answerable and not (item.answers and item.gold_doc_ids and item.gold_evidence):
            problems.append(f"{item.id}: answerable question needs answers, gold_doc_ids and gold_evidence")
        if not item.answerable and (item.answers or item.gold_evidence):
            problems.append(f"{item.id}: unanswerable question should not have answers or evidence")
        for doc_id in item.gold_doc_ids:
            if doc_id not in corpus:
                problems.append(f"{item.id}: unknown gold doc {doc_id}")
        for ev in item.gold_evidence:
            if not any(ev in corpus[d].text for d in item.gold_doc_ids if d in corpus):
                problems.append(f"{item.id}: evidence not found verbatim: {ev[:60]!r}")
    return problems
