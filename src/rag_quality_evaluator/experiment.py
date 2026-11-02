"""Experiment grid: chunking x retriever x k."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from .data import Document, QAItem
from .pipeline import EvalOptions, QuestionResult, RunConfig, run_config, summarize


class GridSpec(BaseModel):
    chunkers: dict[str, dict]
    retrievers: list[str]
    k: list[int]

    def configs(self) -> list[RunConfig]:
        return [
            RunConfig(chunker=name, retriever=r, k=k, chunk_params=params or {})
            for name, params in self.chunkers.items()
            for r in self.retrievers
            for k in self.k
        ]


def load_grid(path: Path | str) -> GridSpec:
    with open(path, encoding="utf-8") as fh:
        return GridSpec.model_validate(yaml.safe_load(fh))


def success_rate(row: dict) -> float:
    """End-to-end success: answered correctly and faithfully, or abstained correctly."""
    return (row["correct"] + row["correct_abstention"]) / row["n_questions"]


def rank_key(row: dict) -> tuple:
    return (
        row["success_rate"],
        row["faithfulness"] or 0.0,
        row["citation_precision"] or 0.0,
        row["recall@k"] or 0.0,
    )


def run_grid(
    grid: GridSpec,
    corpus: dict[str, Document],
    qa: list[QAItem],
    backend,
    options: EvalOptions | None = None,
    embed_backend=None,
    progress=None,
) -> tuple[list[dict], dict[str, list[QuestionResult]]]:
    cache: dict = {}
    rows: list[dict] = []
    per_config: dict[str, list[QuestionResult]] = {}
    for cfg in grid.configs():
        results = run_config(cfg, corpus, qa, backend, options, embed_backend=embed_backend, _cache=cache)
        row = summarize(cfg, results)
        row["success_rate"] = success_rate(row)
        rows.append(row)
        per_config[cfg.label] = results
        if progress:
            progress(cfg, row)
    rows.sort(key=rank_key, reverse=True)
    return rows, per_config


def write_outputs(rows: list[dict], per_config: dict[str, list[QuestionResult]], out_dir: Path, meta: dict) -> dict[str, Path]:
    from .report import render_report

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": out_dir / "summary.csv",
        "summary_json": out_dir / "summary.json",
        "per_question": out_dir / "per_question.jsonl",
        "report": out_dir / "report.md",
    }
    with open(paths["summary_csv"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    paths["summary_json"].write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    with open(paths["per_question"], "w", encoding="utf-8") as fh:
        for results in per_config.values():
            for r in results:
                fh.write(r.model_dump_json() + "\n")
    paths["report"].write_text(render_report(rows, per_config, meta), encoding="utf-8")
    return paths
