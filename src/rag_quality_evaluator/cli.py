"""Command-line interface (``rag-eval``)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .chunking import CHUNKERS, chunk_corpus
from .data import DEFAULT_CORPUS_DIR, DEFAULT_QA_PATH, PROJECT_ROOT, load_corpus, load_qa, validate_qa
from .experiment import load_grid, run_grid, write_outputs
from .llm import BackendUnavailableError, get_backend
from .pipeline import EvalOptions, RunConfig, run_config, summarize

app = typer.Typer(add_completion=False, help="Evaluate RAG pipelines: retrieval, answers, faithfulness, citations.")
console = Console()

CorpusOpt = typer.Option(DEFAULT_CORPUS_DIR, "--corpus", help="Directory of .md/.txt documents.")
QAOpt = typer.Option(DEFAULT_QA_PATH, "--qa", help="QA set in JSONL format.")
BackendOpt = typer.Option("mock", "--backend", "-b", help="mock | ollama | anthropic | openai")
EmbedOpt = typer.Option(None, "--embed-backend", help="Backend used by the 'embed' retriever (defaults to --backend).")


def _backend(name: str):
    try:
        return get_backend(name)
    except BackendUnavailableError as exc:
        console.print(f"[red]Backend error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _default_params(chunker: str) -> dict:
    grid_path = PROJECT_ROOT / "configs" / "grid.yaml"
    if grid_path.exists():
        return load_grid(grid_path).chunkers.get(chunker, {}) or {}
    return {}


@app.command()
def validate(corpus: Path = CorpusOpt, qa: Path = QAOpt) -> None:
    """Check that gold doc ids exist and gold evidence appears verbatim in the corpus."""
    docs, items = load_corpus(corpus), load_qa(qa)
    problems = validate_qa(items, docs)
    n_un = sum(not i.answerable for i in items)
    console.print(f"{len(docs)} documents, {len(items)} questions ({n_un} unanswerable)")
    for p in problems:
        console.print(f"[red]{p}[/red]")
    if problems:
        raise typer.Exit(1)
    console.print("[green]QA set is consistent with the corpus.[/green]")


@app.command()
def chunks(
    strategy: str = typer.Option("heading", "--strategy", "-s", help=f"One of {sorted(CHUNKERS)}"),
    show: int = typer.Option(3, help="Number of example chunks to print."),
    corpus: Path = CorpusOpt,
) -> None:
    """Show chunk statistics for a chunking strategy."""
    docs = load_corpus(corpus)
    cs = chunk_corpus(docs.values(), strategy, **_default_params(strategy))
    lengths = [len(c.text.split()) for c in cs]
    console.print(f"{strategy}: {len(cs)} chunks, words min/mean/max = "
                  f"{min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}")
    for c in cs[:show]:
        console.print(f"[bold]{c.chunk_id}[/bold] {escape(c.text)}\n")


@app.command()
def ask(
    question: str,
    chunker: str = typer.Option("heading", "--chunker", "-c"),
    retriever: str = typer.Option("bm25", "--retriever", "-r"),
    k: int = typer.Option(3, "--k", "-k"),
    backend: str = BackendOpt,
    embed_backend: str = EmbedOpt,
    corpus: Path = CorpusOpt,
) -> None:
    """Answer one question and show retrieval, claim verdicts and citation checks."""
    from .citations import score_citations
    from .claims import score_faithfulness
    from .generation import generate_answer, is_abstention
    from .retrieval import build_retriever

    docs = load_corpus(corpus)
    llm = _backend(backend)
    emb = _backend(embed_backend) if embed_backend else llm
    cs = chunk_corpus(docs.values(), chunker, **_default_params(chunker))
    try:
        hits = build_retriever(retriever, cs, embed_backend=emb).retrieve(question, k)
        answer = generate_answer(llm, question, hits)
    except BackendUnavailableError as exc:
        console.print(f"[red]Backend error:[/red] {exc}")
        raise typer.Exit(2) from exc

    table = Table(title="Retrieved chunks")
    for col in ("rank", "chunk", "score", "text"):
        table.add_column(col)
    for h in hits:
        table.add_row(str(h.rank), h.chunk.chunk_id, f"{h.score:.3f}", escape(h.chunk.text[:100]) + "...")
    console.print(table)
    console.print(f"\n[bold]Answer:[/bold] {escape(answer)}\n")
    if is_abstention(answer):
        console.print("(abstained)")
        return
    faith = score_faithfulness(answer, [(h.chunk.doc_id, h.chunk.text) for h in hits])
    ct = Table(title=f"Claims (faithfulness = {faith.faithfulness:.2f})")
    for col in ("claim", "label", "reason"):
        ct.add_column(col)
    for v in faith.claims:
        color = {"supported": "green", "unsupported": "yellow", "contradicted": "red"}[v.label]
        ct.add_row(escape(v.claim), f"[{color}]{v.label}[/{color}]", escape(v.reason))
    console.print(ct)
    cites = score_citations(answer, docs, {h.chunk.doc_id for h in hits})
    for c in cites.checks:
        verdict = "supports" if c.supported else "does NOT support"
        console.print(escape(f"citation [{c.doc_id}] -> {verdict}: {c.sentence}"))


@app.command()
def evaluate(
    chunker: str = typer.Option("heading", "--chunker", "-c"),
    retriever: str = typer.Option("bm25", "--retriever", "-r"),
    k: int = typer.Option(3, "--k", "-k"),
    backend: str = BackendOpt,
    embed_backend: str = EmbedOpt,
    llm_judge: bool = typer.Option(False, help="Use the backend as an answer-accuracy judge."),
    llm_claims: bool = typer.Option(False, help="Use the backend to split answers into claims."),
    llm_verify: bool = typer.Option(False, help="Use the backend to verify claims."),
    out: Path = typer.Option(None, help="Write per-question results (JSONL) here."),
    corpus: Path = CorpusOpt,
    qa: Path = QAOpt,
) -> None:
    """Evaluate a single chunker/retriever/k configuration."""
    docs, items = load_corpus(corpus), load_qa(qa)
    llm = _backend(backend)
    emb = _backend(embed_backend) if embed_backend else llm
    cfg = RunConfig(chunker=chunker, retriever=retriever, k=k, chunk_params=_default_params(chunker))
    opts = EvalOptions(llm_judge=llm_judge, llm_claims=llm_claims, llm_verify=llm_verify)
    try:
        results = run_config(cfg, docs, items, llm, opts, embed_backend=emb)
    except BackendUnavailableError as exc:
        console.print(f"[red]Backend error:[/red] {exc}")
        raise typer.Exit(2) from exc
    row = summarize(cfg, results)
    table = Table(title=cfg.label)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in row.items():
        if key in ("config", "chunker", "retriever", "k"):
            continue
        table.add_row(key, "-" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value)))
    console.print(table)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(r.model_dump_json() for r in results) + "\n", encoding="utf-8")
        console.print(f"wrote {out}")


@app.command()
def grid(
    config: Path = typer.Option(PROJECT_ROOT / "configs" / "grid.yaml", "--config", help="Grid YAML."),
    backend: str = BackendOpt,
    embed_backend: str = EmbedOpt,
    out: Path = typer.Option(Path("results/grid"), "--out", help="Output directory."),
    llm_judge: bool = typer.Option(False, help="Use the backend as an answer-accuracy judge."),
    llm_claims: bool = typer.Option(False, help="Use the backend to split answers into claims."),
    llm_verify: bool = typer.Option(False, help="Use the backend to verify claims."),
    corpus: Path = CorpusOpt,
    qa: Path = QAOpt,
) -> None:
    """Run the chunker x retriever x k grid and write summary.csv, summary.json, per_question.jsonl and report.md."""
    spec = load_grid(config)
    docs, items = load_corpus(corpus), load_qa(qa)
    llm = _backend(backend)
    emb = _backend(embed_backend) if embed_backend else llm
    opts = EvalOptions(llm_judge=llm_judge, llm_claims=llm_claims, llm_verify=llm_verify)

    def progress(cfg, row):
        console.print(f"  {cfg.label:<26} recall@k={row['recall@k']:.3f} acc={row['accuracy']:.3f} "
                      f"faith={row['faithfulness'] or 0:.3f} success={row['success_rate']:.3f}")

    console.print(f"Running {len(spec.configs())} configurations on backend '{backend}'")
    try:
        rows, per_config = run_grid(spec, docs, items, llm, opts, embed_backend=emb, progress=progress)
    except BackendUnavailableError as exc:
        console.print(f"[red]Backend error:[/red] {exc}")
        raise typer.Exit(2) from exc
    meta = {
        "backend": backend,
        "model": getattr(llm, "model", None),
        "n_docs": len(docs),
        "n_questions": len(items),
        "n_unanswerable": sum(not i.answerable for i in items),
        "chunkers": spec.chunkers,
        "retrievers": spec.retrievers,
        "k": spec.k,
        "options": opts.model_dump(),
    }
    paths = write_outputs(rows, per_config, out, meta)
    table = Table(title="Top configurations")
    for col in ("config", "R@k", "MRR", "Acc", "Faith", "CitP", "Success"):
        table.add_column(col)
    for r in rows[:5]:
        table.add_row(r["config"], *(f"{(r[k] or 0):.3f}" for k in ("recall@k", "mrr", "accuracy", "faithfulness", "citation_precision", "success_rate")))
    console.print(table)
    console.print("Outputs: " + json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    app()
