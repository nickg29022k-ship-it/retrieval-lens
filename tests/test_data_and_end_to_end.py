import json

from typer.testing import CliRunner

from rag_quality_evaluator.cli import app
from rag_quality_evaluator.data import validate_qa
from rag_quality_evaluator.experiment import GridSpec, run_grid, write_outputs
from rag_quality_evaluator.llm import get_backend
from rag_quality_evaluator.pipeline import EvalOptions, RunConfig, run_config, summarize


def test_dataset_shape(corpus, qa):
    assert 20 <= len(corpus) <= 30
    assert 38 <= len(qa) <= 45
    assert sum(not q.answerable for q in qa) >= 5
    assert validate_qa(qa, corpus) == []


def test_single_config_end_to_end(corpus, qa):
    cfg = RunConfig(chunker="heading", retriever="bm25", k=3)
    results = run_config(cfg, corpus, qa, get_backend("mock"), EvalOptions(llm_judge=True))
    assert len(results) == len(qa)
    row = summarize(cfg, results)
    assert 0.8 <= row["recall@k"] <= 1.0
    # the mock injects errors, so grounding metrics must be informative, not perfect
    assert 0.3 < row["faithfulness"] < 1.0
    assert 0.3 < row["citation_precision"] < 1.0
    assert row["n_contradicted_claims"] > 0 and row["n_unsupported_claims"] > 0
    assert row["correct"] > 0 and row["generation_failure"] > 0 and row["correct_abstention"] > 0
    assert sum(row[c] for c in ("correct", "retrieval_failure", "generation_failure", "correct_abstention",
                                "incorrect_abstention", "missed_abstention")) == len(qa)
    assert row["judge_accuracy"] is not None
    for r in results:
        if not r.answerable:
            assert r.retrieval is None and r.answer_scores is None


def test_grid_end_to_end_is_deterministic(corpus, qa, tmp_path):
    grid = GridSpec(chunkers={"fixed": {"size": 50, "overlap": 10}, "heading": {}}, retrievers=["bm25", "tfidf"], k=[1, 3])
    backend = get_backend("mock")
    rows, per_config = run_grid(grid, corpus, qa, backend)
    rows2, _ = run_grid(grid, corpus, qa, backend)
    assert len(rows) == 8
    assert [r["config"] for r in rows] == [r["config"] for r in rows2]
    assert rows == rows2
    # sorted best-first by success rate
    assert rows[0]["success_rate"] == max(r["success_rate"] for r in rows)
    # k=1 can never have better recall than k=3 for the same chunker/retriever
    by = {r["config"]: r for r in rows}
    for ch in ("fixed", "heading"):
        for rt in ("bm25", "tfidf"):
            assert by[f"{ch}/{rt}/k=1"]["recall@k"] <= by[f"{ch}/{rt}/k=3"]["recall@k"]

    paths = write_outputs(rows, per_config, tmp_path, {"backend": "mock"})
    report = paths["report"].read_text()
    for heading in ("## Best configuration", "## Comparison table", "## Failure breakdown", "## Hallucinated claims"):
        assert heading in report
    assert rows[0]["config"] in report
    lines = paths["per_question"].read_text().splitlines()
    assert len(lines) == 8 * len(qa)
    assert json.loads(lines[0])["attribution"]["category"]
    assert json.loads(paths["summary_json"].read_text())["rows"][0]["config"] == rows[0]["config"]


def test_cli_validate_and_evaluate(tmp_path):
    runner = CliRunner()
    res = runner.invoke(app, ["validate"])
    assert res.exit_code == 0, res.output
    out = tmp_path / "res.jsonl"
    res = runner.invoke(app, ["evaluate", "-c", "sentence", "-r", "tfidf", "-k", "2", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "faithfulness" in res.output
    assert out.exists()
    res = runner.invoke(app, ["ask", "What are the names of the two moons of Mars?"])
    assert res.exit_code == 0 and "Phobos" in res.output
