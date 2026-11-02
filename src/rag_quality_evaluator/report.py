"""Markdown report for an experiment grid."""

from __future__ import annotations

from collections import defaultdict

from .attribution import CATEGORIES
from .pipeline import QuestionResult


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


METRIC_COLUMNS = [
    ("recall@k", "R@k"),
    ("precision@k", "P@k"),
    ("mrr", "MRR"),
    ("ndcg@k", "nDCG@k"),
    ("context_relevance", "CtxRel"),
    ("exact_match", "EM"),
    ("token_f1", "F1"),
    ("accuracy", "Acc"),
    ("faithfulness", "Faith"),
    ("citation_precision", "CitP"),
    ("citation_recall", "CitR"),
    ("abstention_accuracy", "AbstAcc"),
    ("success_rate", "Success"),
]


def comparison_table(rows: list[dict]) -> str:
    headers = ["Config"] + [h for _, h in METRIC_COLUMNS]
    return md_table(headers, [[r["config"]] + [r[k] for k, _ in METRIC_COLUMNS] for r in rows])


def failure_table(rows: list[dict]) -> str:
    headers = ["Config"] + list(CATEGORIES) + ["unsupported claims", "contradicted claims"]
    body = [[r["config"]] + [r[c] for c in CATEGORIES] + [r["n_unsupported_claims"], r["n_contradicted_claims"]] for r in rows]
    return md_table(headers, body)


def marginal_table(rows: list[dict], factor: str) -> str:
    keys = ["recall@k", "mrr", "accuracy", "faithfulness", "citation_precision", "success_rate", "retrieval_failure", "generation_failure"]
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[factor]].append(r)
    body = []
    for level in sorted(groups):
        rs = groups[level]
        vals = []
        for k in keys:
            xs = [r[k] for r in rs if r[k] is not None]
            vals.append(sum(xs) / len(xs) if xs else None)
        body.append([level] + vals)
    return md_table([factor] + keys, body)


def render_report(rows: list[dict], per_config: dict[str, list[QuestionResult]], meta: dict) -> str:
    best = rows[0]
    best_results = per_config[best["config"]]
    best_retrieval = max(rows, key=lambda r: ((r["recall@k"] or 0), (r["mrr"] or 0)))
    most_faithful = max(rows, key=lambda r: (r["faithfulness"] or 0))

    out: list[str] = []
    out.append("# RAG Answer Quality Report\n")
    out.append(
        f"- Backend: `{meta.get('backend')}` (model `{meta.get('model')}`)\n"
        f"- Corpus: {meta.get('n_docs')} documents, QA set: {meta.get('n_questions')} questions "
        f"({meta.get('n_unanswerable')} unanswerable)\n"
        f"- Grid: chunkers {meta.get('chunkers')}, retrievers {meta.get('retrievers')}, k {meta.get('k')} "
        f"({len(rows)} configurations)\n"
        f"- Options: {meta.get('options')}\n"
    )

    out.append("## Best configuration\n")
    out.append(
        f"**{best['config']}** has the highest end-to-end success rate "
        f"({fmt(best['success_rate'])}: correct and faithful answers plus correct abstentions, over all questions). "
        f"Ties are broken by faithfulness, then citation precision, then recall@k.\n"
    )
    out.append(
        f"- Retrieval: recall@k {fmt(best['recall@k'])}, MRR {fmt(best['mrr'])}, nDCG@k {fmt(best['ndcg@k'])}\n"
        f"- Answers: accuracy {fmt(best['accuracy'])}, token F1 {fmt(best['token_f1'])}, EM {fmt(best['exact_match'])}\n"
        f"- Grounding: faithfulness {fmt(best['faithfulness'])}, citation precision {fmt(best['citation_precision'])}, "
        f"citation recall {fmt(best['citation_recall'])}\n"
        f"- Best pure retrieval config: {best_retrieval['config']} (recall@k {fmt(best_retrieval['recall@k'])}, MRR {fmt(best_retrieval['mrr'])})\n"
        f"- Most faithful config: {most_faithful['config']} (faithfulness {fmt(most_faithful['faithfulness'])})\n"
    )

    out.append("## Comparison table\n")
    out.append("Sorted by success rate. Retrieval and accuracy columns are averaged over answerable questions; "
               "faithfulness and citation columns over questions that were answered (not abstained).\n")
    out.append(comparison_table(rows) + "\n")

    out.append("## Failure breakdown per configuration\n")
    out.append("Counts of questions in each attribution category, plus claim-level hallucination counts.\n")
    out.append(failure_table(rows) + "\n")

    out.append("## Effect of each factor\n")
    out.append("Each row averages over all configurations that share that setting.\n")
    for factor in ("chunker", "retriever", "k"):
        out.append(f"### By {factor}\n")
        out.append(marginal_table(rows, factor) + "\n")

    out.append(f"## Hallucinated claims in the best configuration ({best['config']})\n")
    hallucinated = [(r, c) for r in best_results for c in r.faithfulness.claims if c.label != "supported"]
    if hallucinated:
        out.append(md_table(
            ["Question", "Claim", "Label", "Reason"],
            [[r.qid, c.claim, c.label, c.reason] for r, c in hallucinated],
        ) + "\n")
    else:
        out.append("No unsupported or contradicted claims.\n")

    out.append(f"## Citation errors in the best configuration ({best['config']})\n")
    bad_cites = [(r, c) for r in best_results for c in r.citations.checks if not c.supported]
    if bad_cites:
        out.append(md_table(
            ["Question", "Cited doc", "In retrieved context", "Sentence"],
            [[r.qid, c.doc_id, c.in_context, c.sentence] for r, c in bad_cites],
        ) + "\n")
    else:
        out.append("Every citation supports its sentence.\n")

    out.append(f"## Failure examples ({best['config']})\n")
    examples = [r for r in best_results if r.attribution.category not in ("correct", "correct_abstention")]
    if examples:
        out.append(md_table(
            ["Question", "Category", "Reason", "Answer", "Gold", "Retrieved docs"],
            [[r.qid, r.attribution.category, r.attribution.reason, r.answer.replace("|", "/"),
              "; ".join(r.gold_answers) or "(unanswerable)", ", ".join(r.retrieved_docs)] for r in examples],
        ) + "\n")
    else:
        out.append("No failures.\n")
    return "\n".join(out)
