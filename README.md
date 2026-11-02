# RAG Answer Quality Evaluator

A small, self-contained framework for evaluating retrieval-augmented generation (RAG) pipelines. It scores a pipeline on five axes:

1. **Retrieval relevance**: recall@k, precision@k, MRR, nDCG@k against gold passages, plus a reference-free context relevance score.
2. **Answer accuracy**: exact match, normalized match, token F1 against gold answers, and an optional LLM-judge score.
3. **Faithfulness**: each answer is split into atomic claims, and each claim is labelled `supported`, `unsupported` or `contradicted` against the retrieved context.
4. **Citation support**: checks whether each `[doc_id]` citation actually supports the sentence it is attached to (citation precision and recall).
5. **Failure attribution**: each wrong answer is classified as a retrieval failure or a generation failure, and unanswerable questions are scored on abstention.

It also includes a minimal RAG pipeline (three chunkers, three retrievers, and a generator that has to cite its sources) and an experiment grid. The grid compares chunking × retriever × k and writes a comparison table and a Markdown report.

Everything runs offline by default on a deterministic mock LLM. Ollama, Anthropic and OpenAI backends are available through plain HTTP.

## Quickstart

```bash
uv sync
uv run rag-eval validate                    # check the QA set against the corpus
uv run rag-eval chunks --strategy heading   # chunk statistics
uv run rag-eval ask "How many vacuum tubes did ENIAC contain?" -c heading -r bm25 -k 3
uv run rag-eval evaluate -c sentence -r tfidf -k 3 --llm-judge
uv run rag-eval grid --out results/grid     # full grid, writes report.md
uv run pytest
```

`ask` prints the retrieved chunks, the answer, a verdict for each claim and a check for each citation. `evaluate` scores one configuration over the whole QA set. `grid` runs every combination in `configs/grid.yaml` and writes the following files:

- `summary.csv` / `summary.json`: one row per configuration with every metric and failure count
- `per_question.jsonl`: the full record for every question and configuration (retrieved chunks, answer, claim verdicts, citation checks, attribution)
- `report.md`: comparison table, best configuration, failure breakdown, per-factor averages, hallucinated claims, citation errors and failure examples

A grid run on the mock backend is committed in [`examples/mock_grid/`](examples/mock_grid/report.md).

## Data

- `data/corpus/` has 28 short Markdown documents. Sixteen form a handbook for a fictional logistics company, Lumen Freight: PTO, remote work, expenses, security, travel, benefits, a driver handbook, contractor rules and so on. The other twelve are short factual pieces on science (water cycle, tides, Mars, the Moon and more) and computing history (ENIAC, Colossus, ARPANET, UNIX and more). Some documents are deliberate distractors. The driver handbook and the contractor guidelines, for example, give different PTO, allowance and hotel numbers than the general policies.
- `data/qa.jsonl` has 42 questions. 36 are answerable and come with gold answers (with aliases), gold document ids and gold evidence sentences that appear verbatim in the corpus. The other 6 are unanswerable but on-topic, for example "How many vacation days do interns receive?". About a third of the answerable questions are paraphrased so they don't share vocabulary with the source. `rag-eval validate` checks that every evidence string appears in its document.

Gold relevance is defined at the evidence level, so it works for every chunking strategy. A chunk counts as a gold passage if it comes from a gold document and contains at least 70% of the content terms of a gold evidence sentence. The 70% threshold lets a fixed-size window that clips a word or two still count.

## Architecture

```
src/rag_quality_evaluator/
  text.py         tokenizer, stemmer, sentence splitter, number/entity extraction, answer normalisation
  data.py         corpus + QA loading and validation
  chunking.py     fixed (word window + overlap), sentence (packed sentences), heading (markdown sections)
  retrieval.py    BM25 (rank-bm25), TF-IDF cosine (scikit-learn), dense embeddings (any backend with embed())
  llm.py          get_backend(name): mock | ollama | anthropic | openai
  mock.py         deterministic extractive "LLM" with injected hallucinations and citation errors
  generation.py   citation-required prompt, abstention detection, citation parsing
  metrics.py      ranking metrics, gold-passage matching, context relevance, answer metrics, LLM judge
  claims.py       claim splitting (heuristic or LLM) and claim verification (heuristic or LLM)
  citations.py    citation precision / recall
  attribution.py  failure attribution rules
  pipeline.py     run one configuration, aggregate per-config metrics
  experiment.py   grid runner and output writers
  report.py       Markdown report
  cli.py          typer CLI (console script `rag-eval`)
```

For each question in a configuration, the pipeline runs these steps:

```
question -> retriever.retrieve(k) -> prompt with "Document [doc_id]:" blocks -> backend.complete()
         -> retrieval metrics (vs gold evidence)
         -> answer metrics (vs gold answers, optional LLM judge)
         -> claims -> verify each against the retrieved chunks -> faithfulness
         -> per-sentence citations -> verify against the cited documents -> citation P/R
         -> failure attribution
```

The generator prompt requires a `[doc_id]` citation at the end of every sentence. It also requires one fixed abstention sentence when the context does not contain the answer.

## Metric definitions

Retrieval metrics are averaged over answerable questions. In the definitions, *top-k* is the list of retrieved chunks.

| Metric | Definition |
|---|---|
| recall@k | fraction of gold evidence sentences covered by at least one top-k chunk |
| precision@k | (number of top-k chunks that are gold passages) / k |
| MRR | 1 / rank of the first gold passage (0 if none) |
| nDCG@k | binary-relevance DCG with a log2(rank+1) discount, normalised by the ideal DCG of min(k, #gold chunks in the index) relevant items |
| doc recall@k | fraction of gold documents that appear among the top-k chunks |
| context relevance | fraction of retrieved sentences containing at least 40% of the question's content terms. It needs no gold labels and measures how much of the context is on-topic |

The answer metrics are listed below.

| Metric | Definition |
|---|---|
| exact match | normalised prediction equals a normalised gold answer. Normalisation is SQuAD-style: lowercase, no punctuation, articles or citations, and thousands separators removed |
| token F1 | SQuAD token-overlap F1, maximised over gold aliases |
| accuracy | heuristic correctness: exact match, or the normalised gold answer occurs as a token span in the prediction, or every gold token appears in the prediction, or F1 ≥ 0.6. With `--llm-judge` the judge's 0/1 score is used instead |
| judge accuracy | mean LLM-judge score (abstentions count as 0) |

Faithfulness and claim grounding work in three steps.

- **Claim splitting.** The answer is split into sentences, and each sentence is split at clause boundaries (`;`, `, and/but/while/whereas/...`, and `and` followed by a new subject). Fragments shorter than three words are merged back. With `--llm-claims` the backend splits the answer instead, and each claim inherits the citations of the sentence it overlaps most.
- **Claim verification.** The heuristic verifier first finds the context sentence, or adjacent pair of sentences, that covers the most of the claim's content terms.
  - The claim is **contradicted** if coverage is at least 0.5 and the claim contains a number that the evidence does not, while the evidence does contain numbers. A polarity mismatch (negation) under the same coverage condition also counts as a contradiction.
  - The claim is **supported** if coverage is at least 0.6, all of its numbers appear in the evidence, and every proper noun appears somewhere in the context.
  - Every other claim is **unsupported**.

  With `--llm-verify` the backend returns the label instead.
- **Faithfulness** = supported claims / all claims. It is averaged over answered questions. All unsupported and contradicted claims are listed in the report as hallucinated claims.

Citation support follows the ALCE definitions.

- A citation is a (sentence, doc_id) pair. It is *supporting* if at least one claim of the sentence is supported by the full text of the cited document. Unknown document ids are never supporting.
- **Citation precision** = supporting citations / all citations.
- **Citation recall** = sentences whose claims are all supported by the union of their cited documents / all answer sentences. Uncited sentences count against recall.
- Each check also records whether the cited document was in the retrieved context. A citation to a document that was never retrieved is a strong hallucination signal.

## Failure attribution

Each question gets exactly one category. The rules are applied in this order:

| # | Question | Situation | Category |
|---|---|---|---|
| 1 | unanswerable | system abstained | `correct_abstention` |
| 2 | unanswerable | system answered | `missed_abstention` (made-up answer) |
| 3 | answerable | abstained, gold evidence not retrieved | `retrieval_failure` |
| 4 | answerable | abstained, gold evidence retrieved | `incorrect_abstention` |
| 5 | answerable | correct and no hallucinated claims | `correct` |
| 6 | answerable | correct but has unsupported/contradicted claims | `generation_failure` (subtype `unfaithful`) |
| 7 | answerable | wrong, gold evidence not retrieved | `retrieval_failure` |
| 8 | answerable | wrong, gold evidence retrieved | `generation_failure` (subtype `wrong_answer`) |

"Gold evidence retrieved" means every gold evidence sentence is covered by a top-k chunk. The rules keep the two failure sources apart. If the retriever never surfaced the evidence, the generator is not blamed. If the evidence was in the context, a wrong or unfaithful answer is the generator's fault. Rule 3 treats an abstention without evidence as a retrieval failure, because abstaining was the right behaviour given what the generator saw.

The grid ranks configurations by **success rate** = (correct + correct_abstention) / all questions. Ties are broken by faithfulness, then citation precision, then recall@k.

## Sample results (mock backend)

The table shows the top six configurations from `examples/mock_grid/report.md` (42 questions, 18 configurations).

| Config | R@k | P@k | MRR | CtxRel | Acc | Faith | CitP | CitR | Success |
|---|---|---|---|---|---|---|---|---|---|
| sentence/tfidf/k=1 | 0.917 | 0.917 | 0.917 | 0.343 | 0.611 | 0.741 | 0.764 | 0.708 | 0.476 |
| sentence/bm25/k=1 | 0.917 | 0.917 | 0.917 | 0.343 | 0.611 | 0.733 | 0.757 | 0.700 | 0.476 |
| heading/bm25/k=1 | 0.944 | 0.944 | 0.944 | 0.528 | 0.611 | 0.727 | 0.736 | 0.708 | 0.476 |
| heading/bm25/k=3 | 0.972 | 0.324 | 0.954 | 0.278 | 0.611 | 0.727 | 0.681 | 0.653 | 0.476 |
| fixed/bm25/k=3 | 0.972 | 0.389 | 0.931 | 0.176 | 0.556 | 0.741 | 0.697 | 0.671 | 0.429 |
| fixed/tfidf/k=1 | 0.861 | 0.861 | 0.861 | 0.322 | 0.500 | 0.715 | 0.750 | 0.724 | 0.381 |

The failure breakdown for a few of those configurations:

| Config | correct | retrieval_failure | generation_failure | correct_abstention | incorrect_abstention | missed_abstention |
|---|---|---|---|---|---|---|
| heading/bm25/k=3 | 17 | 1 | 15 | 3 | 3 | 3 |
| fixed/bm25/k=1 | 15 | 4 | 16 | 2 | 1 | 4 |
| fixed/tfidf/k=1 | 14 | 5 | 16 | 2 | 1 | 4 |

These are the main patterns in the mock run.

- **Chunking.** Heading-based chunks give the best retrieval (mean MRR 0.951 vs 0.907 for fixed windows). Fixed 50-word windows cut sentences apart and merge section headings into the body, which costs recall at k=1 and produces more retrieval failures.
- **k.** Going from k=1 to k=3 raises recall from 0.912 to 0.972 and cuts retrieval failures from about 3 to 1 per configuration. The extra context lowers context relevance and precision@k, and citation precision drops from 0.749 to 0.689 because the generator now has other documents to mis-cite. k=5 adds nothing over k=3 on this corpus.
- **Retriever.** BM25 and TF-IDF are nearly tied on a corpus this small. BM25 is slightly ahead with fixed chunks.
- **Attribution.** Once k ≥ 3 almost every remaining error is a generation failure. Most of these are injected number changes (for example "within 4 hour" where the source says 1 hour), fabricated sentences, or picking the wrong sentence from a correct passage. Three or four of the six unanswerable questions draw a confident answer instead of an abstention, such as answering the "Asian offices" meal allowance question with the general allowance.
- **Exact match** is 0 everywhere because the mock generator returns whole extracted sentences rather than short spans. That is why the report also shows token F1 and the containment-based accuracy.

The mock numbers show that the metrics separate the error types. They say nothing about any real model.

## Switching backends

```bash
# Ollama (start the server yourself; this project never starts one)
export OLLAMA_MODEL=llama3.1:8b              # default
export OLLAMA_EMBED_MODEL=nomic-embed-text   # used by the embed retriever
uv run rag-eval grid --backend ollama --out results/ollama

# dense retrieval with Ollama embeddings and a different generator
uv run rag-eval evaluate -r embed --embed-backend ollama --backend anthropic

# Anthropic
export ANTHROPIC_API_KEY=...                 # model: ANTHROPIC_MODEL, default claude-haiku-4-5-20251001
uv run rag-eval grid --backend anthropic --llm-judge --llm-verify --out results/anthropic

# OpenAI
export OPENAI_API_KEY=...                    # model: OPENAI_MODEL, default gpt-4o-mini
uv run rag-eval evaluate --backend openai --llm-judge
```

All backends implement `complete(prompt, system=None, temperature=0.0) -> str`. The Ollama, OpenAI and mock backends also implement `embed(texts)`. Network backends raise `BackendUnavailableError` with a clear message when the server is unreachable or the API key is missing, and the CLI exits with code 2. The `--llm-judge`, `--llm-claims` and `--llm-verify` flags reuse the selected backend. With the mock backend they run heuristic stand-ins, so the code paths can be tested offline.

To run the embedding retriever in the grid, uncomment `embed` in `configs/grid.yaml` and pass `--embed-backend ollama`.

## The mock backend

`MockBackend` reads the task header in each prompt. For answering, it acts as a weak extractive reader. It scores every context sentence by overlap with the question terms, adds a small bonus for numbers on "how many / when" questions, and applies a slight position bias toward earlier documents. It then returns the best sentence (sometimes two) with the source document's citation. If no sentence covers at least 40% of the question terms, it abstains. To give the grounding metrics something to find, it injects three kinds of error, each seeded by a hash of the question so that runs are reproducible:

| Error | Default rate | What it does |
|---|---|---|
| contradiction | 8% | changes a number in the answer, or inserts a negation |
| fabrication | 8% | appends a plausible sentence that is in no document |
| wrong citation | 10% | swaps the citation for another document in the context |

Because the rates apply per question with a fixed seed, the realised counts on 42 questions differ from the nominal rates. The committed run has 8 contradicted and 5 unsupported claims in most configurations. The mock embedding is signed feature hashing of unigrams and bigrams. It lets the dense-retriever code run offline, but it is lexical, not semantic.

## Limitations

- The heuristic claim verifier is lexical. It catches number swaps, negation flips, invented entities and off-topic sentences. It misses paraphrased support, which shows up as false "unsupported" labels, and subtle semantic contradictions such as antonyms or swapped roles. `--llm-verify` is the intended fix when a real backend is available.
- Gold relevance uses a 70% term-coverage rule rather than human passage labels.
- Context relevance is a lexical proxy, not a judgment of usefulness.
- The corpus and QA set are small and synthetic. Retrieval is close to saturated at k ≥ 3, so differences between configurations come mostly from k=1 and from chunking.
- The real-backend code paths (Ollama, Anthropic, OpenAI) are implemented, but the test suite only covers their failure modes. Tests never make network calls.
