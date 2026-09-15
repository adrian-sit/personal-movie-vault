# Retrieval Evaluation

This folder holds author-curated cases for comparing retrieval strategies and
embedding indexes. It evaluates retrieval separately from answer generation:
the question is successful when it retrieves the chunks you marked relevant,
regardless of how a chat model later phrases the answer.

## Case structure

Each entry in `cases.yaml` has this shape:

```yaml
- id: unique-and-stable-case-name
  type: extraction | compare | filter | rank | combine
  question: Natural-language question asked of the system
  relevant_document_ids: [movie-id:section:chunk-number]
  expected_answer: Optional author reference, for manual generation review
  automated_retrieval: true | false
  generation_review: manual
  notes: Scope, interpretation, and grading guidance
```

Use document IDs from `data/processed/documents.jsonl` after rebuilding the
index. Leave `relevant_document_ids` empty and set `automated_retrieval: false`
when an answer requires every matching movie, a complete ranking, or an
interpretive judgment that cannot be represented by a finite relevant set.
The runner stops before scoring if a marked document ID is absent from the
selected index, preventing stale chunks from producing misleading metrics.

## What the evaluation measures

Use `type` to organize cases (`extraction`, `compare`, `filter`, `rank`, or
`combine`). The scoring itself has two deliberately separate parts.

### Automatic retrieval metrics

Automatic evaluation answers one question: **did the retriever return the
author-labeled documents, and did it place them near the top?**

- **Recall@k**: what fraction of the marked relevant documents appear in the
  first *k* results.
- **MRR**: how early the first relevant document appears.

Only add `relevant_document_ids` when you can name the relevant chunks. Leave
them empty for a complete filter, tie-sensitive ranking, or open-ended opinion;
those cases still generate an answer but do not affect automatic averages.

### Manual answer review

Review `generated_answer` after each run and record a 0–10 score plus one short
note in the result JSON. Your current review style maps naturally to this
rubric:

| Score | Meaning |
|---:|---|
| 10 | Correct or good answer; it answers the question from the relevant evidence. |
| 7–9 | Core answer is correct, with a minor inaccurate, irrelevant, or missing detail. |
| 3–6 | Some useful evidence or a partial answer, but a major omission, error, or off-topic result remains. |
| 1–2 | Relevant material was retrieved, but the answer substantially fails the question. |
| 0 | No answer or a wrong answer. |

Use the note to identify the failure plainly—for example: “No answer,” “did not
retrieve the Sinners note,” or “included an unrelated movie.” This is more
useful than a generic quality label when reading the report later.

For exact count, filter, and metadata-ranking questions, prefer the structured
analytics layer over RAG. They are not top-*k* retrieval tests: the chat model
interprets natural-language questions against deterministic statistics. Evaluate
them separately for correct record selection and presentation.

## Running comparisons

### 1. Build the indexes

First generate the shared source-derived files, then build one named index per
document-embedding model. Keep the same raw data, `--chunk-model`, cases, and
`--top-k` value across runs. The chunk model must be fixed because it determines
the document IDs that `cases.yaml` labels.

```powershell
python src\yaml_to_json.py

ollama pull nomic-embed-text
python src\rag.py build --embed-model nomic-embed-text --chunk-model nomic-embed-text --index data\processed\rag_index_nomic.json

ollama pull mxbai-embed-large
python src\rag.py build --embed-model mxbai-embed-large --chunk-model nomic-embed-text --index data\processed\rag_index_mxbai.json
```

Use a new descriptive `--index` filename for every model. Do not build each
experiment to `rag_index.json`, because that replaces the prior index. A change
to raw data or the chunk model requires rebuilding every compared index and
checking the case document IDs again.

### 2. Run retrieval comparisons

```powershell
# Semantic retrieval: compare the two embedding indexes with identical settings.
python eval\run_retrieval_eval.py --retrieval semantic --index data\processed\rag_index_nomic.json --top-k 5 --output eval\results\semantic-nomic.json
python eval\run_retrieval_eval.py --retrieval semantic --index data\processed\rag_index_mxbai.json --top-k 5 --output eval\results\semantic-mxbai.json

# Lexical BM25 is model-independent, so run it once as a baseline.
python eval\run_retrieval_eval.py --retrieval bm25 --index data\processed\rag_index_nomic.json --top-k 5 --output eval\results\bm25.json

# Hybrid retrieval: pair the same BM25 baseline with each embedding index.
python eval\run_retrieval_eval.py --retrieval hybrid --hybrid-weight 0.5 --index data\processed\rag_index_nomic.json --top-k 5 --output eval\results\hybrid-nomic-50.json
python eval\run_retrieval_eval.py --retrieval hybrid --hybrid-weight 0.5 --index data\processed\rag_index_mxbai.json --top-k 5 --output eval\results\hybrid-mxbai-50.json
```

Compare `mean_recall_at_k` and `mean_reciprocal_rank` between result files.
Only change one variable per comparison: embedding model, retrieval method, or
hybrid weight. For example, test hybrid weights `0.25`, `0.5`, and `0.75`
against the same index rather than changing the model and weight together.

### 3. Add manual reviews

Every retrieval-evaluation run generates an answer by default, so the same JSON
result includes retrieval metrics and the actual local-model response. Each case
also includes its retrieved source IDs, expected answer (where defined), case
notes, and empty `manual_review` fields. This makes a run slower because it
calls the chat model once per automatically scored case.

```powershell
python eval\run_retrieval_eval.py --retrieval semantic --index data\processed\rag_index_nomic.json --top-k 5 --include-manual-cases --chat-model qwen2.5:3b --output eval\results\semantic-nomic-with-answers.json
```

Open the resulting JSON and score each `generated_answer` using the nearby
`retrieved_sources`, `expected_answer`, and `case_notes`. The supplied
`manual_review.score` and `manual_review.notes` fields are blank deliberately:
replace them after reading the answer:

```json
"manual_review": {
  "score": 8,
  "notes": "Correct answer, but included an unrelated format detail."
}
```

Do not change the retrieval fields or generated answer after the run; add only
your manual review. Save the edited JSON in `eval/results/`.
Cases without relevance labels appear when `--include-manual-cases` is set, but
their retrieval metrics are `null` and they do not affect the aggregate scores.
For a fast retrieval-only run, add `--no-generate-answers`.

### 4. Create the report

After reviewing one or more result files, run:

```powershell
python eval\summarize_results.py
```

It writes [RESULTS.md](RESULTS.md), containing method-level automatic/manual
averages, pooled test-type patterns, and every review below 8/10 with its note.

The runner reports:

- `mean_recall_at_k`: fraction of each case's author-marked relevant chunks
  retrieved in the first *k* results. This rewards evidence coverage.
- `mean_reciprocal_rank` (MRR): rewards placing the first relevant chunk early.
- Per-case retrieved and missing IDs, which make failures inspectable.

Current findings from the saved runs are summarized in [RESULTS.md](RESULTS.md).

## Current results snapshot

The following summary is based on the five reviewed `top_k: 5` runs currently
saved in `eval/results/`. Full per-type findings and review notes are in
[RESULTS.md](RESULTS.md).

| Method / index | Recall@5 | MRR | Manual average (/10) |
|---|---:|---:|---:|
| semantic / mxbai | **0.800** | **0.758** | **8.60** |
| semantic / nomic | 0.750 | 0.633 | 7.70 |
| hybrid / nomic (0.5) | 0.675 | 0.670 | 8.35 |
| hybrid / mxbai (0.5) | 0.625 | 0.695 | 7.65 |
| BM25 / nomic corpus | 0.475 | 0.500 | 5.70 |

Semantic retrieval with the mxbai index is the strongest overall configuration
in this snapshot. Extraction is the most reliable test type; comparison and
multi-chunk combination cases remain the main retrieval weaknesses. BM25 is a
useful exact-term baseline, but it is not the preferred main method for these
notes.
