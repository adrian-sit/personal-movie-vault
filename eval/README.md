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
note in the result JSON. The review follows the rubric:

| Score | Meaning |
|---:|---|
| 10 | Correct or good answer; it answers the question from the relevant evidence. |
| 7–9 | Core answer is correct, with a minor inaccurate, irrelevant, or missing detail. |
| 3–6 | Some useful evidence or a partial answer, but a major omission, error, or off-topic result remains. |
| 1–2 | Relevant material was retrieved, but the answer substantially fails the question. |
| 0 | No answer or a wrong answer. |



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

### 2. Run review-ready retrieval comparisons

The commands below are the standard comparison workflow. Each one writes a
single JSON file containing **all** cases:

- labeled cases, which receive automatic Recall@5 and MRR scores; and
- manual-only cases, such as complete filters and rating rankings, which have
  `null` retrieval metrics but still include retrieved sources, a generated
  answer, and blank `manual_review` fields.

`--include-manual-cases` is therefore required for every reviewed comparison.
Answer generation is on by default. Use a new output filename if you want to
preserve an earlier reviewed run rather than replace it.

```powershell
# Semantic retrieval: compare the two embedding indexes with identical settings.
python eval\run_retrieval_eval.py --retrieval semantic --index data\processed\rag_index_nomic.json --top-k 5 --include-manual-cases --output eval\results\semantic-nomic-reviewed.json
python eval\run_retrieval_eval.py --retrieval semantic --index data\processed\rag_index_mxbai.json --top-k 5 --include-manual-cases --output eval\results\semantic-mxbai-reviewed.json

# Lexical BM25 is model-independent, so run it once as a baseline.
python eval\run_retrieval_eval.py --retrieval bm25 --index data\processed\rag_index_nomic.json --top-k 5 --include-manual-cases --output eval\results\bm25-reviewed.json

# Hybrid retrieval: pair the same BM25 baseline with each embedding index.
python eval\run_retrieval_eval.py --retrieval hybrid --hybrid-weight 0.5 --index data\processed\rag_index_nomic.json --top-k 5 --include-manual-cases --output eval\results\hybrid-nomic-50-reviewed.json
python eval\run_retrieval_eval.py --retrieval hybrid --hybrid-weight 0.5 --index data\processed\rag_index_mxbai.json --top-k 5 --include-manual-cases --output eval\results\hybrid-mxbai-50-reviewed.json
```

Compare `mean_recall_at_k` and `mean_reciprocal_rank` between result files.
Only change one variable per comparison: embedding model, retrieval method, or
hybrid weight. For example, test hybrid weights `0.25`, `0.5`, and `0.75`
against the same index rather than changing the model and weight together.

### 3. Review every generated answer

For each `*-reviewed.json` file, review every `generated_answer` using its
nearby `retrieved_sources`, `expected_answer` (when supplied), and `case_notes`.
The runner leaves these two fields blank deliberately; fill them in without
changing the retrieval fields or generated answer:

```json
"manual_review": {
  "score": 8,
  "notes": "Correct answer, but included an unrelated format detail."
}
```

For example, assess `rank-highest-rated` by whether the response selects the
right movies, puts rating groups in descending order, and handles ties clearly.
It has `recall_at_k: null` and `reciprocal_rank: null` by design, but its manual
score contributes to the method's overall manual average and the `rank`
type-level manual average in the final report.

For a fast, automatic-retrieval-only experiment, omit `--include-manual-cases`
and add `--no-generate-answers`; do not mix those output files with reviewed
comparison files when interpreting manual averages.

### 4. Create the report

After adding manual reviews to every comparison JSON, run:

```powershell
python eval\summarize_results.py
```

It writes [RESULTS.md](RESULTS.md), containing method-level automatic/manual
averages, pooled test-type patterns, and every review below 8/10 with its note.
It uses the manual score from all reviewed cases, including manual-only cases,
while Recall@k and MRR use only the labeled cases.

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
