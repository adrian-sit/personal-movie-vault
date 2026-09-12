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

## Test categories

| Type | Purpose | What to score automatically | What to review manually |
|---|---|---|---|
| `extraction` | Find one explicit fact in a note | Retrieval of the known source chunk; optionally exact deterministic extraction | Whether the generated wording is faithful and cited |
| `compare` | Bring evidence for two or more movies together | Recall of every marked evidence chunk | Balance, completeness, and whether the comparison invents a conclusion |
| `filter` | Return every record matching a condition | Only if you define the complete matching metadata-ID set | Completeness and interpretation of terms such as “IMAX” |
| `rank` | Order records by a value or subjective criterion | Retrieval presence for subjective evidence only | Ordering, ties, and subjective ranking rationale |
| `combine` | Join metadata with notes, or synthesize several facts | Recall across the required chunks | Multi-hop reasoning, citations, and answer completeness |

For exact count, filter, and metadata-ranking questions, prefer the structured
analytics layer over RAG. They are not top-*k* retrieval tests: the chat model
interprets natural-language questions against deterministic statistics. Evaluate
them separately for correct record selection and presentation, ideally with a
small author-reviewed set of varied phrasings rather than one fixed template.

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

The runner reports:

- `mean_recall_at_k`: fraction of each case's author-marked relevant chunks
  retrieved in the first *k* results. This rewards evidence coverage.
- `mean_reciprocal_rank` (MRR): rewards placing the first relevant chunk early.
- Per-case retrieved and missing IDs, which make failures inspectable.

The runner deliberately does not score generated prose. After selecting a
retrieval configuration, use the `expected_answer` and `notes` fields as a
small manual rubric for factual grounding, citation quality, completeness, and
whether an interpretation matches your intent.
