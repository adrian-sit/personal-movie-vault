# Evaluation Results

Snapshot based on the five saved runs in `eval/results/`, all at `top_k: 5`.
Automatic metrics measure retrieval against the author-labeled document IDs.
Manual averages use the 0–10 scores recorded in those files.

## Overall comparison

| Method / index | Recall@5 | MRR | Manual average (/10) | Result |
|---|---:|---:|---:|---|
| semantic / mxbai | **0.800** | **0.758** | **8.60** | Best overall run |
| semantic / nomic | 0.750 | 0.633 | 7.70 | Strong retrieval; weaker ranking and combined answers |
| hybrid / nomic (0.5) | 0.675 | 0.670 | 8.35 | Strong manual comparison answers, but lower retrieval coverage |
| hybrid / mxbai (0.5) | 0.625 | 0.695 | 7.65 | Good retrieval ordering, inconsistent comparisons |
| BM25 / nomic corpus | 0.475 | 0.500 | 5.70 | Useful lexical baseline, but weakest overall |

Semantic retrieval with `mxbai-embed-large` is the leading configuration in
this snapshot: it has the highest Recall@5, MRR, and manual average. The
hybrid Nomic run is the closest manual-quality competitor, especially on the
two comparison cases, so it is worth retesting if comparisons are a priority.

## Performance by test type

These averages pool all five methods. They describe broad patterns rather than
a single method's score.

| Test type | Mean Recall@5 | Mean manual score (/10) | Interpretation |
|---|---:|---:|---|
| extraction | **0.860** | **8.78** | The most reliable category: focused facts usually retrieve the right section and generate usable answers. |
| compare | 0.650 | 5.40 | Retrieval can find both movies, but generation and irrelevant statistic chunks often weaken the comparison. |
| rank | 0.300 | 6.80 | Subjective ranking is unstable; a relevant chunk does not guarantee a defensible ordering. |
| combine | **0.325** | 7.25 | The hardest retrieval task: required evidence is spread across several metadata and note chunks. |

## Patterns to investigate

- BM25 failed every comparison case in this snapshot and several note-based
  extraction/combination cases. It remains a useful baseline for exact terms,
  not the preferred main strategy.
- The semantic mxbai index was consistently strongest for extraction. Its
  remaining weak points were the Cars/Toy Story comparison and noisy evidence in
  the immersive-ranking question.
- Statistic chunks can displace personal notes for IMAX questions. This helps
  factual format questions, but can hurt questions that combine format data with
  personal reactions; the hybrid mxbai IMAX comparison review identifies this
  directly.
- Reviews below 8/10 repeatedly mention a missing note, an irrelevant movie or
  statistic, or an answer that gives up despite retrieving related material.
  Treat these as prompt/context-selection and multi-hop-retrieval issues, not
  only embedding-model failures.

Regenerate this report after adding reviews or new runs:

```powershell
python eval\summarize_results.py
```
