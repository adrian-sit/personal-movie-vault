# Personal Movie Vault

## Content Warning

Some personal movie notes may discuss sensitive topics such as violence, assault, or other difficult themes. The project does not promote, encourage, or endorse harmful behavior. Descriptions of sensitive topics will be kept limited and will avoid unnecessary graphic or explicit language wherever possible.

## Overview

The Personal Movie Vault is a local-first knowledge retrieval system built around my viewing history, ratings and personal thoughts of the movies I have watched in movie theatres since 2025.

The project combines structured movie and viewing metadata with semi-structured personal reflections containing categorized but free-form text. The system uses this information to support semantic search, and question answering through RAG.

The project focuses on the data and retrieval pipeline behind the system, including data ingestion, vector embeddings, and retrieval evaluation.

## Design Consideration

Although this project's data content is relatively small and could fit in a modern LLM's context window (including small local models), using a RAG system reduces unnecessary context and improves relevancy and efficiency in generation. Also, this avoids any long context problems, for example lost in the middle, the main goal is to balance accuracy while running on consumer hardware.

## Goals

- Building an end-to-end data ingestion and transformation pipeline
- Combining structured and unstructured data
- Semantic search using vector embeddings
- RAG with a local LLM
- Evaluating different retrieval strategies

## Planned Features

- [ ] Organize personal ratings and notes
- [ ] Build a movie knowledge base
- [ ] Semantic search
- [ ] Vector embeddings
- [ ] Hybrid keyword and semantic retrieval
- [ ] Retrieval-augmented generation (RAG)
- [ ] Build a semi-automatic retrieval evaluation pipeline
- [ ] Perform data processing and conduct exploratory data analysis
      

## Raw Dataset Format

The hand-authored dataset is stored under `data/raw/` in two complementary
formats. These files are the source of truth; everything in `data/processed/`
can be regenerated from them.

### Movie metadata: `movies.yaml`

One YAML object represents one movie. It records structured facts and
viewing history:

```yaml
- id: the-odyssey
  title: The Odyssey
  year: 2026
  director: Christopher Nolan
  genre: [Adventure, Action, Fantasy]
  rating: 8.5
  viewings:
    - date_watched: July 2026
      location: Cineplex Cinemas Langley
      format: IMAX 70MM
      rewatch: false
```

`id` is the stable identifier used to connect this record with its personal
note. `rating` follows the personal [rating scale](docs/rating-scale.md), which
is kept separately.

### Personal notes: `<movie-id>.md`

Each movie can have a Markdown note file named after its ID. YAML frontmatter
links it to the corresponding `movies.yaml` record, and Markdown headings keep
free-form reflections in recognizable categories:

```md
---
movie_id: the-odyssey
title: The Odyssey
---

## Notes
Free-form viewing reactions, interpretations, and observations.

## Favorite Character
Odysseus

## Favorite Scene
The bow sequence when Odysseus returned.

## Likes
- Great visuals

## Dislikes
- Any negative reactions or criticisms.
```

The note schema is intentionally flexible. Common sections include `Notes`,
`Favorite Character`, `Favorite Scene`, `Questions`, `Likes`, `Dislikes`, and
`Details: Facts Or Theories`; sections may be empty or omitted. During indexing,
the frontmatter `movie_id` joins each note to its structured metadata, while
the note text is semantically chunked for retrieval.


## Project Structure

```text
data/raw/                 # Hand-authored source of truth
  movies.yaml             # Structured metadata, ratings, and viewings
  <movie-id>.md           # Personal notes, linked with movie_id frontmatter
data/processed/
  movies.json             # YAML converted to JSON
  viewing_statistics.json # Exact aggregate counts derived from YAML
  documents.jsonl         # Inspectable retrieval chunks (generated)
  rag_index*.json         # Local embeddings and chunks (generated, ignored)
src/
  yaml_to_json.py         # Structured-data transformation
  rag.py                  # Index construction, retrieval, and generation
```

## RAG Design

The system is intentionally framework-free: it uses Python's standard library,
PyYAML, and Ollama's local HTTP API rather than a hosted RAG service, LangChain,
or a separate vector database. This keeps the corpus, embeddings, and model
inference on the local machine.

### Ingestion and chunking

`src/yaml_to_json.py` converts `data/raw/movies.yaml` into `movies.json` and
derives `viewing_statistics.json`. The statistics file contains exact totals,
unique-movie and viewing counts per cinema, rewatch counts, formats, and viewing
years. It is the source used for quantitative questions; the language model is
not asked to count RAG chunks.

During `rag.py build`, each movie becomes one **metadata chunk** containing its
title, year, director, genres, rating, and viewing history. Markdown
frontmatter `movie_id` must match a YAML movie `id`; it is the join key between
the two sources.

Notes use **semantic chunking** rather than fixed paragraph chunks. The script
preserves each Markdown heading (`## Notes`, `## Likes`, `## Favorite Scene`,
and so on) as metadata, then splits prose into sentences and list sections into
individual list items. It embeds those units locally and measures the semantic
distance between each neighbouring pair. Large meaning shifts become candidate
chunk boundaries.

The chunker only accepts a semantic boundary after at least approximately 80
words, choosing boundaries in the most dissimilar 20% of neighbouring unit
pairs. A 280-word maximum prevents a long passage from becoming one broad
chunk; an unusually long sentence is split by words only as a final fallback.
This means a change from reactions about the cast to a theatre-format
observation can form two chunks even when both occur in one paragraph, while a
short `Likes` list remains together. Every chunk retains its movie title,
section, source path, position, and total number of chunks in the section.
`documents.jsonl` exposes the exact chunks used for indexing.

By default, semantic boundaries use `nomic-embed-text` via `--chunk-model`.
This is intentionally separate from `--embed-model`: keep the chunk model
fixed when comparing document embedding models, so the evaluation measures
retrieval rather than a changed corpus.

### Embedding index

The default embedding model is Ollama's `nomic-embed-text`. It turns each chunk
into a numerical vector that represents its semantic meaning. `rag.py build`
stores the vector together with the chunk text and source metadata in a local
JSON index. This is a deliberately simple vector store: search scans every
vector and calculates cosine similarity. It is transparent and sufficiently
fast for a personal library; a dedicated vector database such as Qdrant or
Chroma would be a later scaling choice, not a requirement now.

### Answer generation

At question time, the top-ranked chunks are passed to the local `qwen2.5:3b`
model with instructions to answer only from that context and cite it as `[1]`,
`[2]`, and so on. The printed source list lets you verify the evidence behind an
answer. The notes are explicitly treated as personal opinions, not objective
movie facts.

## Retrieval Variants

All variants use this same semantically chunked corpus and return the same
number of results, making them suitable for a controlled retrieval evaluation.

| Variant | Command option | Method | What it tests |
|---|---|---|---|
| Semantic baseline | `--retrieval semantic` | Cosine similarity between the query and `nomic-embed-text` vectors | Meaning-based matching, including paraphrases |
| Lexical baseline | `--retrieval bm25` | BM25 term-frequency ranking, implemented locally | Exact titles, character names, places, and other rare terms |
| Hybrid | `--retrieval hybrid` | Min–max-normalized semantic and BM25 scores combined with a weighted sum | Whether semantic meaning plus exact-term matching improves recall |

For hybrid retrieval, `--hybrid-weight` is the semantic contribution: `0` is
purely lexical, `1` is purely semantic, and the default `0.5` weights both
equally. BM25 needs no embedding request at query time, while semantic and
hybrid search embed the question through Ollama.

To compare embedding models without overwriting a prior index, build named
indexes. For example, `nomic-embed-text` is a compact baseline; an alternative
such as `mxbai-embed-large` may improve retrieval but uses more resources.

## Setup

This project uses [Ollama](https://ollama.com/) so both the embeddings and the
language model stay on your computer. Install Ollama, then from PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python src\yaml_to_json.py
python src\rag.py build
```

`qwen2.5:3b` is a good small default. On a machine with more RAM, you can use
`qwen2.5:7b` by passing `--chat-model qwen2.5:7b` when asking a question.

## Usage

Add a movie's factual data to `data/raw/movies.yaml` and its reflection to a
Markdown file in `data/raw/`. The Markdown frontmatter must include the same
`movie_id` as the YAML entry. Run the two build commands above after changing
either source.

When changing a source file, the semantic chunking model, or the document
embedding model, rebuild the index before asking questions:

```powershell
python src\yaml_to_json.py
python src\rag.py build
```

Ask a question with retrieval and a local answer:

```powershell
python src\rag.py ask "Which movie did I find most immersive, and why?"
python src\rag.py ask "What did I dislike about Sinners?" --top-k 3
```

### Exact viewing-statistics questions

Count questions are routed to the deterministic aggregates in
`viewing_statistics.json`, rather than retrieval and generation. This prevents
the common RAG failure mode of counting only the retrieved documents.

```powershell
python src\rag.py ask "How many movies have I watched in total?"
python src\rag.py ask "How many movies did I watch at Cineplex Cinemas Langley?"
python src\rag.py ask "How many viewings did I record at Scotiabank Theatre Vancouver?"
```

For these questions, “movies” means distinct movies and “viewings” includes
rewatches. Run `python src\yaml_to_json.py` after editing `movies.yaml` so the
statistics remain current.

The response includes numbered citations and the matched local files. The
derived `documents.jsonl` and vector index are intentionally ignored by Git;
the original YAML and Markdown remain the source of truth.

## Evaluation

Author-curated retrieval cases and a comparison runner live in
[`eval/`](eval/README.md). They support consistent evaluation of semantic,
BM25, and hybrid retrieval across multiple embedding indexes, while keeping
subjective answer quality as a separate manual review.

## Privacy & Copyright

This project is designed as a local-first system.

The data used is either 100% written by me, or publicly available information, such as movie title, director, genre. This project does not involve any APIs or copyrighted/restricted-use content.

The project does not distribute copies of movies, scripts, or other copyrighted media.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
