# Personal Movie Vault

## Content Warning

Some personal movie notes may discuss sensitive topics such as violence, assault, or other difficult themes. The project does not promote, encourage, or endorse harmful behavior. Descriptions of sensitive topics will be kept limited and will avoid unnecessary graphic or explicit language wherever possible.

## Overview

The Personal Movie Vault is a local-first knowledge retrieval system built around my viewing history, ratings and personal thoughts of the movies I have watched in movie theatres since 2025.

The project combines structured movie and viewing metadata with semi-structured personal reflections containing categorized but free-form text. The system uses this information to support semantic search, and question answering through RAG.

The project focuses on the data and retrieval pipeline behind the system, including data ingestion, vector embeddings, and retrieval evaluation.

## Design Consideration

Although this project's data content is relatively small and could potentially fit in a modern LLM's context window, using a RAG system reduces unnecessary context and improves relevancy and efficiency in generation. Plus, this project focuses on smaller local LLMs, the main goal is to balance accuracy while running on consumer hardware.

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
- [ ] Enhance quantitative information retrieval accuracy through data processing
- [ ] Conduct exploratory data analysis
      

## Data sources

- Movie Title
- Director
- Genre

Personal Data:
- Personal ratings
- Viewing dates
- Viewing location and format
- Personal observations
- Favorite characters
- Favorite scenes
- Questions
- Likes and dislikes
- Details most people missed

## Rating Scale

Ratings are given on a scale of **0–10**

- **<5** — Horribly made movie
- **5-5.5** — Boring / made me physically cringe / feels like a waste of time and money, but not poorly made
- **6-6.5** — Includes some elements I enjoy, but overall not very enjoyable
- **7-7.5** — A movie I enjoyed, but not necessarily recommend 
- **8-8.5** — Holds my attention, immersive, emotionally resonant, or provides an escape from reality
- **9-10** — Exceptional experience; completely captivating, deeply emotional, and unforgettable


## Project Structure

```text
data/raw/                 # Hand-authored source of truth
  movies.yaml             # Structured metadata, ratings, and viewings
  <movie-id>.md           # Personal notes, linked with movie_id frontmatter
data/processed/
  movies.json             # YAML converted to JSON
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

`src/yaml_to_json.py` converts `data/raw/movies.yaml` into `movies.json`.
During `rag.py build`, each movie becomes one **metadata chunk** containing its
title, year, director, genres, rating, and viewing history. Every non-empty
Markdown heading (`## Notes`, `## Likes`, `## Favorite Scene`, and so on)
becomes a separate **note chunk**. The Markdown frontmatter `movie_id` must
match a YAML movie `id`; it is the join key between the two sources.

Section-level chunks keep one topic together while preventing an answer about a
dislike or favourite scene from being dominated by an entire review. The
generated `documents.jsonl` exposes these exact chunks for inspection.

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

All variants use the same chunks and return the same number of results, making
them suitable for a controlled retrieval evaluation.

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

Ask a question with retrieval and a local answer:

```powershell
python src\rag.py ask "Which movie did I find most immersive, and why?"
python src\rag.py ask "What did I dislike about Sinners?" --top-k 3
```

### Comparing retrieval and embedding approaches

Use the same question and `--top-k` value for every run, then compare the
printed sources before judging answer quality:

```powershell
# Same embedding index, three retrieval methods
python src\rag.py ask "What did I dislike about Sinners?" --retrieval semantic --top-k 3
python src\rag.py ask "What did I dislike about Sinners?" --retrieval bm25 --top-k 3
python src\rag.py ask "What did I dislike about Sinners?" --retrieval hybrid --hybrid-weight 0.5 --top-k 3

# A second embedding-model index kept alongside the default baseline
ollama pull mxbai-embed-large
python src\rag.py build --embed-model mxbai-embed-large --index data\processed\rag_index_mxbai.json
python src\rag.py ask "What did I dislike about Sinners?" --retrieval semantic --index data\processed\rag_index_mxbai.json --top-k 3
```

The response includes numbered citations and the matched local files. The
derived `documents.jsonl` and vector index are intentionally ignored by Git;
the original YAML and Markdown remain the source of truth.

## Privacy & Copyright

This project is designed as a local-first system.

The data used is either 100% written by me, or publicly available information, such as movie title, director, genre. This project does not involve any APIs or copyrighted/restricted-use content.

The project does not distribute copies of movies, scripts, or other copyrighted media.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
