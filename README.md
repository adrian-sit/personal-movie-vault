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
- [ ] Retrieval evaluation
- [ ] Enhance quantitative information retrieval accuracy through data processing
- [ ] Conduct exploratory data analysis
- [ ] Experiment with fine-tuned embedding models
- [ ] Experiment with multiple local LLMs
      

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

The response includes numbered citations and the matched local files. The
derived `documents.jsonl` and vector index are intentionally ignored by Git;
the original YAML and Markdown remain the source of truth.

## Privacy & Copyright

This project is designed as a local-first system.

The data used is either 100% written by me, or publicly available information, such as movie title, director, genre. This project does not involve any APIs or copyrighted/restricted-use content.

The project does not distribute copies of movies, scripts, or other copyrighted media.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
