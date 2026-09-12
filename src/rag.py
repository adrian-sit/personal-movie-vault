"""Local RAG for the Personal Movie Vault.

Requires a running Ollama server only when building embeddings or generating
answers. The generated index contains your source text and vectors locally;
it is never uploaded by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MOVIES = PROCESSED / "movies.json"
DOCUMENTS = PROCESSED / "documents.jsonl"
INDEX = PROCESSED / "rag_index.json"
STATISTICS = PROCESSED / "viewing_statistics.json"
OLLAMA_URL = "http://localhost:11434"
SEMANTIC_CHUNK_MIN_WORDS = 80
SEMANTIC_CHUNK_MAX_WORDS = 280
SEMANTIC_BREAK_PERCENTILE = 0.80
BM25_STOP_WORDS = {
    "a", "an", "and", "are", "at", "did", "do", "does", "for", "from", "have", "how", "i", "in", "is",
    "it", "many", "me", "my", "of", "on", "or", "show", "the", "to", "was", "watch", "watched", "what",
    "which", "with", "would", "you",
}


def ollama(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach Ollama at http://localhost:11434. Start Ollama and "
            "pull the models listed in the README."
        ) from exc


def embed(texts: list[str], model: str) -> list[list[float]]:
    # /api/embed accepts a batch and is available in current Ollama releases.
    data = ollama("/api/embed", {"model": model, "input": texts})
    return data["embeddings"]


def parse_note(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, flags=re.S)
    frontmatter: dict[str, str] = {}
    body = text
    if match:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = match.group(2)
    sections = re.split(r"^##\s+(.+?)\s*$", body, flags=re.M)
    pairs: list[tuple[str, str]] = []
    for position in range(1, len(sections), 2):
        heading, content = sections[position].strip(), sections[position + 1].strip()
        if content:
            pairs.append((heading, content))
    return frontmatter, pairs


def word_count(text: str) -> int:
    return len(re.findall(r"[\w']+", text))


def semantic_units(content: str) -> list[str]:
    """Split prose into sentences and preserve list items as individual units."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if re.search(r"(?m)^(?:[-*+] |\d+[.)] )", paragraph):
            candidates = [line.strip() for line in paragraph.splitlines() if line.strip()]
        else:
            candidates = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
        for candidate in candidates:
            if word_count(candidate) <= SEMANTIC_CHUNK_MAX_WORDS:
                units.append(candidate)
            else:
                words = candidate.split()
                units.extend(" ".join(words[start:start + SEMANTIC_CHUNK_MAX_WORDS])
                             for start in range(0, len(words), SEMANTIC_CHUNK_MAX_WORDS))
    return units


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def semantic_chunk_section(content: str, model: str) -> list[str]:
    """Create chunks at large local-embedding meaning shifts between note units."""
    units = semantic_units(content)
    if len(units) < 2:
        return units
    vectors = embed(units, model)
    distances = [1 - cosine(left, right) for left, right in zip(vectors, vectors[1:])]
    # Avoid treating the only boundary in a short note as inherently meaningful.
    semantic_threshold = percentile(distances, SEMANTIC_BREAK_PERCENTILE) if len(distances) >= 3 else math.inf
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for position, unit in enumerate(units):
        unit_size = word_count(unit)
        should_break_for_size = current and current_size + unit_size > SEMANTIC_CHUNK_MAX_WORDS
        previous_distance = distances[position - 1] if position else 0.0
        should_break_for_meaning = (
            current_size >= SEMANTIC_CHUNK_MIN_WORDS
            and previous_distance >= semantic_threshold
        )
        if should_break_for_size or should_break_for_meaning:
            chunks.append("\n\n".join(current))
            current, current_size = [], 0
        current.append(unit)
        current_size += unit_size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def movie_text(movie: dict[str, Any]) -> str:
    viewings = "; ".join(
        f"{v.get('date_watched', 'unknown date')} at {v.get('location', 'unknown location')} "
        f"({v.get('format', 'unknown format')}, rewatch: {v.get('rewatch', False)})"
        for v in movie.get("viewings", [])
    )
    return (
        f"Movie: {movie['title']} ({movie.get('year', 'unknown')})\n"
        f"Director: {movie.get('director', 'unknown')}\n"
        f"Genres: {', '.join(movie.get('genre', []))}\n"
        f"Personal rating: {movie.get('rating', 'unrated')}/10\n"
        f"Viewings: {viewings or 'none recorded'}"
    )


def make_documents(chunk_model: str) -> list[dict[str, Any]]:
    if not MOVIES.exists():
        raise RuntimeError("data/processed/movies.json is missing. Run src/yaml_to_json.py first.")
    movies = json.loads(MOVIES.read_text(encoding="utf-8"))
    by_id = {movie["id"]: movie for movie in movies}
    documents = make_statistics_documents()
    for movie in movies:
        documents.append({
            "id": f"{movie['id']}:metadata",
            "movie_id": movie["id"],
            "title": movie["title"],
            "section": "Metadata",
            "source": "data/raw/movies.yaml",
            "chunk_index": 1,
            "chunk_count": 1,
            "text": movie_text(movie),
        })
    for note_path in RAW.glob("*.md"):
        frontmatter, sections = parse_note(note_path)
        movie_id = frontmatter.get("movie_id")
        if not movie_id or movie_id not in by_id:
            print(f"Skipping {note_path.name}: no matching movie_id", file=sys.stderr)
            continue
        for heading, content in sections:
            chunks = semantic_chunk_section(content, chunk_model)
            heading_id = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            for position, chunk in enumerate(chunks, start=1):
                documents.append({
                    "id": f"{movie_id}:{heading_id}:{position}",
                    "movie_id": movie_id,
                    "title": by_id[movie_id]["title"],
                    "section": heading,
                    "source": f"data/raw/{note_path.name}",
                    "chunk_index": position,
                    "chunk_count": len(chunks),
                    "text": f"Movie: {by_id[movie_id]['title']}\nSection: {heading}\n{chunk}",
                })
    return documents


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def tokenize(text: str) -> list[str]:
    """Keep meaningful terms for the lexical BM25 baseline."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in BM25_STOP_WORDS]


def bm25_scores(question: str, documents: list[dict[str, Any]]) -> list[float]:
    """Return lexical BM25 scores without adding a search-engine dependency."""
    query_terms = tokenize(question)
    tokenized_documents = [tokenize(document["text"]) for document in documents]
    document_count = len(documents)
    average_length = sum(map(len, tokenized_documents)) / document_count if document_count else 0
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized_documents)
        for term in set(query_terms)
    }
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in tokenized_documents:
        term_frequency = {term: tokens.count(term) for term in set(query_terms)}
        score = 0.0
        for term in set(query_terms):
            frequency = term_frequency[term]
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (document_count - document_frequency[term] + 0.5) /
                                         (document_frequency[term] + 0.5))
            length_factor = k1 * (1 - b + b * len(tokens) / average_length)
            score += inverse_frequency * frequency * (k1 + 1) / (frequency + length_factor)
        scores.append(score)
    return scores


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 if score else 0.0 for score in scores]
    return [(score - low) / (high - low) for score in scores]


def index_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def make_statistics_documents() -> list[dict[str, Any]]:
    """Expose deterministic aggregates as retrievable, citable RAG documents."""
    if not STATISTICS.exists():
        raise RuntimeError("Viewing statistics are missing. Run `python src/yaml_to_json.py` first.")
    statistics = json.loads(STATISTICS.read_text(encoding="utf-8"))
    source = "data/processed/viewing_statistics.json"
    totals = statistics["totals"]
    documents = [{
        "id": "statistics:overview",
        "movie_id": None,
        "title": "Viewing Statistics",
        "section": "Overview",
        "source": source,
        "statistics_dimension": "overview",
        "statistics_value": "total",
        "chunk_index": 1,
        "chunk_count": 1,
        "text": (
            "Structured data answer for: how many movies have I watched in total? "
            "Deterministic viewing statistics. "
            f"Total unique movies: {totals['unique_movies']}. "
            f"Total viewings: {totals['viewing_count']}. "
            f"Rewatches: {totals['rewatch_count']}."
        ),
    }]

    def add_group_documents(groups: list[dict[str, Any]], group_key: str, section: str) -> None:
        for group in groups:
            value = group[group_key]
            document_id = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
            titles = "; ".join(movie["title"] for movie in group["movies"])
            documents.append({
                "id": f"statistics:{section.lower()}:{document_id}",
                "movie_id": None,
                "title": "Viewing Statistics",
                "section": section,
                "source": source,
                "statistics_dimension": section.lower(),
                "statistics_value": value,
                "chunk_index": 1,
                "chunk_count": 1,
                "text": (
                    f"Structured data answer for: which movies did I watch in {value}, "
                    f"and how many movies or viewings did I record there? "
                    f"Deterministic statistics for {section.lower()} {value}. "
                    f"Unique movies: {group['unique_movies']}. "
                    f"Viewings: {group['viewing_count']}. "
                    f"Movies: {titles or 'none'}."
                ),
            })

    add_group_documents(statistics["cinemas"], "location", "Cinema")
    add_group_documents(statistics["formats"], "format", "Format")
    for year, count in statistics["viewings_by_year"].items():
        documents.append({
            "id": f"statistics:year:{year}",
            "movie_id": None,
            "title": "Viewing Statistics",
            "section": "Viewing Year",
            "source": source,
            "statistics_dimension": "viewing year",
            "statistics_value": year,
            "chunk_index": 1,
            "chunk_count": 1,
            "text": f"Deterministic viewing statistics for {year}. Total viewings: {count}.",
        })
    return documents


def build(args: argparse.Namespace) -> None:
    documents = make_documents(args.chunk_model)
    vectors = embed([doc["text"] for doc in documents], args.embed_model)
    for document, vector in zip(documents, vectors):
        document["embedding"] = vector
    output_index = index_path(args.index)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_index.write_text(json.dumps({"embed_model": args.embed_model, "documents": documents}), encoding="utf-8")
    with DOCUMENTS.open("w", encoding="utf-8") as output:
        for document in documents:
            output.write(json.dumps({key: value for key, value in document.items() if key != "embedding"}) + "\n")
    print(f"Indexed {len(documents)} documents in {output_index.relative_to(ROOT)}")


def search(question: str, top_k: int, retrieval: str, hybrid_weight: float, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError("Index not found. Run `python src/rag.py build` first.")
    index = json.loads(path.read_text(encoding="utf-8"))
    documents = index["documents"]
    lexical_scores = bm25_scores(question, documents)
    semantic_scores = [0.0] * len(documents)
    if retrieval in {"semantic", "hybrid"}:
        query_vector = embed([question], index["embed_model"])[0]
        semantic_scores = [cosine(query_vector, document["embedding"]) for document in documents]
    if retrieval == "semantic":
        scores = semantic_scores
    elif retrieval == "bm25":
        scores = lexical_scores
    else:
        dense = min_max_normalize(semantic_scores)
        lexical = min_max_normalize(lexical_scores)
        scores = [hybrid_weight * dense_score + (1 - hybrid_weight) * lexical_score
                  for dense_score, lexical_score in zip(dense, lexical)]
    question_terms = set(tokenize(question))
    for position, document in enumerate(documents):
        value = document.get("statistics_value")
        if not value:
            continue
        value_terms = set(tokenize(str(value))) - {"cinema", "cinemas", "theatre", "theatres", "theater"}
        if value_terms and value_terms <= question_terms:
            # Prefer an exact structured-data value (IMAX) over a related one (IMAX 70MM).
            scores[position] += 100
    ranked = sorted(
        ({**document, "score": score} for document, score in zip(documents, scores)),
        key=lambda item: item["score"], reverse=True,
    )
    return ranked[:top_k]


def ask(args: argparse.Namespace) -> None:
    results = search(args.question, args.top_k, args.retrieval, args.hybrid_weight, index_path(args.index))
    context = "\n\n".join(
        f"[{i}] {item['title']} — {item['section']} ({item['source']})\n{item['text']}"
        for i, item in enumerate(results, 1)
    )
    prompt = (
        "Answer only from the supplied personal movie-vault context. If it does not "
        "contain the answer, say so. Treat notes as the viewer's opinion, not "
        "objective fact. When a source is viewing_statistics.json, treat it as the "
        "authoritative deterministic source for counts, filters, formats, locations, "
        "and viewing records; do not calculate from partial movie chunks. For a "
        "question asking which movies match a statistics entry, return every title "
        "in that entry's `Movies:` list, not one example. Respect exact labels: "
        "`IMAX` and `IMAX 70MM` are distinct formats unless the question explicitly "
        "asks to combine them. Cite each claim with [1], [2], etc.\n\n"
        f"Context:\n{context}\n\nQuestion: {args.question}\nAnswer:"
    )
    answer = ollama("/api/generate", {"model": args.chat_model, "prompt": prompt, "stream": False})["response"].strip()
    print(answer)
    print(f"\nSources ({args.retrieval} retrieval):")
    for i, item in enumerate(results, 1):
        print(f"[{i}] {item['id']} — {item['title']} — {item['section']} ({item['source']}; score {item['score']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a local RAG index for movie notes.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Combine YAML metadata and Markdown notes and embed them")
    build_parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    build_parser.add_argument("--chunk-model", default="nomic-embed-text",
                              help="Ollama embedding model used to find semantic chunk boundaries")
    build_parser.add_argument("--index", default=str(INDEX.relative_to(ROOT)), help="Output path for this embedding index")
    ask_parser = commands.add_parser("ask", help="Retrieve notes and ask the local LLM")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--chat-model", default="qwen2.5:3b", help="Ollama chat model")
    ask_parser.add_argument("--retrieval", choices=("semantic", "bm25", "hybrid"), default="semantic")
    ask_parser.add_argument("--hybrid-weight", type=float, default=0.5,
                            help="Semantic weight for hybrid retrieval, from 0 to 1")
    ask_parser.add_argument("--index", default=str(INDEX.relative_to(ROOT)), help="Embedding index to query")
    args = parser.parse_args()
    if args.command == "ask" and not 0 <= args.hybrid_weight <= 1:
        parser.error("--hybrid-weight must be between 0 and 1")
    if args.command == "build":
        build(args)
    else:
        ask(args)


if __name__ == "__main__":
    main()
