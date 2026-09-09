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
OLLAMA_URL = "http://localhost:11434"


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


def make_documents() -> list[dict[str, Any]]:
    if not MOVIES.exists():
        raise RuntimeError("data/processed/movies.json is missing. Run src/yaml_to_json.py first.")
    movies = json.loads(MOVIES.read_text(encoding="utf-8"))
    by_id = {movie["id"]: movie for movie in movies}
    documents: list[dict[str, Any]] = []
    for movie in movies:
        documents.append({
            "id": f"{movie['id']}:metadata",
            "movie_id": movie["id"],
            "title": movie["title"],
            "section": "Metadata",
            "source": "data/raw/movies.yaml",
            "text": movie_text(movie),
        })
    for note_path in RAW.glob("*.md"):
        frontmatter, sections = parse_note(note_path)
        movie_id = frontmatter.get("movie_id")
        if not movie_id or movie_id not in by_id:
            print(f"Skipping {note_path.name}: no matching movie_id", file=sys.stderr)
            continue
        for heading, content in sections:
            documents.append({
                "id": f"{movie_id}:{heading.lower().replace(' ', '-')}",
                "movie_id": movie_id,
                "title": by_id[movie_id]["title"],
                "section": heading,
                "source": f"data/raw/{note_path.name}",
                "text": f"Movie: {by_id[movie_id]['title']}\nSection: {heading}\n{content}",
            })
    return documents


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def build(args: argparse.Namespace) -> None:
    documents = make_documents()
    vectors = embed([doc["text"] for doc in documents], args.embed_model)
    for document, vector in zip(documents, vectors):
        document["embedding"] = vector
    INDEX.write_text(json.dumps({"embed_model": args.embed_model, "documents": documents}), encoding="utf-8")
    with DOCUMENTS.open("w", encoding="utf-8") as output:
        for document in documents:
            output.write(json.dumps({key: value for key, value in document.items() if key != "embedding"}) + "\n")
    print(f"Indexed {len(documents)} documents in {INDEX.relative_to(ROOT)}")


def search(question: str, top_k: int) -> list[dict[str, Any]]:
    if not INDEX.exists():
        raise RuntimeError("Index not found. Run `python src/rag.py build` first.")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    query_vector = embed([question], index["embed_model"])[0]
    ranked = sorted(
        ({**document, "score": cosine(query_vector, document["embedding"])} for document in index["documents"]),
        key=lambda item: item["score"], reverse=True,
    )
    return ranked[:top_k]


def ask(args: argparse.Namespace) -> None:
    results = search(args.question, args.top_k)
    context = "\n\n".join(
        f"[{i}] {item['title']} — {item['section']} ({item['source']})\n{item['text']}"
        for i, item in enumerate(results, 1)
    )
    prompt = (
        "Answer only from the supplied personal movie-vault context. If it does not "
        "contain the answer, say so. Treat the notes as the viewer's opinion, not "
        "objective fact. Cite each claim with [1], [2], etc.\n\n"
        f"Context:\n{context}\n\nQuestion: {args.question}\nAnswer:"
    )
    answer = ollama("/api/generate", {"model": args.chat_model, "prompt": prompt, "stream": False})["response"].strip()
    print(answer)
    print("\nSources:")
    for i, item in enumerate(results, 1):
        print(f"[{i}] {item['title']} — {item['section']} ({item['source']}; score {item['score']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a local RAG index for movie notes.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Combine YAML metadata and Markdown notes and embed them")
    build_parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    ask_parser = commands.add_parser("ask", help="Retrieve notes and ask the local LLM")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--chat-model", default="qwen2.5:3b", help="Ollama chat model")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        ask(args)


if __name__ == "__main__":
    main()
