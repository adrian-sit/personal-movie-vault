"""Deterministic aggregates derived from the structured movie-viewing data."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def build_viewing_statistics(movies: list[dict[str, Any]]) -> dict[str, Any]:
    """Create exact, reproducible counts for questions RAG should not answer."""
    cinemas: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"movie_ids": set(), "movies": {}, "viewing_count": 0, "rewatch_count": 0}
    )
    formats_detail: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"movie_ids": set(), "movies": {}, "viewing_count": 0}
    )
    formats: dict[str, int] = defaultdict(int)
    years_watched: dict[str, int] = defaultdict(int)
    viewing_records: list[dict[str, Any]] = []
    viewing_count = 0
    rewatch_count = 0

    for movie in movies:
        for viewing in movie.get("viewings", []):
            viewing_count += 1
            location = viewing.get("location", "Unknown location").strip() or "Unknown location"
            cinema = cinemas[location]
            cinema["viewing_count"] += 1
            cinema["movie_ids"].add(movie["id"])
            cinema["movies"][movie["id"]] = movie["title"]
            if viewing.get("rewatch", False):
                rewatch_count += 1
                cinema["rewatch_count"] += 1
            viewing_format = viewing.get("format", "Unknown format").strip() or "Unknown format"
            formats[viewing_format] += 1
            format_detail = formats_detail[viewing_format]
            format_detail["viewing_count"] += 1
            format_detail["movie_ids"].add(movie["id"])
            format_detail["movies"][movie["id"]] = movie["title"]
            year = re.search(r"\b\d{4}\b", str(viewing.get("date_watched", "")))
            if year:
                years_watched[year.group()] += 1
            viewing_records.append({
                "movie_id": movie["id"],
                "title": movie["title"],
                "movie_year": movie.get("year"),
                "rating": movie.get("rating"),
                "genres": movie.get("genre", []),
                "date_watched": viewing.get("date_watched"),
                "location": location,
                "format": viewing_format,
                "rewatch": viewing.get("rewatch", False),
            })

    cinema_rows = [
        {
            "location": location,
            "unique_movies": len(data["movie_ids"]),
            "viewing_count": data["viewing_count"],
            "rewatch_count": data["rewatch_count"],
            "movies": [
                {"id": movie_id, "title": title}
                for movie_id, title in sorted(data["movies"].items(), key=lambda item: item[1].lower())
            ],
        }
        for location, data in sorted(cinemas.items(), key=lambda item: item[0].lower())
    ]
    format_rows = [
        {
            "format": viewing_format,
            "unique_movies": len(data["movie_ids"]),
            "viewing_count": data["viewing_count"],
            "movies": [
                {"id": movie_id, "title": title}
                for movie_id, title in sorted(data["movies"].items(), key=lambda item: item[1].lower())
            ],
        }
        for viewing_format, data in sorted(formats_detail.items(), key=lambda item: item[0].lower())
    ]
    return {
        "description": "Deterministic statistics derived from data/raw/movies.yaml.",
        "totals": {
            "unique_movies": len({movie["id"] for movie in movies}),
            "viewing_count": viewing_count,
            "rewatch_count": rewatch_count,
        },
        "cinemas": cinema_rows,
        "formats": format_rows,
        "viewings_by_format": dict(sorted(formats.items())),
        "viewings_by_year": dict(sorted(years_watched.items())),
        "viewing_records": sorted(viewing_records, key=lambda record: (record["title"].lower(), record["date_watched"] or "")),
    }
