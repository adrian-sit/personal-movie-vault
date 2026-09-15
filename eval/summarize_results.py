"""Summarize saved retrieval-evaluation JSON files into a Markdown report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent


def average(values: list[float]) -> str:
    return f"{mean(values):.3f}" if values else "—"


def method_name(path: Path, summary: dict) -> str:
    index = Path(summary["index"]).stem.replace("rag_index_", "")
    return f"{summary['retrieval']} / {index}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Markdown summary of retrieval evaluation results.")
    parser.add_argument("--results-dir", default="eval/results")
    parser.add_argument("--output", default="eval/RESULTS.md")
    args = parser.parse_args()

    runs = []
    for path in sorted((ROOT / args.results_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" in data and "cases" in data:
            runs.append((path, data))
    if not runs:
        raise SystemExit(f"No evaluation JSON files found in {args.results_dir}.")

    lines = [
        "# Evaluation Results",
        "",
        "Generated from the saved JSON runs in `eval/results/`. Automatic metrics use author-labeled document IDs; manual scores use the 0–10 reviews in each result file.",
        "",
        "## Overall comparison",
        "",
        "| Method / index | Recall@5 | MRR | Manual average (/10) | Reviewed cases |",
        "|---|---:|---:|---:|---:|",
    ]
    type_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    low_reviews = []
    for path, data in runs:
        summary = data["summary"]
        reviews = [
            case["manual_review"]["score"]
            for case in data["cases"]
            if isinstance(case.get("manual_review", {}).get("score"), (int, float))
        ]
        lines.append(
            f"| {method_name(path, summary)} | {summary['mean_recall_at_k']:.3f} | "
            f"{summary['mean_reciprocal_rank']:.3f} | {average(reviews)} | {len(reviews)} |"
        )
        for case in data["cases"]:
            if case.get("recall_at_k") is not None:
                type_stats[case["type"]]["recall"].append(case["recall_at_k"])
            score = case.get("manual_review", {}).get("score")
            if isinstance(score, (int, float)):
                type_stats[case["type"]]["manual"].append(score)
                if score < 8:
                    low_reviews.append((score, method_name(path, summary), case["id"], case["manual_review"].get("notes", "")))

    lines.extend([
        "",
        "## Performance by test type",
        "",
        "Values below pool all available runs; they identify broad strengths and weaknesses, not a replacement for per-method inspection.",
        "",
        "| Test type | Mean Recall@5 | Mean manual score (/10) |",
        "|---|---:|---:|",
    ])
    for test_type in sorted(type_stats):
        lines.append(f"| {test_type} | {average(type_stats[test_type]['recall'])} | {average(type_stats[test_type]['manual'])} |")

    lines.extend([
        "",
        "## Manual-review follow-ups",
        "",
        "Cases below 8/10 deserve inspection before choosing a configuration.",
        "",
        "| Score | Method / index | Case | Review note |",
        "|---:|---|---|---|",
    ])
    for score, method, case_id, note in sorted(low_reviews):
        lines.append(f"| {score:g} | {method} | {case_id} | {note or '—'} |")

    output = ROOT / args.output
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)} from {len(runs)} runs")


if __name__ == "__main__":
    main()
