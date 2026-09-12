"""Run retrieval-only metrics over author-curated evaluation cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import rag  # noqa: E402


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_ids:
            return 1 / rank
    return 0.0


def evaluate_case(case: dict, args: argparse.Namespace) -> dict:
    results = rag.search(
        case["question"], args.top_k, args.retrieval, args.hybrid_weight, rag.index_path(args.index)
    )
    retrieved_ids = [result["id"] for result in results]
    relevant_ids = set(case["relevant_document_ids"])
    found_ids = relevant_ids & set(retrieved_ids)
    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "relevant_document_ids": sorted(relevant_ids),
        "retrieved_document_ids": retrieved_ids,
        "recall_at_k": len(found_ids) / len(relevant_ids),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        "missing_relevant_ids": sorted(relevant_ids - found_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic, BM25, or hybrid retrieval against curated cases.")
    parser.add_argument("--cases", default="eval/cases.yaml")
    parser.add_argument("--index", default="data/processed/rag_index.json")
    parser.add_argument("--retrieval", choices=("semantic", "bm25", "hybrid"), default="semantic")
    parser.add_argument("--hybrid-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", help="Optional JSON result path, relative to the project root")
    args = parser.parse_args()
    if not 0 <= args.hybrid_weight <= 1:
        parser.error("--hybrid-weight must be between 0 and 1")

    case_file = ROOT / args.cases
    data = yaml.safe_load(case_file.read_text(encoding="utf-8"))
    automated_cases = [
        case for case in data["cases"]
        if case.get("automated_retrieval") and case.get("relevant_document_ids")
    ]
    index_path = rag.index_path(args.index)
    if not index_path.exists():
        parser.error(f"Index not found: {index_path}. Build it with `python src/rag.py build` first.")
    known_ids = {document["id"] for document in json.loads(index_path.read_text(encoding="utf-8"))["documents"]}
    unknown_ids = sorted({
        document_id
        for case in automated_cases
        for document_id in case["relevant_document_ids"]
        if document_id not in known_ids
    })
    if unknown_ids:
        parser.error(
            "Case document IDs are not in this index. Rebuild the index or update cases.yaml: "
            + ", ".join(unknown_ids)
        )
    evaluations = [evaluate_case(case, args) for case in automated_cases]
    summary = {
        "retrieval": args.retrieval,
        "index": args.index,
        "top_k": args.top_k,
        "case_count": len(evaluations),
        "mean_recall_at_k": mean(item["recall_at_k"] for item in evaluations) if evaluations else 0.0,
        "mean_reciprocal_rank": mean(item["reciprocal_rank"] for item in evaluations) if evaluations else 0.0,
    }
    report = {"summary": summary, "cases": evaluations}
    print(json.dumps(report, indent=2))
    if args.output:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
