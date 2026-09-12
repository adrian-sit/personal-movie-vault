import yaml
import json
from pathlib import Path

from viewing_analytics import build_viewing_statistics

# This file's location: src/yaml_to_json.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent          # go up one level, out of src/

INPUT_PATH = REPO_ROOT / "data" / "raw" / "movies.yaml"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "movies.json"
STATISTICS_PATH = REPO_ROOT / "data" / "processed" / "viewing_statistics.json"

def main():
    with open(INPUT_PATH, "r") as f:
        movies = yaml.safe_load(f)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensure processed/ exists

    with open(OUTPUT_PATH, "w") as f:
        json.dump(movies, f, indent=2, default=str)  # default=str handles date objects

    statistics = build_viewing_statistics(movies)
    with open(STATISTICS_PATH, "w") as f:
        json.dump(statistics, f, indent=2)

    print(f"Wrote {len(movies)} movies to {OUTPUT_PATH}")
    print(f"Wrote viewing statistics to {STATISTICS_PATH}")

if __name__ == "__main__":
    main()
