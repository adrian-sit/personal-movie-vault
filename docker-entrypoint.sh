#!/bin/sh
set -eu

echo "Processing the current raw movie data..."
python src/yaml_to_json.py

echo "Building the retrieval index..."
python src/rag.py build

echo "Starting the Movie Vault web app..."
exec python src/web.py
