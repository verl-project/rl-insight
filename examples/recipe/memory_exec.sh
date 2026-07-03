#!/usr/bin/env bash

set -euo pipefail

MEMORY_DATA_PATH="${MEMORY_DATA_PATH:-}"
if [ -z "${MEMORY_DATA_PATH}" ]; then
    echo "Error: MEMORY_DATA_PATH environment variable is not set or empty." >&2
    exit 1
fi
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "Memory Allocation Timeline Visualization"
echo "=========================================="
echo "Input Path:    ${MEMORY_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

cmd=(
    python -m recipe.main
    input.path="${MEMORY_DATA_PATH}"
    output.path="${OUTPUT_PATH}"
    input.rank_list="${RANK_LIST}"
    memory.parser.type=memory
    memory.visualizer.type=memory_html
)

echo ">>> Generating memory allocation timeline..."
"${cmd[@]}"

if ls "${OUTPUT_PATH}"/memory_timeline_*.html 1> /dev/null 2>&1; then
    echo "=========================================="
    echo ">>> Memory timeline generated successfully!"
    echo ">>> Output saved to: ${OUTPUT_PATH}/"
    echo "=========================================="
else
    echo "=========================================="
    echo ">>> Failed to generate memory timeline"
    echo "=========================================="
    exit 1
fi