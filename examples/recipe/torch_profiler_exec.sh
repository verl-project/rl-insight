#!/usr/bin/env bash

set -euo pipefail

TORCH_PROFILER_DATA_PATH="${TORCH_PROFILER_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
VIS_TYPE="${VIS_TYPE:-html}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "Torch Profiler Cluster Analysis"
echo "=========================================="
echo "Input Path:    ${TORCH_PROFILER_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Vis Type:      ${VIS_TYPE}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

python -m recipe.main \
    input.path="${TORCH_PROFILER_DATA_PATH}" \
    timeline.parser.type=torch \
    input.rank_list="${RANK_LIST}" \
    output.path="${OUTPUT_PATH}" \
    timeline.visualizer.type="${VIS_TYPE}"

echo "=========================================="
echo ">>> Analysis completed successfully!"
echo ">>> Output saved to: ${OUTPUT_PATH}"
echo "=========================================="
