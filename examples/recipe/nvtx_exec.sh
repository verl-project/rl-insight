#!/usr/bin/env bash

set -euo pipefail

NVTX_PROFILER_DATA_PATH="${NVTX_PROFILER_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
VIS_TYPE="${VIS_TYPE:-html}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "Nvtx Profiler Cluster Analysis"
echo "=========================================="
echo "Input Path:    ${NVTX_PROFILER_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Vis Type:      ${VIS_TYPE}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

python -m recipe.main \
    input.path="${NVTX_PROFILER_DATA_PATH}" \
    timeline.parser.type=nvtx \
    input.rank_list="${RANK_LIST}" \
    output.path="${OUTPUT_PATH}" \
    timeline.visualizer.type="${VIS_TYPE}"

echo "=========================================="
echo ">>> Analysis completed successfully!"
echo ">>> Output saved to: ${OUTPUT_PATH}"
echo "=========================================="
