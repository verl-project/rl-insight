#!/usr/bin/env bash

set -euo pipefail

TORCH_PROFILER_DATA_PATH="${TORCH_PROFILER_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
PROFILER_TYPE="${PROFILER_TYPE:-torch}"
VIS_TYPE="${VIS_TYPE:-html}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "Torch Profiler Cluster Analysis"
echo "=========================================="
echo "Input Path:    ${TORCH_PROFILER_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Profiler Type: ${PROFILER_TYPE}"
echo "Vis Type:      ${VIS_TYPE}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

python -m rl_insight.main \
    --input-path "${TORCH_PROFILER_DATA_PATH}" \
    --profiler-type "${PROFILER_TYPE}" \
    --output-path "${OUTPUT_PATH}" \
    --vis-type "${VIS_TYPE}" \
    --rank-list "${RANK_LIST}"

echo "=========================================="
echo ">>> Analysis completed successfully!"
echo ">>> Output saved to: ${OUTPUT_PATH}"
echo "=========================================="
