#!/usr/bin/env bash

set -euo pipefail

NVTX_PROFILER_DATA_PATH="${NVTX_PROFILER_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
PROFILER_TYPE="${PROFILER_TYPE:-nvtx}"
VIS_TYPE="${VIS_TYPE:-html}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "Nvtx Profiler Cluster Analysis"
echo "=========================================="
echo "Input Path:    ${NVTX_PROFILER_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Profiler Type: ${PROFILER_TYPE}"
echo "Vis Type:      ${VIS_TYPE}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

python -m rl_insight.main \
    --input-path "${NVTX_PROFILER_DATA_PATH}" \
    --profiler-type "${PROFILER_TYPE}" \
    --output-path "${OUTPUT_PATH}" \
    --vis-type "${VIS_TYPE}" \
    --rank-list "${RANK_LIST}"

echo "=========================================="
echo ">>> Analysis completed successfully!"
echo ">>> Output saved to: ${OUTPUT_PATH}"
echo "=========================================="
