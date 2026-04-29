#!/usr/bin/env bash

set -euo pipefail

MSTX_PROFILER_DATA_PATH="${MSTX_PROFILER_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
PROFILER_TYPE="${PROFILER_TYPE:-mstx}"
VIS_TYPE="${VIS_TYPE:-html}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "MSTX Profiler Cluster Analysis"
echo "=========================================="
echo "Input Path:    ${MSTX_PROFILER_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Profiler Type: ${PROFILER_TYPE}"
echo "Vis Type:      ${VIS_TYPE}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

# Preprocessing is optional: comment out below code if data is already preprocessed

echo ">>> Start mstx data preprocessing..."

python -m rl_insight.utils.mstx_preprocessing "${MSTX_PROFILER_DATA_PATH}"

echo ">>> Mstx data preprocessing completed."

python -m rl_insight.main \
    --input-path "${MSTX_PROFILER_DATA_PATH}" \
    --profiler-type "${PROFILER_TYPE}" \
    --output-path "${OUTPUT_PATH}" \
    --vis-type "${VIS_TYPE}" \
    --rank-list "${RANK_LIST}"

echo "=========================================="
echo ">>> Analysis completed successfully!"
echo ">>> Output saved to: ${OUTPUT_PATH}"
echo "=========================================="
