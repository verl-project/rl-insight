#!/usr/bin/env bash

set -euo pipefail

MSTX_PROFILER_DATA_PATH="${MSTX_PROFILER_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
VIS_TYPE="${VIS_TYPE:-html}"
RANK_LIST="${RANK_LIST:-all}"

echo "=========================================="
echo "MSTX Profiler Cluster Analysis"
echo "=========================================="
echo "Input Path:    ${MSTX_PROFILER_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Vis Type:      ${VIS_TYPE}"
echo "Rank List:     ${RANK_LIST}"
echo "=========================================="

# Preprocessing is optional: comment out below code if data is already preprocessed

echo ">>> Start mstx data preprocessing..."

python -m recipe.utils.mstx_preprocessing "${MSTX_PROFILER_DATA_PATH}"

echo ">>> Mstx data preprocessing completed."

python -m recipe.main \
    input.path="${MSTX_PROFILER_DATA_PATH}" \
    timeline.parser.type=mstx \
    input.rank_list="${RANK_LIST}" \
    output.path="${OUTPUT_PATH}" \
    timeline.visualizer.type="${VIS_TYPE}"

echo "=========================================="
echo ">>> Analysis completed successfully!"
echo ">>> Output saved to: ${OUTPUT_PATH}"
echo "=========================================="
