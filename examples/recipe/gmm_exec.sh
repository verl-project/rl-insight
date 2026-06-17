#!/usr/bin/env bash

set -euo pipefail

# Configuration with environment variable defaults
GMM_DATA_PATH="${GMM_DATA_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/gmm_heatmap.png}"
RANK_LIST="${RANK_LIST:-all}"
DPI="${DPI:-200}"
CMAP="${CMAP:-viridis}"
GMM_PER_LAYER="${GMM_PER_LAYER:-3}"
STEP="${STEP:-}"
ROLE="${ROLE:-}"

# Display configuration
echo "=========================================="
echo "GMM Expert Load Heatmap Visualization"
echo "=========================================="
echo "Input Path:    ${GMM_DATA_PATH}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Rank List:     ${RANK_LIST}"
echo "DPI:           ${DPI}"
echo "Colormap:      ${CMAP}"
echo "GMM/Layer:     ${GMM_PER_LAYER}"
echo "Step:          ${STEP:-all}"
echo "Role:          ${ROLE:-all}"
echo "=========================================="

# Build command
cmd="python -m recipe.main \
    input.path=\"${GMM_DATA_PATH}\" \
    output.path=\"${OUTPUT_PATH}\" \
    input.rank_list=\"${RANK_LIST}\" \
    heatmap.parser.type=gmm \
    heatmap.visualizer.type=gmm_heatmap \
    heatmap.visualizer.dpi=\"${DPI}\" \
    heatmap.visualizer.cmap=\"${CMAP}\" \
    heatmap.visualizer.gmm_per_layer=\"${GMM_PER_LAYER}\""

if [ -n "${STEP}" ]; then
    cmd="${cmd} \
    heatmap.parser.step=\"${STEP}\""
fi

if [ -n "${ROLE}" ]; then
    cmd="${cmd} \
    heatmap.parser.role=\"${ROLE}\""
fi

# Execute the command
echo ">>> Generating GMM expert load heatmap..."
eval ${cmd}

# Check if the heatmap was generated successfully
if [ -f "${OUTPUT_PATH}" ]; then
    echo "=========================================="
    echo ">>> Heatmap generated successfully!"
    echo ">>> Output saved to: ${OUTPUT_PATH}"
    echo "=========================================="
else
    echo "=========================================="
    echo ">>> Failed to generate heatmap"
    echo "=========================================="
    exit 1
fi
