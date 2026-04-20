#!/bin/bash
# Copyright (c) 2025 verl-project authors.

# Configuration
INPUT_PATH="/opt/tiger/Open-VeOmni/wjw/gmm_dump"
OUTPUT_PATH="/opt/tiger/Open-VeOmni/wjw/output/gmm_group_list_heatmap.png"
RANK=""
DPI=200
CMAP="viridis"
# Optional: specify step and role
STEP=""
ROLE=""

# Run through OfflineInsightPipeline
cmd="python -m rl_insight.main \
    --input-path \"$INPUT_PATH\" \
    --input-type \"gmm_data\" \
    --profiler-type \"gmm\" \
    --vis-type \"gmm_heatmap\" \
    --output-path \"$OUTPUT_PATH\" \
    --rank-list \"$RANK\" \
    --dpi \"$DPI\" \
    --cmap \"$CMAP\""

# Add step and role parameters if specified
if [ -n "$STEP" ]; then
    cmd="$cmd \
    --step \"$STEP\""
fi

if [ -n "$ROLE" ]; then
    cmd="$cmd \
    --role \"$ROLE\""
fi

# Execute the command
eval $cmd

# Check if the heatmap was generated successfully
if [ -f "$OUTPUT_PATH" ]; then
    echo "Heatmap generated successfully at: $OUTPUT_PATH"
else
    echo "Failed to generate heatmap"
    exit 1
fi