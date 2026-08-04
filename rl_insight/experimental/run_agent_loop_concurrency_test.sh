#!/usr/bin/env bash
# Copyright (c) 2026 verl-project authors. Licensed under the Apache License 2.0.
#
# Multi-process Agent Loop fixture (one generate_trace_data.py per worker).
# From repo root:
#   bash rl_insight/experimental/run_agent_loop_concurrency_test.sh
#   bash rl_insight/experimental/run_agent_loop_concurrency_test.sh stop

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:18080}"
BASE_PORT="${BASE_PORT:-19093}"
DIR=.concurrency-test
GEN=rl_insight/experimental/generate_trace_data.py

if [[ "${1:-}" == stop ]]; then
  [[ -f $DIR/pids ]] && xargs -r kill <"$DIR/pids" 2>/dev/null || true
  rm -f "$DIR/pids"
  exit 0
fi

mkdir -p "$DIR"
: >"$DIR/pids"

for i in 0 1 2 3; do
  nohup python3 -u "$GEN" \
    --server-url "$SERVER_URL" \
    --metrics-report-port $((BASE_PORT + i)) \
    --agent-loop-runs 1 \
    --agent-loop-samples 2 \
    --agent-loop-sessions 2 \
    --agent-loop-trajs 2 \
    --agent-loop-turns 8 \
    --agent-loop-seed $((100 + i)) \
    --agent-loop-step-duration 0.02 \
    --agent-loop-step-gap 0.05 \
    >"$DIR/worker-${i}.log" 2>&1 &
  echo $! >>"$DIR/pids"
done

echo "started 4 workers; pids in $DIR/pids; stop: $0 stop"
