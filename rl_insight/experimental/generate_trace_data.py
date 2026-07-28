#!/usr/bin/env python3

# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate displayable spans through ``trace_span`` and ``trace_op``.

The script exercises both public reporting interfaces against a running
RL-Insight stack, waits until Tempo stores every span, and verifies that the
same data is queryable through Grafana's provisioned Tempo datasource.

Usage::

    python rl_insight/experimental/generate_trace_data.py
    python rl_insight/experimental/generate_trace_data.py \
        --server-url http://server-host:18080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any
from urllib.parse import urlparse


_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import rl_insight as insight  # noqa: E402


STEPS = (
    {
        "finish_reason": "tool_calls",
        "type": "tool",
        "tools": ["search"],
        "content": "Search for the relevant implementation.",
    },
    {
        "finish_reason": "tool_calls",
        "type": "tool",
        "tools": ["read"],
        "content": "Read the relevant source files.",
    },
    {
        "finish_reason": "tool_calls",
        "type": "tool",
        "tools": ["calculator"],
        "content": "Check the intermediate result.",
    },
    {
        "finish_reason": "tool_calls",
        "type": "tool",
        "tools": ["bash"],
        "content": "Run the verification command.",
    },
    {
        "finish_reason": "tool_calls",
        "type": "tool",
        "tools": ["write"],
        "content": "Record the verified result.",
    },
    {
        "finish_reason": "stop",
        "type": "llm",
        "tools": [],
        "content": "Return the final answer.",
    },
)


@dataclass(frozen=True)
class _StepCall:
    attributes: dict[str, Any]
    duration: float


def _service_url(server_url: str, port: int) -> str:
    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme or 'http'}://{host}:{port}"


def _attributes(
    *,
    run_id: str,
    interface: str,
    turn: int,
    step: dict[str, Any],
    timing_source: str,
) -> dict[str, Any]:
    finish_reason = str(step["finish_reason"])
    return {
        "run_id": run_id,
        "state_lane_id": run_id,
        "state_name": finish_reason,
        "uid": f"trace-api-demo-{interface}",
        "sample": "0",
        "session": "0",
        "traj": "0",
        "turn": str(turn),
        "finish_reason": finish_reason,
        "type": str(step["type"]),
        "tools": json.dumps(step["tools"]),
        "content": str(step["content"]),
        "monitor.trace_source": "trajectory",
        "trajectory.interface": interface,
        "trajectory.timing_source": timing_source,
    }


def _generate_trace_span_run(step_duration: float) -> str:
    run_id = str(uuid.uuid4())
    duration_ns = max(1, int(step_duration * 1_000_000_000))
    first_start_ns = time.time_ns() - len(STEPS) * duration_ns

    for turn, step in enumerate(STEPS):
        start_time_ns = first_start_ns + turn * duration_ns
        insight.trace_span(
            name=str(step["finish_reason"]),
            start_time_ns=start_time_ns,
            end_time_ns=start_time_ns + duration_ns,
            attributes=_attributes(
                run_id=run_id,
                interface="trace_span",
                turn=turn,
                step=step,
                timing_source="explicit_time",
            ),
        )
    return run_id


def _generate_sync_trace_op_run(step_duration: float) -> str:
    run_id = str(uuid.uuid4())

    @insight.trace_op("agent_step", extra_labels=lambda call: call.attributes)
    def execute_step(call: _StepCall) -> None:
        time.sleep(call.duration)

    for turn, step in enumerate(STEPS):
        call = _StepCall(
            attributes=_attributes(
                run_id=run_id,
                interface="trace_op_sync",
                turn=turn,
                step=step,
                timing_source="execution_time",
            ),
            duration=step_duration,
        )
        execute_step(call)
    return run_id


async def _generate_async_trace_op_run(step_duration: float) -> str:
    run_id = str(uuid.uuid4())

    @insight.trace_op("agent_step", extra_labels=lambda call: call.attributes)
    async def execute_step(call: _StepCall) -> None:
        await asyncio.sleep(call.duration)

    for turn, step in enumerate(STEPS):
        call = _StepCall(
            attributes=_attributes(
                run_id=run_id,
                interface="trace_op_async",
                turn=turn,
                step=step,
                timing_source="execution_time",
            ),
            duration=step_duration,
        )
        await execute_step(call)
    return run_id


def _wait_for_search(
    base_url: str,
    path: str,
    *,
    attribute: str,
    run_id: str,
    expected_spans: int,
    timeout: float,
) -> dict[str, Any]:
    import requests

    deadline = time.monotonic() + timeout
    last_payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"{base_url}{path}",
                params={
                    "q": f'{{ span.{attribute} = "{run_id}" }}',
                    "limit": str(max(20, expected_spans)),
                },
                timeout=3,
            )
            response.raise_for_status()
            last_payload = response.json()
            if len(last_payload.get("traces", [])) >= expected_spans:
                return last_payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"spans for run_id={run_id} were not queryable from {base_url}{path} "
        f"within {timeout}s: last_error={last_error} last_payload={last_payload}"
    )


def _verify_run(
    *,
    tempo_url: str,
    grafana_url: str,
    run_id: str,
    interface: str,
    timeout: float,
) -> None:
    expected_spans = len(STEPS)
    tempo_search = _wait_for_search(
        tempo_url,
        "/api/search",
        attribute="run_id",
        run_id=run_id,
        expected_spans=expected_spans,
        timeout=timeout,
    )
    _wait_for_search(
        grafana_url,
        "/api/datasources/proxy/uid/tempo/api/search",
        attribute="state_lane_id",
        run_id=run_id,
        expected_spans=expected_spans,
        timeout=timeout,
    )

    trace_ids = {trace["traceID"] for trace in tempo_search["traces"]}
    if len(trace_ids) < expected_spans:
        raise RuntimeError(
            f"expected {expected_spans} independent spans for {interface}, "
            f"found {len(trace_ids)}"
        )
    print(f"PASS {interface}: {len(trace_ids)} spans, run_id={run_id}")


def _dashboard_url(grafana_url: str) -> str:
    import requests

    try:
        response = requests.get(
            f"{grafana_url}/api/search",
            params={"query": "quick_start_demo"},
            timeout=3,
        )
        response.raise_for_status()
        for item in response.json():
            if item.get("title") == "quick_start_demo" and item.get("url"):
                return f"{grafana_url}{item['url']}?from=now-5m&to=now"
    except (requests.RequestException, ValueError):
        pass
    return f"{grafana_url}/dashboards"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and verify trace_span/trace_op data in Tempo and Grafana"
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("RL_INSIGHT_SERVER_URL", "http://127.0.0.1:18080"),
        help=(
            "RL-Insight server URL "
            "(default: RL_INSIGHT_SERVER_URL or localhost:18080)"
        ),
    )
    parser.add_argument(
        "--tempo-url",
        default=os.environ.get("RL_INSIGHT_TEMPO_QUERY_URL"),
        help="Tempo query URL (default: server host on port 3200)",
    )
    parser.add_argument(
        "--grafana-url",
        default=os.environ.get("RL_INSIGHT_GRAFANA_URL"),
        help="Grafana URL (default: server host on port 3000)",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=0.15,
        help="Displayed duration of each step in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for Tempo/Grafana queries (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import ray

    from rl_insight.utils.constants import MonitorRayActor

    tempo_url = args.tempo_url or _service_url(args.server_url, 3200)
    grafana_url = args.grafana_url or _service_url(args.server_url, 3000)

    os.environ["RL_INSIGHT_SERVER_URL"] = args.server_url
    ray.init(namespace=MonitorRayActor.NAMESPACE, ignore_reinit_error=True)
    insight.init(project="rl-insight-experimental", experiment_name="trace-api-e2e")
    try:
        runs = {
            "trace_span": _generate_trace_span_run(args.step_duration),
            "trace_op_sync": _generate_sync_trace_op_run(args.step_duration),
            "trace_op_async": asyncio.run(
                _generate_async_trace_op_run(args.step_duration)
            ),
        }
        for interface, run_id in runs.items():
            _verify_run(
                tempo_url=tempo_url,
                grafana_url=grafana_url,
                run_id=run_id,
                interface=interface,
                timeout=args.timeout,
            )
    finally:
        insight.finish()
        ray.shutdown()

    print("\nTempo and Grafana datasource verification passed.")
    print("Open the Grafana dashboard and inspect the three run_id lanes:")
    print(f"  {_dashboard_url(grafana_url)}")
    for interface, run_id in runs.items():
        print(f"  {interface}: {run_id}")


if __name__ == "__main__":
    main()
