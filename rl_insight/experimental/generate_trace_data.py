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
import math
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
        "state_name": "tool_calls",
        "step_outcome": "continue",
        "model_finish_reason": "tool_calls",
        "tools": ["search"],
        "content": "Search for the relevant implementation.",
    },
    {
        "state_name": "tool_calls",
        "step_outcome": "continue",
        "model_finish_reason": "tool_calls",
        "tools": ["read"],
        "content": "Read the relevant source files.",
    },
    {
        "state_name": "tool_calls",
        "step_outcome": "continue",
        "model_finish_reason": "tool_calls",
        "tools": ["calculator"],
        "content": "Check the intermediate result.",
    },
    {
        "state_name": "tool_calls",
        "step_outcome": "continue",
        "model_finish_reason": "tool_calls",
        "tools": ["bash"],
        "content": "Run the verification command.",
    },
    {
        "state_name": "tool_calls",
        "step_outcome": "continue",
        "model_finish_reason": "tool_calls",
        "tools": ["write"],
        "content": "Record the verified result.",
    },
    {
        "state_name": "finished",
        "step_outcome": "finished",
        "model_finish_reason": "stop",
        "tools": [],
        "content": "Return the final answer.",
    },
)

_STEP_IDENTITY_KEYS = (
    "session_id",
    "uid",
    "sample_index",
    "session_index",
    "step_index",
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


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a finite number greater than zero"
        )
    return number


def _step_key(attributes: dict[str, Any]) -> tuple[str, ...]:
    missing = [key for key in _STEP_IDENTITY_KEYS if key not in attributes]
    if missing:
        raise RuntimeError(f"agent step span is missing identity fields: {missing}")
    return tuple(str(attributes[key]) for key in _STEP_IDENTITY_KEYS)


def _attributes(
    *,
    session_id: str,
    interface: str,
    step_index: int,
    step: dict[str, Any],
    timing_source: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "uid": f"trace-api-demo-{interface}",
        "sample_index": "0",
        "session_index": "0",
        "step_index": str(step_index),
        "state_name": str(step["state_name"]),
        "step_outcome": str(step["step_outcome"]),
        "tools": json.dumps(step["tools"], ensure_ascii=False),
        "content": str(step["content"]),
        "monitor.trace_source": "agent_step",
        "model.finish_reason": str(step["model_finish_reason"]),
        "agent_step.interface": interface,
        "agent_step.timing_source": timing_source,
    }


def _generate_trace_span_run(step_duration: float) -> str:
    session_id = f"session-trace-span-{uuid.uuid4().hex}"

    for step_index, step in enumerate(STEPS, start=1):
        start_time_ns = time.time_ns()
        time.sleep(step_duration)
        end_time_ns = time.time_ns()
        insight.trace_span(
            name="agent_step",
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes=_attributes(
                session_id=session_id,
                interface="trace_span",
                step_index=step_index,
                step=step,
                timing_source="execution_time",
            ),
        )
    return session_id


def _generate_sync_trace_op_run(step_duration: float) -> str:
    session_id = f"session-trace-op-sync-{uuid.uuid4().hex}"

    @insight.trace_op("agent_step", extra_labels=lambda call: call.attributes)
    def execute_step(call: _StepCall) -> None:
        time.sleep(call.duration)

    for step_index, step in enumerate(STEPS, start=1):
        call = _StepCall(
            attributes=_attributes(
                session_id=session_id,
                interface="trace_op_sync",
                step_index=step_index,
                step=step,
                timing_source="execution_time",
            ),
            duration=step_duration,
        )
        execute_step(call)
    return session_id


async def _generate_async_trace_op_run(step_duration: float) -> str:
    session_id = f"session-trace-op-async-{uuid.uuid4().hex}"

    @insight.trace_op("agent_step", extra_labels=lambda call: call.attributes)
    async def execute_step(call: _StepCall) -> None:
        await asyncio.sleep(call.duration)

    for step_index, step in enumerate(STEPS, start=1):
        call = _StepCall(
            attributes=_attributes(
                session_id=session_id,
                interface="trace_op_async",
                step_index=step_index,
                step=step,
                timing_source="execution_time",
            ),
            duration=step_duration,
        )
        await execute_step(call)
    return session_id


def _wait_for_search(
    base_url: str,
    path: str,
    *,
    session_id: str,
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
                    "q": f'{{ span.session_id = "{session_id}" }}',
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
        f"spans for session_id={session_id} were not queryable from {base_url}{path} "
        f"within {timeout}s: last_error={last_error} last_payload={last_payload}"
    )


def _verify_run(
    *,
    tempo_url: str,
    grafana_url: str,
    session_id: str,
    interface: str,
    timeout: float,
) -> None:
    expected_spans = len(STEPS)
    tempo_search = _wait_for_search(
        tempo_url,
        "/api/search",
        session_id=session_id,
        expected_spans=expected_spans,
        timeout=timeout,
    )
    _wait_for_search(
        grafana_url,
        "/api/datasources/proxy/uid/tempo/api/search",
        session_id=session_id,
        expected_spans=expected_spans,
        timeout=timeout,
    )

    trace_ids = {trace["traceID"] for trace in tempo_search["traces"]}
    if len(trace_ids) < expected_spans:
        raise RuntimeError(
            f"expected {expected_spans} independent spans for {interface}, "
            f"found {len(trace_ids)}"
        )
    _verify_tempo_contract(
        tempo_url=tempo_url,
        trace_ids=trace_ids,
        session_id=session_id,
        interface=interface,
    )
    print(f"PASS {interface}: {len(trace_ids)} spans, session_id={session_id}")


def _verify_tempo_contract(
    *,
    tempo_url: str,
    trace_ids: set[str],
    session_id: str,
    interface: str,
) -> None:
    import requests

    spans_by_step: dict[tuple[str, ...], tuple[dict[str, Any], dict[str, Any]]] = {}
    for trace_id in trace_ids:
        response = requests.get(f"{tempo_url}/api/traces/{trace_id}", timeout=3)
        response.raise_for_status()
        trace = response.json()
        for batch in trace.get("batches", []):
            for scope in batch.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    attributes = {
                        item["key"]: item.get("value", {}).get("stringValue")
                        for item in span.get("attributes", [])
                    }
                    if attributes.get("session_id") != session_id:
                        continue
                    step_key = _step_key(attributes)
                    if step_key in spans_by_step:
                        raise RuntimeError(
                            f"duplicate agent step={step_key!r} for {interface}"
                        )
                    spans_by_step[step_key] = (span, attributes)

    expected_step_keys = {
        (
            session_id,
            f"trace-api-demo-{interface}",
            "0",
            "0",
            str(step_index),
        )
        for step_index in range(1, len(STEPS) + 1)
    }
    if set(spans_by_step) != expected_step_keys:
        raise RuntimeError(
            f"unexpected agent steps for {interface}, session_id={session_id}: "
            f"expected={sorted(expected_step_keys)} actual={sorted(spans_by_step)}"
        )

    for step_index, step in enumerate(STEPS, start=1):
        step_key = (
            session_id,
            f"trace-api-demo-{interface}",
            "0",
            "0",
            str(step_index),
        )
        span, actual_attributes = spans_by_step[step_key]
        expected_attributes = _attributes(
            session_id=session_id,
            interface=interface,
            step_index=step_index,
            step=step,
            timing_source="execution_time",
        )
        for key, expected_value in expected_attributes.items():
            if actual_attributes.get(key) != expected_value:
                raise RuntimeError(
                    f"unexpected {key} for {interface} step={step_index}: "
                    f"expected={expected_value!r} "
                    f"actual={actual_attributes.get(key)!r}"
                )
        expected_name = "agent_step"
        if span.get("name") != expected_name:
            raise RuntimeError(
                f"unexpected span name for {interface} step={step_index}: "
                f"expected={expected_name!r} actual={span.get('name')!r}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and verify trace_span/trace_op data in Tempo and Grafana"
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("RL_INSIGHT_SERVER_URL", "http://127.0.0.1:18080"),
        help=(
            "RL-Insight server URL (default: RL_INSIGHT_SERVER_URL or localhost:18080)"
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
        type=_positive_float,
        default=0.15,
        help="Displayed duration of each step in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
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
        for interface, session_id in runs.items():
            _verify_run(
                tempo_url=tempo_url,
                grafana_url=grafana_url,
                session_id=session_id,
                interface=interface,
                timeout=args.timeout,
            )
    finally:
        insight.finish()
        ray.shutdown()

    print("\nTempo and Grafana datasource verification passed.")
    print("Open Grafana Explore and inspect the three generated runs:")
    print(f"  {grafana_url}/explore")
    for interface, session_id in runs.items():
        print(f"  {interface}: {session_id}")


if __name__ == "__main__":
    main()
