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

"""Generate and verify ``trace_span`` / ``trace_op`` data against a live stack.

Also emits the Agent Loop nested-Repeat fixture:
``rl_insight_monitor_agent_loop_*_info`` gauges + Tempo turn spans.
Tree cardinality defaults to random (``--agent-loop-*`` overrides);
last turn is always terminal; fabricated duration+gap keeps timeline bars visible.

Usage::

    python rl_insight/experimental/generate_trace_data.py
    python rl_insight/experimental/generate_trace_data.py \
        --server-url http://server-host:18080 --metrics-report-port 9094
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
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


@dataclass(frozen=True)
class _AgentLoopTurn:
    turn: int
    finish_reason: str
    tools: list[str]
    content: str


@dataclass
class _AgentLoopTraj:
    traj: str
    reward: float
    turns: list[_AgentLoopTurn]

    @property
    def success(self) -> bool:
        return self.reward > 0.0


@dataclass
class _AgentLoopSession:
    session: str
    trajectories: list[_AgentLoopTraj]


@dataclass
class _AgentLoopSample:
    sample: str
    sessions: list[_AgentLoopSession]


_AGENT_LOOP_TOOLS = ("search", "read", "calculator", "bash", "write")
_AGENT_LOOP_THOUGHTS = tuple(str(s["content"]) for s in STEPS[:-1])


@dataclass(frozen=True)
class _AgentLoopShape:
    """``None`` cardinality fields are drawn randomly (runs stay fixed)."""

    runs: int = 1
    samples: int | None = None
    sessions: int | None = None
    trajs: int | None = None
    turns: int | None = None
    seed: int = 42


def _lane_id(run_id: str, sample: str, session: str, traj: str) -> str:
    return f"run={run_id}/sample={sample}/session={session}/traj={traj}"


def _sess_key(sample: str, session: str) -> str:
    return f"sample={sample}/session={session}"


def _leaf(sample: str, session: str, traj: str) -> str:
    return f"sample={sample}/session={session}/traj={traj}"


def _traj_stats(trajs: list[_AgentLoopTraj]) -> tuple[int, int, int]:
    """Return (success_count, total, turn_count)."""
    succ = sum(1 for t in trajs if t.success)
    turns = sum(len(t.turns) for t in trajs)
    return succ, len(trajs), turns


def _build_trajectory(
    *,
    traj_index: int,
    turn_count: int | None,
    sample_success: bool,
    is_last_in_session: bool,
    rng: random.Random,
) -> _AgentLoopTraj:
    """N turns: mid ``tool_calls``, last always terminal."""
    n = max(1, turn_count if turn_count is not None else rng.randint(4, 10))
    if sample_success and (is_last_in_session or rng.random() < 0.4):
        reward, terminal = 1.0, "stop"
    else:
        reward = 0.0
        roll = rng.random()
        terminal = (
            "length" if roll < 0.3 else "max_step_limit" if roll < 0.5 else "stop"
        )

    turns: list[_AgentLoopTurn] = []
    for step_i in range(1, n + 1):
        if step_i == n:
            turns.append(
                _AgentLoopTurn(
                    turn=step_i,
                    finish_reason=terminal,
                    tools=[],
                    content=(
                        "Return the final answer."
                        if terminal == "stop"
                        else "Stopped before finishing the task."
                    ),
                )
            )
        else:
            turns.append(
                _AgentLoopTurn(
                    turn=step_i,
                    finish_reason="tool_calls",
                    tools=[rng.choice(_AGENT_LOOP_TOOLS)],
                    content=_AGENT_LOOP_THOUGHTS[
                        (step_i - 1) % len(_AGENT_LOOP_THOUGHTS)
                    ],
                )
            )
    return _AgentLoopTraj(traj=str(traj_index), reward=reward, turns=turns)


def _build_sample_tree(shape: _AgentLoopShape) -> list[_AgentLoopSample]:
    rng = random.Random(shape.seed)
    n_samples = shape.samples if shape.samples is not None else rng.randint(2, 3)
    out: list[_AgentLoopSample] = []
    for si in range(n_samples):
        ok = rng.random() < 0.35
        n_sess = (
            shape.sessions
            if shape.sessions is not None
            else (2 if si == 0 else rng.randint(2, 4))
        )
        sessions = []
        for sess_i in range(n_sess):
            n_traj = shape.trajs if shape.trajs is not None else rng.randint(1, 3)
            trajs = [
                _build_trajectory(
                    traj_index=ti,
                    turn_count=shape.turns,
                    sample_success=ok,
                    is_last_in_session=(ti == n_traj - 1),
                    rng=rng,
                )
                for ti in range(n_traj)
            ]
            sessions.append(_AgentLoopSession(str(sess_i), trajs))
        out.append(_AgentLoopSample(str(si), sessions))
    return out


def _publish_agent_loop_prom(run_id: str, samples: list[_AgentLoopSample]) -> None:
    """Emit ``agent_loop_*_info`` gauges (titles drive Repeat row text)."""
    run_trajs = [
        traj
        for sample in samples
        for session in sample.sessions
        for traj in session.trajectories
    ]
    r_succ, r_total, _ = _traj_stats(run_trajs)
    insight.metric_gauge(
        "agent_loop_run_info",
        1.0,
        run_id=run_id,
        title=f"Run · {run_id} · samples {len(samples)} · success {r_succ}/{r_total}",
    )
    for sample in samples:
        s_trajs = [t for c in sample.sessions for t in c.trajectories]
        s_succ, s_total, s_turns = _traj_stats(s_trajs)
        insight.metric_gauge(
            "agent_loop_sample_info",
            1.0,
            run_id=run_id,
            sample=sample.sample,
            title=(
                f"Sample {sample.sample} · success {s_succ}/{s_total} · "
                f"{s_turns} turns · {len(sample.sessions)} sessions"
            ),
        )
        for session in sample.sessions:
            c_succ, c_total, c_turns = _traj_stats(session.trajectories)
            insight.metric_gauge(
                "agent_loop_session_info",
                1.0,
                run_id=run_id,
                sample=sample.sample,
                session=session.session,
                sess_key=_sess_key(sample.sample, session.session),
                title=(
                    f"Session {session.session} · success {c_succ}/{c_total} · "
                    f"{c_turns} turns · {c_total} trajectories"
                ),
            )
            for traj in session.trajectories:
                insight.metric_gauge(
                    "agent_loop_traj_info",
                    1.0,
                    run_id=run_id,
                    sample=sample.sample,
                    session=session.session,
                    traj=traj.traj,
                    leaf=_leaf(sample.sample, session.session, traj.traj),
                    title=(
                        f"Trajectory #{traj.traj} · reward {traj.reward} · "
                        f"{len(traj.turns)} turns"
                    ),
                )


def _emit_agent_loop_tempo_turns(
    *,
    run_id: str,
    samples: list[_AgentLoopSample],
    step_duration_s: float,
    step_gap_s: float,
) -> tuple[list[str], dict[str, int], int, float, float]:
    """Tempo turns with fabricated duration + gap (visible on state-timeline).

    Returns ``(lanes, lane_turns, span_count, first_turn_unix, last_turn_unix)``.
    """
    duration_ns = max(int(step_duration_s * 1_000_000_000), 250_000_000)
    gap_ns = max(int(step_gap_s * 1_000_000_000), 0)
    anchor_end_ns = int((time.time() - 60.0) * 1_000_000_000)
    lanes: list[str] = []
    lane_turns: dict[str, int] = {}
    span_count = 0
    min_start_ns: int | None = None
    max_end_ns: int | None = None
    for sample in samples:
        for session in sample.sessions:
            for traj in session.trajectories:
                lane = _lane_id(run_id, sample.sample, session.session, traj.traj)
                lanes.append(lane)
                lane_turns[lane] = len(traj.turns)
                clock = anchor_end_ns - (
                    len(traj.turns) * duration_ns + max(0, len(traj.turns) - 1) * gap_ns
                )
                for turn in traj.turns:
                    start_ns, end_ns = clock, clock + duration_ns
                    clock = end_ns + gap_ns
                    min_start_ns = (
                        start_ns
                        if min_start_ns is None
                        else min(min_start_ns, start_ns)
                    )
                    max_end_ns = (
                        end_ns if max_end_ns is None else max(max_end_ns, end_ns)
                    )
                    insight.trace_span(
                        name=turn.finish_reason,
                        start_time_ns=start_ns,
                        end_time_ns=end_ns,
                        attributes={
                            "run_id": run_id,
                            "sample": sample.sample,
                            "session": session.session,
                            "traj": traj.traj,
                            "state_lane_id": lane,
                            "turn": str(turn.turn),
                            "type": "tool" if turn.tools else "llm",
                            "tools": json.dumps(turn.tools, ensure_ascii=False),
                            "finish_reason": turn.finish_reason,
                            "content": turn.content,
                            "monitor.trace_source": "agent_loop_dashboard",
                            "monitor.trace_segment": "state_interval",
                        },
                    )
                    span_count += 1
    if min_start_ns is None or max_end_ns is None:
        now = time.time()
        return lanes, lane_turns, span_count, now, now
    return (
        lanes,
        lane_turns,
        span_count,
        min_start_ns / 1_000_000_000,
        max_end_ns / 1_000_000_000,
    )


def _publish_agent_loop_activity_window(
    run_id: str, *, first_turn_unix: float, last_turn_unix: float
) -> None:
    """Prom gauges used to hide Repeat rows when the dashboard range misses turns."""
    insight.metric_gauge(
        "agent_loop_first_turn_unixtime",
        float(first_turn_unix),
        run_id=run_id,
    )
    insight.metric_gauge(
        "agent_loop_last_turn_unixtime",
        float(last_turn_unix),
        run_id=run_id,
    )


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


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be an integer greater than zero")
    return number


def _tcp_port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer in 1..65535")
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
    auth: tuple[str, str] | None = None,
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
                auth=auth,
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
        auth=("admin", "admin"),
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


def _new_run_id() -> str:
    """Unique run id per process/invocation (matches Repeat POC export-*)."""
    return f"export-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _generate_agent_loop_dashboard_tree(
    *,
    shape: _AgentLoopShape,
    step_duration_s: float,
    step_gap_s: float,
) -> dict[str, Any]:
    """Emit Prom ``*_info`` + Tempo turns (random tree unless CLI overrides)."""
    run_ids = [_new_run_id() for _ in range(shape.runs)]
    all_lanes: list[str] = []
    lane_turns: dict[str, int] = {}
    span_count = 0
    session_series = 0
    samples_per_run = 0
    for run_index, run_id in enumerate(run_ids):
        tree = _build_sample_tree(
            _AgentLoopShape(
                runs=1,
                samples=shape.samples,
                sessions=shape.sessions,
                trajs=shape.trajs,
                turns=shape.turns,
                seed=shape.seed + run_index * 10_007,
            )
        )
        samples_per_run = len(tree)
        session_series += sum(len(sample.sessions) for sample in tree)
        _publish_agent_loop_prom(run_id, tree)
        lanes, turns_map, n_spans, first_unix, last_unix = _emit_agent_loop_tempo_turns(
            run_id=run_id,
            samples=tree,
            step_duration_s=step_duration_s,
            step_gap_s=step_gap_s,
        )
        _publish_agent_loop_activity_window(
            run_id, first_turn_unix=first_unix, last_turn_unix=last_unix
        )
        all_lanes.extend(lanes)
        lane_turns.update(turns_map)
        span_count += n_spans
    return {
        "runs": run_ids,
        "samples": samples_per_run,
        "session_series": session_series,
        "lanes": all_lanes,
        "lane_turns": lane_turns,
        "spans": span_count,
    }


def _wait_for_prom_series(
    prometheus_url: str,
    query: str,
    *,
    expected: int,
    timeout: float,
) -> list[dict[str, Any]]:
    import requests

    deadline = time.monotonic() + timeout
    last_payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=3,
            )
            response.raise_for_status()
            last_payload = response.json()
            result = last_payload.get("data", {}).get("result", [])
            if len(result) >= expected:
                return result
        except (requests.RequestException, ValueError, AttributeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"Prometheus query not ready: query={query!r} expected>={expected} "
        f"within {timeout}s: last_error={last_error} last_payload={last_payload}"
    )


def _verify_agent_loop_dashboard(
    *,
    prometheus_url: str,
    tempo_url: str,
    fixture: dict[str, Any],
    timeout: float,
) -> None:
    import requests

    run_filter = "|".join(fixture["runs"])
    samples = int(fixture["samples"])
    session_series = int(fixture["session_series"])
    lane_turns: dict[str, int] = fixture["lane_turns"]
    for metric, expected in (
        ("rl_insight_monitor_agent_loop_run_info", len(fixture["runs"])),
        (
            "rl_insight_monitor_agent_loop_sample_info",
            len(fixture["runs"]) * samples,
        ),
        ("rl_insight_monitor_agent_loop_session_info", session_series),
        (
            "rl_insight_monitor_agent_loop_traj_info",
            len(fixture["lanes"]),
        ),
        (
            "rl_insight_monitor_agent_loop_first_turn_unixtime",
            len(fixture["runs"]),
        ),
        (
            "rl_insight_monitor_agent_loop_last_turn_unixtime",
            len(fixture["runs"]),
        ),
    ):
        series = _wait_for_prom_series(
            prometheus_url,
            f'{metric}{{run_id=~"{run_filter}"}}',
            expected=expected,
            timeout=timeout,
        )
        if len(series) != expected:
            raise RuntimeError(
                f"unexpected {metric} cardinality: expected={expected} "
                f"actual={len(series)}"
            )

    # Spot-check one leaf lane in Tempo (attributes forwarded by public APIs).
    lane = fixture["lanes"][0]
    turns = int(lane_turns[lane])
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"{tempo_url}/api/search",
                params={
                    "q": f'{{ span.state_lane_id = "{lane}" }}',
                    "limit": str(turns + 2),
                },
                timeout=3,
            )
            response.raise_for_status()
            traces = response.json().get("traces", [])
            if len(traces) >= turns:
                print(
                    f"PASS agent_loop_dashboard: runs={fixture['runs']} "
                    f"lanes={len(fixture['lanes'])} spans={fixture['spans']} "
                    f"tempo_lane={lane}"
                )
                return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"Tempo lane {lane!r} not queryable within {timeout}s: last_error={last_error}"
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
        "--metrics-report-port",
        type=_tcp_port,
        default=None,
        help=(
            "MonitorHub Prometheus scrape port "
            f"(default: {os.environ.get('RL_INSIGHT_METRICS_REPORT_PORT') or '9092'}; "
            "use when 9092 is already bound)"
        ),
    )
    parser.add_argument(
        "--step-duration",
        type=_positive_float,
        default=0.15,
        help="Live sleep for trace_span/trace_op API e2e steps (default: 0.15)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=60.0,
        help="Seconds to wait for Tempo/Grafana queries (default: 60)",
    )
    parser.add_argument(
        "--agent-loop-runs",
        type=_positive_int,
        default=1,
        help="Nested Repeat demo: fixed number of runs (default: 1)",
    )
    parser.add_argument(
        "--agent-loop-samples",
        type=_positive_int,
        default=None,
        help="Samples per run (default: random 2–3)",
    )
    parser.add_argument(
        "--agent-loop-sessions",
        type=_positive_int,
        default=None,
        help="Sessions per sample (default: random 2–4)",
    )
    parser.add_argument(
        "--agent-loop-trajs",
        type=_positive_int,
        default=None,
        help="Trajectories per session (default: random 1–3)",
    )
    parser.add_argument(
        "--agent-loop-turns",
        type=_positive_int,
        default=None,
        help=(
            "Turns per trajectory (default: random 4–10). "
            "Last turn is always a terminal finish_reason (stop/length/…)."
        ),
    )
    parser.add_argument(
        "--agent-loop-seed",
        type=int,
        default=42,
        help="RNG seed for random tree shape (default: 42)",
    )
    parser.add_argument(
        "--agent-loop-step-duration",
        type=_positive_float,
        default=1.0,
        help="Fabricated active span length for Agent Loop turns (default: 1.0s)",
    )
    parser.add_argument(
        "--agent-loop-step-gap",
        type=_positive_float,
        default=30.0,
        help="Fabricated gap between Agent Loop turns (default: 30s)",
    )
    parser.add_argument(
        "--exit-after-verify",
        action="store_true",
        help=(
            "Exit after verification (CI). Default keeps the MonitorHub alive so "
            "Prometheus can scrape agent_loop_*_info for the Grafana dashboard."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import ray

    from rl_insight.utils.constants import MonitorDefaults, MonitorRayActor

    tempo_url = args.tempo_url or _service_url(args.server_url, 3200)
    grafana_url = args.grafana_url or _service_url(args.server_url, 3000)
    prometheus_url = _service_url(args.server_url, 9090)

    metrics_port = args.metrics_report_port
    if metrics_port is None:
        env_port = os.environ.get("RL_INSIGHT_METRICS_REPORT_PORT")
        metrics_port = (
            _tcp_port(env_port) if env_port else MonitorDefaults.METRICS_REPORT_PORT
        )

    os.environ["RL_INSIGHT_SERVER_URL"] = args.server_url
    ray.init(namespace=MonitorRayActor.NAMESPACE, ignore_reinit_error=True)
    insight.init(
        project="rl-insight-experimental",
        experiment_name="trace-api-e2e",
        config={"prometheus": {"metrics_report_port": metrics_port}},
    )
    print(f"MonitorHub metrics_report_port={metrics_port}")
    agent_loop: dict[str, Any] | None = None
    runs: dict[str, str] = {}
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
        agent_loop = _generate_agent_loop_dashboard_tree(
            shape=_AgentLoopShape(
                runs=args.agent_loop_runs,
                samples=args.agent_loop_samples,
                sessions=args.agent_loop_sessions,
                trajs=args.agent_loop_trajs,
                turns=args.agent_loop_turns,
                seed=args.agent_loop_seed,
            ),
            step_duration_s=args.agent_loop_step_duration,
            step_gap_s=args.agent_loop_step_gap,
        )
        _verify_agent_loop_dashboard(
            prometheus_url=prometheus_url,
            tempo_url=tempo_url,
            fixture=agent_loop,
            timeout=args.timeout,
        )

        print("\nTempo and Grafana datasource verification passed.")
        print("Open Grafana Explore and inspect the three generated runs:")
        print(f"  {grafana_url}/explore")
        for interface, session_id in runs.items():
            print(f"  {interface}: {session_id}")
        print("\nAgent Loop nested Repeat fixture (dashboard agent_loop_trajectory):")
        print(f"  runs={agent_loop['runs']}")
        print(f"  lanes={len(agent_loop['lanes'])} spans={agent_loop['spans']}")
        print(f"  {grafana_url}/d/a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        if args.exit_after_verify:
            return
        # *_info gauges live on the Ray MonitorHub scrape target. Exiting would
        # drop them. Dashboard filters runs by first/last turn unixtime vs the
        # selected time range (no Tempo rewrite — that would duplicate turns).
        print(
            "\nKeeping MonitorHub alive for Prometheus scrape "
            "(required for dashboard *_info). Ctrl+C to stop."
        )
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        insight.finish()
        ray.shutdown()


if __name__ == "__main__":
    main()
