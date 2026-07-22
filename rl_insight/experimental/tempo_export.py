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

"""Black-box SampleRecord tree → Tempo state spans.

PR#120 (``generate_data`` / ``TrajectoryBuilder`` / ``SampleRecord``) is treated
as an opaque producer. This module only reads the finished sample tree and
writes OTLP spans. Visualization must not import this module's Record logic;
it only sees attributes that land in Tempo (including ``run_id``).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import urllib.error
import urllib.request
from typing import Any, Iterable

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from rl_insight.experimental.agent_loop_constants import SERVICE_NAME_VALUE
from rl_insight.experimental.samples.sample import SampleRecord, Step, TrajectoryRecord

logger = logging.getLogger(__name__)

# Re-export for older imports.
_TRACER_NAME = "rl-insight.experimental.tempo_export"


def new_run_id() -> str:
    """Allocate a mapper-side run id (not from upstream generate_data)."""
    return f"export-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def lane_id(
    run_id: str,
    sample_index: int,
    session_index: int,
    trajectory_index: int,
) -> str:
    """State-timeline lane; includes ``run_id`` so multi-run lanes never collide."""
    return (
        f"run={run_id}/sample={sample_index}/"
        f"session={session_index}/traj={trajectory_index}"
    )


def wait_for_otlp(endpoint: str, *, timeout_s: float = 60.0) -> None:
    """Block until the OTLP/HTTP endpoint accepts connections."""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(endpoint, timeout=2)
            return
        except urllib.error.HTTPError:
            return
        except Exception as exc:  # noqa: BLE001 - startup race
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"OTLP endpoint not ready after {timeout_s:.0f}s: {endpoint} ({last_err})"
    )


def _coerce_sample(sample: Any) -> SampleRecord:
    """Accept SampleRecord or PR120 FileSampleRecord / thin wrappers."""
    if isinstance(sample, SampleRecord):
        return sample
    inner = getattr(sample, "_inner", None)
    if isinstance(inner, SampleRecord):
        return inner
    load = getattr(sample, "load", None)
    if callable(load):
        loaded = load()
        if isinstance(loaded, SampleRecord):
            return loaded
    raise TypeError(
        f"tempo_export expects SampleRecord-like output, got {type(sample)!r}"
    )


def _step_type(step: Step) -> str:
    return "tool" if step.tool_results else "llm"


def _tool_names(step: Step) -> list[str]:
    return [tr.name for tr in step.tool_results if tr.name]


def _finish_reason_for_step(
    traj: TrajectoryRecord, step: Step, *, is_last: bool
) -> str:
    if not is_last:
        return step.exit_reason or "tool_calls"
    if step.exit_reason:
        return step.exit_reason
    tag_fr = getattr(getattr(traj, "tag", None), "finish_reason", "") or ""
    return tag_fr or "stop"


def samples_to_span_dicts(
    samples: Iterable[Any],
    *,
    run_id: str,
    step_duration_s: float = 1.0,
    clock_start_ns: int | None = None,
) -> list[dict[str, Any]]:
    """Map a finished sample tree to in-memory span dicts (not yet sent)."""
    clock = clock_start_ns
    if clock is None:
        clock = int((time.time() - 60.0) * 1_000_000_000)
    duration_ns = max(int(step_duration_s * 1_000_000_000), 250_000_000)

    out: list[dict[str, Any]] = []
    for raw in samples:
        sample = _coerce_sample(raw)
        si = int(sample.sample_index)
        uid = str(sample.uid)
        for session in sample.sessions:
            sess_i = int(session.session_index)
            for traj in session.trajectories:
                ti = int(traj.trajectory_index)
                reward = traj.reward_score
                reward_s = "" if reward is None else str(reward)
                steps = list(traj.steps)
                for idx, step in enumerate(steps):
                    start_ns = clock
                    end_ns = start_ns + duration_ns
                    clock = end_ns
                    is_last = idx == len(steps) - 1
                    finish_reason = _finish_reason_for_step(
                        traj, step, is_last=is_last
                    )
                    name = finish_reason or "unknown"
                    attrs: dict[str, Any] = {
                        "run_id": run_id,
                        "state_lane_id": lane_id(run_id, si, sess_i, ti),
                        "sample": str(si),
                        "uid": uid,
                        "session": str(sess_i),
                        "traj": str(ti),
                        "turn": str(step.step_idx),
                        "type": _step_type(step),
                        "tools": json.dumps(_tool_names(step), ensure_ascii=False),
                        "content": (step.thought or step.response or "")[:500],
                        "reward": reward_s,
                        "monitor.trace_segment": "state_interval",
                        "state_name": name,
                        "finish_reason": name,
                    }
                    out.append(
                        {
                            "name": name,
                            "start_time_ns": start_ns,
                            "end_time_ns": end_ns,
                            "attributes": attrs,
                        }
                    )
    return out


def compress_span_times(
    spans: list[dict[str, Any]],
    *,
    window_s: float = 1800.0,
    lag_s: float = 60.0,
) -> list[dict[str, Any]]:
    """Fit spans into ``[now - lag - window, now - lag]`` for Tempo searchability."""
    if not spans or window_s <= 0:
        return spans
    starts = [item["start_time_ns"] for item in spans]
    ends = [item["end_time_ns"] for item in spans]
    min_start = min(starts)
    max_end = max(ends)
    orig_span = max_end - min_start
    if orig_span <= 0:
        return spans

    target_end_ns = int((time.time() - lag_s) * 1_000_000_000)
    target_start_ns = target_end_ns - int(window_s * 1_000_000_000)
    scale = (target_end_ns - target_start_ns) / orig_span

    out: list[dict[str, Any]] = []
    for item in spans:
        start = target_start_ns + int((item["start_time_ns"] - min_start) * scale)
        end = target_start_ns + int((item["end_time_ns"] - min_start) * scale)
        if end <= start:
            end = start + 250_000_000
        cloned = dict(item)
        cloned["start_time_ns"] = start
        cloned["end_time_ns"] = end
        out.append(cloned)
    return out


def flush_span_dicts(
    spans: list[dict[str, Any]],
    *,
    endpoint: str,
    service_name: str = SERVICE_NAME_VALUE,
    batch_size: int = 50,
    pause_s: float = 0.15,
    window_s: float = 1800.0,
) -> int:
    """Chunked OTLP export (SimpleSpanProcessor) so local Tempo does not drop bursts."""
    if not spans:
        return 0
    wait_for_otlp(endpoint)
    compressed = compress_span_times(spans, window_s=window_s)

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name})
    )
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    tracer = provider.get_tracer(_TRACER_NAME)
    logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(
        logging.ERROR
    )
    logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.ERROR)

    total = len(compressed)
    for i in range(0, total, batch_size):
        chunk = compressed[i : i + batch_size]
        for item in chunk:
            span = tracer.start_span(
                item["name"],
                start_time=item["start_time_ns"],
                attributes=item["attributes"],
            )
            span.end(end_time=item["end_time_ns"])
        time.sleep(pause_s)
        logger.info("exported %s/%s spans", min(i + batch_size, total), total)

    provider.force_flush(timeout_millis=30_000)
    provider.shutdown()
    return total


def export_samples_to_tempo(
    samples: Iterable[Any],
    *,
    endpoint: str = "http://127.0.0.1:4318/v1/traces",
    run_id: str | None = None,
    service_name: str = SERVICE_NAME_VALUE,
    step_duration_s: float = 1.0,
    wait: bool = True,
    wait_timeout_s: float = 60.0,
    batch_size: int = 50,
    pause_s: float = 0.15,
    window_s: float = 1800.0,
) -> dict[str, Any]:
    """Map finished samples → Tempo. Returns ``{run_id, spans, service_name}``."""
    rid = run_id or new_run_id()
    if wait:
        wait_for_otlp(endpoint, timeout_s=wait_timeout_s)
    span_dicts = samples_to_span_dicts(
        samples, run_id=rid, step_duration_s=step_duration_s
    )
    n = flush_span_dicts(
        span_dicts,
        endpoint=endpoint,
        service_name=service_name,
        batch_size=batch_size,
        pause_s=pause_s,
        window_s=window_s,
    )
    logger.info(
        "tempo_export done: run_id=%s spans=%s service.name=%s",
        rid,
        n,
        service_name,
    )
    return {"run_id": rid, "spans": n, "service_name": service_name}
