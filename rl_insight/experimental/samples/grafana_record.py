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

"""BaseSample implementation that mirrors SampleRecord and exports steps to Tempo.

``TrajectoryBuilder`` only passes ``Step`` objects into ``add_step``; it does
**not** copy event ``finish_reason`` onto ``Step.exit_reason``. Terminal reasons
(``stop`` / ``length``) arrive on the subsequent ``finish_trajectory`` call.
This class therefore buffers each step until the next step or finish so spans
can be labeled correctly for Grafana State Timeline.

Mock generate_data also has no event timestamps; spans use a synthetic clock
(``step_duration_s``) so batch mode still produces a readable timeline.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from rl_insight.experimental.samples.sample import (
    SampleRecord,
    Step,
    TrajectoryRecord,
    TrainingStatus,
)

logger = logging.getLogger(__name__)

# Isolate demo spans from online-monitor training timelines.
SERVICE_NAME_VALUE = "agent-loop-poc"
_TRACER_NAME = "rl-insight.experimental.grafana_record"

_provider: TracerProvider | None = None
_tracer = None
_endpoint: str | None = None
# Shared synthetic clock across all samples (ns).
_clock_ns: int | None = None


def _lane_id(sample_index: int, session_index: int, trajectory_index: int) -> str:
    return f"sample={sample_index}/session={session_index}/traj={trajectory_index}"


def _step_type(step: Step) -> str:
    return "tool" if step.tool_results else "llm"


def _tool_names(step: Step) -> list[str]:
    return [tr.name for tr in step.tool_results if tr.name]


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


def configure_exporter(
    endpoint: str,
    *,
    service_name: str = SERVICE_NAME_VALUE,
    wait: bool = True,
    wait_timeout_s: float = 60.0,
) -> None:
    """Initialize a process-wide OTLP exporter (call once before generate/stream)."""
    global _provider, _tracer, _endpoint, _clock_ns
    if wait:
        wait_for_otlp(endpoint, timeout_s=wait_timeout_s)
    _endpoint = endpoint
    _provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    _provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    _tracer = _provider.get_tracer(_TRACER_NAME)
    # Anchor near "now" so Tempo's recent-search window picks spans up immediately.
    _clock_ns = int((time.time() - 60.0) * 1_000_000_000)
    logger.info("GrafanaRecord OTLP exporter ready: %s", endpoint)


def shutdown_exporter() -> None:
    """Flush/shutdown the process-wide exporter."""
    global _provider, _tracer
    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _tracer = None


def _next_span_window(step_duration_s: float) -> tuple[int, int]:
    global _clock_ns
    if _clock_ns is None:
        _clock_ns = int((time.time() - 60.0) * 1_000_000_000)
    start = _clock_ns
    duration_ns = max(int(step_duration_s * 1_000_000_000), 250_000_000)
    end = start + duration_ns
    _clock_ns = end
    return start, end


def _emit_span(
    *,
    finish_reason: str,
    start_ns: int,
    end_ns: int,
    attributes: dict[str, Any],
) -> None:
    if _tracer is None:
        raise RuntimeError(
            "GrafanaRecord exporter is not configured; call configure_exporter() first"
        )
    name = finish_reason or "unknown"
    attrs = dict(attributes)
    attrs["monitor.trace_segment"] = "state_interval"
    attrs["state_name"] = name
    attrs["finish_reason"] = name
    span = _tracer.start_span(name, start_time=start_ns, attributes=attrs)
    span.end(end_time=end_ns)


class GrafanaRecord:
    """In-memory BaseSample that also exports each step as a Tempo state span.

    Storage is delegated to an internal ``SampleRecord``. Span export is a
    side effect of ``add_step`` / ``finish_trajectory``.
    """

    def __init__(
        self,
        inner: SampleRecord,
        *,
        step_duration_s: float = 1.0,
    ) -> None:
        self._inner = inner
        self.step_duration_s = step_duration_s
        # key: (session_index, trajectory_index) → pending step waiting for reason
        self._pending: dict[tuple[int, int], Step] = {}
        self._rewards: dict[tuple[int, int], float] = {}

    @classmethod
    def create(
        cls,
        *,
        uid: str,
        sample_index: int = 0,
        step_duration_s: float = 1.0,
    ) -> GrafanaRecord:
        return cls(
            SampleRecord.create(uid=uid, sample_index=sample_index),
            step_duration_s=step_duration_s,
        )

    @property
    def uid(self) -> str:
        return self._inner.uid

    @property
    def sample_index(self) -> int:
        return self._inner.sample_index

    def new_trajectory(self, session_index: int = 0, **kwargs: Any) -> TrajectoryRecord:
        return self._inner.new_trajectory(session_index, **kwargs)

    def get_trajectory(
        self, session_index: int, trajectory_index: int
    ) -> TrajectoryRecord | None:
        return self._inner.get_trajectory(session_index, trajectory_index)

    def add_step(self, session_index: int, trajectory_index: int, step: Step) -> None:
        key = (session_index, trajectory_index)
        previous = self._pending.pop(key, None)
        if previous is not None:
            # Next step arrived ⇒ previous was a non-terminal tool/llm turn.
            self._flush_step(
                session_index, trajectory_index, previous, finish_reason="tool_calls"
            )
        self._pending[key] = step
        self._inner.add_step(session_index, trajectory_index, step)

    def finish_trajectory(
        self,
        session_index: int,
        trajectory_index: int,
        exit_reason: str = "finished",
        status: TrainingStatus = "success",
    ) -> None:
        key = (session_index, trajectory_index)
        pending = self._pending.pop(key, None)
        if pending is not None:
            # Builder calls add_step then finish_trajectory for stop/length.
            reason = exit_reason if exit_reason else "stop"
            self._flush_step(
                session_index, trajectory_index, pending, finish_reason=reason
            )
        self._inner.finish_trajectory(
            session_index, trajectory_index, exit_reason, status
        )

    def set_trajectory_reward(
        self,
        session_index: int,
        trajectory_index: int,
        score: float,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        self._rewards[(session_index, trajectory_index)] = score
        self._inner.set_trajectory_reward(
            session_index, trajectory_index, score, extra_info
        )

    def set_trajectory_token_data(
        self,
        session_index: int,
        trajectory_index: int,
        *,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        response_mask: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        routed_experts: Any = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> None:
        self._inner.set_trajectory_token_data(
            session_index,
            trajectory_index,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
        )

    def flush_pending(self) -> None:
        """Emit any steps never closed by ``finish_trajectory`` (e.g. max_step_limit)."""
        for (si, ti), step in list(self._pending.items()):
            # generate_data may emit finish_reason=max_step_limit; Builder does not
            # treat it as terminal, so we fall back to tool_calls / step.exit_reason.
            reason = step.exit_reason or "tool_calls"
            self._flush_step(si, ti, step, finish_reason=reason)
        self._pending.clear()

    def _flush_step(
        self,
        session_index: int,
        trajectory_index: int,
        step: Step,
        *,
        finish_reason: str,
    ) -> None:
        start_ns, end_ns = _next_span_window(self.step_duration_s)
        reward = self._rewards.get((session_index, trajectory_index))
        tools = _tool_names(step)
        attributes: dict[str, Any] = {
            "state_lane_id": _lane_id(
                self.sample_index, session_index, trajectory_index
            ),
            "sample": str(self.sample_index),
            "uid": self.uid,
            "session": str(session_index),
            "traj": str(trajectory_index),
            "turn": str(step.step_idx),
            "type": _step_type(step),
            "tools": json.dumps(tools, ensure_ascii=False),
            "content": (step.thought or step.response or "")[:500],
            "reward": "" if reward is None else str(reward),
        }
        _emit_span(
            finish_reason=finish_reason,
            start_ns=start_ns,
            end_ns=end_ns,
            attributes=attributes,
        )
