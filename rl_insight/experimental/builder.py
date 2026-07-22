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

"""Event-driven trajectory builder.

Accepts two event types and drives a ``BaseSample`` lifecycle internally.
Works with any ``BaseSample`` implementation (in-memory or filesystem).

Events::

    trajectory_begin   → create a new trajectory in the target session
    step               → append a step to the current trajectory

Usage::

    from rl_insight.experimental import TrajectoryBuilder

    # In-memory (default)
    builder = TrajectoryBuilder()
    builder.feed({"event": "trajectory_begin", "uid": "...", ...})
    builder.feed({"event": "step", "uid": "...", ...})
    samples = builder.samples

    # Filesystem-backed
    builder = TrajectoryBuilder(
        sample_factory=lambda uid, si: FileSampleRecord.create("/data", uid=uid, sample_index=si)
    )

    # Load from JSONL
    builder = TrajectoryBuilder.from_jsonl("events.jsonl")
"""

from __future__ import annotations

from typing import Any, Callable

from rl_insight.experimental.samples.base import BaseSample
from rl_insight.experimental.samples.sample import (
    SampleRecord,
    Step,
    ToolResult,
    TrainingStatus,
)

# Factory signature: (uid: str, sample_index: int) -> BaseSample
SampleFactory = Callable[[str, int], BaseSample]


def _default_factory(uid: str, sample_index: int) -> SampleRecord:
    return SampleRecord.create(uid=uid, sample_index=sample_index)


class TrajectoryBuilder:
    """Ingest trajectory events and build ``BaseSample`` objects.

    Maintains an internal registry of samples keyed by ``uid``. All
    operations on the samples go through the ``BaseSample`` interface.

    The builder does not care whether samples live in memory or on disk.
    """

    def __init__(
        self,
        sample_factory: SampleFactory = _default_factory,
    ) -> None:
        self._samples: dict[str, BaseSample] = {}
        self._factory = sample_factory
        # Per-sample cursor: (session_index, trajectory_index).
        self._cursor: dict[str, tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def samples(self) -> list[BaseSample]:
        """All built samples, in insertion order."""
        return list(self._samples.values())

    def get(self, uid: str) -> BaseSample | None:
        """Return a sample by uid, or None."""
        return self._samples.get(uid)

    def feed(self, event: dict[str, Any]) -> None:
        """Ingest a single event dict.

        Raises ``ValueError`` on unknown event types or missing required fields.
        """
        event_type = event.get("event")
        if event_type == "trajectory_begin":
            self._handle_trajectory_begin(event)
        elif event_type == "step":
            self._handle_step(event)
        else:
            raise ValueError(f"unknown event type: {event_type!r}")

    def feed_jsonl(self, path: str) -> TrajectoryBuilder:
        """Ingest all events from a JSONL file. Returns self for chaining."""
        import json

        with open(path) as f:
            for line in f:
                self.feed(json.loads(line))
        return self

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        sample_factory: SampleFactory = _default_factory,
    ) -> TrajectoryBuilder:
        """Create a builder and load a JSONL file."""
        builder = cls(sample_factory=sample_factory)
        builder.feed_jsonl(path)
        return builder

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_trajectory_begin(self, event: dict[str, Any]) -> None:
        uid = _require(event, "uid", str)
        sample_index = event.get("sample_index", 0)
        session_index = event.get("session_index", 0)
        trajectory_index = event.get("trajectory_index", 0)
        prompt_len = event.get("prompt_len", 0)

        sample = self._get_or_create_sample(uid, sample_index)
        traj = sample.new_trajectory(
            session_index=session_index,
            trajectory_index=trajectory_index,
        )
        # Set initial prompt length (not part of BaseSample, but TrajectoryRecord
        # is a shared return type -- we set it on the returned object).
        traj.prompt_len = prompt_len
        traj.tag.prompt_len = prompt_len

        self._cursor[uid] = (session_index, trajectory_index)

    def _handle_step(self, event: dict[str, Any]) -> None:
        uid = _require(event, "uid", str)
        cursor = self._cursor.get(uid, (0, 0))
        session_index = event.get("session_index", cursor[0])
        trajectory_index = event.get("trajectory_index", cursor[1])
        step_index = event.get("step_index", 0)
        finish_reason = event.get("finish_reason", "tool_calls")
        assistant_msg = event.get("assistant_msg")
        thought = event.get("thought", "")
        tool_results_raw = event.get("tool_results", [])

        sample = self._require_sample(uid)

        # Build step.
        tool_results = [_build_tool_result(tr) for tr in tool_results_raw]
        step = Step(
            step_idx=step_index,
            thought=thought,
            tool_results=tool_results,
            done=False,
            exit_reason="",
        )
        if assistant_msg and isinstance(assistant_msg, dict):
            step.response = assistant_msg.get("content", "")

        sample.add_step(session_index, trajectory_index, step)

        # Finish the trajectory if this step ends it.
        if finish_reason in ("stop", "length"):
            status: TrainingStatus = "success"
            if finish_reason == "length":
                status = "truncated"
            sample.finish_trajectory(
                session_index, trajectory_index, finish_reason, status
            )
            self._cursor[uid] = (session_index, trajectory_index + 1)
        else:
            self._cursor[uid] = (session_index, trajectory_index)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_sample(self, uid: str, sample_index: int) -> BaseSample:
        if uid not in self._samples:
            self._samples[uid] = self._factory(uid, sample_index)
        return self._samples[uid]

    def _require_sample(self, uid: str) -> BaseSample:
        sample = self._samples.get(uid)
        if sample is None:
            raise KeyError(
                f"sample {uid!r} not found -- send a trajectory_begin event first"
            )
        return sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(event: dict[str, Any], key: str, expected_type: type) -> Any:
    value = event.get(key)
    if value is None:
        raise ValueError(f"missing required field {key!r} in event: {event}")
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{key!r} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _build_tool_result(raw: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=raw.get("tool_call_id", ""),
        name=raw.get("name", ""),
        action=raw.get("action", ""),
        observation=raw.get("observation", ""),
        status=raw.get("status", "ok"),
        execution_time=raw.get("execution_time"),
    )
