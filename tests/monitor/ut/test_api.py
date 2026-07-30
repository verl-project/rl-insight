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

"""Unit tests for the public monitor API."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Generator
from typing import Any

import pytest

from rl_insight import api
from rl_insight.utils.constants import MonitorEventKind


class RecordingClient:
    """Small client double that preserves every submitted event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def apply_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def reset_monitor_state() -> Generator[None, None, None]:
    api.finish()
    yield
    api.finish()


@pytest.fixture
def recording_client(monkeypatch: pytest.MonkeyPatch) -> RecordingClient:
    client = RecordingClient()
    monkeypatch.setattr(api, "create_monitor_client", lambda _conf: client)
    api.init(
        project="project-a",
        experiment_name="experiment-a",
        config={"server": {"url": "http://monitor:18080"}},
    )
    return client


def test_init_should_enable_monitoring_when_server_and_client_are_available(
    recording_client: RecordingClient,
) -> None:
    assert api._STATE.enabled is True
    assert api._STATE.client is recording_client
    assert api._STATE.labels == {
        "project": "project-a",
        "experiment_name": "experiment-a",
    }


def test_init_should_leave_monitoring_disabled_when_server_url_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_called = False

    def create_client(_conf: Any) -> RecordingClient:
        nonlocal factory_called
        factory_called = True
        return RecordingClient()

    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    monkeypatch.setattr(api, "create_monitor_client", create_client)

    api.init(config={"server": {"url": ""}})

    assert api._STATE.enabled is False
    assert factory_called is False


def test_metric_helpers_should_emit_typed_events_when_monitoring_is_enabled(
    recording_client: RecordingClient,
) -> None:
    api.metric_count("steps", amount=2, worker="w0")
    api.metric_gauge("reward", value=1.5, documentation="Latest reward", worker="w0")
    api.metric_histogram("latency", value=12, worker="w0")

    assert [event["kind"] for event in recording_client.events] == [
        MonitorEventKind.COUNTER,
        MonitorEventKind.GAUGE,
        MonitorEventKind.HISTOGRAM,
    ]
    assert [event["value"] for event in recording_client.events] == [2.0, 1.5, 12.0]
    assert recording_client.events[0]["documentation"] == "Counter steps"
    assert recording_client.events[1]["documentation"] == "Latest reward"
    assert recording_client.events[2]["documentation"] == "Histogram latency"
    assert recording_client.events[0]["labels"] == {
        "project": "project-a",
        "experiment_name": "experiment-a",
        "worker": "w0",
    }


def test_trace_state_should_merge_same_state_and_ignore_shadow_when_lane_is_busy(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([100, 200])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    with api.trace_state("rollout", state_lane_id="replica-0", step=3):
        with api.trace_state("rollout", state_lane_id="replica-0"):
            pass
        with api.trace_state("shadowed", state_lane_id="replica-0"):
            pass

    assert len(recording_client.events) == 1
    event = recording_client.events[0]
    assert event["kind"] == MonitorEventKind.TRACE
    assert event["name"] == "rollout"
    assert (event["start_time_ns"], event["end_time_ns"]) == (100, 200)
    assert event["attributes"] == {
        "process_id": api._STATE.process_id,
        "project": "project-a",
        "experiment_name": "experiment-a",
        "step": 3,
        "monitor.trace_segment": "state_interval",
        "state_name": "rollout",
        "state_lane_id": "replica-0",
    }


def test_trace_op_sync_should_propagate_exception_and_report_duration_span(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([300, 450])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    @api.trace_op("train", extra_labels=lambda item: {"worker": item}, phase="update")
    def train(worker: str) -> None:
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        train("w1")

    event = recording_client.events[0]
    assert event["name"] == "train"
    assert (event["start_time_ns"], event["end_time_ns"]) == (300, 450)
    assert event["attributes"]["worker"] == "w1"
    assert event["attributes"]["phase"] == "update"
    assert event["attributes"]["monitor.trace_segment"] == "duration"


def test_finish_should_disable_future_events_when_monitoring_was_enabled(
    recording_client: RecordingClient,
) -> None:
    api.finish()
    api.metric_count("ignored")

    assert recording_client.events == []
    assert api._STATE.enabled is False


# ---------------------------------------------------------------------------
# Direct span reporting and sync/async ``trace_op``.
# ---------------------------------------------------------------------------


def test_trace_span_should_emit_complete_trace_event_when_reported_directly(
    recording_client: RecordingClient,
) -> None:
    api.trace_span(
        name="tool_calls",
        start_time_ns=1000,
        end_time_ns=2000,
        attributes={"run_id": "abc", "turn": "5"},
    )

    assert len(recording_client.events) == 1
    event = recording_client.events[0]
    assert event["kind"] == MonitorEventKind.TRACE
    assert event["name"] == "tool_calls"
    assert (event["start_time_ns"], event["end_time_ns"]) == (1000, 2000)
    assert event["attributes"]["run_id"] == "abc"
    assert event["attributes"]["turn"] == "5"
    assert "monitor.trace_segment" not in event["attributes"]


def test_trace_span_should_merge_init_attributes(
    recording_client: RecordingClient,
) -> None:
    api.trace_span(name="span", start_time_ns=1, end_time_ns=2, attributes={"k": "v"})

    attributes = recording_client.events[0]["attributes"]
    assert attributes["process_id"] == api._STATE.process_id
    assert attributes["project"] == "project-a"
    assert attributes["experiment_name"] == "experiment-a"
    assert attributes["k"] == "v"


def test_trace_span_should_copy_attributes_so_later_mutation_is_isolated(
    recording_client: RecordingClient,
) -> None:
    attributes = {"k": "v"}
    api.trace_span(name="span", start_time_ns=1, end_time_ns=2, attributes=attributes)

    attributes["k"] = "changed"
    attributes["added_later"] = "x"

    stored = recording_client.events[0]["attributes"]
    assert stored["k"] == "v"
    assert "added_later" not in stored


def test_trace_op_sync_should_return_value_and_report_duration_span(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([10, 20])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    @api.trace_op("op", phase="update")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5

    event = recording_client.events[0]
    assert event["name"] == "op"
    assert (event["start_time_ns"], event["end_time_ns"]) == (10, 20)
    assert event["attributes"]["phase"] == "update"
    assert event["attributes"]["monitor.trace_segment"] == "duration"


def test_trace_op_async_should_report_span_around_await(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([100, 200])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    @api.trace_op("async_op")
    async def compute() -> str:
        await asyncio.sleep(0)
        return "done"

    assert asyncio.run(compute()) == "done"

    event = recording_client.events[0]
    assert event["name"] == "async_op"
    assert (event["start_time_ns"], event["end_time_ns"]) == (100, 200)
    assert event["attributes"]["monitor.trace_segment"] == "duration"


def test_trace_op_async_should_preserve_coroutine_function_identity(
    recording_client: RecordingClient,
) -> None:
    @api.trace_op()
    async def compute() -> int:
        return 1

    assert inspect.iscoroutinefunction(compute)


def test_trace_op_async_should_propagate_exception_and_still_emit_span(
    recording_client: RecordingClient,
) -> None:
    @api.trace_op("async_boom")
    async def fail() -> None:
        raise ValueError("nope")

    async def run() -> None:
        with pytest.raises(ValueError, match="nope"):
            await fail()

    asyncio.run(run())

    assert recording_client.events[0]["name"] == "async_boom"


def test_trace_op_async_should_emit_span_and_propagate_cancellation(
    recording_client: RecordingClient,
) -> None:
    @api.trace_op("async_cancel")
    async def block() -> None:
        await asyncio.sleep(10)

    async def run() -> None:
        task = asyncio.ensure_future(block())
        await asyncio.sleep(0)  # let the task reach the inner await
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert len(recording_client.events) == 1
    assert recording_client.events[0]["name"] == "async_cancel"


def test_trace_op_should_warn_and_keep_static_labels_when_extra_labels_fails(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([1, 2])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    def boom(_first: Any) -> dict[str, Any]:
        raise RuntimeError("extra broke")

    @api.trace_op("base", extra_labels=boom, static="s")
    def run(_self: str) -> int:
        return 1

    with pytest.warns(RuntimeWarning):
        assert run("self") == 1

    assert recording_client.events[0]["attributes"]["static"] == "s"


def test_trace_op_should_skip_timing_labels_and_emission_when_monitoring_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No ``recording_client`` fixture, so monitoring stays disabled.
    called = {"extra": False}

    def forbidden_time() -> int:
        raise AssertionError("time_ns must not be called when monitoring is disabled")

    monkeypatch.setattr(api.time, "time_ns", forbidden_time)

    def record_extra(_first: Any) -> dict[str, Any]:
        called["extra"] = True
        return {}

    @api.trace_op("op", extra_labels=record_extra)
    def run(_self: str) -> int:
        return 1

    assert run("self") == 1
    assert called == {"extra": False}


def test_trace_span_and_trace_op_should_produce_same_event_shape(
    recording_client: RecordingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([10, 20])
    monkeypatch.setattr(api.time, "time_ns", lambda: next(timestamps))

    @api.trace_op("op")
    def run() -> None:
        return None

    run()
    api.trace_span(name="op", start_time_ns=10, end_time_ns=20, attributes={})

    decorator_event, direct_event = recording_client.events
    assert decorator_event.keys() == direct_event.keys()
    for event in (decorator_event, direct_event):
        assert event["kind"] == MonitorEventKind.TRACE
        assert event["name"] == "op"
        assert isinstance(event["start_time_ns"], int)
        assert isinstance(event["end_time_ns"], int)
        # identical init-level merge on both reporting paths
        assert event["attributes"]["process_id"] == api._STATE.process_id
        assert event["attributes"]["project"] == "project-a"
        assert event["attributes"]["experiment_name"] == "experiment-a"
    # the one intended difference is the compat-only segment marker
    assert decorator_event["attributes"]["monitor.trace_segment"] == "duration"
    assert "monitor.trace_segment" not in direct_event["attributes"]
