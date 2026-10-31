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

"""Unit tests for the hardware monitor background loop."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fakes import RecordingClient, WorkingCollector, wait_for

from rl_insight.hardware.base import HardwareCollector, Sample, register_collector
from rl_insight.hardware.coordinator import HardwareMonitor
from rl_insight.utils.constants import MonitorEventKind, MonitorHardware

_STATUS_METRIC = MonitorHardware.STATUS_METRIC_NAME


class BatchRecordingClient:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    def apply_events(self, events: list[dict[str, Any]]) -> None:
        self.batches.append(list(events))


class FailingClient:
    def apply_events(self, _events: list[dict[str, Any]]) -> None:
        raise RuntimeError("hub unreachable")


class BrokenCollector(HardwareCollector):
    name = "fake_broken"

    def __init__(self) -> None:
        self.collect_calls = 0

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        self.collect_calls += 1
        raise RuntimeError("probe exploded")


def _events_named(client: RecordingClient, name: str) -> list[dict[str, Any]]:
    return [event for event in client.events if event["name"] == name]


def test_monitor_should_start_sample_and_stop() -> None:
    register_collector("fake_working", WorkingCollector)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client,
        labels={"project": "p"},
        collector_names=["fake_working"],
        interval_seconds=60.0,
    )

    assert monitor.start() is True
    assert monitor.start() is False  # already running
    wait_for(lambda: len(client.events) >= 2, "expected a status gauge and a sample")
    monitor.stop()

    assert monitor.is_running is False
    status = _events_named(client, _STATUS_METRIC)
    assert status[0]["kind"] == MonitorEventKind.GAUGE
    assert status[0]["value"] == 1.0
    assert status[0]["labels"] == {"project": "p", "collector": "fake_working"}

    sample = _events_named(client, "rlinsight_fake_gauge")[0]
    assert sample["labels"] == {"project": "p", "device_index": "0"}
    assert sample["documentation"] == "Fake gauge"


def test_monitor_should_report_unavailable_and_unknown_collectors() -> None:
    class Unavailable(HardwareCollector):
        name = "fake_unavailable"

        def discover(self) -> bool:
            return False

        def collect(self) -> list[Sample]:
            raise AssertionError("must not be sampled")

    register_collector(Unavailable.name, Unavailable)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client,
        collector_names=[Unavailable.name, "ghost"],
        interval_seconds=60.0,
    )

    monitor._resolve_and_report()
    monitor._collect_once()
    monitor.stop()

    status = _events_named(client, _STATUS_METRIC)
    assert {event["labels"]["collector"] for event in status} == {
        "fake_unavailable",
        "ghost",
    }
    assert all(event["value"] == 0.0 for event in status)
    assert client.events == status


def test_broken_collector_should_be_deactivated_after_consecutive_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_collector("fake_broken", BrokenCollector)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client, collector_names=["fake_broken"], interval_seconds=60.0
    )

    with caplog.at_level(logging.WARNING, logger="rl_insight.hardware.coordinator"):
        monitor._resolve_and_report()
        for _ in range(3):
            monitor._collect_once()
        monitor._collect_once()  # deactivated: not called again
    monitor.stop()

    entry = monitor._entries[0]
    assert isinstance(entry.collector, BrokenCollector)
    assert entry.collector.collect_calls == 3
    assert entry.available is False
    assert [event["value"] for event in _events_named(client, _STATUS_METRIC)] == [
        1.0,
        0.0,
    ]
    assert any("deactivating" in record.message for record in caplog.records)


def test_one_round_should_be_submitted_as_one_batch_and_drops_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_collector("fake_working", WorkingCollector)
    client = BatchRecordingClient()
    monitor = HardwareMonitor(
        client=client, collector_names=["fake_working"], interval_seconds=60.0
    )

    monitor._resolve_and_report()
    monitor._collect_once()
    monitor.stop()

    assert [event["name"] for event in client.batches[0]] == [_STATUS_METRIC]
    assert [event["name"] for event in client.batches[1]] == ["rlinsight_fake_gauge"]

    monitor = HardwareMonitor(
        client=FailingClient(), collector_names=["fake_working"], interval_seconds=60.0
    )
    with caplog.at_level(logging.WARNING, logger="rl_insight.hardware.coordinator"):
        monitor._resolve_and_report()
        monitor._collect_once()
    monitor.stop()

    assert monitor.dropped_events == 2
    assert any("failed to forward" in record.message for record in caplog.records)
