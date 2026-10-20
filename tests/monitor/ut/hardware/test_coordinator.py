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
from typing import Any, Mapping

import pytest

from fakes import RecordingClient, WorkingCollector, wait_for

from rl_insight.hardware.base import (
    COLLECTOR_REGISTRY,
    HardwareCollector,
    Sample,
    register_collector,
)
from rl_insight.hardware.coordinator import HardwareMonitor
from rl_insight.utils.constants import MonitorEventKind, MonitorHardware

_STATUS_METRIC = MonitorHardware.STATUS_METRIC_NAME


class BatchRecordingClient:
    """Client double recording each ``apply_events`` batch as one entry."""

    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    def apply_events(self, events: list[dict[str, Any]]) -> None:
        self.batches.append(list(events))


class FailingClient:
    """Client double whose transport always raises."""

    def apply_events(self, _events: list[dict[str, Any]]) -> None:
        raise RuntimeError("hub unreachable")


class BrokenCollector(HardwareCollector):
    """Fake collector whose sampling always raises."""

    name = "fake_broken"

    def __init__(self) -> None:
        self.collect_calls = 0

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        self.collect_calls += 1
        raise RuntimeError("probe exploded")


class UnavailableCollector(HardwareCollector):
    """Fake collector reporting no capability on this node."""

    name = "fake_unavailable"

    def discover(self) -> bool:
        return False

    def collect(self) -> list[Sample]:
        raise AssertionError("must not be sampled when unavailable")


class MalformedSampleCollector(HardwareCollector):
    name = "fake_malformed"

    def __init__(self) -> None:
        self.collect_calls = 0

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        self.collect_calls += 1
        return [None]  # type: ignore[list-item]


def _events_named(client: RecordingClient, name: str) -> list[dict[str, Any]]:
    return [event for event in client.events if event["name"] == name]


def test_thread_should_emit_status_and_samples_after_start() -> None:
    register_collector("fake_working", WorkingCollector)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client,
        labels={"project": "p"},
        collector_names=["fake_working"],
        interval_seconds=60.0,
    )

    assert monitor.start() is True
    assert monitor.is_running is True
    wait_for(
        lambda: len(client.events) >= 2,
        "expected one status gauge and one sample event",
    )
    monitor.stop()
    assert monitor.is_running is False

    status = _events_named(client, _STATUS_METRIC)
    assert len(status) == 1
    assert status[0]["kind"] == MonitorEventKind.GAUGE
    assert status[0]["value"] == 1.0
    assert status[0]["labels"] == {"project": "p", "collector": "fake_working"}

    samples = _events_named(client, "rlinsight_fake_gauge")
    assert len(samples) == 1
    assert samples[0]["value"] == 7.5
    assert samples[0]["documentation"] == "Fake gauge"
    assert samples[0]["labels"] == {"project": "p", "device_index": "0"}


def test_lifecycle_should_be_idempotent_at_every_step() -> None:
    client = RecordingClient()
    monitor = HardwareMonitor(client=client, interval_seconds=60.0)

    monitor.stop()  # safe before start
    assert monitor.start() is True
    assert monitor.start() is False  # second start is a no-op
    monitor.stop()
    monitor.stop()  # idempotent
    assert monitor.is_running is False


def test_collect_once_should_merge_base_labels_and_stringify_values() -> None:
    register_collector("fake_working", WorkingCollector)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client,
        labels={"project": "p", "node": "node-1"},
        collector_names=["fake_working"],
        interval_seconds=60.0,
    )

    monitor._resolve_and_report()
    monitor._collect_once()
    monitor.stop()

    event = _events_named(client, "rlinsight_fake_gauge")[0]
    assert event["kind"] == MonitorEventKind.GAUGE
    assert event["labels"] == {"project": "p", "node": "node-1", "device_index": "0"}


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
        monitor._collect_once()  # deactivated: must not call collect again
    monitor.stop()

    entry = monitor._entries[0]
    assert isinstance(entry.collector, BrokenCollector)
    assert entry.collector.collect_calls == 3
    assert entry.available is False

    status_values = [event["value"] for event in _events_named(client, _STATUS_METRIC)]
    assert status_values == [1.0, 0.0]
    assert any("deactivating" in record.message for record in caplog.records)


def test_malformed_samples_should_not_stop_the_loop() -> None:
    register_collector(MalformedSampleCollector.name, MalformedSampleCollector)
    register_collector("fake_working", WorkingCollector)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client,
        collector_names=[MalformedSampleCollector.name, "fake_working"],
        interval_seconds=60.0,
    )

    monitor._resolve_and_report()
    for _ in range(3):
        monitor._collect_once()
    monitor._collect_once()  # deactivated: must not be called again
    monitor.stop()

    malformed = next(
        entry
        for entry in monitor._entries
        if entry.name == MalformedSampleCollector.name
    )
    healthy = next(entry for entry in monitor._entries if entry.name == "fake_working")
    assert malformed.available is False
    assert malformed.collector.collect_calls == 3
    assert healthy.available is True
    assert len(_events_named(client, "rlinsight_fake_gauge")) == 4


def test_unavailable_collector_should_be_reported_but_never_sampled() -> None:
    register_collector("fake_unavailable", UnavailableCollector)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client, collector_names=["fake_unavailable"], interval_seconds=60.0
    )

    monitor._resolve_and_report()
    monitor._collect_once()
    monitor.stop()

    status = _events_named(client, _STATUS_METRIC)
    assert len(status) == 1
    assert status[0]["value"] == 0.0
    assert client.events == status  # no sample events at all


def test_unknown_collector_name_should_report_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client, collector_names=["ghost"], interval_seconds=60.0
    )

    with caplog.at_level(logging.INFO, logger="rl_insight.hardware.coordinator"):
        monitor._resolve_and_report()
    monitor.stop()

    status = _events_named(client, _STATUS_METRIC)
    assert len(status) == 1
    assert status[0]["value"] == 0.0
    assert status[0]["labels"]["collector"] == "ghost"
    assert any("not registered" in record.message for record in caplog.records)


def test_default_collector_names_should_follow_the_registry() -> None:
    register_collector("fake_working", WorkingCollector)
    monitor = HardwareMonitor(client=RecordingClient(), interval_seconds=60.0)

    monitor._resolve_and_report()
    monitor.stop()

    # Built-in system collectors are registered alongside the fake above.
    assert [entry.name for entry in monitor._entries] == sorted(COLLECTOR_REGISTRY)
    assert {"cpu", "memory", "network", "disk", "fake_working"} <= set(
        COLLECTOR_REGISTRY
    )


def test_collector_options_should_be_injected_before_discovery() -> None:
    class OptionsRecorder(HardwareCollector):
        """Fake collector that records the options it was configured with."""

        name = "fake_options"

        def __init__(self) -> None:
            self.options: dict[str, Any] | None = None

        def configure(self, options: Mapping[str, Any]) -> None:
            self.options = dict(options)

        def discover(self) -> bool:
            return True

        def collect(self) -> list[Sample]:
            return []

    register_collector("fake_options", OptionsRecorder)
    client = RecordingClient()
    monitor = HardwareMonitor(
        client=client,
        collector_names=["fake_options"],
        interval_seconds=60.0,
        collector_options={"fake_options": {"path": "/data"}},
    )

    monitor._resolve_and_report()
    monitor.stop()

    collector = monitor._entries[0].collector
    assert isinstance(collector, OptionsRecorder)
    assert collector.options == {"path": "/data"}


def test_non_positive_interval_should_fall_back_to_default() -> None:
    monitor = HardwareMonitor(client=RecordingClient(), interval_seconds=0)

    assert monitor.interval_seconds == MonitorHardware.INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# Batched emission and backpressure.
# ---------------------------------------------------------------------------


def test_one_round_should_be_submitted_as_a_single_batch() -> None:
    register_collector("fake_working", WorkingCollector)
    client = BatchRecordingClient()
    monitor = HardwareMonitor(
        client=client, collector_names=["fake_working"], interval_seconds=60.0
    )

    monitor._resolve_and_report()  # one batch: one status gauge
    monitor._collect_once()  # one batch: one sample
    monitor.stop()

    assert len(client.batches) == 2
    assert [event["name"] for event in client.batches[0]] == [_STATUS_METRIC]
    assert [event["name"] for event in client.batches[1]] == ["rlinsight_fake_gauge"]


def test_failed_batches_should_be_dropped_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_collector("fake_working", WorkingCollector)
    monitor = HardwareMonitor(
        client=FailingClient(), collector_names=["fake_working"], interval_seconds=60.0
    )

    with caplog.at_level(logging.WARNING, logger="rl_insight.hardware.coordinator"):
        monitor._resolve_and_report()  # status batch fails -> dropped
        monitor._collect_once()  # sample batch fails -> dropped
    monitor.stop()

    assert monitor.dropped_events == 2
    assert any("failed to forward" in record.message for record in caplog.records)
    assert any("dropping them" in record.message for record in caplog.records)
