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

"""Unit tests for hardware monitor startup and the ``init``/``finish``
lifecycle outside Ray."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from fakes import RecordingClient, WorkingCollector, wait_for

from rl_insight import api
from rl_insight.hardware import start_hardware_monitor
from rl_insight.hardware.base import register_collector
from rl_insight.hardware.coordinator import HardwareMonitor
from rl_insight.utils.constants import MonitorHardware
from rl_insight.utils.monitor_config_loader import load_monitor_config


def _conf(**hardware: Any) -> Any:
    section = {
        "enabled": True,
        "interval_seconds": MonitorHardware.INTERVAL_SECONDS,
        "collectors": ["fake_working"],
    }
    section.update(hardware)
    return load_monitor_config(
        {"server": {"url": "http://monitor:18080"}, "hardware": section}
    )


def test_start_should_honour_disabled_and_missing_client() -> None:
    assert start_hardware_monitor(_conf(enabled=False), RecordingClient()) is None
    assert start_hardware_monitor(_conf(), None) is None


@pytest.mark.parametrize(
    ("hardware", "expected_names"),
    [
        ({"interval_seconds": 1}, ["fake_working"]),
        ({"collectors": "fake_working"}, ["fake_working"]),
        ({"collectors": 3}, None),
    ],
    ids=["clamped-interval", "scalar-collector", "invalid-collector-type"],
)
def test_start_should_validate_configuration(
    hardware: dict[str, Any], expected_names: list[str] | None
) -> None:
    with pytest.warns(RuntimeWarning):
        monitor = start_hardware_monitor(_conf(**hardware), RecordingClient())

    assert isinstance(monitor, HardwareMonitor)
    if "interval_seconds" in hardware:
        assert monitor.interval_seconds == MonitorHardware.MIN_INTERVAL_SECONDS
    assert monitor._collector_names == expected_names
    monitor.stop()


@pytest.fixture(autouse=True)
def reset_monitor_state() -> Generator[None, None, None]:
    api.finish()
    yield
    api.finish()


def _init_with_client(
    monkeypatch: pytest.MonkeyPatch, client: RecordingClient, config: dict[str, Any]
) -> None:
    monkeypatch.setattr(api, "create_monitor_client", lambda _conf: client)
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    api.init(project="project-a", experiment_name="experiment-a", config=config)


def test_init_should_start_and_finish_should_stop_hardware_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_collector("fake_working", WorkingCollector)
    client = RecordingClient()
    _init_with_client(
        monkeypatch,
        client,
        {
            "server": {"url": "http://monitor:18080"},
            "hardware": {"enabled": True, "collectors": ["fake_working"]},
        },
    )

    monitor = api._STATE.hardware_monitor
    assert isinstance(monitor, HardwareMonitor)
    assert monitor.is_running is True
    wait_for(
        lambda: any(event["name"] == "rlinsight_fake_gauge" for event in client.events),
        "expected a hardware sample",
    )
    event = next(
        event for event in client.events if event["name"] == "rlinsight_fake_gauge"
    )
    assert event["labels"]["project"] == "project-a"
    assert event["labels"]["experiment_name"] == "experiment-a"
    assert event["labels"]["node"]

    api.finish()
    assert monitor.is_running is False
    assert api._STATE.hardware_monitor is None


def test_init_should_skip_hardware_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_with_client(
        monkeypatch,
        RecordingClient(),
        {
            "server": {"url": "http://monitor:18080"},
            "hardware": {"enabled": False},
        },
    )

    assert api._STATE.hardware_monitor is None
