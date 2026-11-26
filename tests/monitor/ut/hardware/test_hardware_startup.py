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

import logging
from collections.abc import Generator
from contextlib import nullcontext
from typing import Any

import pytest
from fakes import RecordingClient, WorkingCollector, wait_for

import rl_insight.hardware as hardware_module
from rl_insight import api
from rl_insight.hardware import start_hardware_monitor
from rl_insight.hardware.base import register_collector
from rl_insight.hardware.coordinator import HardwareMonitor
from rl_insight.utils.constants import MonitorHardware
from rl_insight.utils.monitor_config_loader import load_monitor_config


def _conf(**hardware: Any) -> Any:
    section = {
        "enable": True,
        "interval_seconds": MonitorHardware.INTERVAL_SECONDS,
        "collectors": ["fake_working"],
    }
    section.update(hardware)
    return load_monitor_config(
        {"server": {"url": "http://monitor:18080"}, "hardware": section}
    )


def test_start_should_honour_disabled_and_missing_client() -> None:
    assert start_hardware_monitor(_conf(enable=False), RecordingClient()) is None
    assert start_hardware_monitor(_conf(), None) is None


def test_start_failure_should_log_and_return_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(hardware_module, "_start_hardware_monitor", explode)
    with caplog.at_level(logging.ERROR, logger="rl_insight.hardware"):
        assert start_hardware_monitor(_conf(), RecordingClient()) is None

    assert any(
        "failed to start hardware monitoring" in record.message
        for record in caplog.records
    )
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize(
    ("hardware", "expected_names", "expects_warning"),
    [
        ({"interval_seconds": 1}, ["fake_working"], False),
        ({"collectors": "fake_working"}, ["fake_working"], True),
        ({"collectors": None}, list(MonitorHardware.COLLECTORS), False),
    ],
    ids=["custom-interval", "scalar-collector", "default-collectors"],
)
def test_start_should_normalize_configuration(
    hardware: dict[str, Any],
    expected_names: list[str],
    expects_warning: bool,
) -> None:
    expectation = pytest.warns(RuntimeWarning) if expects_warning else nullcontext()
    with expectation:
        monitor = start_hardware_monitor(_conf(**hardware), RecordingClient())

    assert isinstance(monitor, HardwareMonitor)
    if "interval_seconds" in hardware:
        assert monitor.interval_seconds == hardware["interval_seconds"]
    assert monitor._collector_names == expected_names
    monitor.stop()


@pytest.mark.parametrize("interval", [0, -1, "not-a-number"])
def test_start_should_fall_back_to_default_for_invalid_interval(
    interval: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="rl_insight.hardware"):
        monitor = start_hardware_monitor(
            _conf(interval_seconds=interval), RecordingClient()
        )

    assert isinstance(monitor, HardwareMonitor)
    assert monitor.interval_seconds == MonitorHardware.INTERVAL_SECONDS
    assert any("interval_seconds" in record.message for record in caplog.records)
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
            "hardware": {"enable": True, "collectors": ["fake_working"]},
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

    # Disabling hardware in the config must not create a monitor at all.
    _init_with_client(
        monkeypatch,
        RecordingClient(),
        {
            "server": {"url": "http://monitor:18080"},
            "hardware": {"enable": False},
        },
    )
    assert api._STATE.hardware_monitor is None
