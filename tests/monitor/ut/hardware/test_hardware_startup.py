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

"""Unit tests for hardware monitor startup: the config entry point and the
``rl_insight.init``/``finish`` lifecycle (in-process mode, Ray stubbed out)."""

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
    """Build a merged monitor config with a custom ``hardware`` section."""
    section = {
        "enabled": True,
        "interval_seconds": MonitorHardware.INTERVAL_SECONDS,
        "collectors": ["fake_working"],
    }
    section.update(hardware)
    return load_monitor_config(
        {"server": {"url": "http://monitor:18080"}, "hardware": section}
    )


# ---------------------------------------------------------------------------
# ``start_hardware_monitor`` config entry point.
# ---------------------------------------------------------------------------


def test_start_should_return_none_when_disabled_or_without_client() -> None:
    assert start_hardware_monitor(_conf(enabled=False), RecordingClient()) is None
    assert start_hardware_monitor(_conf(), None) is None


def test_start_should_clamp_interval_below_the_minimum() -> None:
    conf = _conf(interval_seconds=1)

    with pytest.warns(RuntimeWarning, match="interval_seconds"):
        monitor = start_hardware_monitor(conf, RecordingClient())

    assert isinstance(monitor, HardwareMonitor)
    assert monitor.interval_seconds == MonitorHardware.MIN_INTERVAL_SECONDS
    monitor.stop()


def test_start_should_read_a_scalar_collector_name_as_one_name() -> None:
    # ``collectors: cpu`` is a common typo for ``collectors: [cpu]``; iterating
    # the scalar would yield the pseudo names c/p/u and run nothing.
    conf = _conf(collectors="fake_working")

    with pytest.warns(RuntimeWarning, match="collectors"):
        monitor = start_hardware_monitor(conf, RecordingClient())

    assert isinstance(monitor, HardwareMonitor)
    assert monitor._collector_names == ["fake_working"]
    monitor.stop()


def test_start_should_ignore_a_collectors_value_that_is_not_a_name_list() -> None:
    # ``OmegaConf`` replaces the default list with any scalar, e.g. a number
    # typed by mistake; it must not be iterated into pseudo names either.
    conf = _conf(collectors=3)

    with pytest.warns(RuntimeWarning, match="collectors"):
        monitor = start_hardware_monitor(conf, RecordingClient())

    assert isinstance(monitor, HardwareMonitor)
    # ``None`` resolves to every registered collector instead of iterating a
    # non-list value.
    assert monitor._collector_names is None
    monitor.stop()


# ---------------------------------------------------------------------------
# ``rl_insight.init``/``finish`` lifecycle.
# ---------------------------------------------------------------------------


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


def test_init_should_start_hardware_monitor_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    _init_with_client(monkeypatch, client, {"server": {"url": "http://monitor:18080"}})

    monitor = api._STATE.hardware_monitor
    assert isinstance(monitor, HardwareMonitor)
    assert monitor.is_running is True

    # Every default collector name gets one status gauge: the built-in
    # system collectors report available, vendor names report unavailable.
    wait_for(
        lambda: len(
            [
                event
                for event in client.events
                if event["name"] == MonitorHardware.STATUS_METRIC_NAME
            ]
        )
        >= len(MonitorHardware.COLLECTORS),
        "expected one status gauge per default collector",
    )


def test_init_should_skip_hardware_monitor_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    _init_with_client(
        monkeypatch,
        client,
        {
            "server": {"url": "http://monitor:18080"},
            "hardware": {"enabled": False},
        },
    )

    assert api._STATE.hardware_monitor is None


def test_finish_should_stop_the_hardware_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    _init_with_client(monkeypatch, client, {"server": {"url": "http://monitor:18080"}})
    monitor = api._STATE.hardware_monitor
    assert isinstance(monitor, HardwareMonitor)

    api.finish()

    assert monitor.is_running is False
    assert api._STATE.hardware_monitor is None


def test_hardware_events_should_carry_init_labels_and_node_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_collector("fake_working", WorkingCollector)
    client = RecordingClient()
    _init_with_client(
        monkeypatch,
        client,
        {
            "server": {"url": "http://monitor:18080"},
            "hardware": {"collectors": ["fake_working"]},
        },
    )

    wait_for(
        lambda: any(event["name"] == "rlinsight_fake_gauge" for event in client.events),
        "expected a fake sample event through the hardware path",
    )

    event = next(
        event for event in client.events if event["name"] == "rlinsight_fake_gauge"
    )
    assert event["labels"]["project"] == "project-a"
    assert event["labels"]["experiment_name"] == "experiment-a"
    assert event["labels"]["node"]
