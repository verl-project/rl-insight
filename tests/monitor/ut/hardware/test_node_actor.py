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

"""Unit tests for per-node hardware actor startup and lifecycle."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from fakes import RecordingClient

import rl_insight.hardware as hardware_module
import rl_insight.hardware.node_actor as node_actor_module
from rl_insight.hardware import start_hardware_monitor
from rl_insight.hardware.coordinator import HardwareMonitor
from rl_insight.hardware.node_actor import NodeActorSet, NodeHardwareActor
from rl_insight.utils.constants import MonitorHardware

NodeHardwareImplementation = cast(
    Any, NodeHardwareActor
).__ray_metadata__.modified_class


def _conf(**hardware: Any) -> Any:
    """Build a merged monitor config with hardware enabled."""
    section = {
        "enabled": True,
        "interval_seconds": MonitorHardware.INTERVAL_SECONDS,
        "collectors": ["cpu"],
    }
    section.update(hardware)
    return OmegaConf.create(
        {"server": {"url": "http://monitor:18080"}, "hardware": section}
    )


@pytest.fixture(autouse=True)
def ray_stubs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Default stubs: Ray down, so tests opt into node-actor mode explicitly."""
    monkeypatch.setattr(hardware_module, "_ray_is_initialized", lambda: False)
    monkeypatch.setattr(hardware_module, "_alive_node_ids", lambda: [])
    factory = MagicMock()
    monkeypatch.setattr(hardware_module, "_create_node_actor", factory)
    return factory


@pytest.fixture(autouse=True)
def clear_worker_role_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the host environment from leaking a worker_role label into tests."""
    monkeypatch.delenv("RL_INSIGHT_WORKER_ROLE", raising=False)


def _enable_ray(monkeypatch: pytest.MonkeyPatch, node_ids: list[str]) -> None:
    """Stub a live Ray cluster with the given alive nodes."""
    monkeypatch.setattr(hardware_module, "_ray_is_initialized", lambda: True)
    monkeypatch.setattr(hardware_module, "_alive_node_ids", lambda: node_ids)


def test_start_should_create_one_actor_per_alive_node(
    ray_stubs: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_ray(monkeypatch, ["node-a", "node-b"])
    handles = [MagicMock(), MagicMock()]
    ray_stubs.side_effect = lambda node_id, _conf, _labels, _params: handles[
        0 if node_id == "node-a" else 1
    ]

    monitor = start_hardware_monitor(_conf(), RecordingClient(), {"project": "p"})

    assert isinstance(monitor, NodeActorSet)
    assert monitor.node_count == 2
    called_node_ids = [call.args[0] for call in ray_stubs.call_args_list]
    assert called_node_ids == ["node-a", "node-b"]
    # Init-level labels flow to the actors; node/run_id are detected on them.
    for call in ray_stubs.call_args_list:
        assert call.args[2] == {"project": "p"}


def test_start_should_route_worker_role_to_the_self_process_collector(
    ray_stubs: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_ray(monkeypatch, ["node-a"])
    ray_stubs.side_effect = lambda *_args: MagicMock()

    start_hardware_monitor(
        _conf(worker_role="rollout"), RecordingClient(), {"project": "p"}
    )

    call = ray_stubs.call_args_list[0]
    # The role describes the reporting process, not the node, so it must not
    # leak into the node-level labels shared by every sample.
    assert call.args[2] == {"project": "p"}
    assert call.args[3]["collector_options"]["self_process"] == {
        "worker_role": "rollout"
    }


def test_start_should_let_worker_role_env_override_the_config(
    ray_stubs: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RL_INSIGHT_WORKER_ROLE", "ref")
    _enable_ray(monkeypatch, ["node-a"])
    ray_stubs.side_effect = lambda *_args: MagicMock()

    start_hardware_monitor(
        _conf(worker_role="rollout"), RecordingClient(), {"project": "p"}
    )

    options = ray_stubs.call_args_list[0].args[3]["collector_options"]
    assert options["self_process"]["worker_role"] == "ref"


def test_start_should_omit_worker_role_when_unset() -> None:
    monitor = start_hardware_monitor(_conf(), RecordingClient(), {"project": "p"})

    assert isinstance(monitor, HardwareMonitor)
    assert "worker_role" not in monitor._labels
    assert "self_process" not in monitor._collector_options
    monitor.stop()


@pytest.mark.parametrize(
    ("ray_up", "alive_nodes"),
    [(False, []), (True, [])],
    ids=["ray-down", "ray-up-without-nodes"],
)
def test_start_should_fall_back_to_in_process_monitor_without_node_actors(
    ray_stubs: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    ray_up: bool,
    alive_nodes: list[str],
) -> None:
    if ray_up:
        _enable_ray(monkeypatch, alive_nodes)

    monitor = start_hardware_monitor(_conf(), RecordingClient(), {"project": "p"})

    assert isinstance(monitor, HardwareMonitor)
    assert monitor.is_running is True
    ray_stubs.assert_not_called()
    monitor.stop()


def test_failed_node_actor_should_only_skip_that_node(
    ray_stubs: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_ray(monkeypatch, ["node-a", "node-b", "node-c"])
    good = MagicMock()
    ray_stubs.side_effect = [RuntimeError("node gone"), good, RuntimeError("boom")]

    monitor = start_hardware_monitor(_conf(), RecordingClient())

    assert isinstance(monitor, NodeActorSet)
    assert monitor.node_count == 1


def test_node_actor_set_stop_should_kill_every_actor_and_be_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill = MagicMock()
    monkeypatch.setattr(node_actor_module.ray, "kill", kill)
    handles = [MagicMock(), MagicMock()]
    monitor = NodeActorSet(handles)

    assert monitor.is_running is True
    monitor.stop()

    for handle in handles:
        kill.assert_any_call(handle)
    assert monitor.is_running is False
    assert monitor.node_count == 0
    monitor.stop()  # idempotent


def test_node_actor_init_should_start_a_monitor_with_node_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf = _conf()
    seen: list[Any] = []
    client = MagicMock()
    monkeypatch.setattr(
        "rl_insight.client.create_monitor_client",
        lambda passed: seen.append(passed) or client,
    )
    monkeypatch.setattr(
        node_actor_module,
        "identity_labels",
        lambda: {"node": "10.0.0.8", "run_id": "job-42"},
    )
    monitor_cls = MagicMock()
    monkeypatch.setattr(node_actor_module, "HardwareMonitor", monitor_cls)

    instance = NodeHardwareImplementation.__new__(NodeHardwareImplementation)
    instance.__init__(
        conf,
        {"project": "p"},
        {
            "interval_seconds": 7.0,
            "collector_names": ["cpu"],
        },
    )

    assert seen == [conf]
    monitor_cls.assert_called_once()
    kwargs = monitor_cls.call_args.kwargs
    assert kwargs["client"] is client
    assert kwargs["labels"] == {"project": "p", "node": "10.0.0.8", "run_id": "job-42"}
    assert kwargs["interval_seconds"] == 7.0
    assert kwargs["collector_names"] == ["cpu"]
    monitor_cls.return_value.start.assert_called_once()


def test_node_actor_should_skip_monitoring_without_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rl_insight.client.create_monitor_client", lambda _conf: None)
    monitor_cls = MagicMock()
    monkeypatch.setattr(node_actor_module, "HardwareMonitor", monitor_cls)

    instance = NodeHardwareImplementation.__new__(NodeHardwareImplementation)
    instance.__init__(
        _conf(),
        {},
        {
            "interval_seconds": 5.0,
            "collector_names": ["cpu"],
        },
    )

    monitor_cls.assert_not_called()
