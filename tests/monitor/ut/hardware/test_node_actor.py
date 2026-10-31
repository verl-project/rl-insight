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

import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fakes import RecordingClient
from omegaconf import OmegaConf

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
    section = {
        "enabled": True,
        "interval_seconds": MonitorHardware.INTERVAL_SECONDS,
        "collectors": ["cpu"],
    }
    section.update(hardware)
    return OmegaConf.create(
        {"server": {"url": "http://monitor:18080"}, "hardware": section}
    )


@pytest.fixture()
def ray_stubs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setattr(hardware_module, "_ray_is_initialized", lambda: False)
    monkeypatch.setattr(hardware_module, "_job_node_ids", list)
    monkeypatch.setattr(node_actor_module.ray, "kill", MagicMock())
    factory = MagicMock()
    monkeypatch.setattr(hardware_module, "_create_node_actor", factory)
    return factory


def _enable_ray(monkeypatch: pytest.MonkeyPatch, node_ids: list[str]) -> None:
    monkeypatch.setattr(hardware_module, "_ray_is_initialized", lambda: True)
    monkeypatch.setattr(hardware_module, "_job_node_ids", lambda: node_ids)


def test_start_should_create_one_actor_per_job_node(
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
    assert monitor.node_ids == ["node-a", "node-b"]
    assert [call.args[0] for call in ray_stubs.call_args_list] == ["node-a", "node-b"]
    for call in ray_stubs.call_args_list:
        assert call.args[2] == {"project": "p"}
    monitor.stop()


@pytest.mark.parametrize(
    ("ray_up", "alive_nodes"),
    [(False, []), (True, [])],
    ids=["ray-down", "ray-up-without-nodes"],
)
def test_start_should_fall_back_to_a_local_monitor_without_node_actors(
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
    ray_stubs.side_effect = [
        RuntimeError("node gone"),
        MagicMock(),
        RuntimeError("boom"),
    ]

    monitor = start_hardware_monitor(_conf(), RecordingClient())

    assert isinstance(monitor, NodeActorSet)
    assert monitor.node_count == 1
    monitor.stop()


def test_node_actor_set_stop_should_not_kill_shared_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill = MagicMock()
    monkeypatch.setattr(node_actor_module.ray, "kill", kill)
    handles = [MagicMock(), MagicMock()]
    monitor = NodeActorSet(handles)

    monitor.stop()

    kill.assert_not_called()
    assert monitor.is_running is False
    assert monitor.node_count == 0


def test_job_node_ids_should_only_include_current_job_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_module = types.ModuleType("ray.util.state")
    state_module.list_actors = lambda **_kwargs: [  # type: ignore[attr-defined]
        types.SimpleNamespace(node_id="node-b", class_name="CriticWorker"),
        types.SimpleNamespace(node_id="node-c", class_name="NodeHardwareActor"),
    ]
    ray_module = types.ModuleType("ray")
    ray_module.get_runtime_context = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_job_id=lambda: "job-1",
        get_node_id=lambda: "node-a",
    )
    ray_module.nodes = lambda: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("must not use the cluster-wide node list")
    )
    util_module = types.ModuleType("ray.util")
    util_module.state = state_module  # type: ignore[attr-defined]
    ray_module.util = util_module  # type: ignore[attr-defined]

    for name, module in (
        ("ray", ray_module),
        ("ray.util", util_module),
        ("ray.util.state", state_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    assert hardware_module._job_node_ids() == ["node-a", "node-b"]


def test_job_node_ids_should_return_none_when_state_api_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_module = types.ModuleType("ray.util.state")

    def _fail(**_kwargs: Any) -> list[Any]:
        raise RuntimeError("state api unavailable")

    state_module.list_actors = _fail  # type: ignore[attr-defined]
    ray_module = types.ModuleType("ray")
    ray_module.get_runtime_context = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_job_id=lambda: "job-1",
        get_node_id=lambda: "node-a",
    )
    util_module = types.ModuleType("ray.util")
    util_module.state = state_module  # type: ignore[attr-defined]
    ray_module.util = util_module  # type: ignore[attr-defined]

    for name, module in (
        ("ray", ray_module),
        ("ray.util", util_module),
        ("ray.util.state", state_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    assert hardware_module._job_node_ids() is None


def test_create_node_actor_should_not_reserve_cpu_and_should_be_restartable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeActor:
        @classmethod
        def options(cls, **kwargs: Any) -> FakeActor:
            captured["options"] = kwargs
            return cls()

        def remote(self, *args: Any) -> str:
            captured["args"] = args
            return "handle"

    class FakeStrategy:
        def __init__(self, **kwargs: Any) -> None:
            captured["strategy"] = kwargs

    scheduling_module = types.ModuleType("ray.util.scheduling_strategies")
    scheduling_module.NodeAffinitySchedulingStrategy = FakeStrategy  # type: ignore[attr-defined]
    util_module = types.ModuleType("ray.util")
    util_module.scheduling_strategies = scheduling_module  # type: ignore[attr-defined]
    ray_module = types.ModuleType("ray")
    ray_module.util = util_module  # type: ignore[attr-defined]
    for name, module in (
        ("ray", ray_module),
        ("ray.util", util_module),
        ("ray.util.scheduling_strategies", scheduling_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(node_actor_module, "NodeHardwareActor", FakeActor)
    monkeypatch.setattr(hardware_module, "detect_run_id_label", lambda: "job-1")

    handle = hardware_module._create_node_actor(
        "node-a",
        _conf(),
        {"project": "p"},
        {"interval_seconds": 10.0, "collector_names": ["cpu"]},
    )

    assert handle == "handle"
    assert captured["options"]["num_cpus"] == 0
    assert captured["options"]["max_restarts"] >= 1
    assert captured["options"]["lifetime"] == "detached"
    assert captured["options"]["name"] == "RLInsightHardware_job-1_node-a"
    assert captured["strategy"] == {"node_id": "node-a", "soft": False}


class _FakeReadyMethod:
    def __init__(self, name: str) -> None:
        self._name = name

    def remote(self) -> str:
        return f"ready-{self._name}"


class _FakeHandle:
    def __init__(self, name: str) -> None:
        self.name = name
        self.__ray_ready__ = _FakeReadyMethod(name)


def test_node_actor_set_should_reconcile_job_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    killed: list[_FakeHandle] = []
    monkeypatch.setattr(
        node_actor_module.ray,
        "wait",
        lambda refs, timeout=None: (list(refs), []),
    )
    monkeypatch.setattr(node_actor_module.ray, "get", lambda refs, timeout=None: None)
    monkeypatch.setattr(
        node_actor_module.ray, "kill", lambda handle: killed.append(handle)
    )

    desired = ["node-a"]
    handles = {"node-a": _FakeHandle("node-a")}

    def create_actor(node_id: str) -> _FakeHandle:
        created.append(node_id)
        return _FakeHandle(node_id)

    monitor = NodeActorSet(
        handles,
        create_actor=create_actor,
        node_ids_provider=lambda: desired,
        reconcile_interval_seconds=3600.0,
    )

    monitor._reconcile_once()
    assert monitor.node_ids == ["node-a"]

    desired.append("node-b")
    monitor._reconcile_once()
    assert monitor.node_ids == ["node-a", "node-b"]
    assert created == ["node-b"]

    desired.remove("node-a")
    monitor._reconcile_once()
    assert monitor.node_ids == ["node-b"]
    assert [handle.name for handle in killed] == ["node-a"]

    monitor.stop()
    assert monitor.is_running is False
    assert [handle.name for handle in killed] == ["node-a"]


def test_node_actor_set_should_recreate_an_actor_that_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    killed: list[_FakeHandle] = []
    failed = {"old": True}

    monkeypatch.setattr(
        node_actor_module.ray,
        "wait",
        lambda refs, timeout=None: (list(refs), []),
    )

    def fake_get(ref: str, timeout: float | None = None) -> None:
        _ = timeout
        if ref == "ready-old" and failed["old"]:
            failed["old"] = False
            raise RuntimeError("actor exited")

    monkeypatch.setattr(node_actor_module.ray, "get", fake_get)
    monkeypatch.setattr(
        node_actor_module.ray, "kill", lambda handle: killed.append(handle)
    )

    old = _FakeHandle("old")
    handles = {"node-a": old}

    def create_actor(node_id: str) -> _FakeHandle:
        created.append(node_id)
        return _FakeHandle("new")

    monitor = NodeActorSet(
        handles,
        create_actor=create_actor,
        node_ids_provider=lambda: ["node-a"],
        reconcile_interval_seconds=3600.0,
    )

    monitor._reconcile_once()
    assert created == []
    assert [handle.name for handle in killed] == ["old"]

    monitor._reconcile_once()
    assert created == ["node-a"]
    assert monitor.node_ids == ["node-a"]

    monitor.stop()


def test_node_actor_should_start_a_monitor_with_node_labels(
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
        {"interval_seconds": 7.0, "collector_names": ["cpu"]},
    )

    assert seen == [conf]
    kwargs = monitor_cls.call_args.kwargs
    assert kwargs["client"] is client
    assert kwargs["labels"] == {
        "project": "p",
        "experiment_name": "",
        "node": "10.0.0.8",
        "run_id": "job-42",
    }
    assert kwargs["interval_seconds"] == 7.0
    monitor_cls.return_value.start.assert_called_once()


def test_node_actor_should_keep_fixed_label_keys_without_init_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every node actor must publish the same label keys, even without init labels."""
    client = MagicMock()
    monkeypatch.setattr(
        "rl_insight.client.create_monitor_client", lambda _conf: client
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
        _conf(), {}, {"interval_seconds": 5.0, "collector_names": ["cpu"]}
    )

    assert monitor_cls.call_args.kwargs["labels"] == {
        "project": "",
        "experiment_name": "",
        "node": "10.0.0.8",
        "run_id": "job-42",
    }


def test_node_actor_watchdog_should_exit_when_the_job_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exited: list[bool] = []
    state_module = types.ModuleType("ray.util.state")
    state_module.get_job = lambda job_id: types.SimpleNamespace(  # type: ignore[attr-defined]
        status="JobStatus.SUCCEEDED"
    )
    monkeypatch.setitem(sys.modules, "ray.util.state", state_module)
    monkeypatch.setattr(node_actor_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        node_actor_module.ray,
        "get_runtime_context",
        lambda: types.SimpleNamespace(get_job_id=lambda: "job-1"),
    )
    monkeypatch.setattr(
        node_actor_module.ray.actor, "exit_actor", lambda: exited.append(True)
    )

    instance = NodeHardwareImplementation.__new__(NodeHardwareImplementation)
    instance._job_watchdog()

    assert exited == [True]


def test_node_actor_should_skip_monitoring_without_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rl_insight.client.create_monitor_client", lambda _conf: None)
    monitor_cls = MagicMock()
    monkeypatch.setattr(node_actor_module, "HardwareMonitor", monitor_cls)

    instance = NodeHardwareImplementation.__new__(NodeHardwareImplementation)
    instance.__init__(
        _conf(), {}, {"interval_seconds": 5.0, "collector_names": ["cpu"]}
    )

    monitor_cls.assert_not_called()
