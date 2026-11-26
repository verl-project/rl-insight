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

import logging
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
from rl_insight.hardware.base import HardwareSettings
from rl_insight.hardware.coordinator import HardwareMonitor
from rl_insight.hardware.node_actor import NodeActorSet, NodeHardwareActor
from rl_insight.utils.constants import MonitorHardware

NodeHardwareImplementation = cast(
    Any, NodeHardwareActor
).__ray_metadata__.modified_class


def _conf(**hardware: Any) -> Any:
    section = {
        "enable": True,
        "interval_seconds": MonitorHardware.INTERVAL_SECONDS,
        "collectors": ["host"],
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


def test_start_should_create_actors_or_fall_back_to_local(
    ray_stubs: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ray is not initialized: one in-process monitor for this node.
    monitor = start_hardware_monitor(_conf(), RecordingClient(), {"project": "p"})
    assert isinstance(monitor, HardwareMonitor)
    assert monitor.is_running is True
    monitor.stop()
    ray_stubs.assert_not_called()

    # Ray is up but the job has no alive nodes: same local fallback.
    _enable_ray(monkeypatch, [])
    monitor = start_hardware_monitor(_conf(), RecordingClient(), {"project": "p"})
    assert isinstance(monitor, HardwareMonitor)
    monitor.stop()
    ray_stubs.assert_not_called()

    # Ray is up with alive nodes: one actor per job node.
    _enable_ray(monkeypatch, ["node-a", "node-b"])
    handles = [MagicMock(), MagicMock()]
    ray_stubs.side_effect = lambda node_id, _conf, _labels, _params: handles[
        0 if node_id == "node-a" else 1
    ]

    monitor = start_hardware_monitor(_conf(), RecordingClient(), {"project": "p"})

    assert isinstance(monitor, NodeActorSet)
    assert monitor._reconcile_interval_seconds == 30.0
    assert monitor.node_ids == ["node-a", "node-b"]
    assert [call.args[0] for call in ray_stubs.call_args_list] == ["node-a", "node-b"]
    for call in ray_stubs.call_args_list:
        assert call.args[2] == {"project": "p"}
    monitor.stop()


def test_start_should_tolerate_actor_creation_failures(
    ray_stubs: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Individual failures only skip that node.
    _enable_ray(monkeypatch, ["node-a", "node-b", "node-c"])
    ray_stubs.side_effect = [
        RuntimeError("node gone"),
        MagicMock(),
        RuntimeError("boom"),
    ]

    monitor = start_hardware_monitor(_conf(), RecordingClient())

    assert isinstance(monitor, NodeActorSet)
    assert monitor.node_ids == ["node-b"]
    monitor.stop()

    # If every initial creation fails, the set still reconciles the node later.
    _enable_ray(monkeypatch, ["node-a"])
    ray_stubs.side_effect = [RuntimeError("unavailable"), _FakeHandle("recovered")]
    monkeypatch.setattr(node_actor_module.ray, "wait", lambda refs, **_: ([], refs))

    monitor = start_hardware_monitor(_conf(), RecordingClient())

    assert isinstance(monitor, NodeActorSet)
    monitor._reconcile_once()
    assert monitor.node_ids == ["node-a"]
    monitor.stop()


def _install_ray_state(monkeypatch: pytest.MonkeyPatch, list_actors: Any) -> None:
    state_module = types.ModuleType("ray.util.state")
    state_module.list_actors = list_actors  # type: ignore[attr-defined]
    ray_module = types.ModuleType("ray")
    ray_module.is_initialized = lambda: True  # type: ignore[attr-defined]
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


def test_job_node_ids_should_scope_to_current_job_and_degrade_on_state_api_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _fail(**_kwargs: Any) -> list[Any]:
        raise RuntimeError("state api unavailable")

    _install_ray_state(monkeypatch, _fail)
    with caplog.at_level(logging.WARNING, logger="rl_insight.hardware"):
        assert hardware_module._job_node_ids() is None

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "Ray State API is unavailable" in warnings[0].message

    _install_ray_state(
        monkeypatch,
        lambda **_kwargs: [
            types.SimpleNamespace(node_id="node-b", class_name="CriticWorker"),
            types.SimpleNamespace(node_id="node-c", class_name="NodeHardwareActor"),
        ],
    )
    assert hardware_module._job_node_ids() == ["node-a", "node-b"]


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
    monkeypatch.setattr(hardware_module, "current_ray_job_id", lambda: "job-1")

    handle = hardware_module._create_node_actor(
        "node-a",
        _conf(),
        {"project": "p"},
        HardwareSettings(interval_seconds=10.0, collector_names=["host"]),
    )

    assert handle == "handle"
    assert captured["options"]["num_cpus"] == 0
    assert captured["options"]["max_restarts"] >= 1
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


def test_node_actor_set_should_reconcile_missing_and_failed_pending_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    killed: list[_FakeHandle] = []
    probed: list[str] = []

    monkeypatch.setattr(
        node_actor_module.ray,
        "wait",
        lambda refs, timeout=None: (list(refs), []),
    )

    def fake_get(ref: str, timeout: float | None = None) -> None:
        _ = timeout
        probed.append(ref)
        if ref == "ready-old":
            raise RuntimeError("actor exited")

    monkeypatch.setattr(node_actor_module.ray, "get", fake_get)
    monkeypatch.setattr(
        node_actor_module.ray, "kill", lambda handle: killed.append(handle)
    )

    old = _FakeHandle("old")

    def create_actor(node_id: str) -> _FakeHandle:
        created.append(node_id)
        return _FakeHandle(f"new-{node_id}")

    monitor = NodeActorSet(
        {"node-a": old},
        create_actor=create_actor,
        node_ids=["node-a", "node-b"],
        reconcile_interval_seconds=3600.0,
    )

    # node-b is missing from the initial set; node-a fails its readiness probe.
    monitor._reconcile_once()
    assert created == ["node-b"]
    assert [handle.name for handle in killed] == ["old"]
    assert monitor.node_ids == ["node-b"]
    assert probed == ["ready-old", "ready-new-node-b"]

    # node-a is recreated; the already-ready node-b is not probed again.
    monitor._reconcile_once()
    assert created == ["node-b", "node-a"]
    assert monitor.node_ids == ["node-a", "node-b"]
    assert probed == ["ready-old", "ready-new-node-b", "ready-new-node-a"]

    # With no pending actors, reconciliation performs no readiness RPCs.
    probed.clear()
    monitor._reconcile_once()
    assert probed == []
    assert monitor.node_ids == ["node-a", "node-b"]

    # stop() only drops local references; actors are reclaimed with their
    # creating process.
    monitor.stop()
    assert [handle.name for handle in killed] == ["old"]
    assert monitor.node_ids == []


def test_reconcile_should_log_persistent_creation_failures_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_node_id: str) -> _FakeHandle:
        raise RuntimeError("node gone")

    monitor = NodeActorSet(
        {},
        create_actor=fail,
        node_ids=["node-a"],
        reconcile_interval_seconds=3600.0,
    )

    with caplog.at_level(logging.WARNING, logger="rl_insight.hardware.node_actor"):
        monitor._reconcile_once()
        monitor._reconcile_once()
    monitor.stop()

    failures = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "failed to create the hardware actor" in record.message
    ]
    assert len(failures) == 1


def test_node_actor_should_publish_fixed_labels_and_skip_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every node actor publishes the same label keys, with or without init labels."""
    conf = _conf()
    seen: list[Any] = []
    client = MagicMock()

    def make_client(passed: Any) -> Any:
        seen.append(passed)
        return client

    monkeypatch.setattr("rl_insight.client.create_monitor_client", make_client)
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
        HardwareSettings(interval_seconds=7.0, collector_names=["host"]),
    )

    kwargs = monitor_cls.call_args.kwargs
    assert kwargs["client"] is client
    assert kwargs["labels"] == {
        "project": "p",
        "experiment_name": "",
        "node": "10.0.0.8",
        "run_id": "job-42",
    }
    assert kwargs["interval_seconds"] == 7.0

    instance = NodeHardwareImplementation.__new__(NodeHardwareImplementation)
    instance.__init__(
        conf,
        {},
        HardwareSettings(interval_seconds=5.0, collector_names=["host"]),
    )

    assert monitor_cls.call_args.kwargs["labels"] == {
        "project": "",
        "experiment_name": "",
        "node": "10.0.0.8",
        "run_id": "job-42",
    }
    assert seen == [conf, conf]
    assert monitor_cls.return_value.start.call_count == 2

    # Without a client the actor must not start a monitor.
    monkeypatch.setattr("rl_insight.client.create_monitor_client", lambda _conf: None)
    monitor_cls.reset_mock()
    instance = NodeHardwareImplementation.__new__(NodeHardwareImplementation)
    instance.__init__(
        conf,
        {},
        HardwareSettings(interval_seconds=5.0, collector_names=["host"]),
    )
    monitor_cls.assert_not_called()
