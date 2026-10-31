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

"""Unit tests for collector registration, rate helpers, and the
built-in psutil-based system collectors."""

from __future__ import annotations

import logging
import os
import sys
import types
from collections import namedtuple
from typing import Any
from unittest.mock import Mock

import pytest

import rl_insight.hardware.system_collectors as system_collectors_module
from rl_insight.hardware.base import (
    COLLECTOR_REGISTRY,
    HardwareCollector,
    Sample,
    register_collector,
)
from rl_insight.hardware.system_collectors import (
    CpuCollector,
    DiskCollector,
    MemoryCollector,
    NetworkCollector,
    ProcessCollector,
    RayProcessCollector,
    SystemCollector,
)
from rl_insight.hardware.vendor import DeviceCollector

_NetCounters = namedtuple("_NetCounters", ["bytes_recv", "bytes_sent"])


class _FakeCollector(HardwareCollector):
    name = "fake"

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        return []


def test_collector_registry_should_register_builtin_and_custom_collectors() -> None:
    # CPU, memory, network and disk share one host collector name; accelerators
    # share one auto-detecting device collector.
    assert COLLECTOR_REGISTRY["host"] is SystemCollector
    assert COLLECTOR_REGISTRY["device"] is DeviceCollector
    assert "cpu" not in COLLECTOR_REGISTRY
    assert "memory" not in COLLECTOR_REGISTRY
    assert "network" not in COLLECTOR_REGISTRY
    assert "disk" not in COLLECTOR_REGISTRY
    assert "npu" not in COLLECTOR_REGISTRY
    assert "gpu" not in COLLECTOR_REGISTRY
    assert COLLECTOR_REGISTRY["self_process"] is ProcessCollector
    assert COLLECTOR_REGISTRY["ray_actor_process"] is RayProcessCollector

    register_collector("fake", _FakeCollector)
    assert "fake" in COLLECTOR_REGISTRY
    assert isinstance(COLLECTOR_REGISTRY["fake"](), _FakeCollector)
    with pytest.raises(ValueError, match="already registered"):
        register_collector("fake", _FakeCollector)


def test_disk_failures_should_not_stop_cpu_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disk = Mock(side_effect=OSError("disk unavailable"))
    monkeypatch.setattr(DiskCollector, "collect", disk)
    collector = SystemCollector()
    for _ in range(4):
        names = {sample.name for sample in collector.collect()}
        assert "rlinsight_cpu_utilization_percent" in names
    assert disk.call_count == 3


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeState:
    def __init__(self, pid: int, class_name: str = "", name: str = "") -> None:
        self.pid = pid
        self.class_name = class_name
        self.name = name


class _FakeProcess:
    def __init__(
        self,
        pid: int,
        *,
        rss: int = 2048,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.pid = pid
        self._rss = rss
        self._environ = environ or {}

    def cpu_percent(self, interval: float | None = None) -> float:
        return 33.0

    def memory_info(self) -> Any:
        return types.SimpleNamespace(rss=self._rss)

    def environ(self) -> dict[str, str]:
        return self._environ


def _fake_ray(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialized: bool,
    node_id: str = "node-a",
    job_id: str = "job-1",
) -> None:
    module = types.ModuleType("ray")
    module.is_initialized = lambda: initialized  # type: ignore[attr-defined]
    module.get_runtime_context = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_node_id=lambda: node_id,
        get_job_id=lambda: job_id,
    )
    monkeypatch.setitem(sys.modules, "ray", module)


def _ray_process_collector(
    monkeypatch: pytest.MonkeyPatch,
    states: list[_FakeState],
    processes: dict[int, _FakeProcess] | None = None,
) -> RayProcessCollector:
    fakes = processes if processes is not None else {}

    def _process(pid: int) -> _FakeProcess:
        if pid not in fakes:
            fakes[pid] = _FakeProcess(pid)
        return fakes[pid]

    monkeypatch.setattr(
        system_collectors_module,
        "_list_node_actor_states",
        lambda _node_id, _job_id: states,
    )
    monkeypatch.setattr(system_collectors_module.psutil, "Process", _process)
    collector = RayProcessCollector()
    collector._node_id = "node-a"
    collector._job_id = "job-1"
    return collector


# ---------------------------------------------------------------------------
# Built-in system collectors.
# ---------------------------------------------------------------------------


def test_host_collectors_should_report_cpu_memory_process_and_disk_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _VirtualMemory:
        total = 1000
        available = 400
        used = 700
        percent = 60.0

    monkeypatch.setattr(
        system_collectors_module.psutil, "virtual_memory", lambda: _VirtualMemory()
    )

    # CPU exposes the utilization gauge from psutil.
    assert CpuCollector().discover() is True
    cpu = {sample.name: sample.value for sample in CpuCollector().collect()}
    assert "rlinsight_cpu_utilization_percent" in cpu

    # Memory reports total - available rather than psutil's own "used" value.
    memory = {sample.name: sample.value for sample in MemoryCollector().collect()}
    assert memory["rlinsight_memory_used_bytes"] == 600.0
    assert memory["rlinsight_memory_utilization_percent"] == 60.0

    # The self-process collector labels the current pid without a Ray role.
    process = ProcessCollector().collect()
    assert {
        "rlinsight_process_cpu_percent",
        "rlinsight_process_rss_bytes",
    } <= {sample.name for sample in process}
    for sample in process:
        assert sample.labels["pid"] == str(os.getpid())
        assert sample.labels["worker_role"] == ""

    # Disk reports utilization for the configured root path.
    disk = DiskCollector()
    disk.collect()  # seed the I/O rates
    utilization = next(
        sample
        for sample in disk.collect()
        if sample.name == "rlinsight_disk_utilization_percent"
    )
    assert utilization.labels["path"] == "/"

    # The umbrella system collector composes the host collectors.
    names = {sample.name for sample in SystemCollector().collect()}
    assert {
        "rlinsight_cpu_utilization_percent",
        "rlinsight_memory_used_bytes",
        "rlinsight_disk_utilization_percent",
    } <= names

    # Network rates need two rounds: the first seeds the counters.
    counters = {"lo": _NetCounters(1000, 2000), "eth0": _NetCounters(100, 200)}
    monkeypatch.setattr(
        system_collectors_module.psutil,
        "net_io_counters",
        lambda pernic=False: counters,
    )
    clock = FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    network = NetworkCollector()
    assert network.collect() == []
    clock.advance(1.0)
    counters["eth0"] = _NetCounters(300, 500)
    assert {sample.name for sample in network.collect()} == {
        "rlinsight_network_receive_bytes_per_second",
        "rlinsight_network_transmit_bytes_per_second",
    }


# ---------------------------------------------------------------------------
# Ray actor process collector.
# ---------------------------------------------------------------------------


def test_worker_role_should_be_inferred_from_the_actor_class() -> None:
    infer = system_collectors_module._worker_role

    assert infer("FusedWorker_actor_rollout_ref") == "actor_rollout_ref"
    assert infer("CriticWorker") == "critic"
    assert infer("create_colocated_worker_cls.<locals>.WorkerDict") == "worker"
    assert infer("MyCustomWorker") == "MyCustomWorker"
    assert infer("", "aB3xK9FusedWorker_actor_rollout_ref_0:0") == "actor_rollout_ref"


def test_ray_process_collector_should_require_a_job_and_scope_actor_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # collect() must scope the State API call to this node and job.
    captured: dict[str, Any] = {}
    state_module = types.ModuleType("ray.util.state")

    def _list_actors(**kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return []

    state_module.list_actors = _list_actors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray.util.state", state_module)
    collector = RayProcessCollector()
    collector._node_id = "node-a"
    collector._job_id = "job-42"

    assert collector.collect() == []
    assert captured["filters"] == [
        ("node_id", "=", "node-a"),
        ("job_id", "=", "job-42"),
        ("state", "=", "ALIVE"),
    ]

    # discover() stays unavailable outside an initialized job.
    _fake_ray(monkeypatch, initialized=False)
    assert RayProcessCollector().discover() is False
    _fake_ray(monkeypatch, initialized=True, job_id="")
    assert RayProcessCollector().discover() is False


def test_ray_process_collector_should_reuse_actors_then_fail_after_state_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    collector = _ray_process_collector(monkeypatch, [_FakeState(101, "CriticWorker")])
    assert {sample.labels["pid"] for sample in collector.collect()} == {"101"}

    def _fail(_node_id: str, _job_id: str) -> list[Any]:
        raise RuntimeError("state api unavailable")

    monkeypatch.setattr(system_collectors_module, "_list_node_actor_states", _fail)
    with caplog.at_level(
        logging.WARNING, logger="rl_insight.hardware.system_collectors"
    ):
        # Transient failures keep the previous actor set.
        assert {sample.labels["pid"] for sample in collector.collect()} == {"101"}
        assert {sample.labels["pid"] for sample in collector.collect()} == {"101"}
        # Persistent failures surface to the monitor's deactivation policy.
        with pytest.raises(RuntimeError):
            collector.collect()

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "Ray State API is unavailable" in warnings[0].message


def test_ray_process_collector_should_report_workers_with_device_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _FakeState(101, "WorkerDict", "aB3xK9WorkerDict_0:2"),
        _FakeState(os.getpid(), "NodeHardwareActor"),
    ]
    processes = {101: _FakeProcess(101, environ={"ASCEND_RT_VISIBLE_DEVICES": "6"})}
    collector = _ray_process_collector(monkeypatch, states, processes)

    first = collector.collect()
    assert all(sample.labels["pid"] == "101" for sample in first)
    labels = first[0].labels
    assert labels["worker_role"] == "worker"
    assert labels["local_rank"] == "2"
    assert labels["device_index"] == "6"
    # A newly seen pid is only seeded for CPU, so that round has no CPU value.
    assert "rlinsight_ray_process_cpu_percent" not in {sample.name for sample in first}

    second = collector.collect()
    cpu = {
        sample.labels["pid"]: sample.value
        for sample in second
        if sample.name == "rlinsight_ray_process_cpu_percent"
    }
    assert cpu == {"101": 33.0}
