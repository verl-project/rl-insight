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

"""Unit tests for collector registration, rate/cgroup helpers, and the
built-in psutil-based system collectors."""

from __future__ import annotations

import os
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

import rl_insight.hardware.system_collectors as system_collectors_module
from rl_insight.hardware.base import (
    COLLECTOR_REGISTRY,
    HardwareCollector,
    Sample,
    create_collector,
    list_collector_names,
    register_collector,
)
from rl_insight.hardware.cgroup import (
    CgroupCpuThrottling,
    read_cgroup_cpu_throttling,
    read_cgroup_memory_limit,
    read_cgroup_memory_usage,
)
from rl_insight.hardware.rate import RateCalculator
from rl_insight.hardware.system_collectors import (
    CpuCollector,
    DiskCollector,
    MemoryCollector,
    NetworkCollector,
    ProcessCollector,
    RayProcessCollector,
    SystemCollector,
)

_NetCounters = namedtuple("_NetCounters", ["bytes_recv", "bytes_sent"])


class _FakeCollector(HardwareCollector):
    name = "fake"

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        return []


def test_collector_registry_should_register_and_create_collectors() -> None:
    register_collector("fake", _FakeCollector)

    assert "fake" in list_collector_names()
    assert isinstance(create_collector("fake"), _FakeCollector)
    assert create_collector("ghost") is None
    with pytest.raises(ValueError, match="already registered"):
        register_collector("fake", _FakeCollector)


def test_builtin_collectors_should_register_under_config_names() -> None:
    # CPU, memory, network and disk share one collector name.
    assert COLLECTOR_REGISTRY["cpu"] is SystemCollector
    assert "memory" not in COLLECTOR_REGISTRY
    assert "network" not in COLLECTOR_REGISTRY
    assert "disk" not in COLLECTOR_REGISTRY
    assert COLLECTOR_REGISTRY["self_process"] is ProcessCollector
    assert COLLECTOR_REGISTRY["ray_actor_process"] is RayProcessCollector


def test_system_collector_should_report_cpu_memory_and_disk() -> None:
    samples = SystemCollector().collect()
    names = {sample.name for sample in samples}

    assert {
        "rlinsight_cpu_utilization_percent",
        "rlinsight_memory_used_bytes",
        "rlinsight_disk_utilization_percent",
    } <= names


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
        threads: int = 4,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.pid = pid
        self._rss = rss
        self._threads = threads
        self._environ = environ or {}

    def cpu_percent(self, interval: float | None = None) -> float:
        return 33.0

    def memory_info(self) -> Any:
        return types.SimpleNamespace(rss=self._rss)

    def num_threads(self) -> int:
        return self._threads

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
# Rate and cgroup helpers.
# ---------------------------------------------------------------------------


def test_rate_should_handle_baseline_reset_and_zero_elapsed_time() -> None:
    clock = FakeClock()
    calculator = RateCalculator(clock=clock)

    assert calculator.rate(100.0) is None
    assert calculator.rate(150.0) is None  # zero elapsed time
    clock.advance(10.0)
    assert calculator.rate(250.0) == pytest.approx(10.0)
    clock.advance(5.0)
    assert calculator.rate(40.0) is None  # counter reset
    clock.advance(5.0)
    assert calculator.rate(90.0) == pytest.approx(10.0)


def test_cgroup_v2_files_should_be_parsed(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("2147483648\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("123456789\n", encoding="utf-8")
    (tmp_path / "cpu.stat").write_text(
        "nr_periods 10\nnr_throttled 3\nthrottled_usec 2000000\n",
        encoding="utf-8",
    )

    assert read_cgroup_memory_limit(tmp_path) == 2147483648
    assert read_cgroup_memory_usage(tmp_path) == 123456789
    assert read_cgroup_cpu_throttling(tmp_path) == CgroupCpuThrottling(
        throttled_periods=3, throttled_seconds=2.0
    )


def test_cgroup_v1_files_should_be_parsed(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory.limit_in_bytes").write_text("8589934592\n", encoding="utf-8")
    (memory / "memory.usage_in_bytes").write_text("456\n", encoding="utf-8")
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "cpu.stat").write_text(
        "nr_periods 10\nnr_throttled 4\nthrottled_time 1500000000\n",
        encoding="utf-8",
    )

    assert read_cgroup_memory_limit(tmp_path) == 8589934592
    assert read_cgroup_memory_usage(tmp_path) == 456
    assert read_cgroup_cpu_throttling(tmp_path) == CgroupCpuThrottling(
        throttled_periods=4, throttled_seconds=1.5
    )


def test_missing_or_unlimited_cgroup_files_should_return_none(
    tmp_path: Path,
) -> None:
    assert read_cgroup_memory_limit(tmp_path) is None
    assert read_cgroup_memory_usage(tmp_path) is None
    assert read_cgroup_cpu_throttling(tmp_path) is None

    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    assert read_cgroup_memory_limit(tmp_path) is None


# ---------------------------------------------------------------------------
# Built-in system collectors.
# ---------------------------------------------------------------------------


def test_cpu_collector_should_report_host_and_cgroup_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_collectors_module,
        "read_cgroup_cpu_throttling",
        lambda: CgroupCpuThrottling(throttled_periods=7, throttled_seconds=1.25),
    )

    assert CpuCollector().discover() is True
    values = {sample.name: sample.value for sample in CpuCollector().collect()}

    assert values["rlinsight_cpu_cgroup_throttled_periods"] == 7.0
    assert values["rlinsight_cpu_cgroup_throttled_seconds"] == 1.25
    assert "rlinsight_process_cpu_percent" not in values


def test_memory_collector_should_report_total_minus_available(
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

    values = {sample.name: sample.value for sample in MemoryCollector().collect()}

    assert values["rlinsight_memory_used_bytes"] == 600.0
    assert values["rlinsight_memory_available_bytes"] == 400.0
    assert values["rlinsight_memory_utilization_percent"] == 60.0


def test_process_collector_should_label_pid_without_a_worker_role() -> None:
    samples = ProcessCollector().collect()

    assert {
        "rlinsight_process_cpu_percent",
        "rlinsight_process_thread_count",
        "rlinsight_process_rss_bytes",
        "rlinsight_process_rss_percent",
    } <= {sample.name for sample in samples}
    for sample in samples:
        assert sample.labels["pid"] == str(os.getpid())
        assert sample.labels["worker_role"] == ""


def test_network_collector_should_report_non_loopback_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = {"lo": _NetCounters(1000, 2000), "eth0": _NetCounters(100, 200)}
    monkeypatch.setattr(
        system_collectors_module.psutil,
        "net_io_counters",
        lambda pernic=False: counters,
    )
    clock = FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    collector = NetworkCollector()

    assert collector.collect() == []  # first round seeds the counters
    clock.advance(1.0)
    counters["eth0"] = _NetCounters(300, 500)
    samples = collector.collect()

    assert {sample.name for sample in samples} == {
        "rlinsight_network_receive_bytes_per_second",
        "rlinsight_network_transmit_bytes_per_second",
    }


def test_disk_collector_should_report_utilization_for_the_root_path() -> None:
    collector = DiskCollector()
    collector.collect()  # seed the I/O rates

    samples = collector.collect()
    utilization = next(
        sample
        for sample in samples
        if sample.name == "rlinsight_disk_utilization_percent"
    )

    assert utilization.labels["path"] == "/"


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


def test_ray_process_collector_should_be_unavailable_without_a_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_ray(monkeypatch, initialized=False)
    assert RayProcessCollector().discover() is False

    _fake_ray(monkeypatch, initialized=True, job_id="")
    assert RayProcessCollector().discover() is False


def test_ray_process_collector_should_query_only_the_current_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ray_process_collector_should_tolerate_transient_state_api_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _ray_process_collector(monkeypatch, [_FakeState(101, "CriticWorker")])
    assert collector.collect()

    def _boom(_node_id: str, _job_id: str) -> list[_FakeState]:
        raise RuntimeError("state api down")

    monkeypatch.setattr(system_collectors_module, "_list_node_actor_states", _boom)
    samples = collector.collect()
    assert {sample.labels["pid"] for sample in samples} == {"101"}
