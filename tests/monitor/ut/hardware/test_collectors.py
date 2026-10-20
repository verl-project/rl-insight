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
)
from rl_insight.utils.constants import MonitorHardware


_NetCounters = namedtuple("_NetCounters", ["bytes_recv", "bytes_sent"])


class _FakeCollector(HardwareCollector):
    """Minimal concrete collector used to exercise the registry."""

    name = "fake"

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        return []


# ---------------------------------------------------------------------------
# Collector registry (base.py).
# ---------------------------------------------------------------------------


def test_register_and_create_collector_round_trip() -> None:
    register_collector("fake", _FakeCollector)

    # Built-in system and vendor collectors are registered at import time.
    assert list_collector_names() == [
        "cpu",
        "disk",
        "fake",
        "gpu",
        "memory",
        "network",
        "npu",
        "ray_actor_process",
        "self_process",
    ]
    collector = create_collector("fake")
    assert isinstance(collector, _FakeCollector)
    assert collector.discover() is True
    assert collector.collect() == []


def test_create_collector_should_return_none_for_unknown_name() -> None:
    assert create_collector("ghost") is None


def test_register_collector_should_reject_duplicate_names() -> None:
    register_collector("fake", _FakeCollector)

    with pytest.raises(ValueError, match="already registered"):
        register_collector("fake", _FakeCollector)


# ---------------------------------------------------------------------------
# Counter-to-rate conversion (rate.py).
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic monotonic clock double."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_first_rate_call_should_return_none() -> None:
    calculator = RateCalculator(clock=FakeClock())

    assert calculator.rate(100.0) is None


def test_rate_should_divide_counter_delta_by_elapsed_time() -> None:
    clock = FakeClock()
    calculator = RateCalculator(clock=clock)

    calculator.rate(100.0)
    clock.advance(10.0)

    assert calculator.rate(250.0) == pytest.approx(15.0)


def test_zero_elapsed_time_should_return_none() -> None:
    clock = FakeClock()
    calculator = RateCalculator(clock=clock)

    calculator.rate(100.0)
    # No clock advance: the elapsed time is zero.
    assert calculator.rate(150.0) is None


def test_counter_reset_should_return_none_and_rebaseline() -> None:
    clock = FakeClock()
    calculator = RateCalculator(clock=clock)

    calculator.rate(100.0)
    clock.advance(5.0)
    # Counter went backwards (reset/wrap): no rate for this round.
    assert calculator.rate(40.0) is None

    clock.advance(5.0)
    # The baseline was refreshed, so rates resume from the new counter.
    assert calculator.rate(90.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Cgroup v1/v2 memory readers (cgroup.py).
# ---------------------------------------------------------------------------


def test_v2_limit_should_be_read_from_memory_max(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("2147483648\n", encoding="utf-8")

    assert read_cgroup_memory_limit(tmp_path) == 2147483648


def test_v2_max_should_mean_no_limit(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    assert read_cgroup_memory_limit(tmp_path) is None


def test_v1_limit_should_be_read_from_memory_limit_in_bytes(
    tmp_path: Path,
) -> None:
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("8589934592\n", encoding="utf-8")

    assert read_cgroup_memory_limit(tmp_path) == 8589934592


def test_v1_unlimited_sentinel_should_mean_no_limit(tmp_path: Path) -> None:
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n", encoding="utf-8")

    assert read_cgroup_memory_limit(tmp_path) is None


def test_missing_files_should_return_none(tmp_path: Path) -> None:
    assert read_cgroup_memory_limit(tmp_path) is None
    assert read_cgroup_memory_usage(tmp_path) is None


def test_v2_usage_should_be_read_from_memory_current(tmp_path: Path) -> None:
    (tmp_path / "memory.current").write_text("123456789\n", encoding="utf-8")

    assert read_cgroup_memory_usage(tmp_path) == 123456789


def test_v1_usage_should_be_read_from_memory_usage_in_bytes(
    tmp_path: Path,
) -> None:
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.usage_in_bytes").write_text("456\n", encoding="utf-8")

    assert read_cgroup_memory_usage(tmp_path) == 456


# ---------------------------------------------------------------------------
# Cgroup v1/v2 CPU throttling reader (cgroup.py).
# ---------------------------------------------------------------------------


def test_v2_throttling_should_be_read_from_cpu_stat(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat").write_text(
        "usage_usec 1000\n"
        "user_usec 500\n"
        "system_usec 500\n"
        "nr_periods 10\n"
        "nr_throttled 3\n"
        "throttled_usec 2000000\n",
        encoding="utf-8",
    )

    result = read_cgroup_cpu_throttling(tmp_path)

    # v2 throttling time is microseconds.
    assert result == CgroupCpuThrottling(throttled_periods=3, throttled_seconds=2.0)


def test_v1_throttling_should_be_read_from_cpu_dir(tmp_path: Path) -> None:
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.stat").write_text(
        "nr_periods 10\nnr_throttled 4\nthrottled_time 1500000000\n",
        encoding="utf-8",
    )

    result = read_cgroup_cpu_throttling(tmp_path)

    # v1 throttling time is nanoseconds.
    assert result == CgroupCpuThrottling(throttled_periods=4, throttled_seconds=1.5)


def test_missing_cpu_stat_should_return_none(tmp_path: Path) -> None:
    assert read_cgroup_cpu_throttling(tmp_path) is None


def test_cpu_stat_without_throttling_fields_should_return_none(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat").write_text("usage_usec 1000\n", encoding="utf-8")

    assert read_cgroup_cpu_throttling(tmp_path) is None


def test_cpu_stat_malformed_lines_should_be_ignored(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat").write_text(
        "nr_throttled 5\nnot a number pair here\nthrottled_usec nope\n",
        encoding="utf-8",
    )

    result = read_cgroup_cpu_throttling(tmp_path)

    assert result == CgroupCpuThrottling(throttled_periods=5, throttled_seconds=None)


def test_cpu_collector_should_report_cgroup_throttling_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_collectors_module,
        "read_cgroup_cpu_throttling",
        lambda: CgroupCpuThrottling(throttled_periods=7, throttled_seconds=1.25),
    )

    values = {sample.name: sample.value for sample in CpuCollector().collect()}

    assert values["rlinsight_cpu_cgroup_throttled_periods"] == 7.0
    assert values["rlinsight_cpu_cgroup_throttled_seconds"] == 1.25


# ---------------------------------------------------------------------------
# Built-in system collectors (system_collectors.py).
# ---------------------------------------------------------------------------


def test_builtin_collectors_should_register_under_config_names() -> None:
    assert COLLECTOR_REGISTRY["cpu"] is CpuCollector
    assert COLLECTOR_REGISTRY["memory"] is MemoryCollector
    assert COLLECTOR_REGISTRY["network"] is NetworkCollector
    assert COLLECTOR_REGISTRY["disk"] is DiskCollector
    assert COLLECTOR_REGISTRY["self_process"] is ProcessCollector
    assert COLLECTOR_REGISTRY["ray_actor_process"] is RayProcessCollector


def test_cpu_collector_should_report_host_level_samples_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_cpu_throttling", lambda: None
    )

    collector = CpuCollector()
    assert collector.discover() is True

    collector.collect()  # first round measures over the seeded window
    names = {sample.name for sample in collector.collect()}

    # Physical core count may be unavailable on some virtual machines.
    assert names <= {
        "rlinsight_cpu_utilization_percent",
        "rlinsight_cpu_logical_count",
        "rlinsight_cpu_physical_count",
    }
    assert "rlinsight_cpu_utilization_percent" in names
    assert "rlinsight_process_cpu_percent" not in names


def test_process_collector_should_report_process_samples() -> None:
    collector = ProcessCollector()
    assert collector.discover() is True

    collector.collect()  # first round measures over the seeded window
    samples = collector.collect()
    names = {sample.name for sample in samples}

    assert {
        "rlinsight_process_cpu_percent",
        "rlinsight_process_thread_count",
        "rlinsight_process_rss_bytes",
        "rlinsight_process_rss_percent",
    } <= names
    # The labels describe this process, not the node.
    for sample in samples:
        assert sample.labels["pid"] == str(os.getpid())
        # The key stays present with an empty value, so the family's label set
        # does not change when a role is configured.
        assert sample.labels["worker_role"] == ""


def test_process_collector_should_label_the_configured_worker_role() -> None:
    collector = ProcessCollector()
    collector.configure({"worker_role": "rollout"})

    for sample in collector.collect():
        assert sample.labels["worker_role"] == "rollout"


def test_worker_role_should_be_inferred_from_the_actor_name() -> None:
    infer = system_collectors_module._worker_role

    # Fused path: the class name carries the role set.
    assert infer("FusedWorker_actor_rollout_ref") == "actor_rollout_ref"
    assert infer("FusedWorker_actor_rollout_ref_critic") == "actor_rollout_ref_critic"
    # Non-colocated: one worker class per role.
    assert infer("CriticWorker") == "critic"
    assert infer("ActorRolloutRefWorker") == "actor_rollout_ref"
    # Unknown classes stay distinguishable instead of collapsing to "unknown".
    assert infer("MyCustomWorker") == "MyCustomWorker"
    # Without a class name the actor name is the last resort.
    assert infer("", "aB3xK9FusedWorker_actor_rollout_ref_0:0") == "actor_rollout_ref"
    assert infer("", "RLInsightHardware_run_node-a") == "unknown"


def test_worker_role_should_shorten_qualified_class_names() -> None:
    infer = system_collectors_module._worker_role

    # Ray reports classes defined inside a function fully qualified, which is
    # how verl's default colocated workers show up.
    assert infer("create_colocated_worker_cls.<locals>.WorkerDict") == "worker"
    assert infer("mod.CriticWorker") == "critic"
    assert (
        infer("verl.single_controller.ray.base.<locals>.FusedWorker_actor_rollout_ref")
        == "actor_rollout_ref"
    )


class _FakeState:
    """Stand-in for a Ray ``ActorState`` record."""

    def __init__(self, pid: int, class_name: str = "", name: str = "") -> None:
        self.pid = pid
        self.class_name = class_name
        self.name = name


class _FakeProcess:
    """Stand-in for ``psutil.Process``."""

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
        self.cpu_calls = 0

    def cpu_percent(self, interval: float | None = None) -> float:
        self.cpu_calls += 1
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
    """Install a stub ``ray`` module for one test."""
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
    """A collector already bound to a node, with the state API stubbed."""
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


def test_ray_process_collector_should_be_unavailable_without_a_ray_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_ray(monkeypatch, initialized=False)

    assert RayProcessCollector().discover() is False


def test_ray_process_collector_should_bind_to_the_current_node_and_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_ray(monkeypatch, initialized=True, node_id="node-a", job_id="job-42")

    collector = RayProcessCollector()

    assert collector.discover() is True
    assert collector._node_id == "node-a"
    assert collector._job_id == "job-42"


def test_ray_process_collector_should_be_unavailable_without_a_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ray_process_collector_should_report_one_process_per_alive_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _FakeState(101, "FusedWorker_actor_rollout_ref"),
        _FakeState(202, "CriticWorker"),
        # The monitor's own actor is covered by the self-process collector.
        _FakeState(os.getpid(), "NodeHardwareActor"),
    ]
    collector = _ray_process_collector(monkeypatch, states)

    first = collector.collect()
    # A newly seen pid is only seeded for CPU, so that round has no CPU value.
    by_pid = {(sample.labels["pid"], sample.name) for sample in first}
    assert (str(os.getpid()), "rlinsight_ray_process_cpu_percent") not in by_pid
    assert ("101", "rlinsight_ray_process_rss_bytes") in by_pid
    assert ("202", "rlinsight_ray_process_cpu_percent") not in by_pid
    # The Ray actor families are distinct from the self-process ones, so the
    # two collectors never share a series name.
    assert all(sample.name.startswith("rlinsight_ray_process_") for sample in first)
    roles = {sample.labels["pid"]: sample.labels["worker_role"] for sample in first}
    assert roles == {"101": "actor_rollout_ref", "202": "critic"}

    second = collector.collect()
    cpu = {
        sample.labels["pid"]: sample.value
        for sample in second
        if sample.name == "rlinsight_ray_process_cpu_percent"
    }
    assert cpu == {"101": 33.0, "202": 33.0}


def test_ray_process_collector_should_label_the_assigned_device_and_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``verl`` names worker-group members ``<prefix><Class>_<pg_idx>:<local_rank>``
    # and Ray exports the assigned card to the worker's own environment.
    states = [_FakeState(101, "WorkerDict", "aB3xK9WorkerDict_0:2")]
    processes = {101: _FakeProcess(101, environ={"ASCEND_RT_VISIBLE_DEVICES": "6"})}
    collector = _ray_process_collector(monkeypatch, states, processes)

    labels = {sample.labels["pid"]: sample.labels for sample in collector.collect()}

    assert labels["101"]["worker_role"] == "worker"
    assert labels["101"]["local_rank"] == "2"
    # The same physical id the NPU/GPU collectors publish, so the panels can be
    # filtered by card instead of by pid.
    assert labels["101"]["device_index"] == "6"


def test_ray_process_collector_should_report_empty_rank_and_device_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        # Not a worker-group member: nothing identifies a rank.
        _FakeState(101, "vLLMHttpServer"),
        # Holds several cards, so there is no single device index to report.
        _FakeState(202, "CriticWorker", "aB3xK9CriticWorker_0:1"),
    ]
    processes = {202: _FakeProcess(202, environ={"CUDA_VISIBLE_DEVICES": "0,1"})}
    collector = _ray_process_collector(monkeypatch, states, processes)

    labels = {sample.labels["pid"]: sample.labels for sample in collector.collect()}

    # Unknown values keep their label key with an empty value so every worker of
    # the family shares one label set (the hub registers a single gauge).
    assert labels["101"]["local_rank"] == ""
    assert labels["101"]["device_index"] == ""
    assert labels["202"]["local_rank"] == "1"
    assert labels["202"]["device_index"] == ""
    assert set(labels["101"]) == set(labels["202"])


def test_ray_process_collector_should_skip_rl_insights_own_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _FakeState(101, "rl_insight.collector.ray_monitor_hub.MonitorHubActor"),
        _FakeState(202, "NodeHardwareActor"),
        _FakeState(303, "CriticWorker"),
    ]
    collector = _ray_process_collector(monkeypatch, states)

    samples = collector.collect()

    # Reporting the monitor's own actors would present its footprint as a
    # worker's.
    assert {sample.labels["pid"] for sample in samples} == {"303"}


def test_ray_process_collector_should_forget_processes_that_went_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [_FakeState(101, "CriticWorker"), _FakeState(202, "CriticWorker")]
    collector = _ray_process_collector(monkeypatch, states)
    collector.collect()

    states.pop()
    samples = collector.collect()

    assert {sample.labels["pid"] for sample in samples} == {"101"}
    assert set(collector._processes) == {101}


def test_ray_process_collector_should_tolerate_transient_state_api_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [_FakeState(101, "CriticWorker")]
    collector = _ray_process_collector(monkeypatch, states)
    assert collector.collect()

    def _boom(_node_id: str, _job_id: str) -> list[_FakeState]:
        raise RuntimeError("state api down")

    monkeypatch.setattr(system_collectors_module, "_list_node_actor_states", _boom)
    # A few failed rounds keep the last known actor set instead of dropping
    # every series and flapping the panels.
    for _ in range(system_collectors_module._MAX_LIST_FAILURES - 1):
        samples = collector.collect()
        assert {sample.labels["pid"] for sample in samples} == {"101"}

    # A persistent failure is handed to the monitor's failure policy, which
    # marks the collector unavailable.
    with pytest.raises(RuntimeError, match="state api down"):
        collector.collect()


def test_memory_collector_should_report_host_level_samples_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_memory_limit", lambda: None
    )
    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_memory_usage", lambda: None
    )

    collector = MemoryCollector()
    assert collector.discover() is True

    samples = collector.collect()
    names = {sample.name for sample in samples}

    assert {
        "rlinsight_memory_used_bytes",
        "rlinsight_memory_available_bytes",
        "rlinsight_memory_utilization_percent",
    } <= names
    assert "rlinsight_process_rss_bytes" not in names


def test_memory_collector_should_report_cgroup_fields_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_memory_limit", lambda: 8589934592
    )
    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_memory_usage", lambda: 1048576
    )

    values = {sample.name: sample.value for sample in MemoryCollector().collect()}

    assert values["rlinsight_memory_cgroup_limit_bytes"] == 8589934592.0
    assert values["rlinsight_memory_cgroup_usage_bytes"] == 1048576.0


def test_memory_collector_should_report_total_minus_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # psutil's ``used`` (total - free - buffers - cached) drifts from both
    # psutil's own ``percent`` and the node-exporter fallback
    # (MemTotal - MemAvailable), so the collector reports total - available.
    class _VirtualMemory:
        total = 1000
        available = 400
        used = 700
        percent = 60.0

    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_memory_limit", lambda: None
    )
    monkeypatch.setattr(
        system_collectors_module, "read_cgroup_memory_usage", lambda: None
    )
    monkeypatch.setattr(
        system_collectors_module.psutil, "virtual_memory", lambda: _VirtualMemory()
    )

    values = {sample.name: sample.value for sample in MemoryCollector().collect()}

    assert values["rlinsight_memory_used_bytes"] == 600.0
    assert values["rlinsight_memory_available_bytes"] == 400.0


def test_network_collector_should_omit_first_round_then_report_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = {"eth0": _NetCounters(100, 200)}
    monkeypatch.setattr(
        system_collectors_module.psutil,
        "net_io_counters",
        lambda pernic=False: counters,
    )
    # A deterministic clock keeps the rate gate independent of the platform's
    # monotonic timer resolution (coarse on Windows, which used to yield a
    # zero elapsed time and drop the samples).
    clock = FakeClock()
    monkeypatch.setattr("time.monotonic", clock)

    instance = NetworkCollector()

    # Rates need a previous counter round to diff against.
    assert instance.collect() == []

    clock.advance(1.0)
    counters["eth0"] = _NetCounters(300, 500)

    samples = instance.collect()
    assert {sample.name for sample in samples} == {
        "rlinsight_network_receive_bytes_per_second",
        "rlinsight_network_transmit_bytes_per_second",
    }
    assert all(sample.value >= 0.0 for sample in samples)


def test_non_loopback_totals_should_skip_the_loopback_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Loopback traffic mirrors in-host IPC rather than the job's real network
    # load and is excluded by the node-exporter fallback too, so both sources
    # stay comparable.
    counters = {
        "lo": _NetCounters(1000, 2000),
        "eth0": _NetCounters(100, 200),
        "eth1": _NetCounters(10, 20),
    }
    monkeypatch.setattr(
        system_collectors_module.psutil,
        "net_io_counters",
        lambda pernic=False: counters,
    )

    assert system_collectors_module._non_loopback_totals() == (110, 220)


def test_non_loopback_totals_should_return_none_without_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_collectors_module.psutil, "net_io_counters", lambda pernic=False: {}
    )

    assert system_collectors_module._non_loopback_totals() is None


def test_disk_collector_should_report_utilization_immediately(
    tmp_path: Path,
) -> None:
    instance = DiskCollector()
    instance.configure({"path": str(tmp_path)})

    # Filesystem utilization is instantaneous, so it is present from the
    # first round even though the I/O rates need a previous counter round.
    samples = instance.collect()

    utilization = [
        sample
        for sample in samples
        if sample.name == "rlinsight_disk_utilization_percent"
    ]
    assert len(utilization) == 1
    assert utilization[0].labels["path"] == str(tmp_path)
    assert 0.0 <= utilization[0].value <= 100.0


def test_disk_collector_should_default_to_the_default_path() -> None:
    instance = DiskCollector()
    instance.configure({})

    samples = instance.collect()

    utilization = next(
        sample
        for sample in samples
        if sample.name == "rlinsight_disk_utilization_percent"
    )
    assert utilization.labels["path"] == MonitorHardware.DISK_PATH


def test_disk_collector_should_report_rates_after_the_first_round() -> None:
    instance = DiskCollector()
    if not instance.discover():
        pytest.skip("disk counters unavailable on this platform")

    instance.collect()  # seed the counter baselines
    names = {sample.name for sample in instance.collect()}

    assert names <= {
        "rlinsight_disk_utilization_percent",
        "rlinsight_disk_read_bytes_per_second",
        "rlinsight_disk_write_bytes_per_second",
    }
    assert "rlinsight_disk_utilization_percent" in names
    assert all(
        sample.value >= 0.0
        for sample in instance.collect()
        if sample.name.endswith("_per_second") or sample.name.endswith("_percent")
    )
