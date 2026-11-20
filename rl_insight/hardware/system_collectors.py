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

"""Built-in CPU, memory, network, disk, and process collectors based on psutil."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, NamedTuple

import psutil

from ..utils.constants import MonitorRayActor
from .base import (
    HardwareCollector,
    Sample,
    current_ray_job_id,
    optional_sample,
    drop_missing_samples,
    register_collector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_MAX_COMPONENT_FAILURES = 3

__all__ = [
    "CpuCollector",
    "DiskCollector",
    "MemoryCollector",
    "NetworkCollector",
    "ProcessCollector",
    "RayProcessCollector",
    "SystemCollector",
]


class RateCalculator:
    """Convert one monotonically increasing counter into a per-second rate.

    The previous ``(timestamp, counter)`` pair is kept between calls. Counter
    resets/wraps and non-positive elapsed time return ``None``; the baseline is
    refreshed so the next round recovers.
    """

    def __init__(self) -> None:
        self._previous: tuple[float, float] | None = None

    def rate(self, counter: float) -> float | None:
        """Return the per-second rate since the last call, or ``None``."""
        now = time.monotonic()
        previous = self._previous
        self._previous = (now, float(counter))
        if previous is None:
            return None
        previous_time, previous_counter = previous
        elapsed = now - previous_time
        if elapsed <= 0.0:
            return None
        delta = float(counter) - previous_counter
        if delta < 0.0:
            return None
        return delta / elapsed


class CpuCollector(HardwareCollector):
    """Host CPU utilization and topology metrics.

    Building block of :class:`SystemCollector`; only the umbrella is registered.
    """

    def __init__(self) -> None:
        # Seed the interval-less CPU counter for the first collect.
        psutil.cpu_percent(interval=None)

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        return drop_missing_samples(
            [
                optional_sample(
                    "rlinsight_cpu_utilization_percent",
                    psutil.cpu_percent(interval=None),
                    documentation="System CPU utilization percent across all cores",
                ),
            ]
        )


class ProcessCollector(HardwareCollector):
    """CPU and memory metrics of the process hosting the monitor.

    Under Ray this is the ``NodeHardwareActor``, not a training worker. Enable
    with ``hardware.collectors=['self_process']``. Samples carry ``pid`` and an
    empty ``worker_role``.
    """

    name = "self_process"

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        # Keep the label keys identical across samples: the hub registers one
        # gauge per metric name, so mixing label sets would drop series.
        labels: dict[str, str] = {
            "pid": str(self._process.pid),
            "worker_role": "",
        }
        return drop_missing_samples(
            [
                optional_sample(
                    "rlinsight_process_cpu_percent",
                    self._process.cpu_percent(interval=None),
                    labels,
                    documentation=(
                        "CPU percent of the monitored process (can exceed 100)"
                    ),
                ),
                optional_sample(
                    "rlinsight_process_rss_bytes",
                    self._process.memory_info().rss,
                    labels,
                    documentation=(
                        "Resident set size of the monitored process in bytes"
                    ),
                ),
            ]
        )


# Colocated (default) ``verl`` workers are named ``FusedWorker_<roles>``.
_FUSED_MARK = "FusedWorker_"
_UNKNOWN_ROLE = "unknown"

# Non-colocated worker class -> role; unmapped classes report their own name.
_ACTOR_CLASS_ROLES = {
    "TrainingWorker": "trainer",
    "TinkerTrainingWorker": "trainer",
    "ActorRolloutRefWorker": "actor_rollout_ref",
    "TinkerActorRolloutRefWorker": "actor_rollout_ref",
    "DetachActorWorker": "actor",
    "CriticWorker": "critic",
    "RewardModelWorker": "rm",
    "CheckpointEngineWorker": "checkpoint_engine",
    # Colocated workers have no role set; distinguish them by pid.
    "WorkerDict": "worker",
}

# Consecutive state-API failures tolerated before the monitor deactivates the
# collector through its own failure policy.
_MAX_LIST_FAILURES = 3

# Ray exports the assigned devices of every accelerator worker through these
# variables, so a worker's own environment names the card it runs on.
_DEVICE_ENV_VARS = ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# ``verl`` names every worker-group member ``<prefix><Class>_<pg_idx>:<local_rank>``.
_ACTOR_RANK_PATTERN = re.compile(r"_(\d+):(\d+)$")


class _Actor(NamedTuple):
    """One alive Ray actor of the current node."""

    pid: int
    worker_role: str
    # Worker index inside its node; only worker-group actors have the suffix.
    local_rank: str | None = None


def _short_class_name(class_name: str) -> str:
    """Drop the scope qualifier Ray adds to classes defined inside a function."""
    return (class_name or "").strip().rsplit(".", 1)[-1].strip()


def _worker_role(class_name: str, actor_name: str = "") -> str:
    """Infer a worker role from the Ray actor class or actor name.

    ``FusedWorker_<roles>`` carries the roles in its class name; non-colocated
    workers are mapped through ``_ACTOR_CLASS_ROLES``, and unmapped classes keep
    their own name. ``WorkerDict`` has no role and is reported as ``worker``.
    The actor name is consulted only when the class name is empty.
    """
    text = _short_class_name(class_name)
    if _FUSED_MARK in text:
        return text.split(_FUSED_MARK, 1)[1] or _UNKNOWN_ROLE
    if text:
        return _ACTOR_CLASS_ROLES.get(text, text)
    actor = (actor_name or "").strip()
    if _FUSED_MARK in actor:
        # e.g. ``aB3xK9FusedWorker_actor_rollout_ref_0:0``
        tail = actor.split(_FUSED_MARK, 1)[1]
        return _ACTOR_RANK_PATTERN.sub("", tail) or _UNKNOWN_ROLE
    return _UNKNOWN_ROLE


def _actor_local_rank(actor_name: str) -> str | None:
    """Return the local rank encoded in a worker-group actor name.

    ``verl`` names members ``<prefix><Class>_<pg_idx>:<local_rank>``. The local
    rank is the worker's ordinal inside its node and matches its device ordinal.
    Actors created outside a worker group have no suffix.
    """
    match = _ACTOR_RANK_PATTERN.search((actor_name or "").strip())
    return match.group(2) if match else None


def _actor_device_index(process: psutil.Process) -> str | None:
    """Return the device assigned to a worker, read from its environment.

    Ray sets the accelerator visibility variable for each worker. Only a
    single-device value is used; a worker holding several cards has no single
    index. Returns ``None`` when the value is unavailable.
    """
    try:
        env = process.environ()
    except psutil.Error:
        return None
    for name in _DEVICE_ENV_VARS:
        raw = (env.get(name) or "").strip()
        if not raw or "," in raw:
            continue
        return raw
    return None


def _list_node_actor_states(node_id: str, job_id: str) -> list[Any]:
    """Return the node's alive actors for ``job_id`` through the Ray state API.

    ``list_actors`` is cluster-wide, so the query filters both node and job.
    """
    from ray.util.state import list_actors

    return list_actors(
        filters=[
            ("node_id", "=", node_id),
            ("job_id", "=", job_id),
            ("state", "=", "ALIVE"),
        ],
        limit=MonitorRayActor.STATE_API_ACTOR_LIMIT,
    )


class RayProcessCollector(HardwareCollector):
    """Per-actor CPU and memory metrics of the Ray workers on this node.

    Reads the alive actors of the current job on this node, infers each role
    from its class name, and samples the backing process with psutil. Samples
    carry ``pid``, ``worker_role``, ``local_rank`` and ``device_index``. Enable
    with ``hardware.collectors=['ray_actor_process']``; outside a Ray job the
    collector is unavailable. Transient state-API failures reuse the previous
    actor set.
    """

    name = "ray_actor_process"

    def __init__(self) -> None:
        # The monitor's own actor is in this list; skip it (self_process covers it).
        self._self_pid = os.getpid()
        self._node_id = ""
        self._job_id = ""
        self._processes: dict[int, psutil.Process] = {}
        self._devices: dict[int, str | None] = {}
        self._actors: list[_Actor] = []
        self._list_failures = 0

    def discover(self) -> bool:
        try:
            import ray
        except Exception:
            return False
        if not ray.is_initialized():
            return False
        try:
            self._node_id = str(ray.get_runtime_context().get_node_id() or "")
        except Exception:
            return False
        self._job_id = current_ray_job_id()
        return bool(self._node_id and self._job_id)

    def collect(self) -> list[Sample]:
        actors = self._alive_actors()
        if actors is None:
            actors = self._actors
        else:
            self._actors = actors
        samples: list[Sample] = []
        for actor in actors:
            samples.extend(self._collect_process(actor))
        self._forget_dead(actors)
        return samples

    def _alive_actors(self) -> list[_Actor] | None:
        """Return this job's alive actors on this node.

        Returns ``None`` on a transient query failure. Raises after
        ``_MAX_LIST_FAILURES`` consecutive failures so the monitor can
        deactivate the collector.
        """
        try:
            states = _list_node_actor_states(self._node_id, self._job_id)
        except Exception:
            self._list_failures += 1
            if self._list_failures >= _MAX_LIST_FAILURES:
                raise
            if self._list_failures == 1:
                logger.warning(
                    "[rl-insight] Ray State API is unavailable; the "
                    "ray_actor_process collector reuses the previous actor set "
                    "and will be deactivated if the failure persists. Install "
                    "'ray[default]' and keep the Ray dashboard enabled."
                )
            logger.debug(
                "[rl-insight] failed to list the Ray actors of this node.",
                exc_info=True,
            )
            return None
        self._list_failures = 0
        actors: list[_Actor] = []
        for state in states:
            pid = getattr(state, "pid", None)
            if not pid or int(pid) == self._self_pid:
                continue
            class_name = getattr(state, "class_name", "")
            if _short_class_name(class_name) in MonitorRayActor.INTERNAL_CLASSES:
                continue
            actors.append(
                _Actor(
                    pid=int(pid),
                    worker_role=_worker_role(class_name, getattr(state, "name", "")),
                    local_rank=_actor_local_rank(getattr(state, "name", "")),
                )
            )
        return actors

    def _collect_process(self, actor: _Actor) -> list[Sample]:
        """Samples of one worker process, or ``[]`` once it is gone."""
        pid = actor.pid
        process = self._processes.get(pid)
        fresh = process is None
        if process is None:
            try:
                process = psutil.Process(pid)
                # Seed the interval-less CPU counter; this round has no window.
                process.cpu_percent(interval=None)
            except psutil.Error:
                return []
            self._processes[pid] = process
        # Keep the label keys identical across workers: the hub registers one
        # gauge per metric name, so differing label sets would drop the series.
        # Unknown values are reported as empty strings.
        labels = {
            "pid": str(pid),
            "worker_role": actor.worker_role,
            "local_rank": actor.local_rank or "",
            "device_index": self._device_index(pid, process) or "",
        }
        try:
            cpu = None if fresh else process.cpu_percent(interval=None)
            rss = process.memory_info().rss
        except psutil.Error:
            return []
        return drop_missing_samples(
            [
                optional_sample(
                    "rlinsight_ray_process_cpu_percent",
                    cpu,
                    labels,
                    documentation=(
                        "CPU percent of the Ray worker process (can exceed 100)"
                    ),
                ),
                optional_sample(
                    "rlinsight_ray_process_rss_bytes",
                    rss,
                    labels,
                    documentation=(
                        "Resident set size of the Ray worker process in bytes"
                    ),
                ),
            ]
        )

    def _device_index(self, pid: int, process: psutil.Process) -> str | None:
        """Return a worker's device, resolved once and cached."""
        if pid not in self._devices:
            self._devices[pid] = _actor_device_index(process)
        return self._devices[pid]

    def _forget_dead(self, actors: list[_Actor]) -> None:
        """Drop cached psutil handles for actors that are no longer alive."""
        alive = {actor.pid for actor in actors}
        for pid in list(self._processes):
            if pid not in alive:
                del self._processes[pid]
                self._devices.pop(pid, None)


class MemoryCollector(HardwareCollector):
    """Host memory usage metrics."""

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        virtual = psutil.virtual_memory()
        # Use total - available rather than psutil's ``used``, which excludes
        # buffers/cache and drifts from ``percent``.
        used = virtual.total - virtual.available
        return drop_missing_samples(
            [
                optional_sample(
                    "rlinsight_memory_used_bytes",
                    used,
                    documentation="System memory used in bytes (total - available)",
                ),
                optional_sample(
                    "rlinsight_memory_utilization_percent",
                    virtual.percent,
                    documentation="System memory utilization percent",
                ),
            ]
        )


_LOOPBACK_INTERFACE = "lo"


def _non_loopback_totals() -> tuple[int, int] | None:
    """Sum receive/transmit bytes over non-loopback interfaces.

    ``psutil.net_io_counters(pernic=False)`` includes ``lo``, whose traffic
    mirrors in-host IPC and can dominate a single-node total.
    """
    counters = psutil.net_io_counters(pernic=True)
    if not counters:
        return None
    receive = 0
    transmit = 0
    for name, stats in counters.items():
        if name == _LOOPBACK_INTERFACE:
            continue
        receive += stats.bytes_recv
        transmit += stats.bytes_sent
    return receive, transmit


class NetworkCollector(HardwareCollector):
    """Host network throughput across non-loopback interfaces.

    Building block of :class:`SystemCollector`; only the umbrella is registered.
    """

    def __init__(self) -> None:
        self._receive_rate = RateCalculator()
        self._transmit_rate = RateCalculator()

    def discover(self) -> bool:
        return _non_loopback_totals() is not None

    def collect(self) -> list[Sample]:
        totals = _non_loopback_totals()
        if totals is None:
            return []
        receive, transmit = totals
        return drop_missing_samples(
            [
                optional_sample(
                    "rlinsight_network_receive_bytes_per_second",
                    self._receive_rate.rate(receive),
                    documentation=(
                        "Network receive bytes per second across all "
                        "non-loopback interfaces"
                    ),
                ),
                optional_sample(
                    "rlinsight_network_transmit_bytes_per_second",
                    self._transmit_rate.rate(transmit),
                    documentation=(
                        "Network transmit bytes per second across all "
                        "non-loopback interfaces"
                    ),
                ),
            ]
        )


class DiskCollector(HardwareCollector):
    """Host disk throughput and root-filesystem utilization.

    Building block of :class:`SystemCollector`; only the umbrella is registered.
    """

    _PATH = "/"

    def __init__(self) -> None:
        self._read_rate = RateCalculator()
        self._write_rate = RateCalculator()

    def discover(self) -> bool:
        return psutil.disk_io_counters() is not None or self._disk_usage() is not None

    def _disk_usage(self) -> Any | None:
        """Return ``psutil.disk_usage`` for the root path, or ``None``."""
        try:
            return psutil.disk_usage(self._PATH)
        except OSError:
            return None

    def collect(self) -> list[Sample]:
        usage = self._disk_usage()
        counters = psutil.disk_io_counters()
        samples: list[Sample | None] = [
            optional_sample(
                "rlinsight_disk_utilization_percent",
                None if usage is None else usage.percent,
                {"path": self._PATH},
                documentation=(
                    f"Filesystem utilization percent for the root path {self._PATH}"
                ),
            )
        ]
        if counters is not None:
            samples.extend(
                [
                    optional_sample(
                        "rlinsight_disk_read_bytes_per_second",
                        self._read_rate.rate(counters.read_bytes),
                        documentation="Disk read bytes per second across all devices",
                    ),
                    optional_sample(
                        "rlinsight_disk_write_bytes_per_second",
                        self._write_rate.rate(counters.write_bytes),
                        documentation="Disk write bytes per second across all devices",
                    ),
                ]
            )
        return drop_missing_samples(samples)


class SystemCollector(HardwareCollector):
    """Host CPU, memory, network, and disk metrics from psutil.

    All four families share the collector name ``host``; the component
    collectors keep their own sampling code.
    """

    name = "host"

    def __init__(self) -> None:
        self._collectors = (
            CpuCollector(),
            MemoryCollector(),
            NetworkCollector(),
            DiskCollector(),
        )
        self._component_failures: dict[str, int] = {}
        self._disabled_components: set[str] = set()

    def discover(self) -> bool:
        return any(collector.discover() for collector in self._collectors)

    def collect(self) -> list[Sample]:
        samples: list[Sample] = []
        for collector in self._collectors:
            name = type(collector).__name__
            if name in self._disabled_components:
                continue
            try:
                samples.extend(collector.collect())
            except Exception:
                failures = self._component_failures.get(name, 0) + 1
                self._component_failures[name] = failures
                if failures >= _MAX_COMPONENT_FAILURES:
                    self._disabled_components.add(name)
                    logger.warning(
                        "[rl-insight] host collector component %r failed %d "
                        "times; disabling it for this run.",
                        name,
                        failures,
                        exc_info=True,
                    )
                elif failures == 1:
                    logger.warning(
                        "[rl-insight] host collector component %r failed; "
                        "it will be disabled after %d consecutive failures.",
                        name,
                        _MAX_COMPONENT_FAILURES,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "[rl-insight] host collector component %r failed %d "
                        "times in a row.",
                        name,
                        failures,
                        exc_info=True,
                    )
            else:
                self._component_failures.pop(name, None)
        return samples


register_collector(SystemCollector.name, SystemCollector)
register_collector(ProcessCollector.name, ProcessCollector)
register_collector(RayProcessCollector.name, RayProcessCollector)
