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
from typing import Any, Mapping, NamedTuple

import psutil

from ..utils.constants import MonitorHardware
from .base import (
    HardwareCollector,
    Sample,
    optional_sample,
    present_samples,
    register_collector,
)
from .cgroup import (
    read_cgroup_cpu_throttling,
    read_cgroup_memory_limit,
    read_cgroup_memory_usage,
)
from .rate import RateCalculator

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = [
    "CpuCollector",
    "DiskCollector",
    "MemoryCollector",
    "NetworkCollector",
    "ProcessCollector",
    "RayProcessCollector",
]


class CpuCollector(HardwareCollector):
    """Host-level CPU utilization and topology metrics plus cgroup throttling."""

    name = "cpu"

    def __init__(self) -> None:
        # Seed the interval-less CPU percent counter so the first collect
        # already reports a value measured over a real window.
        psutil.cpu_percent(interval=None)
        # Topology is fixed for the process lifetime; querying it is
        # expensive on many-core hosts, so cache it once.
        self._logical_count = psutil.cpu_count(logical=True)
        self._physical_count = psutil.cpu_count(logical=False)

    def discover(self) -> bool:
        # psutil is a hard dependency; CPU counters always exist.
        return True

    def collect(self) -> list[Sample]:
        throttling = read_cgroup_cpu_throttling()
        return present_samples(
            [
                optional_sample(
                    "rlinsight_cpu_utilization_percent",
                    psutil.cpu_percent(interval=None),
                    documentation="System CPU utilization percent across all cores",
                ),
                optional_sample(
                    "rlinsight_cpu_logical_count",
                    self._logical_count,
                    documentation="Logical CPU count",
                ),
                optional_sample(
                    "rlinsight_cpu_physical_count",
                    self._physical_count,
                    documentation="Physical CPU core count",
                ),
                optional_sample(
                    "rlinsight_cpu_cgroup_throttled_periods",
                    None if throttling is None else throttling.throttled_periods,
                    documentation=(
                        "Container cgroup CPU throttled scheduling periods "
                        "since start (cumulative)"
                    ),
                ),
                optional_sample(
                    "rlinsight_cpu_cgroup_throttled_seconds",
                    None if throttling is None else throttling.throttled_seconds,
                    documentation=(
                        "Container cgroup CPU throttled time in seconds "
                        "since start (cumulative)"
                    ),
                ),
            ]
        )


class ProcessCollector(HardwareCollector):
    """CPU and memory metrics of the process hosting the monitor itself.

    This is the monitor's *own* process, not a training worker: under Ray the
    monitor runs inside a ``NodeHardwareActor``, so what it measures is that
    actor. Use :class:`RayProcessCollector` for the per-worker view. Enable it
    explicitly through ``hardware.collectors=['self_process']`` when worker
    processes call ``rl_insight.init`` themselves. Samples carry the process
    ``pid`` and, when ``hardware.worker_role`` is configured, its
    ``worker_role``: both are process-level facts, unlike the ``node`` label
    which describes the host.
    """

    name = "self_process"

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)
        self._worker_role = ""

    def configure(self, options: Mapping[str, Any]) -> None:
        """Apply the configured role of this process (``hardware.worker_role``)."""
        self._worker_role = str(options.get("worker_role") or "").strip()

    def discover(self) -> bool:
        # psutil is a hard dependency; the current process always exists.
        return True

    def collect(self) -> list[Sample]:
        # Keep the label-name set fixed across processes even when the role is
        # unset: the hub registers one gauge per metric name, so a mix of
        # samples with and without ``worker_role`` would drop series.
        labels: dict[str, str] = {
            "pid": str(self._process.pid),
            "worker_role": self._worker_role,
        }
        return present_samples(
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
                    "rlinsight_process_thread_count",
                    self._process.num_threads(),
                    labels,
                    documentation="Thread count of the monitored process",
                ),
                optional_sample(
                    "rlinsight_process_rss_bytes",
                    self._process.memory_info().rss,
                    labels,
                    documentation=(
                        "Resident set size of the monitored process in bytes"
                    ),
                ),
                optional_sample(
                    "rlinsight_process_rss_percent",
                    self._process.memory_percent(),
                    labels,
                    documentation="Process RSS as percent of system memory",
                ),
            ]
        )


# Colocated (default) ``verl`` workers are named ``FusedWorker_<roles>``.
_FUSED_MARK = "FusedWorker_"
_UNKNOWN_ROLE = "unknown"

# Fallback mapping for the non-colocated path, where one worker class serves
# one role; classes outside this table report their own class name so an
# unmapped worker is still distinguishable.
_ACTOR_CLASS_ROLES = {
    "TrainingWorker": "trainer",
    "TinkerTrainingWorker": "trainer",
    "ActorRolloutRefWorker": "actor_rollout_ref",
    "TinkerActorRolloutRefWorker": "actor_rollout_ref",
    "DetachActorWorker": "actor",
    "CriticWorker": "critic",
    "RewardModelWorker": "rm",
    "CheckpointEngineWorker": "checkpoint_engine",
    # verl's colocated path fuses several roles into one dynamically created
    # class, so Ray exposes no role set for those actors: they are reported
    # generically and told apart by pid.
    "WorkerDict": "worker",
}

# Upper bound on the actors inspected per round, so one ``collect`` stays
# cheap on clusters with a very large actor count.
_MAX_ACTORS = 10000

# Consecutive state-API failures tolerated before the error is handed to the
# monitor, which applies its own failure policy and marks the collector
# unavailable instead of sampling it with no result forever.
_MAX_LIST_FAILURES = 3

# rl-insight's own actors run next to the training workers; reporting them
# would present the monitor's footprint as if it were a worker's.
_OWN_ACTOR_CLASSES = frozenset({"MonitorHubActor", "NodeHardwareActor"})

# Ray exports the assigned devices of every accelerator worker through these
# variables, so a worker's own environment names the card it runs on.
_DEVICE_ENV_VARS = ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

# ``verl`` names every worker-group member ``<prefix><Class>_<pg_idx>:<local_rank>``.
_ACTOR_RANK_PATTERN = re.compile(r"_(\d+):(\d+)$")


class _Actor(NamedTuple):
    """One alive Ray actor of the current node."""

    pid: int
    worker_role: str
    # Worker index inside its node; ``None`` for actors that no worker group
    # created (the suffix is only added by ``verl/Ray`` worker groups).
    local_rank: str | None = None


def _short_class_name(class_name: str) -> str:
    """Drop the scope qualifier Ray puts on classes defined inside a function.

    Ray reports such classes qualified, e.g.
    ``create_colocated_worker_cls.<locals>.WorkerDict``.
    """
    return (class_name or "").strip().rsplit(".", 1)[-1].strip()


def _worker_role(class_name: str, actor_name: str = "") -> str:
    """Infer the worker role of a Ray actor from its class or actor name.

    ``verl`` encodes the role in the actor's class name: the fused path names
    it ``FusedWorker_<roles>``, the non-colocated path after the role itself
    (``CriticWorker``, ...). The default colocated path fuses the roles into
    an anonymous ``WorkerDict``, which carries no role, so such workers are
    reported as ``worker``. The actor name embeds the same class token plus a
    random prefix, so it is only consulted when the class name is missing.

    Args:
        class_name: ``ActorState.class_name`` of the actor.
        actor_name: ``ActorState.name`` of the actor.

    Returns:
        The inferred role, or ``"unknown"`` when nothing could be derived.
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
        return re.sub(r"_\d+:\d+$", "", tail) or _UNKNOWN_ROLE
    return _UNKNOWN_ROLE


def _actor_local_rank(actor_name: str) -> str | None:
    """Worker index inside the node, as encoded in a worker-group actor name.

    ``verl`` names every worker-group member ``<prefix><Class>_<pg_idx>:<local_rank>``
    (``verl/single_controller/ray/base.py``). ``local_rank`` is the worker's
    ordinal inside its node, and because a device pool hands out one placement
    group bundle per device, it is the device ordinal there too. Actors created
    outside a worker group carry no such suffix.

    Args:
        actor_name: ``ActorState.name`` of the actor.

    Returns:
        The local rank, or ``None`` when the name has no rank suffix.
    """
    match = _ACTOR_RANK_PATTERN.search((actor_name or "").strip())
    return match.group(2) if match else None


def _actor_device_index(process: psutil.Process) -> str | None:
    """Device a worker was assigned, read from the worker's own environment.

    Ray sets the accelerator visibility variable of every worker to the
    devices it hands out, so the variable the process started with names the
    card it runs on—the same physical id the NPU/GPU collectors publish under
    ``device_index``. Only a single-device value is used: a worker holding
    several cards has no single index to report.

    Args:
        process: Handle of the worker process.

    Returns:
        The device id, or ``None`` when it cannot be determined.
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
    """Alive actor records of one job scheduled on one node, via the Ray state API.

    Args:
        node_id: Ray node id to filter on.
        job_id: Ray job id to filter on; ``list_actors`` is cluster-wide, so
            without it actors from other jobs sharing the node would be
            attributed to this job's metrics.

    Returns:
        The ``ActorState`` records of the node's alive actors in this job.
    """
    from ray.util.state import list_actors

    return list_actors(
        filters=[
            ("node_id", "=", node_id),
            ("job_id", "=", job_id),
            ("state", "=", "ALIVE"),
        ],
        limit=_MAX_ACTORS,
    )


class RayProcessCollector(HardwareCollector):
    """Per-actor CPU and memory metrics of the Ray workers on this node.

    Node-level collectors describe the whole host; this one answers which
    worker is consuming the CPUs. It reads the actors alive on the current
    node **that belong to the current Ray job** from the Ray state API, derives
    each one's role from its class name (see :func:`_worker_role`), and samples
    the backing OS process with psutil, emitting the ``rlinsight_ray_process_*``
    families with ``pid`` and ``worker_role`` labels.
    Every actor is identified further by ``local_rank``
    (its ordinal inside the node) and ``device_index`` (the card it was
    assigned, matching the NPU/GPU panels' label), so a series can be traced
    back to a node and a card instead of a bare pid. The label set is what
    makes these samples process-level: on a shared node several workers report
    at once. The metric prefix is deliberately distinct from the self-process
    ``rlinsight_process_*`` families so the two collectors never share a series
    name, and rl-insight's own actors are skipped so the monitor does not
    report itself as a worker.

    Enabled by default; drop ``ray_actor_process`` from ``hardware.collectors``
    to turn it off. ``discover`` returns ``False`` outside a Ray job. A
    transient state query failure keeps the previous actor set instead of
    dropping every series for one round; once it keeps failing the error is
    raised so the monitor marks the collector unavailable.
    """

    name = "ray_actor_process"

    def __init__(self) -> None:
        # Under Ray the monitor runs inside the node hardware actor, which is
        # one of the actors listed here; it is the monitor's own process rather
        # than a worker, so it is skipped (self_process covers it).
        self._self_pid = os.getpid()
        self._node_id = ""
        self._job_id = ""
        self._processes: dict[int, psutil.Process] = {}
        self._devices: dict[int, str | None] = {}
        self._seeded: set[int] = set()
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
            context = ray.get_runtime_context()
            self._node_id = str(context.get_node_id() or "")
            job_id = context.get_job_id()
        except Exception:
            return False
        if not job_id:
            return False
        # Stay unavailable rather than attributing other jobs' actors to this one.
        self._job_id = job_id if isinstance(job_id, str) else job_id.hex()
        return bool(self._node_id)

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
        """Alive actors of this job on this node, ``None`` on a transient query failure.

        Raises:
            Exception: The state API error, once it failed
                ``_MAX_LIST_FAILURES`` rounds in a row, so the monitor's
                failure policy can deactivate the collector.
        """
        try:
            states = _list_node_actor_states(self._node_id, self._job_id)
        except Exception:
            self._list_failures += 1
            if self._list_failures >= _MAX_LIST_FAILURES:
                raise
            logger.warning(
                "[rl-insight] failed to list the Ray actors of this node; "
                "reusing the previous actor set (%d/%d).",
                self._list_failures,
                _MAX_LIST_FAILURES,
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
            if _short_class_name(class_name) in _OWN_ACTOR_CLASSES:
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
        if process is None:
            try:
                process = psutil.Process(pid)
            except psutil.Error:
                return []
            self._processes[pid] = process
        # The label-name set must stay identical across every worker of the
        # family: the hub registers one Prometheus gauge per metric name, so a
        # worker whose labels differ (e.g. one without a resolved device) would
        # raise DuplicateTimeseries and drop both series. Unknown values are
        # therefore reported as empty strings rather than omitted.
        labels = {
            "pid": str(pid),
            "worker_role": actor.worker_role,
            "local_rank": actor.local_rank or "",
            "device_index": self._device_index(pid, process) or "",
        }
        cpu: float | None = None
        try:
            if pid in self._seeded:
                cpu = process.cpu_percent(interval=None)
            else:
                # Seed the interval-less counter; this round has no window yet.
                process.cpu_percent(interval=None)
                self._seeded.add(pid)
            rss = process.memory_info().rss
            threads = process.num_threads()
        except psutil.Error:
            return []
        return present_samples(
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
                    "rlinsight_ray_process_thread_count",
                    threads,
                    labels,
                    documentation="Thread count of the Ray worker process",
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
        """Device of one worker, resolved once and kept for its lifetime.

        A process's environment is fixed when it starts, so the lookup is
        cached; a failed lookup caches ``None`` instead of retrying—and
        raising—every round.
        """
        if pid not in self._devices:
            self._devices[pid] = _actor_device_index(process)
        return self._devices[pid]

    def _forget_dead(self, actors: list[_Actor]) -> None:
        """Drop cached psutil handles for actors that are no longer alive."""
        alive = {actor.pid for actor in actors}
        for pid in list(self._processes):
            if pid not in alive:
                del self._processes[pid]
                del self._devices[pid]
                self._seeded.discard(pid)


class MemoryCollector(HardwareCollector):
    """Host-level memory usage plus container cgroup limits."""

    name = "memory"

    def discover(self) -> bool:
        # psutil is a hard dependency; memory counters always exist.
        return True

    def collect(self) -> list[Sample]:
        virtual = psutil.virtual_memory()
        # Report total - available rather than psutil's ``used``: the latter
        # only subtracts free + buffers + cached, so it drifts from both
        # psutil's own ``percent`` ((total - available) / total) and the
        # node-exporter fallback (MemTotal - MemAvailable), making the panel
        # step whenever the data source changed.
        used = virtual.total - virtual.available
        return present_samples(
            [
                optional_sample(
                    "rlinsight_memory_used_bytes",
                    used,
                    documentation=("System memory used in bytes (total - available)"),
                ),
                optional_sample(
                    "rlinsight_memory_available_bytes",
                    virtual.available,
                    documentation="System memory available in bytes",
                ),
                optional_sample(
                    "rlinsight_memory_utilization_percent",
                    virtual.percent,
                    documentation="System memory utilization percent",
                ),
                optional_sample(
                    "rlinsight_memory_cgroup_limit_bytes",
                    read_cgroup_memory_limit(),
                    documentation="Container cgroup memory limit in bytes",
                ),
                optional_sample(
                    "rlinsight_memory_cgroup_usage_bytes",
                    read_cgroup_memory_usage(),
                    documentation="Container cgroup memory usage in bytes",
                ),
            ]
        )


_LOOPBACK_INTERFACE = "lo"


def _non_loopback_totals() -> tuple[int, int] | None:
    """Sum ``(bytes_recv, bytes_sent)`` over every non-loopback interface.

    ``psutil.net_io_counters(pernic=False)`` includes ``lo``, whose traffic
    mirrors in-host IPC (including other host-network containers) rather than
    the job's real network load; on a single node it dominates the total. The
    node-exporter fallback already excludes ``lo``, so dropping it here keeps
    both sources on the same footing.

    Returns:
        The summed receive/transmit byte counters, or ``None`` when no
        interface is visible.
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
    """Host-wide network throughput across non-loopback interfaces."""

    name = "network"

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
        return present_samples(
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
    """Host-wide disk throughput plus filesystem utilization of a path."""

    name = "disk"

    def __init__(self) -> None:
        self._read_rate = RateCalculator()
        self._write_rate = RateCalculator()
        self._path = MonitorHardware.DISK_PATH

    def configure(self, options: Mapping[str, Any]) -> None:
        """Override the sampled path (``hardware.disk_path``)."""
        path = options.get("path")
        if path:
            self._path = str(path)

    def discover(self) -> bool:
        return psutil.disk_io_counters() is not None or self._disk_usage() is not None

    def _disk_usage(self) -> Any | None:
        """Return ``psutil.disk_usage`` for the configured path, or ``None``."""
        try:
            return psutil.disk_usage(self._path)
        except OSError:
            return None

    def collect(self) -> list[Sample]:
        usage = self._disk_usage()
        counters = psutil.disk_io_counters()
        samples: list[Sample | None] = [
            optional_sample(
                "rlinsight_disk_utilization_percent",
                None if usage is None else usage.percent,
                {"path": self._path},
                documentation=(
                    f"Filesystem utilization percent for the configured path {self._path}"
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
        return present_samples(samples)


register_collector(CpuCollector.name, CpuCollector)
register_collector(MemoryCollector.name, MemoryCollector)
register_collector(NetworkCollector.name, NetworkCollector)
register_collector(DiskCollector.name, DiskCollector)
register_collector(ProcessCollector.name, ProcessCollector)
register_collector(RayProcessCollector.name, RayProcessCollector)
