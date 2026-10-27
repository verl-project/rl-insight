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

"""Hardware monitoring for RL-Insight.

``rl_insight.init`` starts hardware monitoring when the ``hardware`` config
section is enabled. In a Ray job, one node-scoped
:class:`NodeHardwareActor` runs per alive node so host samples are collected
exactly once per node; outside Ray the monitor runs as a thread inside the
caller's process. Both modes resolve the configured collector names against
``COLLECTOR_REGISTRY``, probe each collector, and emit batched gauge events
through the regular monitor client. Collector modules (system ``psutil``
collectors, vendor adapters such as NVIDIA/Ascend) register themselves via
``register_collector`` at import time.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Mapping, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..utils.constants import MonitorHardware, MonitorRayActor
from .identity import detect_run_id_label, identity_labels

if TYPE_CHECKING:
    from .node_actor import NodeActorSet

# Side-effect imports: register the built-in system and vendor collectors.
from . import system_collectors, vendor  # noqa: F401
from .base import (
    COLLECTOR_REGISTRY,
    CollectorFactory,
    HardwareCollector,
    Sample,
    create_collector,
    list_collector_names,
    register_collector,
)
from .coordinator import HardwareMonitor

__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFactory",
    "HardwareCollector",
    "HardwareMonitor",
    "Sample",
    "create_collector",
    "list_collector_names",
    "register_collector",
    "start_hardware_monitor",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Upper bound for the cluster-wide actor query used to discover job nodes.
# Ray's state API server rejects requests above
# ``RAY_MAX_LIMIT_FROM_API_SERVER`` (10000 by default) with HTTP 500, so this
# must stay within that default rather than exceeding it.
_MAX_JOB_ACTORS = 10_000
# Minimum delay between node-set reconciliations. A larger sampling interval
# also slows down reconciliation to keep the State API load low.
_MIN_RECONCILE_INTERVAL_SECONDS = 10.0
# Actors created by RL-Insight itself must not keep a node in the job node set
# after its training actors have gone away.
_MONITOR_ACTOR_CLASSES = frozenset({"MonitorHubActor", "NodeHardwareActor"})


def _resolve_hardware_params(hardware_conf: Any) -> dict[str, Any]:
    """Resolve and validate the ``hardware`` config section once.

    Args:
        hardware_conf: The ``hardware`` node of the merged monitor config.

    Returns:
        Dict with ``interval_seconds`` and ``collector_names`` ready for
        ``HardwareMonitor``.
    """
    raw_interval = OmegaConf.select(hardware_conf, "interval_seconds")
    interval = (
        float(raw_interval)
        if raw_interval is not None
        else float(MonitorHardware.INTERVAL_SECONDS)
    )
    if interval < MonitorHardware.MIN_INTERVAL_SECONDS:
        warnings.warn(
            f"[rl-insight] hardware interval_seconds {interval} is below the "
            f"minimum {MonitorHardware.MIN_INTERVAL_SECONDS}; clamping to the "
            "minimum.",
            RuntimeWarning,
            stacklevel=2,
        )
        interval = float(MonitorHardware.MIN_INTERVAL_SECONDS)

    raw_collectors = OmegaConf.select(hardware_conf, "collectors")
    collector_names = _resolve_collector_names(raw_collectors)

    return {
        "interval_seconds": interval,
        "collector_names": collector_names,
    }


def _resolve_collector_names(raw: Any) -> list[str] | None:
    """Resolve ``hardware.collectors`` into collector names; ``None`` means every registered collector.

    A bare scalar (``collectors: cpu``) is a common YAML typo for
    ``collectors: [cpu]``; iterating it would produce the pseudo names ``c``,
    ``p`` and ``u`` and leave every real collector unresolvable, so a scalar is
    read as one name. Any other non-list value is ignored with a warning rather
    than iterated, so a malformed value degrades to the default collector set
    instead of reporting meaningless status series.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            return None
        warnings.warn(
            f"[rl-insight] hardware.collectors should be a list of names; "
            f"interpreting {name!r} as a single collector.",
            RuntimeWarning,
            stacklevel=3,
        )
        return [name]
    if isinstance(raw, (ListConfig, list, tuple)):
        return [str(name) for name in raw]
    warnings.warn(
        f"[rl-insight] hardware.collectors {raw!r} is not a list of names; "
        "using every registered collector instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return None


def _ray_is_initialized() -> bool:
    """Whether a Ray cluster is available for node-scoped actors."""
    try:
        import ray

        return ray.is_initialized()
    except Exception:
        return False


def _local_node_id() -> str | None:
    """Node id of the current Ray process, or ``None`` outside Ray."""
    try:
        import ray

        return str(ray.get_runtime_context().get_node_id() or "") or None
    except Exception:
        return None


def _job_node_ids() -> list[str] | None:
    """Node ids currently used by the current Ray job.

    Only the driver node and nodes hosting alive actors of the current job are
    returned. ``ray.nodes()`` is intentionally not used here: it lists every
    node in the cluster, including nodes used by other jobs.

    Returns:
        The job node ids, or ``None`` when the Ray state API is unavailable.
        ``None`` means "do not reconcile/remove nodes this round".
    """
    try:
        import ray

        context = ray.get_runtime_context()
        job_id = context.get_job_id()
        job_id = job_id if isinstance(job_id, str) else job_id.hex()
        local_node = str(context.get_node_id() or "")
    except Exception:
        return None
    if not job_id:
        return None

    try:
        from ray.util.state import list_actors

        states = list_actors(
            filters=[("job_id", "=", job_id), ("state", "=", "ALIVE")],
            limit=_MAX_JOB_ACTORS,
        )
    except Exception:
        logger.warning(
            "[rl-insight] failed to query Ray actors for hardware node "
            "discovery; keeping the current node set.",
            exc_info=True,
        )
        return None

    node_ids = {local_node} if local_node else set()
    for state in states:
        class_name = str(getattr(state, "class_name", "") or "").rsplit(".", 1)[-1]
        if class_name in _MONITOR_ACTOR_CLASSES:
            continue
        node_id = getattr(state, "node_id", None)
        if node_id:
            node_ids.add(str(node_id))
    return sorted(node_ids)


def _node_actor_name(node_id: str) -> str:
    """Job-scoped, node-scoped hardware actor name."""
    run_id = detect_run_id_label() or "default"
    return f"RLInsightHardware_{run_id}_{node_id}"


def _create_node_actor(
    node_id: str,
    conf: DictConfig,
    labels: dict[str, Any],
    params: dict[str, Any],
) -> Any:
    """Create (or reuse) the hardware actor pinned to one Ray node.

    Args:
        node_id: Target Ray node id.
        conf: Merged monitor config passed through to the actor.
        labels: Init-level labels without node/run identity.
        params: Resolved hardware params (see ``_resolve_hardware_params``).

    Returns:
        A Ray actor handle for the node's hardware monitor.
    """
    # Imported here so this module stays importable without Ray-side client
    # dependencies (e.g. from the server command line).
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from .node_actor import NodeHardwareActor

    name = _node_actor_name(node_id)
    options = {
        "name": name,
        "namespace": MonitorRayActor.NAMESPACE,
        # A per-process handle would otherwise be the actor's only lifeline:
        # short-lived workers call ``init``/``finish`` and Ray would destroy
        # the actor mid-job as soon as the last handle is gone. Detached
        # actors keep sampling for the whole job and stop themselves when the
        # job ends (see ``NodeHardwareActor._job_watchdog``).
        "lifetime": "detached",
        # The monitor must be schedulable on a node whose CPUs are fully used
        # by training workers, and it must not take a CPU away from them.
        "num_cpus": 0,
        "max_restarts": 3,
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False
        ),
    }
    try:
        actor_cls = cast(Any, NodeHardwareActor)
        return actor_cls.options(**options).remote(conf, labels, params)
    except ValueError:
        logger.info(
            "[rl-insight] hardware actor %r was created concurrently; "
            "connecting to it.",
            name,
        )
        return ray.get_actor(name, namespace=MonitorRayActor.NAMESPACE)


def _start_node_actors(
    conf: DictConfig,
    labels: dict[str, Any],
    params: dict[str, Any],
    node_ids: list[str],
) -> "NodeActorSet | None":
    """Start one hardware actor per current-job node and reconcile later changes."""
    # Imported here so this module stays importable without Ray (e.g. from
    # the server command line).
    from .node_actor import NodeActorSet

    handles: dict[str, Any] = {}
    for node_id in node_ids:
        try:
            handles[node_id] = _create_node_actor(node_id, conf, labels, params)
        except Exception:
            logger.warning(
                "[rl-insight] failed to create the hardware actor on node "
                "%s; skipping it.",
                node_id,
                exc_info=True,
            )
    if not handles:
        return None
    monitor = NodeActorSet(
        handles,
        create_actor=lambda node_id: _create_node_actor(node_id, conf, labels, params),
        node_ids_provider=_job_node_ids,
        reconcile_interval_seconds=max(
            _MIN_RECONCILE_INTERVAL_SECONDS, float(params["interval_seconds"])
        ),
    )
    logger.info(
        "[rl-insight] hardware monitoring started on %d node(s); "
        "new job nodes will be added automatically.",
        len(handles),
    )
    return monitor


def start_hardware_monitor(
    conf: DictConfig,
    client: Any,
    labels: Mapping[str, Any] | None = None,
) -> "HardwareMonitor | NodeActorSet | None":
    """Build and start hardware monitoring from the merged monitor config.

    In a Ray job this creates one node-scoped hardware actor on the driver node
    and on nodes hosting alive actors of the current job; the node set is
    reconciled while the job runs. The given ``client`` is used only for the
    non-Ray monitor so tests and non-Ray clients keep the simple thread model.

    Args:
        conf: Fully merged monitor config containing a ``hardware`` section.
        client: Monitor client used by the non-Ray monitor; ``None``
            disables hardware monitoring entirely.
        labels: Init-level labels (project, experiment name) merged with the
            auto-detected ``node`` and ``run_id`` labels.

    Returns:
        The started monitor (``NodeActorSet`` under Ray, ``HardwareMonitor``
        otherwise), or ``None`` when disabled or there is nothing to report
        through.

    Note:
        ``interval_seconds`` values below ``MonitorHardware.MIN_INTERVAL_SECONDS``
        are clamped to the minimum with a ``RuntimeWarning``.
    """
    hardware_conf = OmegaConf.select(conf, "hardware")
    if client is None or hardware_conf is None:
        return None
    if not bool(OmegaConf.select(hardware_conf, "enabled") or False):
        return None

    params = _resolve_hardware_params(hardware_conf)
    base_labels = {str(k): str(v) for k, v in dict(labels or {}).items()}

    if _ray_is_initialized():
        node_ids = _job_node_ids()
        if node_ids is None:
            local_node = _local_node_id()
            node_ids = [local_node] if local_node else []
            logger.warning(
                "[rl-insight] could not discover nodes used by the current Ray "
                "job; monitoring the local node only."
            )
        if node_ids:
            return _start_node_actors(conf, base_labels, params, node_ids)

    # Non-Ray mode (tests, non-Ray clients): the caller's own process is the
    # sampling node.
    base_labels.update(identity_labels())
    monitor = HardwareMonitor(
        client=client,
        labels=base_labels,
        interval_seconds=params["interval_seconds"],
        collector_names=params["collector_names"],
        collector_options=params.get("collector_options"),
    )
    monitor.start()
    return monitor
