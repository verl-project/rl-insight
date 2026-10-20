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

"""In-process hardware monitoring for RL-Insight.

``rl_insight.init`` starts hardware monitoring when the ``hardware`` config
section is enabled. In a Ray job, one node-scoped
:class:`NodeHardwareActor` runs per alive node so host samples are collected
exactly once per node; outside Ray the monitor falls back to a thread inside
the caller's process. Both modes resolve the configured collector names
against ``COLLECTOR_REGISTRY``, probe each collector, and emit batched gauge
events through the regular monitor client. Collector modules (system
``psutil`` collectors, vendor adapters such as NVIDIA/Ascend) register
themselves via ``register_collector`` at import time.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Mapping, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..utils.constants import MonitorEnv, MonitorHardware, MonitorRayActor
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


def _resolve_hardware_params(hardware_conf: Any) -> dict[str, Any]:
    """Resolve and validate the ``hardware`` config section once.

    Args:
        hardware_conf: The ``hardware`` node of the merged monitor config.

    Returns:
        Dict with ``interval_seconds``, ``collector_names``, and
        ``collector_options`` ready for ``HardwareMonitor``.
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

    raw_disk_path = OmegaConf.select(hardware_conf, "disk_path")
    disk_path = str(raw_disk_path) if raw_disk_path else str(MonitorHardware.DISK_PATH)
    collector_options = {"disk": {"path": disk_path}}
    # The role describes the process that reports, not the node: on a shared
    # node several workers report at once, so it belongs on the self-process
    # collector instead of the monitor's base labels.
    worker_role = _resolve_worker_role(hardware_conf)
    if worker_role:
        collector_options["self_process"] = {"worker_role": worker_role}
    return {
        "interval_seconds": interval,
        "collector_names": collector_names,
        "collector_options": collector_options,
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


def _resolve_worker_role(hardware_conf: Any) -> str:
    """Resolve the ``worker_role`` label of the reporting process.

    The environment variable wins so a launcher can tag each worker process
    without rebuilding the config; otherwise ``hardware.worker_role`` is used.
    An empty value means the label is omitted. This is a process-level fact,
    so it is handed to the process collectors rather than broadcast to every
    sample through the monitor's base labels.
    """
    env_role = os.environ.get(MonitorEnv.WORKER_ROLE, "").strip()
    if env_role:
        return env_role
    return str(OmegaConf.select(hardware_conf, "worker_role") or "").strip()


def _ray_is_initialized() -> bool:
    """Whether a Ray cluster is available for node-scoped actors."""
    try:
        import ray

        return ray.is_initialized()
    except Exception:
        return False


def _alive_node_ids() -> list[str]:
    """Node ids of the alive nodes in the current Ray cluster."""
    import ray

    return [str(node["NodeID"]) for node in ray.nodes() if node.get("Alive")]


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
    """Start one hardware actor per alive node; failures only skip a node."""
    # Imported here so this module stays importable without Ray (e.g. from
    # the server command line).
    from .node_actor import NodeActorSet

    handles = []
    for node_id in node_ids:
        try:
            handles.append(_create_node_actor(node_id, conf, labels, params))
        except Exception:
            logger.warning(
                "[rl-insight] failed to create the hardware actor on node "
                "%s; skipping it.",
                node_id,
                exc_info=True,
            )
    if not handles:
        return None
    logger.info("[rl-insight] hardware monitoring started on %d node(s).", len(handles))
    return NodeActorSet(handles)


def start_hardware_monitor(
    conf: DictConfig,
    client: Any,
    labels: Mapping[str, Any] | None = None,
) -> "HardwareMonitor | NodeActorSet | None":
    """Build and start hardware monitoring from the merged monitor config.

    In a Ray job with alive nodes this creates one node-scoped hardware actor
    per node (deduplicated by actor name), and the given ``client`` is used
    only for the in-process fallback so tests and non-Ray clients keep the
    simple thread model.

    Args:
        conf: Fully merged monitor config containing a ``hardware`` section.
        client: Monitor client used by the in-process fallback; ``None``
            disables hardware monitoring entirely.
        labels: Init-level labels (project, experiment name) merged with the
            auto-detected ``node`` and ``run_id`` labels.

    Returns:
        The started monitor (``NodeActorSet`` under Ray, ``HardwareMonitor``
        in-process), or ``None`` when disabled or there is nothing to report
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
        node_ids = _alive_node_ids()
        if node_ids:
            return _start_node_actors(conf, base_labels, params, node_ids)

    # In-process fallback (tests, non-Ray clients): the caller's own process
    # is the sampling node.
    base_labels.update(identity_labels())
    monitor = HardwareMonitor(
        client=client,
        labels=base_labels,
        interval_seconds=params["interval_seconds"],
        collector_names=params["collector_names"],
        collector_options=params["collector_options"],
    )
    monitor.start()
    return monitor
