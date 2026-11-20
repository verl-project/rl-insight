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

``rl_insight.init`` starts monitoring when the ``hardware`` section is enabled.
In a Ray job one ``NodeHardwareActor`` runs per node, so host samples are
collected once per node; outside Ray the monitor runs as a thread in the
caller. Both modes resolve ``hardware.collectors`` against the registry and
emit batched gauge events through the monitor client.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..utils.constants import MonitorHardware, MonitorRayActor
from .base import HardwareSettings, current_ray_job_id, identity_labels

if TYPE_CHECKING:
    from .node_actor import NodeActorSet

# Side-effect imports: register the built-in system and vendor collectors.
from . import system_collectors, vendor  # noqa: F401
from .coordinator import HardwareMonitor

__all__ = ["start_hardware_monitor"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Minimum delay between node-actor reconciliation passes. A ready actor is
# probed only once; the interval mainly bounds how quickly a failed actor is
# recreated and how often pending actors are re-checked.
_MIN_RECONCILE_INTERVAL_SECONDS = 30.0


def _resolve_hardware_settings(hardware_conf: Any) -> HardwareSettings:
    """Resolve and validate the ``hardware`` config section."""
    raw_interval = OmegaConf.select(
        hardware_conf,
        "interval_seconds",
        default=MonitorHardware.INTERVAL_SECONDS,
    )
    interval = float(raw_interval)
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

    return HardwareSettings(
        interval_seconds=interval,
        collector_names=collector_names,
    )


def _resolve_collector_names(raw: Any) -> list[str]:
    """Normalize ``hardware.collectors`` into a list of names.

    A scalar is read as one name (``collectors: host`` is a common YAML typo).
    Missing or invalid values use the default collector set.
    """
    defaults = list(MonitorHardware.COLLECTORS)
    if raw is None:
        return defaults
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            return defaults
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
        "using the default collectors instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return defaults


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

        if not ray.is_initialized():
            return None
        return str(ray.get_runtime_context().get_node_id() or "") or None
    except Exception:
        return None


def _job_node_ids() -> list[str] | None:
    """Return node ids used by the current Ray job.

    Only the driver node and nodes hosting alive actors of this job are
    returned; ``ray.nodes()`` would include nodes used by other jobs. Returns
    ``None`` when the state API is unavailable.
    """
    job_id = current_ray_job_id()
    if not job_id:
        return None

    try:
        from ray.util.state import list_actors

        states = list_actors(
            filters=[("job_id", "=", job_id), ("state", "=", "ALIVE")],
            limit=MonitorRayActor.STATE_API_ACTOR_LIMIT,
        )
    except Exception as exc:
        logger.warning(
            "[rl-insight] Ray State API is unavailable (%s); hardware node "
            "discovery falls back to the local node. Install 'ray[default]' "
            "and keep the Ray dashboard enabled for multi-node monitoring.",
            type(exc).__name__,
        )
        logger.debug(
            "[rl-insight] Ray State API discovery failure detail.",
            exc_info=True,
        )
        return None

    node_ids: set[str] = set()
    local_node = _local_node_id()
    if local_node:
        node_ids.add(local_node)
    for state in states:
        class_name = str(getattr(state, "class_name", "") or "").rsplit(".", 1)[-1]
        if class_name in MonitorRayActor.INTERNAL_CLASSES:
            continue
        node_id = getattr(state, "node_id", None)
        if node_id:
            node_ids.add(str(node_id))
    return sorted(node_ids)


def _node_actor_name(node_id: str) -> str:
    """Job-scoped, node-scoped hardware actor name."""
    job_id = current_ray_job_id() or "default"
    return f"RLInsightHardware_{job_id}_{node_id}"


def _create_node_actor(
    node_id: str,
    conf: DictConfig,
    labels: dict[str, Any],
    settings: HardwareSettings,
) -> Any:
    """Create or reuse the hardware actor pinned to one Ray node."""
    # Keep Ray-side imports out of this module's import path.
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from .node_actor import NodeHardwareActor

    name = _node_actor_name(node_id)
    options = {
        "name": name,
        "namespace": MonitorRayActor.NAMESPACE,
        # Stay schedulable on a fully used node and take no CPU from training.
        "num_cpus": 0,
        "max_restarts": 3,
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False
        ),
    }
    try:
        actor_cls = cast(Any, NodeHardwareActor)
        return actor_cls.options(**options).remote(conf, labels, settings)
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
    settings: HardwareSettings,
    node_ids: list[str],
) -> NodeActorSet:
    """Start one actor per initially discovered job node and keep them ready."""
    # Keep node_actor (and its Ray imports) out of this module's import path.
    from .node_actor import NodeActorSet

    handles: dict[str, Any] = {}
    for node_id in node_ids:
        try:
            handles[node_id] = _create_node_actor(node_id, conf, labels, settings)
        except Exception:
            logger.warning(
                "[rl-insight] failed to create the hardware actor on node "
                "%s; skipping it.",
                node_id,
                exc_info=True,
            )
    monitor = NodeActorSet(
        handles,
        create_actor=lambda node_id: _create_node_actor(
            node_id, conf, labels, settings
        ),
        node_ids=node_ids,
        reconcile_interval_seconds=max(
            _MIN_RECONCILE_INTERVAL_SECONDS, settings.interval_seconds
        ),
    )
    logger.info(
        "[rl-insight] hardware monitoring started on %d node(s).",
        len(handles),
    )
    return monitor


def start_hardware_monitor(
    conf: DictConfig,
    client: Any,
    labels: Mapping[str, Any] | None = None,
) -> HardwareMonitor | NodeActorSet | None:
    """Start hardware monitoring from the merged monitor config.

    Under Ray, creates one node-scoped actor per discovered job node. Outside
    Ray, runs a monitor thread in the caller's process. Returns ``None`` when
    disabled or when no client is available.

    ``interval_seconds`` below ``MonitorHardware.MIN_INTERVAL_SECONDS`` is
    clamped to the minimum with a ``RuntimeWarning``.
    """
    hardware_conf = OmegaConf.select(conf, "hardware")
    if client is None or hardware_conf is None:
        return None
    if not OmegaConf.select(hardware_conf, "enabled", default=False):
        return None

    settings = _resolve_hardware_settings(hardware_conf)
    # Normalized once more by ``HardwareMonitor`` / ``NodeHardwareActor``.
    base_labels = dict(labels or {})

    if _ray_is_initialized():
        node_ids = _job_node_ids()
        if node_ids is None:
            local_node = _local_node_id()
            node_ids = [local_node] if local_node else []
        if node_ids:
            return _start_node_actors(conf, base_labels, settings, node_ids)

    # Non-Ray mode (tests, non-Ray clients): the caller's own process is the
    # sampling node.
    base_labels.update(identity_labels())
    monitor = HardwareMonitor(
        client=client,
        labels=base_labels,
        interval_seconds=settings.interval_seconds,
        collector_names=settings.collector_names,
    )
    monitor.start()
    return monitor
