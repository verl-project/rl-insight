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

"""Ray client for the shared monitor hub."""

from __future__ import annotations

import logging
from typing import Any, cast

import ray
from omegaconf import DictConfig

from ..collector.ray_monitor_hub import MonitorHubActor
from ..utils.constants import MonitorRayActor
from .base import MonitorClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorRayClient", "create_ray_monitor_client", "get_or_create_monitor_hub"]


def _current_job_actor_name() -> str:
    """Return a hub actor name scoped to the current Ray job."""
    job_id = ray.get_runtime_context().get_job_id()
    job_id_str = job_id if isinstance(job_id, str) else job_id.hex()
    return f"{MonitorRayActor.NAME}_{job_id_str}"


def get_or_create_monitor_hub(conf: DictConfig) -> Any:
    """Get or create a job-scoped MonitorHubActor with health check.

    Each Ray job gets its own hub actor named
    ``MonitorRayActor.NAME_{job_id}``. The actor is not detached, so it
    cleans up automatically when the job exits.

    Args:
        conf: Merged trainer monitor config passed to the actor constructor.

    Returns:
        Ray actor handle for ``MonitorHubActor``.

    Raises:
        RuntimeError: If Ray is not initialized.
    """

    actor_name = _current_job_actor_name()
    namespace = MonitorRayActor.NAMESPACE

    try:
        handle = ray.get_actor(actor_name, namespace=namespace)
        logger.info(
            "[rl-insight] Connected to existing monitor hub actor %r.", actor_name
        )
        return handle
    except ValueError:
        logger.info(
            "[rl-insight] No existing monitor hub actor %r found; creating one.",
            actor_name,
        )
    actor_options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
    }

    try:
        actor_cls = cast(Any, MonitorHubActor)
        return actor_cls.options(**actor_options).remote(conf)
    except ValueError:
        logger.info(
            "[rl-insight] Monitor hub actor %r was created concurrently; "
            "connecting to it.",
            actor_name,
        )
        return ray.get_actor(actor_name, namespace=namespace)


def create_ray_monitor_client(conf: DictConfig) -> MonitorRayClient | None:
    """Build a client that talks to ``MonitorHubActor`` over Ray.

    Args:
        conf: Merged monitor configuration.

    Returns:
        Client instance, or ``None`` if Ray is not initialized (monitoring disabled).
    """
    if not ray.is_initialized():
        logger.warning("[rl-insight] Ray is not initialized; monitoring is disabled.")
        return None

    handle = get_or_create_monitor_hub(conf)
    return MonitorRayClient(handle)


class MonitorRayClient(MonitorClient):
    """Ray facade: ``apply_event`` submits work to the hub without blocking on completion."""

    def __init__(self, actor_handle: Any) -> None:
        """
        Args:
            actor_handle: Return value of ``get_or_create_monitor_hub``.
        """
        self._actor = actor_handle

    def apply_event(self, event: dict[str, Any]) -> None:
        """Submit ``MonitorHubActor.apply_event`` on the actor (fire-and-forget; no ``ray.get``).

        Args:
            event: Serialized monitor event (see ``metric_*`` / ``trace_*`` helpers for shapes).

        Note:
            Errors on the hub side are not surfaced here. Ordering follows Ray actor scheduling.
        """
        self._actor.apply_event.remote(event)
