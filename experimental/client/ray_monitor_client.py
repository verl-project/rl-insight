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
from omegaconf import DictConfig, OmegaConf

from ..collector.ray_monitor_hub import MonitorHubActor
from ..config import MONITOR_HUB_ACTOR_NAME, MONITOR_RAY_NAMESPACE

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorRayClient", "create_monitor_client", "get_or_create_monitor_hub"]


def get_or_create_monitor_hub(conf: DictConfig) -> Any:
    """Get the detached ``MonitorHubActor`` handle, creating it on first use (race-safe).

    Args:
        conf: Merged trainer monitor config passed to the actor constructor.

    Returns:
        Ray actor handle for ``MonitorHubActor``.

    Raises:
        RuntimeError: If Ray is not initialized.
    """
    if not ray.is_initialized():
        raise RuntimeError(
            "Ray is not initialized. Call ray.init() before using monitor helpers."
        )

    actor_name = MONITOR_HUB_ACTOR_NAME
    namespace = MONITOR_RAY_NAMESPACE

    try:
        handle = ray.get_actor(actor_name, namespace=namespace)
        logger.info("Connected to existing monitor hub actor %r.", actor_name)
        return handle
    except ValueError:
        logger.info("No existing monitor hub actor %r found; creating one.", actor_name)

    actor_options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
        "lifetime": "detached",
    }

    try:
        actor_cls = cast(Any, MonitorHubActor)
        return actor_cls.options(**actor_options).remote(
            OmegaConf.to_container(conf, resolve=True)
        )
    except ValueError:
        logger.info(
            "Monitor hub actor %r was created concurrently; connecting to it.",
            actor_name,
        )
        return ray.get_actor(actor_name, namespace=namespace)


def create_monitor_client(conf: DictConfig) -> "MonitorRayClient | None":
    """Build a client that talks to ``MonitorHubActor`` over Ray.

    Args:
        conf: Merged monitor configuration.

    Returns:
        Client instance, or ``None`` if Ray is not initialized (monitoring disabled).
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized; monitoring is disabled.")
        return None

    handle = get_or_create_monitor_hub(conf)
    return MonitorRayClient(handle)


class MonitorRayClient:
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
            event: Serialized monitor event (see ``experimental.api`` helpers for shapes).

        Note:
            Errors on the hub side are not surfaced here. Ordering follows Ray actor scheduling.
        """
        self._actor.apply_event.remote(event)
