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

from .. import __version__
from ..collector.ray_monitor_hub import MonitorHubActor
from ..server.http_api import initialize_server_connection
from ..utils.constants import MonitorRayActor, MonitorServer
from ..utils.protocol import RunContext, require_version
from .base import MonitorClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = [
    "MonitorRayClient",
    "create_ray_monitor_client",
    "get_current_job_context",
    "get_or_create_monitor_hub",
    "initialize_hub_connection",
]

# Named handles are weak; retain creating handles for the Ray job's lifetime.
_hub_owners: list[Any] = []


def _current_job_actor_name() -> str:
    """Return a hub actor name scoped to the current Ray job."""
    job_id = ray.get_runtime_context().get_job_id()
    job_id_str = job_id if isinstance(job_id, str) else job_id.hex()
    return f"{MonitorRayActor.NAME}_{job_id_str}"


def initialize_hub_connection(
    hub: Any, *, timeout: float, conf: DictConfig | None = None
) -> None:
    """Verify the Hub version before applying task configuration."""
    version = ray.get(hub.connect.remote(__version__), timeout=timeout)
    require_version(version, "Ray Hub")
    if conf is not None:
        ray.get(hub.init.remote(conf), timeout=timeout)


def get_or_create_monitor_hub() -> Any:
    """Get or create the current job's Hub, reusing concurrent creations."""
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
        hub = actor_cls.options(**actor_options).remote()
    except ValueError:
        logger.info(
            "[rl-insight] Monitor hub actor %r was created concurrently; "
            "connecting to it.",
            actor_name,
        )
        return ray.get_actor(actor_name, namespace=namespace)
    _hub_owners.append(hub)
    return hub


def get_current_job_context(timeout: float = 10.0) -> RunContext | None:
    """Return the current job's Run identity without creating or initializing a Hub.

    Processes that never call ``init()`` (such as rollout workers) use this to
    read the Run name the Hub stamps on telemetry. Returns ``None`` when Ray is
    not initialized, the job has no Hub, or the Hub is not initialized yet.
    """
    if not ray.is_initialized():
        return None
    try:
        handle: Any = ray.get_actor(
            _current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE
        )
    except ValueError:
        return None
    return ray.get(handle.get_context.remote(), timeout=timeout)


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

    task_conf = conf if conf.get("run_metadata") is not None else None
    if task_conf is not None:
        if initialize_server_connection(str(task_conf.server.url)) is None:
            return None
    handle = get_or_create_monitor_hub()
    client = MonitorRayClient(handle, task_conf)
    if task_conf is not None and client.get_context() is None:
        logger.error(
            "[rl-insight] Monitor Hub did not initialize; monitoring is disabled "
            "for this process. Check the Hub logs and retry init()."
        )
        return None
    return client


class MonitorRayClient(MonitorClient):
    """Ray facade: ``apply_event`` submits work to the hub without blocking on completion."""

    def __init__(self, actor_handle: Any, conf: DictConfig | None = None) -> None:
        """
        Args:
            actor_handle: Return value of ``get_or_create_monitor_hub``.
            conf: Formal task configuration, or ``None`` to only connect.
        """
        self._actor = actor_handle
        # Includes service discovery and actor scheduling during initialization.
        self._timeout = (
            MonitorServer.SERVICE_DISCOVERY_RETRIES
            * (
                MonitorServer.SERVICE_DISCOVERY_TIMEOUT_SECONDS
                + MonitorServer.SERVICE_DISCOVERY_RETRY_DELAY_SECONDS
            )
            + 30
        )

        initialize_hub_connection(self._actor, timeout=self._timeout, conf=conf)

    def get_context(self) -> RunContext | None:
        return ray.get(self._actor.get_context.remote(), timeout=self._timeout)

    def apply_event(self, event: dict[str, Any]) -> None:
        """Submit ``MonitorHubActor.apply_event`` on the actor (fire-and-forget; no ``ray.get``).

        Args:
            event: Serialized monitor event (see ``metric_*`` / ``trace_*`` helpers for shapes).

        Note:
            Errors on the hub side are not surfaced here. Ordering follows Ray actor scheduling.
        """
        self._actor.apply_event.remote(event)

    def close(self) -> None:
        """Drain the shared Hub after event producers have stopped."""
        ray.get(self._actor.flush.remote(), timeout=self._timeout)
