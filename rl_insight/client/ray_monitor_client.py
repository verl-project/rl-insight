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
import threading
from typing import Any, cast

import ray
from omegaconf import DictConfig

from .. import __version__
from ..collector.ray_monitor_hub import MonitorHubActor
from ..server.http_api import initialize_server_connection
from ..utils.constants import MonitorRayActor, MonitorServer
from ..utils.protocol import RunContext, RunError, require_version
from .base import MonitorClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorRayClient", "create_ray_monitor_client", "get_or_create_monitor_hub"]

# Named handles are weak; retain creating handles for the Ray job's lifetime.
_hub_owners: list[Any] = []


def _current_job_actor_name() -> str:
    """Return a hub actor name scoped to the current Ray job."""
    job_id = ray.get_runtime_context().get_job_id()
    job_id_str = job_id if isinstance(job_id, str) else job_id.hex()
    return f"{MonitorRayActor.NAME}_{job_id_str}"


def _initialize_hub_connection(
    hub: Any, *, timeout: float, conf: DictConfig | None = None
) -> None:
    """Verify the Hub version before applying task configuration."""
    version = ray.get(hub.connect.remote(__version__), timeout=timeout)
    require_version(version, "Ray Hub")
    if conf is not None:
        ray.get(hub.init.remote(conf), timeout=timeout)


def get_or_create_monitor_hub() -> Any:
    """Reuse the current job's Hub; Ray handles concurrent creation."""
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
        "get_if_exists": True,
    }
    hub = cast(Any, MonitorHubActor).options(**actor_options).remote()
    _hub_owners.append(hub)
    return hub


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
        initialize_server_connection(str(task_conf.server.url))
    handle = get_or_create_monitor_hub()
    client = MonitorRayClient(handle)
    client.initialize(task_conf)
    return client


class MonitorRayClient(MonitorClient):
    """Ray facade: ``apply_event`` submits work to the hub without blocking on completion."""

    def __init__(self, actor_handle: Any) -> None:
        """
        Args:
            actor_handle: Return value of ``get_or_create_monitor_hub``.
        """
        self._actor = actor_handle
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._closed = False
        self._compatibility_error: RunError | None = None
        self._initialized = False
        # Includes service discovery and actor scheduling during initialization.
        self._timeout = (
            MonitorServer.SERVICE_DISCOVERY_RETRIES
            * (
                MonitorServer.SERVICE_DISCOVERY_TIMEOUT_SECONDS
                + MonitorServer.SERVICE_DISCOVERY_RETRY_DELAY_SECONDS
            )
            + 30
        )

    def initialize(self, conf: DictConfig | None = None) -> None:
        """Establish and validate the Hub connection before collecting data."""
        with self._init_lock:
            if self._compatibility_error is not None:
                raise self._compatibility_error
            if self._initialized:
                return
            try:
                _initialize_hub_connection(
                    self._actor, timeout=self._timeout, conf=conf
                )
            except RunError as exc:
                self._compatibility_error = exc
                raise
            self._initialized = True

    def get_context(self) -> RunContext | None:
        if self._compatibility_error is not None:
            raise self._compatibility_error
        if not self._initialized:
            return None
        context = ray.get(self._actor.get_context.remote(), timeout=self._timeout)
        if context is not None and not isinstance(context, RunContext):
            raise RunError("Ray Hub returned an invalid Context", 502)
        return context

    def apply_event(self, event: dict[str, Any]) -> None:
        """Submit ``MonitorHubActor.apply_event`` on the actor (fire-and-forget; no ``ray.get``).

        Args:
            event: Serialized monitor event (see ``metric_*`` / ``trace_*`` helpers for shapes).

        Note:
            Errors on the hub side are not surfaced here. Ordering follows Ray actor scheduling.
        """
        with self._lock:
            if self._compatibility_error is not None:
                raise self._compatibility_error
            if self._closed:
                raise RunError("Monitor client is closed")
            if not self._initialized:
                raise RunError("Monitor client is not initialized")
            self._actor.apply_event.remote(event)

    def close(self) -> None:
        with self._lock:
            self._closed = True
