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
from collections import deque
from typing import Any, cast
from uuid import uuid4

import ray
from omegaconf import DictConfig

from ..collector.ray_monitor_hub import MonitorHubActor
from ..utils.constants import MonitorEnv, MonitorRayActor, MonitorServer
from ..utils.protocol import RunContext, RunError
from .base import MonitorClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_owned_hubs: dict[str, str] = {}

__all__ = ["MonitorRayClient", "create_ray_monitor_client", "get_or_create_monitor_hub"]


def _current_job_actor_name() -> str:
    """Return a hub actor name scoped to the current Ray job."""
    job_id = ray.get_runtime_context().get_job_id()
    job_id_str = job_id if isinstance(job_id, str) else job_id.hex()
    return f"{MonitorRayActor.NAME}_{job_id_str}"


def current_job_context() -> RunContext | None:
    """Resolve only an existing Hub; native registration must not create a Run."""
    if not ray.is_initialized():
        return None
    try:
        hub: Any = ray.get_actor(
            _current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE
        )
    except ValueError:
        return None
    try:
        context = ray.get(hub.get_context.remote(), timeout=2)
    except (ray.exceptions.GetTimeoutError, ray.exceptions.ActorUnavailableError):
        return None
    if not isinstance(context, RunContext):
        raise RunError("Hub returned an invalid RunContext")
    return context


def get_or_create_monitor_hub(conf: DictConfig, *, create: bool = True) -> Any:
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
        if not create:
            return None
        logger.info(
            "[rl-insight] No existing monitor hub actor %r found; creating one.",
            actor_name,
        )
    # Do not use get_if_exists: only successful creation grants Run-close ownership.
    actor_options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
    }
    server_url = str(conf.server.url).strip()
    if server_url:
        actor_options["runtime_env"] = {
            "env_vars": {MonitorEnv.SERVER_URL: server_url},
        }

    try:
        actor_cls = cast(Any, MonitorHubActor)
        token = uuid4().hex
        handle = actor_cls.options(**actor_options).remote(conf, owner_token=token)
        _owned_hubs[actor_name] = token
        return handle
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

    handle = get_or_create_monitor_hub(conf, create=bool(conf.get("run_metadata")))
    return MonitorRayClient(handle, _owned_hubs.get(_current_job_actor_name()))


class MonitorRayClient(MonitorClient):
    """Ray facade: ``apply_event`` submits work to the hub without blocking on completion."""

    def __init__(self, actor_handle: Any, owner_token: str | None = None) -> None:
        """
        Args:
            actor_handle: Return value of ``get_or_create_monitor_hub``.
        """
        self._actor = actor_handle
        self._owner_token = owner_token
        self._actor_name = _current_job_actor_name() if actor_handle is None else ""
        self._pending: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._waiter: threading.Thread | None = None

    def _resolve_actor(self) -> bool:
        if self._actor is None:
            try:
                self._actor = ray.get_actor(
                    self._actor_name, namespace=MonitorRayActor.NAMESPACE
                )
            except ValueError:
                return False
        return True

    def get_context(self) -> RunContext | None:
        if not self._resolve_actor():
            return None
        # Includes discovery, bounded registration retries and actor scheduling.
        timeout = (
            MonitorServer.SERVICE_DISCOVERY_RETRIES
            * (
                MonitorServer.SERVICE_DISCOVERY_TIMEOUT_SECONDS
                + MonitorServer.SERVICE_DISCOVERY_RETRY_DELAY_SECONDS
            )
            + 30
        )
        context = ray.get(self._actor.get_context.remote(), timeout=timeout)
        if not isinstance(context, RunContext):
            raise RunError("Hub returned an invalid RunContext")
        return context

    def apply_event(self, event: dict[str, Any]) -> None:
        """Submit ``MonitorHubActor.apply_event`` on the actor (fire-and-forget; no ``ray.get``).

        Args:
            event: Serialized monitor event (see ``metric_*`` / ``trace_*`` helpers for shapes).

        Note:
            Errors on the hub side are not surfaced here. Ordering follows Ray actor scheduling.
        """
        with self._lock:
            if self._closed.is_set():
                raise RunError("Monitor client is closed")
            if self._actor is not None and not self._pending:
                self._actor.apply_event.remote(event)
                return
            if len(self._pending) >= 1024:
                raise RunError(
                    "Too many events waiting for configured Run initialization"
                )
            self._pending.append(event)
            if self._waiter is None or not self._waiter.is_alive():
                self._waiter = threading.Thread(
                    target=self._wait_for_hub, name="insight-events", daemon=True
                )
                self._waiter.start()

    def _flush_pending(self) -> bool:
        with self._lock:
            if not self._resolve_actor():
                return False
            while self._pending:
                self._actor.apply_event.remote(self._pending[0])
                self._pending.popleft()
            return True

    def _wait_for_hub(self) -> None:
        delay = 0.1
        while not self._closed.is_set():
            try:
                if self._flush_pending():
                    return
            except ray.exceptions.RayError:
                logger.exception("[rl-insight] Cannot attach pending events to the Run")
            if self._closed.wait(delay):
                return
            delay = min(delay * 2, 2)

    def flush(self) -> None:
        """Drain prior events; only the creator waits for backend collection."""
        if not self._flush_pending():
            return
        # A serial Context read follows this client's prior events. Workers must
        # not each occupy the Hub waiting for another scrape.
        operation = (
            self._actor.prepare_finish.remote(self._owner_token)
            if self._owner_token
            else self._actor.get_context.remote()
        )
        ray.get(operation, timeout=35)

    def close(self) -> None:
        self._closed.set()
        if self._pending and not self._flush_pending():
            logger.warning(
                "[rl-insight] Discarding %d events: no configured Run was initialized",
                len(self._pending),
            )
            self._pending.clear()

        if self._owner_token:
            try:
                ray.get(self._actor.end_run.remote(self._owner_token), timeout=15)
            finally:
                _owned_hubs.pop(_current_job_actor_name(), None)
                self._owner_token = None
