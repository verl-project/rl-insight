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

"""Monitor client factory."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..utils import MonitorBackend
from ..utils.protocol import RunContext
from .base import MonitorClient, create_monitor_client, register_monitor_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["create_monitor_client", "register_targets"]

# Built-in clients. Keys must match ``server.backend`` in trainer config.
try:
    from .ray_monitor_client import MonitorRayClient, create_ray_monitor_client

    register_monitor_client(MonitorBackend.RAY, create_ray_monitor_client)
except ImportError as exc:
    MonitorRayClient = None  # type: ignore[misc,assignment]
    logger.debug("[rl-insight] Ray monitor client is unavailable: %s", exc)

try:
    from .push.client import PushMonitorClient, create_push_monitor_client

    register_monitor_client(MonitorBackend.PUSH, create_push_monitor_client)
except ImportError as exc:  # pragma: no cover - push only needs lightweight deps
    PushMonitorClient = None  # type: ignore[misc,assignment]
    logger.debug("[rl-insight] Push monitor client is unavailable: %s", exc)


def register_targets(
    client: MonitorClient,
    server_addresses: list[str],
    job_name: str | None,
    labels: list[Mapping[str, Any] | None] | None,
    *,
    context: RunContext | None = None,
) -> None:
    """Register scrape targets with the selected monitor client."""
    if MonitorRayClient is not None and isinstance(client, MonitorRayClient):
        import ray

        try:
            ray.get(
                client._actor.submit_targets.remote(
                    server_addresses, job_name, labels, context=context
                ),
                timeout=30,
            )
        except Exception:
            logger.exception(
                "[rl-insight] Cannot register Prometheus targets through Ray"
            )
    elif PushMonitorClient is not None and isinstance(client, PushMonitorClient):
        poller = client._poller
        if poller is not None:
            poller.targets.set_targets(server_addresses, labels)
            logger.info(
                "[rl-insight] Registered %d push scrape targets in-process",
                len(server_addresses),
            )
