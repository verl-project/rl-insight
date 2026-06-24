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

"""Monitor client base class and backend registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

MonitorClientFactory = Callable[[DictConfig], "MonitorClient | None"]

# ``server.backend`` value -> client factory. Populated in ``client.__init__``.
MONITOR_CLIENT_REGISTRY: dict[str, MonitorClientFactory] = {}


class MonitorClient(ABC):
    """Training-side data-collection proxy for monitor events.

    A client runs inside trainer / worker processes, collects metric and trace
    events locally, and forwards them to a collector backend in a separate
    process over RPC. The collector then handles centralized processing such as
    Prometheus export, trace aggregation, or data parsing.
    """

    @abstractmethod
    def apply_event(self, event: dict[str, Any]) -> None:
        """Forward one monitor event to the collector backend."""


def register_monitor_client(backend: str, factory: MonitorClientFactory) -> None:
    """Register a client factory for ``server.backend``."""
    MONITOR_CLIENT_REGISTRY[str(backend)] = factory


def create_monitor_client(conf: DictConfig) -> MonitorClient | None:
    """Create a monitor client from ``server.backend`` in the merged trainer config.

    Args:
        conf: Merged monitor config.

    Returns:
        Backend-specific client, or ``None`` if the factory disables monitoring.

    Raises:
        ValueError: Missing or unknown ``server.backend``.
    """
    backend_type = str(OmegaConf.select(conf, "server.backend") or "").strip()
    if not backend_type:
        raise ValueError("monitor config server.backend is required")

    factory = MONITOR_CLIENT_REGISTRY.get(backend_type)
    if factory is None:
        supported = ", ".join(sorted(MONITOR_CLIENT_REGISTRY)) or "(none installed)"
        raise ValueError(
            f"Unsupported monitor backend {backend_type!r}; supported: {supported}"
        )

    return factory(conf)
