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

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..utils import MonitorBackend

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["create_monitor_client"]


def create_monitor_client(conf: DictConfig) -> Any | None:
    """Factory: pick backend from ``conf.backend`` and return the implementation client.

    Args:
        conf: Merged monitor config; must set ``backend.type`` (only ``ray`` is supported).

    Returns:
        Backend-specific client, or ``None`` if optional Ray deps fail to import.

    Raises:
        ValueError: Unknown backend or missing ``backend.type``.
    """
    backend = conf.backend
    if OmegaConf.is_config(backend):
        typ = backend.get("type")
        if typ is None:
            raise ValueError("monitor config backend.type is required")
        backend_type = str(typ)
    else:
        backend_type = str(backend)
    if backend_type != MonitorBackend.RAY:
        raise ValueError(f"Unsupported monitor backend: {backend_type!r}")

    try:
        from .ray_monitor_client import (
            create_monitor_client as create_ray_monitor_client,
        )
    except ImportError as e:
        logger.warning(
            "Ray monitor client is unavailable; monitoring is disabled: %s", e
        )
        return None

    return create_ray_monitor_client(conf)
