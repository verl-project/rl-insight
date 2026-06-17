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

from ..utils import MonitorBackend
from .base import create_monitor_client, register_monitor_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["create_monitor_client"]

# Built-in clients. Keys must match ``server.backend`` in trainer config.
try:
    from .ray_monitor_client import create_ray_monitor_client

    register_monitor_client(MonitorBackend.RAY, create_ray_monitor_client)
except ImportError as exc:
    logger.debug("Ray monitor client is unavailable: %s", exc)
