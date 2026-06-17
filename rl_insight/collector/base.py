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

"""Monitor collector base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MonitorCollector(ABC):
    """Collector backend that receives monitor events from training-side clients.

    A collector runs as a separate process or service. Clients forward metric and
    trace events over RPC; the collector performs centralized processing such as
    Prometheus export, trace aggregation, or downstream parsing.
    """

    @abstractmethod
    def apply_event(self, event: dict[str, Any]) -> None:
        """Handle one monitor event forwarded by a client."""

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Return a small status snapshot for debugging or health checks."""
