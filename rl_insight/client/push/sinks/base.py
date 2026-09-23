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

"""Sink interface for the direct-emit (push) backend.

A sink receives metric names **relative to its own** :attr:`prefix`; the push
client supplies the stream segment (e.g. ``trainer.`` / ``trace.`` /
``rollout.``) and the sink prepends its deployment-specific root prefix.
Timers always use microseconds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

TagMap = Mapping[str, str]


class PushSink(ABC):
    """Abstract destination for direct metric emission.

    Args:
        prefix: Deployment-specific root prefix prepended to every metric.
    """

    def __init__(self, prefix: str = "") -> None:
        self._prefix = str(prefix or "").strip(".")

    @property
    def prefix(self) -> str:
        """Root metric prefix (no trailing dot)."""
        return self._prefix

    def metric_name(self, suffix: str) -> str:
        """Return the fully qualified metric name for a prefix-relative ``suffix``."""
        suffix = str(suffix).lstrip(".")
        return f"{self._prefix}.{suffix}" if self._prefix else suffix

    @abstractmethod
    def emit_counter(self, name: str, value: float, tags: TagMap) -> None:
        """Add ``value`` to a monotonic counter identified by ``name``."""

    @abstractmethod
    def emit_store(self, name: str, value: float, tags: TagMap) -> None:
        """Record the latest gauge ``value`` for ``name``."""

    @abstractmethod
    def emit_timer(self, name: str, value_us: float, tags: TagMap) -> None:
        """Record a duration ``value_us`` (microseconds) for ``name``."""

    def flush(self) -> None:
        """Best-effort flush of any buffered points. Default: no-op."""

    def close(self) -> None:
        """Release resources. Default: flush then no-op."""
        self.flush()
