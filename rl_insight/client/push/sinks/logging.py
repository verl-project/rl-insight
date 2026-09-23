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

"""Built-in sink that logs every emission; useful for tests and local demos."""

from __future__ import annotations

import logging as _stdlib_logging
from typing import Any, Mapping

from omegaconf import OmegaConf

from .base import PushSink

logger = _stdlib_logging.getLogger("rl_insight.push.logging")


class LoggingSink(PushSink):
    """Write each emission as one structured log line at INFO level."""

    def __init__(
        self, prefix: str = "", log: "_stdlib_logging.Logger | None" = None
    ) -> None:
        super().__init__(prefix)
        self._log = log or logger

    def emit_counter(self, name: str, value: float, tags: Mapping[str, str]) -> None:
        self._log.info(
            "[push] counter %s=%s tags=%s", self.metric_name(name), value, dict(tags)
        )

    def emit_store(self, name: str, value: float, tags: Mapping[str, str]) -> None:
        self._log.info(
            "[push] store %s=%s tags=%s", self.metric_name(name), value, dict(tags)
        )

    def emit_timer(self, name: str, value_us: float, tags: Mapping[str, str]) -> None:
        self._log.info(
            "[push] timer %s=%sus tags=%s",
            self.metric_name(name),
            value_us,
            dict(tags),
        )


def create_sink(conf: Any) -> LoggingSink:
    """Factory for the built-in ``logging`` driver; reads ``prefix`` from sink config."""
    prefix = str(OmegaConf.select(conf, "prefix") or "")
    return LoggingSink(prefix=prefix)
