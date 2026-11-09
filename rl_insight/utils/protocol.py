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

"""Execution identity and live status shared by a job's Hub and workers."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from packaging.version import InvalidVersion, Version

from .. import __version__

logger = logging.getLogger(__name__)

# This global minimum stays fixed when the package version increases.
MIN_SUPPORTED_VERSION = Version("0.4.0")
SDK_VERSION_HEADER = "X-RL-Insight-Client-Version"


class InvalidVersionError(ValueError):
    """The peer does not provide a supported RL-Insight version."""


def create_execution_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S_") + uuid4().hex[:8]


def require_version(value: Any, component: str) -> None:
    try:
        supported = isinstance(value, str) and Version(value) >= MIN_SUPPORTED_VERSION
    except InvalidVersion:
        supported = False
    if not supported:
        message = (
            f"[rl-insight] RL-Insight requires {component} >= {MIN_SUPPORTED_VERSION}; "
            f"received version {value!r}. Upgrade the RL-Insight server and SDK on "
            f"the driver and all workers to >= {MIN_SUPPORTED_VERSION}, "
            "then restart the training job."
        )
        logger.error("%s", message)
        raise InvalidVersionError(message)


@dataclass(frozen=True)
class RunContext:
    project: str
    experiment_name: str
    server_url: str
    original_experiment_name: str | None = None
    actor_name: str | None = None
    namespace: str | None = None
    node_ip: str | None = None
    metrics_endpoint: str | None = None
    otlp_endpoint: str | None = None
    prometheus_metrics_enabled: bool | None = None
    otel_traces_enabled: bool | None = None
    events_applied: int | None = None
    version: str = field(default_factory=lambda: __version__)

    @classmethod
    def create(
        cls,
        project: str,
        experiment_name: str,
        server_url: str,
        *,
        suffix: str | None = None,
    ) -> RunContext:
        suffix = suffix if suffix is not None else create_execution_suffix()
        return cls(
            project,
            f"{experiment_name}_{suffix}",
            server_url,
            original_experiment_name=experiment_name,
        )

    def __post_init__(self) -> None:
        for name in (
            self.project,
            self.original_experiment_name
            if self.original_experiment_name is not None
            else self.experiment_name,
        ):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("[rl-insight] Run names must be non-empty strings")
        try:
            url = urlsplit(self.server_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or any(c.isspace() for c in self.server_url)
            ):
                raise ValueError
            if url.port is not None and not 1 <= url.port <= 65535:
                raise ValueError
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError("[rl-insight] Invalid server URL") from e
        object.__setattr__(self, "server_url", self.server_url.rstrip("/"))

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def labels(self, supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
        labels = dict(supplied or {})
        labels.update(project=self.project, experiment_name=self.experiment_name)
        return labels
