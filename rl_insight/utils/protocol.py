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

"""Runtime-independent identity shared by collectors and query routes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from .. import __version__


class RunError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:[0-9]{8}T[0-9]{6}Z_)?[0-9a-f]{32}", value
    ):
        raise RunError("Invalid Run ID")
    if "_" in value:
        try:
            datetime.strptime(value.split("_", 1)[0], "%Y%m%dT%H%M%SZ")
        except ValueError as exc:
            raise RunError("Invalid Run timestamp") from exc
    return value


@dataclass(frozen=True)
class RunContext:
    run_id: str
    project: str
    experiment_name: str
    server_url: str
    actor_name: str | None = None
    namespace: str | None = None
    node_ip: str | None = None
    metrics_endpoint: str | None = None
    otlp_endpoint: str | None = None
    prometheus_metrics_enabled: bool | None = None
    otel_traces_enabled: bool | None = None
    events_applied: int | None = None

    @classmethod
    def create(cls, project: str, experiment_name: str, server_url: str) -> RunContext:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex
        return cls(run_id, project, experiment_name, server_url)

    def __post_init__(self) -> None:
        validate_id(self.run_id)
        for field_name in (
            "actor_name",
            "namespace",
            "node_ip",
            "metrics_endpoint",
            "otlp_endpoint",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise RunError("Invalid Context connection field")
        for value in (self.prometheus_metrics_enabled, self.otel_traces_enabled):
            if value is not None and type(value) is not bool:
                raise RunError("Invalid Context status flag")
        if self.events_applied is not None and (
            type(self.events_applied) is not int or self.events_applied < 0
        ):
            raise RunError("Invalid Context event count")
        for name in (self.project, self.experiment_name):
            if not isinstance(name, str) or not name.strip() or len(name) > 256:
                raise RunError(
                    "Run names must be non-empty strings of at most 256 characters"
                )
            if any(ord(c) < 32 for c in name):
                raise RunError("Run names cannot contain control characters")
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
        except (ValueError, TypeError, AttributeError) as exc:
            raise RunError("Invalid server URL") from exc
        object.__setattr__(self, "server_url", self.server_url.rstrip("/"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": __version__,
            **{k: v for k, v in asdict(self).items() if v is not None},
        }

    def metadata(self) -> dict[str, Any]:
        """Persist stable identity only; runtime status is a query-time snapshot."""
        return {
            "version": __version__,
            **{
                k: getattr(self, k)
                for k in ("run_id", "project", "experiment_name", "server_url")
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunContext:
        try:
            if not isinstance(value, Mapping):
                raise TypeError
            return cls(
                **{
                    field.name: value[field.name]
                    for field in fields(cls)
                    if field.name in value
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunError("Invalid RunContext") from exc

    def labels(self, supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
        labels = dict(supplied or {})
        for key in ("run_id", "rl_insight.run_id", "exported_run_id"):
            if key in labels and labels[key] != self.run_id:
                raise RunError("run_id conflicts with RunContext")
        labels.pop("rl_insight.run_id", None)
        labels.pop("exported_run_id", None)
        labels["run_id"] = self.run_id
        return labels
