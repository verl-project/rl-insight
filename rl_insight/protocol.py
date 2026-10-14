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
import ipaddress
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit


class RunError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise RunError("Invalid Run/request ID")
    return value


@dataclass(frozen=True)
class RunContext:
    run_id: str
    project: str
    experiment_name: str
    server_url: str

    def __post_init__(self) -> None:
        validate_id(self.run_id)
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
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunContext:
        try:
            if value.get("version") != 1:
                raise ValueError
            return cls(
                **{
                    name: value[name]
                    for name in ("run_id", "project", "experiment_name", "server_url")
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


def proof_headers(context: RunContext, incarnation: str) -> dict[str, str]:
    validate_id(incarnation)
    return {"X-RLInsight-Run": context.run_id, "X-RLInsight-Incarnation": incarnation}


def endpoint_description(
    context: RunContext, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a publisher's facts, independent of any framework or runtime."""
    for key in ("component", "instance"):
        name = value.get(key)
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 256
            or any(ord(c) < 32 for c in name)
        ):
            raise RunError("Invalid endpoint " + key)
    incarnation = validate_id(value.get("incarnation", ""))
    url = value.get("metrics_url")
    if not isinstance(url, str):
        raise RunError("Invalid metrics URL")
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
            or parsed.query
            or any(c.isspace() for c in url)
        ):
            raise ValueError
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
        if "%" in parsed.hostname:
            raise ValueError
        ipaddress.ip_address(parsed.hostname)
    except (ValueError, TypeError, AttributeError) as exc:
        raise RunError(
            "Metrics URL must contain a literal IPv4/IPv6 address and valid port"
        ) from exc
    supplied = value.get("labels", {})
    if not isinstance(supplied, Mapping):
        raise RunError("Endpoint labels must be an object")
    labels = context.labels(supplied)
    for key, val in labels.items():
        if (
            not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", key)
            or key.startswith("__")
            or not isinstance(val, (str, int, float, bool))
        ):
            raise RunError("Invalid endpoint label")
    if any(
        key in labels
        for key in ("endpoint_id", "component", "exported_job", "exported_instance")
    ):
        raise RunError("Reserved endpoint label")
    return {
        "run_id": context.run_id,
        "component": value["component"],
        "instance": value["instance"],
        "incarnation": incarnation,
        "metrics_url": url,
        "labels": {k: str(v) for k, v in labels.items()},
    }
