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
