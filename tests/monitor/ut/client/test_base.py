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

"""Unit tests for monitor client registration and selection."""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from rl_insight.client import base


class DummyClient(base.MonitorClient):
    def apply_event(self, event: dict[str, Any]) -> None:
        pass


def test_create_monitor_client_should_use_registered_factory_when_backend_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = DummyClient()
    conf = OmegaConf.create({"server": {"backend": "test"}})
    monkeypatch.setattr(base, "MONITOR_CLIENT_REGISTRY", {})
    base.register_monitor_client(
        "test", lambda received: expected if received is conf else None
    )

    assert base.create_monitor_client(conf) is expected


@pytest.mark.parametrize("backend", [None, "", "   "])
def test_create_monitor_client_should_raise_when_backend_is_missing(
    backend: str | None,
) -> None:
    conf = OmegaConf.create({"server": {"backend": backend}})

    with pytest.raises(ValueError, match="server.backend is required"):
        base.create_monitor_client(conf)


def test_create_monitor_client_should_list_supported_backends_when_backend_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf = OmegaConf.create({"server": {"backend": "missing"}})
    monkeypatch.setattr(base, "MONITOR_CLIENT_REGISTRY", {"ray": lambda _conf: None})

    with pytest.raises(ValueError, match="supported: ray"):
        base.create_monitor_client(conf)
