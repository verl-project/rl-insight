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

"""Unit tests for trainer-side monitor config loading."""

from __future__ import annotations

import pytest

from rl_insight.utils.constants import MonitorBackend
from rl_insight.utils.monitor_config_loader import load_monitor_config


@pytest.fixture(autouse=True)
def clear_monitor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    monkeypatch.delenv("RL_INSIGHT_SERVER_BACKEND", raising=False)


def test_load_monitor_config_should_default_to_ray_backend() -> None:
    conf = load_monitor_config()

    assert conf.server.backend == MonitorBackend.RAY
    assert conf.server.url == ""


def test_load_monitor_config_should_merge_user_config() -> None:
    conf = load_monitor_config({"server": {"backend": "custom", "url": "http://a"}})

    assert conf.server.backend == "custom"
    assert conf.server.url == "http://a"


def test_server_url_env_should_override_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", " http://env:18080 ")

    conf = load_monitor_config({"server": {"url": "http://config"}})

    assert conf.server.url == "http://env:18080"


def test_server_backend_env_should_apply_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Worker processes often call ``rl_insight.init()`` without the trainer
    # config, so the environment is the only way to select their backend.
    monkeypatch.setenv("RL_INSIGHT_SERVER_BACKEND", " custom ")

    conf = load_monitor_config()

    assert conf.server.backend == "custom"


def test_server_backend_env_should_override_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RL_INSIGHT_SERVER_BACKEND", "custom")

    conf = load_monitor_config({"server": {"backend": MonitorBackend.RAY}})

    assert conf.server.backend == "custom"


def test_empty_server_backend_env_should_be_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RL_INSIGHT_SERVER_BACKEND", "  ")

    conf = load_monitor_config({"server": {"backend": "custom"}})

    assert conf.server.backend == "custom"
