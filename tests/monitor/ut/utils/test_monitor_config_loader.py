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

"""Unit tests for monitor config layering (defaults < RL_INSIGHT_CONFIG < init < env URL)."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from rl_insight.utils.constants import MonitorBackend, MonitorEnv
from rl_insight.utils.monitor_config_loader import load_monitor_config


@pytest.fixture(autouse=True)
def _clear_config_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    monkeypatch.delenv(MonitorEnv.CONFIG_PATH, raising=False)
    monkeypatch.delenv(MonitorEnv.SERVER_URL, raising=False)
    yield


def test_load_monitor_config_should_default_to_ray_backend() -> None:
    conf = load_monitor_config()
    assert conf.server.backend == MonitorBackend.RAY
    assert conf.server.url == ""


def test_load_monitor_config_should_merge_external_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    yaml_path = tmp_path / "push.yaml"
    yaml_path.write_text(
        "server:\n  backend: push\npush:\n  metric_prefix: trainer.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(MonitorEnv.CONFIG_PATH, str(yaml_path))

    conf = load_monitor_config()

    assert conf.server.backend == MonitorBackend.PUSH
    assert conf.push.metric_prefix == "trainer."
    # Defaults not shadowed by unrelated external keys.
    assert conf.server.namespace == "rl_insight_monitor"


def test_load_monitor_config_should_let_init_config_override_external_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    yaml_path = tmp_path / "push.yaml"
    yaml_path.write_text(
        "server:\n  backend: push\npush:\n  metric_prefix: from-file.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(MonitorEnv.CONFIG_PATH, str(yaml_path))

    conf = load_monitor_config({"push": {"metric_prefix": "from-arg."}})

    assert conf.server.backend == MonitorBackend.PUSH
    assert conf.push.metric_prefix == "from-arg."


def test_load_monitor_config_should_ignore_missing_external_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(MonitorEnv.CONFIG_PATH, str(tmp_path / "missing.yaml"))

    conf = load_monitor_config()

    assert conf.server.backend == MonitorBackend.RAY


def test_load_monitor_config_should_ignore_broken_external_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("server: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv(MonitorEnv.CONFIG_PATH, str(bad_path))

    conf = load_monitor_config()

    assert conf.server.backend == MonitorBackend.RAY


def test_load_monitor_config_should_let_server_url_env_win(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    yaml_path = tmp_path / "push.yaml"
    yaml_path.write_text("server:\n  url: http://from-file\n", encoding="utf-8")
    monkeypatch.setenv(MonitorEnv.CONFIG_PATH, str(yaml_path))
    monkeypatch.setenv(MonitorEnv.SERVER_URL, "http://from-env")

    conf = load_monitor_config()

    assert conf.server.url == "http://from-env"
