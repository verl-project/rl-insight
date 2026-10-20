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

"""Unit tests for monitor config layering.

Precedence (lowest to highest): built-in defaults, the external YAML named by
``RL_INSIGHT_CONFIG``, the explicit ``init(config)`` mapping, and finally the
``RL_INSIGHT_SERVER_URL`` / ``RL_INSIGHT_SERVER_BACKEND`` /
``RL_INSIGHT_HARDWARE_ENABLED`` environment variables.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from rl_insight.utils.constants import MonitorBackend, MonitorEnv, MonitorHardware
from rl_insight.utils.monitor_config_loader import load_monitor_config


@pytest.fixture(autouse=True)
def _clear_config_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    monkeypatch.delenv(MonitorEnv.CONFIG_PATH, raising=False)
    monkeypatch.delenv(MonitorEnv.SERVER_URL, raising=False)
    monkeypatch.delenv(MonitorEnv.SERVER_BACKEND, raising=False)
    monkeypatch.delenv(MonitorEnv.HARDWARE_ENABLED, raising=False)
    yield


def test_load_monitor_config_should_default_to_ray_backend() -> None:
    conf = load_monitor_config()
    assert conf.server.backend == MonitorBackend.RAY
    assert conf.server.url == ""


def test_load_monitor_config_should_merge_user_config() -> None:
    conf = load_monitor_config({"server": {"backend": "custom", "url": "http://a"}})

    assert conf.server.backend == "custom"
    assert conf.server.url == "http://a"


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


def test_server_url_env_should_override_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MonitorEnv.SERVER_URL, " http://env:18080 ")

    conf = load_monitor_config({"server": {"url": "http://config"}})

    assert conf.server.url == "http://env:18080"


def test_load_monitor_config_should_let_server_url_env_win(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    yaml_path = tmp_path / "push.yaml"
    yaml_path.write_text("server:\n  url: http://from-file\n", encoding="utf-8")
    monkeypatch.setenv(MonitorEnv.CONFIG_PATH, str(yaml_path))
    monkeypatch.setenv(MonitorEnv.SERVER_URL, "http://from-env")

    conf = load_monitor_config()

    assert conf.server.url == "http://from-env"


def test_server_backend_env_should_apply_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Worker processes often call ``rl_insight.init()`` without the trainer
    # config, so the environment is the only way to select their backend.
    monkeypatch.setenv(MonitorEnv.SERVER_BACKEND, " custom ")

    conf = load_monitor_config()

    assert conf.server.backend == "custom"


def test_server_backend_env_should_override_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MonitorEnv.SERVER_BACKEND, "custom")

    conf = load_monitor_config({"server": {"backend": MonitorBackend.RAY}})

    assert conf.server.backend == "custom"


def test_empty_server_backend_env_should_be_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MonitorEnv.SERVER_BACKEND, "  ")

    conf = load_monitor_config({"server": {"backend": "custom"}})

    assert conf.server.backend == "custom"


def test_load_monitor_config_should_default_hardware_section() -> None:
    conf = load_monitor_config()

    assert conf.hardware.enabled == MonitorHardware.ENABLED
    assert conf.hardware.interval_seconds == MonitorHardware.INTERVAL_SECONDS
    assert list(conf.hardware.collectors) == list(MonitorHardware.COLLECTORS)
    assert conf.hardware.disk_path == MonitorHardware.DISK_PATH
    assert conf.hardware.worker_role == MonitorHardware.WORKER_ROLE


def test_load_monitor_config_should_merge_hardware_overrides() -> None:
    conf = load_monitor_config(
        {
            "server": {"url": "http://a"},
            "hardware": {
                "enabled": False,
                "interval_seconds": 5,
                "collectors": ["cpu"],
                "disk_path": "/data",
            },
        }
    )

    assert conf.hardware.enabled is False
    assert conf.hardware.interval_seconds == 5
    assert list(conf.hardware.collectors) == ["cpu"]
    assert conf.hardware.disk_path == "/data"


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", " OFF "])
def test_hardware_enabled_env_should_disable(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(MonitorEnv.HARDWARE_ENABLED, value)

    conf = load_monitor_config()

    assert conf.hardware.enabled is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_hardware_enabled_env_should_enable_over_config(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(MonitorEnv.HARDWARE_ENABLED, value)

    conf = load_monitor_config({"hardware": {"enabled": False}})

    assert conf.hardware.enabled is True


def test_empty_hardware_enabled_env_should_be_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MonitorEnv.HARDWARE_ENABLED, "  ")

    conf = load_monitor_config({"hardware": {"enabled": False}})

    assert conf.hardware.enabled is False


def test_invalid_hardware_enabled_env_should_be_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MonitorEnv.HARDWARE_ENABLED, "maybe")

    conf = load_monitor_config()

    assert conf.hardware.enabled == MonitorHardware.ENABLED
