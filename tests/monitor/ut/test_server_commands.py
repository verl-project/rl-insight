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

"""Unit tests for RL-Insight server command handlers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import requests
from omegaconf import DictConfig, OmegaConf

from rl_insight import cli
from rl_insight.server import commands as commands_module
from rl_insight.utils.prometheus_utils import PrometheusTarget


def _patch_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        commands_module,
        "local_addresses",
        lambda: {
            "ipv4": "10.0.0.5",
            "ipv6": "",
            "host": "10.0.0.5",
            "bind": "0.0.0.0",
            "loopback": "127.0.0.1",
            "family": "ipv4",
        },
    )


def _stack_config(
    *,
    server_port: int = 18080,
    prometheus_port: int = 9090,
    tempo_port: int = 3200,
    otel_port: int = 4318,
    grafana_port: int = 3000,
    grafana: bool = True,
) -> DictConfig:
    return OmegaConf.create(
        {
            "server": {"enable": True, "backend": "local", "port": server_port},
            "prometheus": {"enable": True, "prometheus_port": prometheus_port},
            "tempo": {"enable": True, "query_port": tempo_port},
            "otel": {"otel_port": otel_port},
            "grafana": {"enable": grafana, "port": grafana_port},
        }
    )


def _running_state(
    runtime_dir: Path, services: list[tuple[str, int, str]]
) -> dict[str, object]:
    return {
        "runtime_dir": str(runtime_dir),
        "services": [
            {"name": name, "pid": pid, "log_file": log} for name, pid, log in services
        ],
    }


def _start_args(config: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        extra_dashboard_dir=None,
        log_dir=None,
        detach=True,
        attach_logs=False,
        auto_port=False,
    )


def test_parser_should_accept_targets_add_command(tmp_path) -> None:
    target_file = tmp_path / "targets.yaml"

    args = cli._build_parser().parse_args(
        ["server", "targets", "add", str(target_file)]
    )

    assert args.target_file == target_file
    assert args.func.__name__ == "add_targets"


@pytest.mark.parametrize("flags", [[], ["--auto-port", "--detach"]])
def test_parser_should_accept_log_dir_for_server_start(tmp_path, flags) -> None:
    log_dir = tmp_path / "rl-insight-data"

    args = cli._build_parser().parse_args(
        ["server", "start", "--log-dir", str(log_dir), *flags]
    )

    assert args.log_dir == log_dir
    assert args.detach == bool(flags)
    assert args.auto_port == bool(flags)
    assert args.func.__name__ == "start"


def test_parser_should_accept_relative_extra_dashboard_dir() -> None:
    args = cli._build_parser().parse_args(
        ["server", "start", "--extra-dashboard-dir", "relative/dashboards"]
    )

    assert args.extra_dashboard_dir == Path("relative/dashboards")


def test_apply_log_dir_should_override_server_data_dir(tmp_path) -> None:
    conf = OmegaConf.create({"server": {"data_dir": None}})

    commands_module.ServerCommands._apply_log_dir(conf, tmp_path / "logs")

    assert conf.server.data_dir == str(tmp_path / "logs")


def test_data_dir_should_resolve_explicit_and_default_paths(tmp_path) -> None:
    explicit = OmegaConf.create({"server": {"data_dir": str(tmp_path / "logs")}})
    default = OmegaConf.create({"server": {}})

    assert commands_module.ServerCommands._data_dir(explicit) == tmp_path / "logs"
    assert commands_module.ServerCommands._data_dir(default) == (
        commands_module.DEFAULT_STATE_ROOT / "data"
    )


def test_add_targets_should_register_each_job_and_reload_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target_file = tmp_path / "targets.yaml"
    target_file.write_text(
        """
jobs:
  - job_name: npu-exporter
    targets:
      - target: node-a:8082
        labels:
          node: node-a
  - job_name: node-exporter
    targets:
      - node-a:9100
""".strip(),
        encoding="utf-8",
    )
    runtime_config = OmegaConf.create({"prometheus": {"prometheus_port": 54321}})
    OmegaConf.save(runtime_config, tmp_path / "server.yaml")
    monkeypatch.setattr(
        commands_module.ServerServiceManager,
        "active_state",
        lambda self: {"runtime_dir": str(tmp_path)},
    )
    store = MagicMock()
    factory = MagicMock(return_value=store)
    monkeypatch.setattr(
        commands_module.PrometheusTargetStore,
        "from_config",
        factory,
    )

    result = commands_module.ServerCommands().add_targets(
        argparse.Namespace(target_file=target_file, config=None)
    )

    assert result == 0
    actual = factory.call_args.args[0]
    assert actual.prometheus.prometheus_port == 54321
    assert store.register.call_args_list == [
        call(
            "npu-exporter",
            [PrometheusTarget("node-a:8082", {"node": "node-a"})],
        ),
        call(
            "node-exporter",
            [PrometheusTarget("node-a:9100")],
        ),
    ]
    store.reload.assert_called_once_with()


def test_add_targets_should_reject_empty_jobs_before_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    target_file = tmp_path / "targets.yaml"
    target_file.write_text("jobs: []\n", encoding="utf-8")
    store_factory = MagicMock()
    monkeypatch.setattr(
        commands_module.PrometheusTargetStore, "from_config", store_factory
    )

    result = commands_module.ServerCommands().add_targets(
        argparse.Namespace(target_file=target_file, config=None)
    )

    assert result == 2
    store_factory.assert_not_called()
    assert "jobs must be a non-empty list" in capsys.readouterr().err


def test_add_targets_should_fail_when_reload_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    target_file = tmp_path / "targets.yaml"
    target_file.write_text(
        "jobs:\n  - job_name: node-exporter\n    targets: [node-a:9100]\n",
        encoding="utf-8",
    )
    store = MagicMock()
    store.reload.side_effect = requests.ConnectionError("connection refused")
    monkeypatch.setattr(
        commands_module.PrometheusTargetStore,
        "from_config",
        MagicMock(return_value=store),
    )

    result = commands_module.ServerCommands().add_targets(
        argparse.Namespace(target_file=target_file, config=None)
    )

    assert result == 1
    assert "Failed to add Prometheus targets" in capsys.readouterr().err


def test_start_should_print_full_details_when_services_already_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _patch_addresses(monkeypatch)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    OmegaConf.save(_stack_config(), runtime_dir / "server.yaml")
    state = _running_state(
        runtime_dir,
        [
            ("prometheus", 4242, "/logs/prometheus.log"),
            ("tempo", 4243, "/logs/tempo.log"),
        ],
    )
    monkeypatch.setattr(
        commands_module.ServerServiceManager, "active_state", lambda self: state
    )
    start = MagicMock()
    check = MagicMock()
    monkeypatch.setattr(commands_module.ServerServiceManager, "start", start)
    monkeypatch.setattr(
        commands_module.ServerServiceManager, "check_dependencies", check
    )

    result = commands_module.ServerCommands().start(_start_args())

    out = capsys.readouterr().out
    assert result == 0
    assert "already appear to be running" in out
    assert re.search(r"Status\s+running", out)
    for name, port in (
        ("RL-Insight server", 18080),
        ("Prometheus", 9090),
        ("Tempo", 3200),
        ("OTLP traces", 4318),
        ("Grafana", 3000),
    ):
        assert name in out
        assert re.search(rf"\b{port}\b", out)
    assert "4242" in out
    assert "/logs/prometheus.log" in out
    assert "4243" in out
    assert "/logs/tempo.log" in out
    assert "RL_INSIGHT_SERVER_URL" in out
    assert re.search(r"http://10\.0\.0\.5:18080", out)
    assert "http://127.0.0.1:3000" in out
    assert "http://10.0.0.5:3000" in out
    start.assert_not_called()
    check.assert_not_called()


def test_start_should_use_runtime_ports_when_services_already_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _patch_addresses(monkeypatch)
    config_file = tmp_path / "cli.yaml"
    config_file.write_text(
        "server:\n  port: 18080\n"
        "prometheus:\n  prometheus_port: 9090\n"
        "tempo:\n  query_port: 3200\n"
        "otel:\n  otel_port: 4318\n"
        "grafana:\n  port: 3000\n",
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    OmegaConf.save(
        _stack_config(
            server_port=18181,
            prometheus_port=19090,
            tempo_port=13200,
            otel_port=14318,
            grafana_port=13000,
        ),
        runtime_dir / "server.yaml",
    )
    monkeypatch.setattr(
        commands_module.ServerServiceManager,
        "active_state",
        lambda self: _running_state(runtime_dir, [("prometheus", 5151, "p.log")]),
    )
    monkeypatch.setattr(commands_module.ServerServiceManager, "start", MagicMock())

    result = commands_module.ServerCommands().start(_start_args(config_file))

    out = capsys.readouterr().out
    assert result == 0
    for port in (18181, 19090, 13200, 14318, 13000):
        assert re.search(rf"\b{port}\b", out)
    for port in (18080, 9090, 3200, 4318, 3000):
        assert not re.search(rf"\b{port}\b", out)
    assert "http://127.0.0.1:13000" in out


def test_start_should_fall_back_when_runtime_config_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _patch_addresses(monkeypatch)
    monkeypatch.setattr(
        commands_module.ServerServiceManager,
        "active_state",
        lambda self: _running_state(tmp_path, [("prometheus", 6161, "p.log")]),
    )
    monkeypatch.setattr(commands_module.ServerServiceManager, "start", MagicMock())

    result = commands_module.ServerCommands().start(_start_args())

    out = capsys.readouterr().out
    assert result == 0
    assert "already appear to be running" in out
    assert re.search(r"\b18080\b", out)
    assert re.search(r"http://10\.0\.0\.5:18080", out)


def test_start_should_print_details_when_start_races_with_running_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    _patch_addresses(monkeypatch)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    OmegaConf.save(_stack_config(), runtime_dir / "server.yaml")
    state = _running_state(runtime_dir, [("prometheus", 7171, "/logs/p.log")])
    states: list[dict[str, object] | None] = [None, state]
    monkeypatch.setattr(
        commands_module.ServerServiceManager,
        "active_state",
        lambda self: states.pop(0),
    )
    start = MagicMock(return_value=None)
    monkeypatch.setattr(commands_module.ServerServiceManager, "start", start)
    monkeypatch.setattr(
        commands_module.ServerServiceManager,
        "check_dependencies",
        MagicMock(return_value=[]),
    )

    result = commands_module.ServerCommands().start(_start_args())

    out = capsys.readouterr().out
    assert result == 0
    start.assert_called_once()
    assert out.count("already appear to be running") == 1
    assert "7171" in out
    assert "/logs/p.log" in out
    assert re.search(r"http://10\.0\.0\.5:18080", out)
    assert re.search(r"Status\s+running", out)


def test_print_running_summary_should_accept_plain_service_rows(capsys) -> None:
    conf = _stack_config(grafana=False)

    commands_module.ServerConsole().print_running_summary(
        conf,
        [("prometheus", 8181, "/logs/prometheus.log"), ("tempo", 8182, "t.log")],
    )

    out = capsys.readouterr().out
    assert "prometheus" in out
    assert "8181" in out
    assert "/logs/prometheus.log" in out
    assert "8182" in out
    assert "t.log" in out
    assert "RL_INSIGHT_SERVER_URL" in out
    assert "dashboard" not in out
