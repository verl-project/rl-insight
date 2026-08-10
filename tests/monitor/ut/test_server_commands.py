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
from unittest.mock import MagicMock, call

import pytest
import requests
from omegaconf import OmegaConf
from rl_insight import cli
from rl_insight.server import commands as commands_module
from rl_insight.utils.prometheus_utils import PrometheusTarget


def test_parser_should_accept_targets_add_command(tmp_path) -> None:
    target_file = tmp_path / "targets.yaml"

    args = cli._build_parser().parse_args(
        ["server", "targets", "add", str(target_file)]
    )

    assert args.target_file == target_file
    assert args.func.__name__ == "add_targets"


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
    store = MagicMock()
    monkeypatch.setattr(
        commands_module.PrometheusTargetStore,
        "from_config",
        MagicMock(return_value=store),
    )

    result = commands_module.ServerCommands().add_targets(
        argparse.Namespace(target_file=target_file, config=None)
    )

    assert result == 0
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


def test_start_should_append_diagnostics_without_changing_success_behavior(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    conf = OmegaConf.create({"server": {"enable": True}})
    stack = MagicMock()
    stack.services = []
    manager = MagicMock()
    manager.active_state.return_value = None
    manager.check_dependencies.return_value = []
    manager.missing_dependencies.return_value = []
    manager.start.return_value = stack
    monkeypatch.setattr(
        commands_module.ServerCommands,
        "_load_config",
        staticmethod(lambda _args: conf),
    )
    monkeypatch.setattr(
        commands_module.ServerCommands,
        "_stack_management_enabled",
        staticmethod(lambda _conf, action: True),
    )
    monkeypatch.setattr(
        commands_module, "ServerServiceManager", MagicMock(return_value=manager)
    )
    console = MagicMock()
    validator = MagicMock()
    diagnostics_report = MagicMock()
    diagnostics_runner = MagicMock(return_value=diagnostics_report)

    result = commands_module.ServerCommands(
        validator=validator,
        console=console,
        diagnostics_runner=diagnostics_runner,
    ).start(argparse.Namespace(detach=True, attach_logs=False, config=None))

    assert result == 0
    validator.validate_start.assert_called_once_with(conf)
    console.print_running_summary.assert_called_once_with(conf, stack.services)
    diagnostics_runner.assert_called_once_with(conf, stack.services)
    console.print_startup_diagnostics.assert_called_once_with(diagnostics_report)
    assert "running in background mode" in capsys.readouterr().out


def test_start_should_keep_success_when_diagnostics_raise(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    conf = OmegaConf.create({"server": {"enable": True}})
    stack = MagicMock()
    stack.services = []
    manager = MagicMock()
    manager.active_state.return_value = None
    manager.check_dependencies.return_value = []
    manager.missing_dependencies.return_value = []
    manager.start.return_value = stack
    monkeypatch.setattr(
        commands_module.ServerCommands,
        "_load_config",
        staticmethod(lambda _args: conf),
    )
    monkeypatch.setattr(
        commands_module.ServerCommands,
        "_stack_management_enabled",
        staticmethod(lambda _conf, action: True),
    )
    monkeypatch.setattr(
        commands_module, "ServerServiceManager", MagicMock(return_value=manager)
    )
    console = MagicMock()
    validator = MagicMock()
    diagnostics_runner = MagicMock(side_effect=RuntimeError("diagnostic failure"))

    result = commands_module.ServerCommands(
        validator=validator,
        console=console,
        diagnostics_runner=diagnostics_runner,
    ).start(argparse.Namespace(detach=True, attach_logs=False, config=None))

    captured = capsys.readouterr()
    assert result == 0
    validator.validate_start.assert_called_once_with(conf)
    console.print_running_summary.assert_called_once_with(conf, stack.services)
    console.print_startup_diagnostics.assert_not_called()
    assert "Startup diagnostics unavailable: diagnostic failure" in captured.err
    assert "running in background mode" in captured.out
