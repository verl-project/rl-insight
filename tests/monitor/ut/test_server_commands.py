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


def test_parser_should_accept_log_dir_for_server_start(tmp_path) -> None:
    log_dir = tmp_path / "rl-insight-data"

    args = cli._build_parser().parse_args(
        ["server", "start", "--log-dir", str(log_dir)]
    )

    assert args.log_dir == log_dir
    assert args.detach is False
    assert args.func.__name__ == "start"


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


def test_parser_should_accept_experiments_commands(tmp_path) -> None:
    parser = cli._build_parser()

    list_args = parser.parse_args(["server", "experiments", "list"])
    show_args = parser.parse_args(
        [
            "server",
            "experiments",
            "show",
            "--project",
            "project-a",
            "--experiment-name",
            "exp-1",
        ]
    )
    archive_args = parser.parse_args(
        [
            "server",
            "experiments",
            "archive",
            "--project",
            "project-a",
            "--experiment-name",
            "exp-1",
            "--server-url",
            "http://host:18080",
        ]
    )

    assert list_args.func.__name__ == "experiments_list"
    assert show_args.func.__name__ == "experiments_show"
    assert archive_args.func.__name__ == "experiments_archive"
    assert archive_args.server_url == "http://host:18080"


def test_experiments_list_should_render_rows_from_the_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [
        {
            "project": "project-a",
            "experiment_name": "exp-1",
            "state": "active",
            "target_count": 2,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    get = MagicMock(return_value=response)
    monkeypatch.setattr(commands_module.requests, "get", get)
    args = commands_module._experiments_parser().parse_args(
        ["list", "--server-url", "http://host:18080/"]
    )

    code = commands_module.ServerCommands().experiments_list(args)

    assert code == 0
    get.assert_called_once_with(
        "http://host:18080/api/v1/experiments",
        params={"project": None},
        timeout=10,
    )
    assert "project-a" in capsys.readouterr().out


def test_experiments_archive_should_post_identity_and_warn_when_not_converged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "project": "project-a",
        "experiment_name": "exp-1",
        "state": "archived",
        "target_count": 1,
        "prometheus_converged": False,
    }
    post = MagicMock(return_value=response)
    monkeypatch.setattr(commands_module.requests, "post", post)
    args = commands_module._experiments_parser().parse_args(
        [
            "archive",
            "--project",
            "project-a",
            "--experiment-name",
            "exp-1",
            "--server-url",
            "http://host:18080",
        ]
    )

    code = commands_module.ServerCommands().experiments_archive(args)

    assert code == 0
    post.assert_called_once_with(
        "http://host:18080/api/v1/experiments/archive",
        json={"project": "project-a", "experiment_name": "exp-1"},
        timeout=10,
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "archived" in out
    assert "did not confirm" in out


def test_experiments_archive_should_fail_on_http_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    response = MagicMock()
    error_response = MagicMock()
    error_response.text = "experiment is archived"
    response.raise_for_status.side_effect = requests.HTTPError(
        "409 Client Error", response=error_response
    )
    post = MagicMock(return_value=response)
    monkeypatch.setattr(commands_module.requests, "post", post)
    args = commands_module._experiments_parser().parse_args(
        [
            "archive",
            "--project",
            "project-a",
            "--experiment-name",
            "exp-1",
            "--server-url",
            "http://host:18080",
        ]
    )

    code = commands_module.ServerCommands().experiments_archive(args)

    assert code == 1
    assert "experiment is archived" in capsys.readouterr().err


def test_experiments_archive_and_restore_help_should_note_same_name_limitation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = commands_module._experiments_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    flattened = " ".join(capsys.readouterr().out.split())
    assert "same experiment name" in flattened
