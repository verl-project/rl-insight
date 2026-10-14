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

"""Unit tests for rendering local server runtime files."""

from __future__ import annotations

import os
import socket
from contextlib import ExitStack

import pytest
import yaml
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module
from rl_insight.utils.monitor_config_loader import load_server_config_file

_render_prometheus_config = runtime_module._render_prometheus_config


def _targets_file(tmp_path):
    return tmp_path / "data" / "targets" / "prometheus-targets.yml"


def _prometheus_conf(source, tmp_path):
    return OmegaConf.create(
        {
            "server": {"data_dir": str(tmp_path / "data")},
            "prometheus": {"config_file": str(source)},
        }
    )


def test_render_prometheus_config_should_reference_and_preserve_file_sd_targets(
    tmp_path,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "global": {"scrape_interval": "10s"},
                "scrape_configs": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    targets_file = _targets_file(tmp_path)
    targets_file.parent.mkdir(parents=True)
    existing_targets = [
        {
            "targets": ["node-a:9100"],
            "labels": {"rl_insight_job": "node-exporter"},
        }
    ]
    targets_file.write_text(yaml.safe_dump(existing_targets), encoding="utf-8")
    conf = _prometheus_conf(source, tmp_path)

    rendered_path = _render_prometheus_config(conf, runtime_dir)
    rendered = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))

    assert yaml.safe_load(targets_file.read_text(encoding="utf-8")) == existing_targets
    assert rendered["scrape_configs"] == [
        {
            "job_name": "rl-insight-dynamic",
            "file_sd_configs": [
                {
                    "files": [str(targets_file.resolve())],
                    "refresh_interval": "5s",
                }
            ],
            "relabel_configs": [
                {
                    "source_labels": ["rl_insight_job"],
                    "target_label": "job",
                },
                {"regex": "rl_insight_job", "action": "labeldrop"},
            ],
        },
        {
            "job_name": "rl-insight-runs",
            "honor_labels": True,
            "file_sd_configs": [
                {
                    "files": [str(targets_file.with_name("run-targets.yml"))],
                    "refresh_interval": "5s",
                }
            ],
        },
    ]


def test_render_prometheus_config_should_migrate_existing_static_targets(
    tmp_path,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    source.write_text(
        yaml.safe_dump({"global": {"scrape_interval": "10s"}, "scrape_configs": []}),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    runtime_config = runtime_dir / "prometheus.yml"
    runtime_config.write_text(
        yaml.safe_dump(
            {
                "scrape_configs": [
                    {
                        "job_name": "node-exporter",
                        "static_configs": [
                            {
                                "targets": ["node-a:9100"],
                                "labels": {"node": "node-a"},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    conf = _prometheus_conf(source, tmp_path)

    _render_prometheus_config(conf, runtime_dir)

    targets_file = _targets_file(tmp_path)
    assert yaml.safe_load(targets_file.read_text(encoding="utf-8")) == [
        {
            "targets": ["node-a:9100"],
            "labels": {
                "rl_insight_job": "node-exporter",
                "node": "node-a",
            },
        }
    ]


def test_render_prometheus_config_should_move_legacy_runtime_targets_to_data(
    tmp_path,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    source.write_text("scrape_configs: []\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    legacy_targets = [
        {
            "targets": ["trainer:9092"],
            "labels": {"rl_insight_job": "trainer_metrics"},
        }
    ]
    (runtime_dir / "prometheus-targets.yml").write_text(
        yaml.safe_dump(legacy_targets), encoding="utf-8"
    )

    _render_prometheus_config(_prometheus_conf(source, tmp_path), runtime_dir)

    assert (
        yaml.safe_load(_targets_file(tmp_path).read_text(encoding="utf-8"))
        == legacy_targets
    )


def test_render_prometheus_config_should_migrate_only_runtime_added_targets(
    tmp_path,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    source_job = {
        "job_name": "node-exporter",
        "metrics_path": "/custom-metrics",
        "static_configs": [{"targets": ["node-a:9100"]}],
    }
    source.write_text(
        yaml.safe_dump({"scrape_configs": [source_job]}, sort_keys=False),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "prometheus.yml").write_text(
        yaml.safe_dump(
            {
                "scrape_configs": [
                    {
                        **source_job,
                        "static_configs": [{"targets": ["node-a:9100", "node-b:9100"]}],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    conf = _prometheus_conf(source, tmp_path)

    rendered_path = _render_prometheus_config(conf, runtime_dir)

    rendered = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    assert rendered["scrape_configs"][0] == source_job
    profile_job = rendered["scrape_configs"][1]
    assert profile_job["metrics_path"] == "/custom-metrics"
    assert profile_job["file_sd_configs"] == [
        {
            "files": [str(_targets_file(tmp_path).resolve())],
            "refresh_interval": "5s",
        }
    ]
    assert profile_job["relabel_configs"][:3] == [
        {
            "source_labels": ["rl_insight_job"],
            "regex": "node\\-exporter",
            "action": "keep",
        },
        {"source_labels": ["rl_insight_job"], "target_label": "job"},
        {"regex": "rl_insight_job", "action": "labeldrop"},
    ]
    assert yaml.safe_load(_targets_file(tmp_path).read_text(encoding="utf-8")) == [
        {
            "targets": ["node-b:9100"],
            "labels": {"rl_insight_job": "node-exporter"},
        }
    ]


@pytest.mark.parametrize(
    "job_name,expected",
    [
        (
            "rl-insight-dynamic",
            [
                "rl-insight-dynamic",
                "rl-insight-dynamic-1",
                "rl-insight-dynamic-2",
                "rl-insight-runs",
            ],
        ),
        (
            "rl-insight-runs",
            [
                "rl-insight-runs",
                "rl-insight-dynamic",
                "rl-insight-dynamic-1",
                "rl-insight-runs-1",
            ],
        ),
    ],
)
def test_render_prometheus_config_should_preserve_colliding_custom_job(
    tmp_path,
    job_name,
    expected,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    custom_job = {
        "job_name": job_name,
        "static_configs": [{"targets": ["custom:9090"]}],
    }
    source.write_text(
        yaml.safe_dump({"scrape_configs": [custom_job]}, sort_keys=False),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    conf = _prometheus_conf(source, tmp_path)

    rendered_path = _render_prometheus_config(conf, runtime_dir)

    jobs = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))["scrape_configs"]
    assert jobs[0] == custom_job
    assert [job["job_name"] for job in jobs] == expected


def test_render_prometheus_config_should_migrate_targets_from_colliding_job(
    tmp_path,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    source_job = {
        "job_name": "rl-insight-dynamic",
        "static_configs": [{"targets": ["node-a:9100"]}],
    }
    source.write_text(
        yaml.safe_dump({"scrape_configs": [source_job]}, sort_keys=False),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "prometheus.yml").write_text(
        yaml.safe_dump(
            {
                "scrape_configs": [
                    {
                        **source_job,
                        "static_configs": [{"targets": ["node-a:9100", "node-b:9100"]}],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    conf = _prometheus_conf(source, tmp_path)

    _render_prometheus_config(conf, runtime_dir)

    assert yaml.safe_load(_targets_file(tmp_path).read_text(encoding="utf-8")) == [
        {
            "targets": ["node-b:9100"],
            "labels": {"rl_insight_job": "rl-insight-dynamic"},
        }
    ]


def test_render_prometheus_config_should_not_leave_partial_targets_on_write_failure(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "source-prometheus.yml"
    source.write_text("scrape_configs: []\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    targets_file = _targets_file(tmp_path)
    real_replace = os.replace

    def fail_targets_replace(source_path, target_path) -> None:
        if target_path == targets_file.resolve():
            raise OSError("simulated interrupted migration")
        real_replace(source_path, target_path)

    monkeypatch.setattr(runtime_module.os, "replace", fail_targets_replace)
    conf = _prometheus_conf(source, tmp_path)

    with pytest.raises(OSError, match="simulated interrupted migration"):
        _render_prometheus_config(conf, runtime_dir)

    assert not targets_file.exists()


@pytest.mark.parametrize("udp", [False, True])
def test_auto_ports_and_rendered_config(tmp_path, udp):
    conf = load_server_config_file()
    keys = [
        "server.port",
        "prometheus.prometheus_port",
        "grafana.port",
        "tempo.query_port",
        "otel.otel_port",
        "tempo.grpc_port",
        "tempo.memberlist_port",
    ]
    with ExitStack() as sockets:
        occupied = sockets.enter_context(
            socket.socket(type=socket.SOCK_DGRAM if udp else socket.SOCK_STREAM)
        )
        occupied.bind(("0.0.0.0", 0))
        port = occupied.getsockname()[1]
        for key in keys:
            OmegaConf.update(conf, key, port)
        runtime_module.LocalServiceRuntime(conf, tmp_path)._assign_ports()
        assigned = [OmegaConf.select(conf, key) for key in keys]
        assert len(set(assigned)) == 7
        assert conf.tempo.memberlist_port != port
        if not udp:
            assert port not in assigned
        for value in assigned:
            sockets.enter_context(socket.socket()).bind(("0.0.0.0", value))
        rendered = runtime_module._render_tempo_config(
            conf, tmp_path, tmp_path, "2.6.1"
        )
        tempo = yaml.safe_load(rendered.read_text())
        assert tempo["server"]["grpc_listen_port"] == conf.tempo.grpc_port
        assert tempo["memberlist"]["bind_port"] == conf.tempo.memberlist_port


@pytest.mark.parametrize("races", [0, 1, 3])
def test_auto_port_real_start_retry_and_cleanup(tmp_path, monkeypatch, races):
    conf = load_server_config_file()
    for name in ("prometheus", "tempo", "grafana"):
        conf[name].enable = False
    conf.server.runtime_dir = str(tmp_path / "runtime")
    conf.server.state_file = str(tmp_path / "state.json")
    conf.server.data_dir = str(tmp_path / "data")
    runtime = runtime_module.LocalServiceRuntime(conf, tmp_path)
    spawn_service = runtime_module._spawn_service
    processes = []
    with ExitStack() as sockets:
        occupied = sockets.enter_context(socket.socket())
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        conf.server.port = occupied.getsockname()[1]

        def spawn(name, command, log_file):
            if len(processes) < races:
                blocker = sockets.enter_context(socket.socket())
                blocker.bind(("0.0.0.0", conf.server.port))
                blocker.listen()
            process = spawn_service(name, command, log_file)
            processes.append(process)
            return process

        monkeypatch.setattr(runtime_module, "_spawn_service", spawn)
        try:
            if races == 3:
                with pytest.raises(runtime_module.PortConflictError):
                    runtime.start(detach=True, attach_logs=False, auto_port=True)
            else:
                runtime.start(detach=True, attach_logs=False, auto_port=True)
                assert conf.server.port != occupied.getsockname()[1]
                assert (
                    OmegaConf.load(tmp_path / "runtime/server.yaml").server.port
                    == conf.server.port
                )
                assert runtime.stop()[0] == 0
            assert len(processes) == min(races + 1, 3)
            for process in processes:
                process.wait(timeout=5)
            assert not runtime.state_file.exists()
        finally:
            for process in processes:
                runtime_module._terminate_process(process)
