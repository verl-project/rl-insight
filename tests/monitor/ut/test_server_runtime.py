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

import pytest
import yaml
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module


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
        }
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


def test_render_prometheus_config_should_preserve_colliding_custom_job(
    tmp_path,
) -> None:
    source = tmp_path / "source-prometheus.yml"
    custom_job = {
        "job_name": "rl-insight-dynamic",
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
    assert [job["job_name"] for job in jobs] == [
        "rl-insight-dynamic",
        "rl-insight-dynamic-1",
        "rl-insight-dynamic-2",
    ]


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
