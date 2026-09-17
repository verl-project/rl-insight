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

"""Unit tests for the RL-Insight server HTTP API."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi import HTTPException
from omegaconf import OmegaConf

from rl_insight.server.http_api import create_app
from rl_insight.utils.experiment_targets import ExperimentTargetStore
from rl_insight.utils.prometheus_utils import PrometheusTargetStore


def test_register_targets_should_persist_file_sd_targets_and_reload_prometheus(
    monkeypatch, tmp_path
) -> None:
    conf = OmegaConf.create(
        {
            "server": {
                "runtime_dir": str(tmp_path / "runtime"),
                "data_dir": str(tmp_path / "data"),
            },
            "prometheus": {"prometheus_port": 9090},
        }
    )

    reload_prometheus = MagicMock(return_value=True)
    monkeypatch.setattr(PrometheusTargetStore, "reload", reload_prometheus)
    app = create_app(conf)
    endpoint = next(
        cast(Any, route).endpoint
        for route in app.routes
        if getattr(route, "path", "") == "/api/v1/prometheus/targets"
    )

    result = endpoint(
        {
            "job_name": "node-exporter",
            "targets": ["node-a:9100"],
        }
    )

    assert result["status"] == "ok"
    assert result["prometheus_reloaded"] is True
    reload_prometheus.assert_called_once_with()
    assert (tmp_path / "data" / "targets" / "prometheus-targets.yml").exists()


def _conf(tmp_path: Any) -> Any:
    return OmegaConf.create(
        {
            "server": {
                "runtime_dir": str(tmp_path / "runtime"),
                "data_dir": str(tmp_path / "data"),
            },
            "prometheus": {"prometheus_port": 9090},
        }
    )


def _route(app: Any, path: str, method: str = "POST") -> Any:
    return next(
        cast(Any, route).endpoint
        for route in app.routes
        if getattr(route, "path", "") == path
        and getattr(route, "methods", "")
        and method in route.methods
    )


def test_register_should_route_identified_targets_to_the_experiment_partition(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    endpoint = _route(app, "/api/v1/prometheus/targets")

    result = endpoint(
        {
            "project": "project-a",
            "experiment_name": "exp-1",
            "job_name": "trainer_metrics",
            "targets": [{"target": "10.0.0.1:9092", "labels": {"role": "trainer"}}],
        }
    )

    assert result["status"] == "ok"
    assert result["project"] == "project-a"
    assert result["state"] == "active"
    assert not (tmp_path / "data" / "targets" / "prometheus-targets.yml").exists()
    shown = app.state.experiments.show_targets("project-a", "exp-1")
    assert shown["targets"][0]["labels"]["project"] == "project-a"
    assert shown["targets"][0]["labels"]["experiment_name"] == "exp-1"


def test_register_should_keep_unidentified_targets_in_the_legacy_partition(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    endpoint = _route(app, "/api/v1/prometheus/targets")

    result = endpoint({"job_name": "node-exporter", "targets": ["node-a:9100"]})

    assert result["status"] == "ok"
    assert (tmp_path / "data" / "targets" / "prometheus-targets.yml").exists()
    assert app.state.experiments.list_experiments() == []


def test_register_should_reject_partial_identity(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    endpoint = _route(app, "/api/v1/prometheus/targets")

    with pytest.raises(HTTPException) as missing_experiment:
        endpoint({"project": "project-a", "targets": ["a:9000"]})
    assert missing_experiment.value.status_code == 400

    with pytest.raises(HTTPException) as blank_project:
        endpoint({"project": " ", "experiment_name": "exp-1", "targets": ["a:9000"]})
    assert blank_project.value.status_code == 400


def test_register_should_reject_labels_that_override_reserved_identity(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    endpoint = _route(app, "/api/v1/prometheus/targets")

    with pytest.raises(HTTPException) as default_conflict:
        endpoint(
            {
                "project": "project-a",
                "experiment_name": "exp-1",
                "labels": {"project": "project-b"},
                "targets": ["a:9000"],
            }
        )
    assert default_conflict.value.status_code == 400

    with pytest.raises(HTTPException) as target_conflict:
        endpoint(
            {
                "project": "project-a",
                "experiment_name": "exp-1",
                "targets": [
                    {"target": "a:9000", "labels": {"experiment_name": "exp-9"}}
                ],
            }
        )
    assert target_conflict.value.status_code == 400


def test_register_should_return_409_for_archived_experiments(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    register = _route(app, "/api/v1/prometheus/targets")
    register(
        {
            "project": "project-a",
            "experiment_name": "exp-1",
            "targets": ["a:9000"],
        }
    )
    archive = _route(app, "/api/v1/experiments/archive")
    archive({"project": "project-a", "experiment_name": "exp-1"})

    with pytest.raises(HTTPException) as conflict:
        register(
            {
                "project": "project-a",
                "experiment_name": "exp-1",
                "targets": ["a:9001"],
            }
        )
    assert conflict.value.status_code == 409
    assert "restore" in str(conflict.value.detail)


def test_experiment_lifecycle_endpoints_should_archive_and_restore_idempotently(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    register = _route(app, "/api/v1/prometheus/targets")
    register(
        {"project": "project-a", "experiment_name": "exp-1", "targets": ["a:9000"]}
    )
    archive = _route(app, "/api/v1/experiments/archive")
    restore = _route(app, "/api/v1/experiments/restore")

    archived = archive({"project": "project-a", "experiment_name": "exp-1"})
    assert archived["state"] == "archived"
    assert (
        archive({"project": "project-a", "experiment_name": "exp-1"})["state"]
        == "archived"
    )

    restored = restore({"project": "project-a", "experiment_name": "exp-1"})
    assert restored["state"] == "active"
    assert (
        restore({"project": "project-a", "experiment_name": "exp-1"})["state"]
        == "active"
    )


def test_experiment_endpoints_should_validate_identity_and_unknown_experiments(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    archive = _route(app, "/api/v1/experiments/archive")
    show = _route(app, "/api/v1/experiments/targets", method="GET")
    list_endpoint = _route(app, "/api/v1/experiments", method="GET")

    with pytest.raises(HTTPException) as missing:
        archive({"project": "project-a"})
    assert missing.value.status_code == 400

    with pytest.raises(HTTPException) as unknown_archive:
        archive({"project": "project-a", "experiment_name": "missing"})
    assert unknown_archive.value.status_code == 404

    with pytest.raises(HTTPException) as missing_show:
        show()
    assert missing_show.value.status_code == 400

    with pytest.raises(HTTPException) as blank_show:
        show(project=" ", experiment_name="exp-1")
    assert blank_show.value.status_code == 400

    with pytest.raises(HTTPException) as unknown_show:
        show(project="project-a", experiment_name="missing")
    assert unknown_show.value.status_code == 404

    assert list_endpoint() == []
    assert list_endpoint(project="project-a") == []


def test_list_and_show_should_report_experiments(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    app = create_app(_conf(tmp_path))
    register = _route(app, "/api/v1/prometheus/targets")
    register(
        {"project": "project-a", "experiment_name": "exp-1", "targets": ["a:9000"]}
    )
    register(
        {"project": "project-b", "experiment_name": "exp-1", "targets": ["b:9000"]}
    )
    show = _route(app, "/api/v1/experiments/targets", method="GET")
    list_endpoint = _route(app, "/api/v1/experiments", method="GET")

    rows = list_endpoint()
    assert {(row["project"], row["experiment_name"]) for row in rows} == {
        ("project-a", "exp-1"),
        ("project-b", "exp-1"),
    }
    assert list_endpoint(project="project-b")[0]["project"] == "project-b"

    shown = show(project="project-a", experiment_name="exp-1")
    assert shown["state"] == "active"
    assert shown["target_count"] == 1
    assert shown["targets"][0]["target"] == "a:9000"


def test_create_app_should_migrate_identified_records_from_the_legacy_global_file(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    global_file = tmp_path / "data" / "targets" / "prometheus-targets.yml"
    global_file.parent.mkdir(parents=True, exist_ok=True)
    global_file.write_text(
        yaml.safe_dump(
            [
                {
                    "targets": ["a:9000"],
                    "labels": {
                        "rl_insight_job": "trainer_metrics",
                        "project": "project-a",
                        "experiment_name": "exp-1",
                    },
                },
                {"targets": ["b:9100"], "labels": {"rl_insight_job": "node_exporter"}},
            ]
        ),
        encoding="utf-8",
    )

    create_app(_conf(tmp_path))

    shown = ExperimentTargetStore.from_config(_conf(tmp_path)).show_targets(
        "project-a", "exp-1"
    )
    assert shown["target_count"] == 1
    remaining = yaml.safe_load(global_file.read_text(encoding="utf-8"))
    assert remaining == [
        {"targets": ["b:9100"], "labels": {"rl_insight_job": "node_exporter"}}
    ]


def test_create_app_should_keep_serving_when_legacy_migration_fails(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(PrometheusTargetStore, "reload", MagicMock(return_value=True))
    global_file = tmp_path / "data" / "targets" / "prometheus-targets.yml"
    global_file.parent.mkdir(parents=True, exist_ok=True)
    global_file.write_text("{not: [valid", encoding="utf-8")

    app = create_app(_conf(tmp_path))

    assert _route(app, "/api/v1/experiments", method="GET")() == []
