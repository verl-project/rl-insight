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

from omegaconf import OmegaConf

from rl_insight.server.http_api import create_app
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
