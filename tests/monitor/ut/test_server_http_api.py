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


def test_metric_catalog_preserves_run_and_time_scope(monkeypatch, tmp_path):
    from rl_insight.server import http_api
    from rl_insight.server.registry import RunRegistry
    from rl_insight.server.runtime import run_options
    from rl_insight.utils.protocol import RunContext

    context = RunContext.create("p", "same", "http://server")
    registry = RunRegistry(tmp_path)
    registry.create(context.run_id, "p", "same", "http://server", {})
    calls = []

    def backend(url, fields, **kwargs):
        calls.append(dict(fields))
        return 200, b'{"tagValues":[{"type":"string","value":"loss"}]}'

    monkeypatch.setattr(http_api, "read_backend", backend)
    fields = [
        ("match[]", '{run_id=~"' + context.run_id + '"}'),
        ("start", "1792600000.5"),
        ("end", "1792600300.5"),
    ]
    try:
        assert run_options(registry, OmegaConf.create({}), "metric_name", fields) == [
            "loss"
        ]
        assert context.run_id in calls[-1]["q"]
        assert (calls[-1]["start"], calls[-1]["end"]) == ("1792600000", "1792600300")
    finally:
        registry.close()


def test_timeline_and_step_projections_keep_separate_columns():
    from rl_insight.server.http_api import project_search, trace_search
    from rl_insight.utils.protocol import RunContext

    contexts = [RunContext(c * 32, "p", "same", "http://server") for c in "ab"]
    for metric in (False, True):
        query = (
            '{span."metric.name"!=nil} | select(span."metric.name", span."metric.value", span.global_step)'
            if metric
            else '{span.state_lane_id!=""} | select(span.state_lane_id)'
        )
        scoped = trace_search(query, contexts)
        assert ('span."metric.value"' in scoped) is metric
        payload = {
            "traces": [
                {
                    "traceID": context.run_id,
                    "spanSet": {
                        "matched": 1,
                        "spans": [
                            {
                                "name": "metric_log" if metric else "actor_update",
                                "startTimeUnixNano": "1",
                                "attributes": [
                                    {"key": k, "value": {"stringValue": v}}
                                    for k, v in {
                                        "rl_insight.run_id": context.run_id,
                                        "state_lane_id": "rank_0",
                                        "project": "p",
                                        "experiment_name": "same",
                                        "global_step": "3",
                                        "metric.name": "loss",
                                        "metric.value": "0.5",
                                    }.items()
                                ],
                            }
                        ],
                    },
                }
                for context in contexts
            ]
        }
        project_search(payload, contexts, scoped)
        for trace, context in zip(payload["traces"], contexts):
            span = trace["spanSet"]["spans"][0]
            attributes = {
                a["key"]: a["value"]["stringValue"] for a in span["attributes"]
            }
            assert span["durationNanos"] == "0"
            if metric:
                assert attributes["rl_insight.run_id"] == context.run_id
                assert {
                    "metric.name",
                    "metric.value",
                    "global_step",
                } <= attributes.keys()
            else:
                assert attributes == {"state_lane_id": context.run_id[:8] + "/rank_0"}
