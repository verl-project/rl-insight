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

"""Unit tests for Prometheus metrics and target registration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from rl_insight.utils import prometheus_utils as prometheus_module


def _samples(collector: Any) -> dict[str, Any]:
    return {
        sample.name: sample
        for metric in collector.collect()
        for sample in metric.samples
    }


def test_metric_registry_should_store_real_samples_when_all_metric_types_are_recorded() -> (
    None
):
    registry = prometheus_module.MetricRegistry(namespace="monitor_ut")
    registry.count(
        "steps", "Steps", 2, defaults={"worker": "old"}, labels={"worker": "w0"}
    )
    registry.value("reward", "Reward", 1.5, labels={"worker": "w0"})
    registry.distribution(
        "latency", "Latency", 12, labels={"worker": "w0"}, buckets=(10, 20)
    )

    counter = _samples(next(iter(registry._counters.values())))
    gauge = _samples(next(iter(registry._gauges.values())))
    histogram = _samples(next(iter(registry._histograms.values())))
    assert counter["monitor_ut_steps_total"].value == 2
    assert counter["monitor_ut_steps_total"].labels == {"worker": "w0"}
    assert gauge["monitor_ut_reward"].value == 1.5
    assert histogram["monitor_ut_latency_count"].value == 1
    assert histogram["monitor_ut_latency_sum"].value == 12


def test_register_should_merge_and_sort_targets_when_config_already_exists(
    tmp_path,
) -> None:
    config_file = tmp_path / "prometheus.yml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "scrape_configs": [
                    {
                        "job_name": "trainers",
                        "static_configs": [{"targets": ["host-b:9000"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    store = prometheus_module.PrometheusTargetStore(config_file, 9090)

    result = store.register(
        "trainers",
        [
            prometheus_module.PrometheusTarget("host-a:9000", {"rank": 0}),
            prometheus_module.PrometheusTarget("host-b:9000", {"rank": 1}),
        ],
    )

    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    groups = saved["scrape_configs"][0]["static_configs"]
    assert result["target_count"] == 2
    assert groups == [
        {"targets": ["host-a:9000"], "labels": {"rank": "0"}},
        {"targets": ["host-b:9000"], "labels": {"rank": "1"}},
    ]


def test_reload_should_post_to_local_prometheus_when_store_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    response = MagicMock()
    session_mock = MagicMock()
    session_mock.post = MagicMock(return_value=response)
    session_mock.__enter__ = MagicMock(return_value=session_mock)
    session_mock.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(
        prometheus_module.requests, "Session", MagicMock(return_value=session_mock)
    )
    monkeypatch.setattr(
        prometheus_module, "local_addresses", lambda: {"loopback": "127.0.0.1"}
    )
    store = prometheus_module.PrometheusTargetStore(tmp_path / "prometheus.yml", 9090)

    assert store.reload() is True
    session_mock.post.assert_called_once_with(
        "http://127.0.0.1:9090/-/reload", timeout=5
    )
    response.raise_for_status.assert_called_once_with()


def test_update_prometheus_config_should_send_normalized_targets_when_server_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    post = MagicMock(return_value=response)
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "http://server:18080/")
    monkeypatch.setattr(prometheus_module.requests, "post", post)

    prometheus_module.update_prometheus_config(
        ["host-a:9000", "host-b:9000"],
        job_name="workers",
        labels=[{"rank": 0}, None],
    )

    post.assert_called_once_with(
        "http://server:18080/api/v1/prometheus/targets",
        json={
            "job_name": "workers",
            "targets": [
                {"target": "host-a:9000", "labels": {"rank": "0"}},
                {"target": "host-b:9000"},
            ],
        },
        timeout=10,
    )
    response.raise_for_status.assert_called_once_with()


def test_update_prometheus_config_should_reject_mismatched_labels_when_lengths_differ() -> (
    None
):
    with pytest.raises(ValueError, match="labels length must match"):
        prometheus_module.update_prometheus_config(
            ["host-a:9000", "host-b:9000"], labels=[{"rank": 0}]
        )
