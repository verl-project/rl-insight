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

"""Unit tests for the Ray monitor hub implementation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, call

import pytest
from omegaconf import OmegaConf

from rl_insight.collector import ray_monitor_hub as hub_module
from rl_insight.utils.constants import MonitorEventKind, MonitorRayActor


HubImplementation = cast(
    Any, hub_module.MonitorHubActor
).__ray_metadata__.modified_class


@pytest.fixture
def hub() -> Any:
    instance = HubImplementation.__new__(HubImplementation)
    instance._registry = MagicMock()
    instance._trace_collector = MagicMock(enabled=True)
    instance._events_applied = 0
    instance._event_handlers = {
        MonitorEventKind.COUNTER: instance._handle_counter,
        MonitorEventKind.GAUGE: instance._handle_gauge,
        MonitorEventKind.HISTOGRAM: instance._handle_histogram,
        MonitorEventKind.TRACE: instance._handle_trace,
    }
    return instance


def test_init_should_configure_collectors_and_register_scrape_target_when_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = MagicMock()
    trace_collector = MagicMock(enabled=True)
    start_server = MagicMock()
    update_config = MagicMock()
    monkeypatch.setattr(hub_module, "get_server_services", lambda: {"otlp_port": 4318})
    registry_factory = MagicMock(return_value=registry)
    monkeypatch.setattr(hub_module, "MetricRegistry", registry_factory)
    trace_factory = MagicMock(return_value=trace_collector)
    monkeypatch.setattr(hub_module, "OpenTelemetryTraceCollector", trace_factory)
    monkeypatch.setattr(hub_module, "start_metrics_http_server", start_server)
    monkeypatch.setattr(hub_module, "update_prometheus_config", update_config)
    monkeypatch.setattr(hub_module.ray.util, "get_node_ip_address", lambda: "10.0.0.8")
    conf = OmegaConf.create(
        {
            "server": {"namespace": "trainer", "url": "http://host:18080"},
            "prometheus": {"metrics_report_port": 9092},
        }
    )

    instance = HubImplementation.__new__(HubImplementation)
    instance.__init__(conf)

    registry_factory.assert_called_once_with(namespace="trainer")
    trace_factory.assert_called_once_with(
        namespace="trainer", endpoint="http://host:4318/v1/traces"
    )
    start_server.assert_called_once_with(9092, addr="10.0.0.8")
    update_config.assert_called_once_with(["10.0.0.8:9092"])
    assert instance._registry is registry


@pytest.mark.parametrize(
    ("event", "method", "expected_call"),
    [
        (
            {"kind": "counter", "name": "steps", "value": "2", "labels": {"w": 1}},
            "count",
            call("steps", "", 2.0, {}, {"w": 1}),
        ),
        (
            {
                "kind": "gauge",
                "name": "reward",
                "value": 1.5,
                "documentation": "reward doc",
            },
            "value",
            call("reward", "reward doc", 1.5, {}, {}),
        ),
        (
            {"kind": "histogram", "name": "latency", "value": 12},
            "distribution",
            call("latency", "", 12.0, {}, {}, buckets=None),
        ),
    ],
)
def test_apply_event_should_dispatch_metric_when_kind_is_supported(
    hub: Any,
    event: dict[str, object],
    method: str,
    expected_call: Any,
) -> None:
    hub.apply_event(event)

    assert hub._events_applied == 1
    assert getattr(hub._registry, method).call_args == expected_call


def test_apply_event_should_export_trace_when_trace_collection_is_enabled(
    hub: Any,
) -> None:
    event = {
        "kind": "trace",
        "name": "rollout",
        "start_time_ns": "10",
        "end_time_ns": 25,
        "attributes": {"step": 3},
    }

    hub.apply_event(event)

    hub._trace_collector.record_span.assert_called_once_with(
        "rollout", 10, 25, attributes={"step": 3}
    )


@pytest.mark.parametrize(
    ("event", "message"),
    [({}, "missing required field"), ({"kind": "unknown"}, "Unknown event kind")],
)
def test_apply_event_should_raise_value_error_when_event_is_invalid(
    hub: Any, event: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        hub.apply_event(event)


def test_get_status_should_describe_endpoint_when_hub_is_initialized(hub: Any) -> None:
    hub._node_ip = "10.0.0.8"
    hub._metrics_port = 9092
    hub._events_applied = 4

    status = hub.get_status()

    assert status == {
        "actor_name": MonitorRayActor.NAME,
        "namespace": MonitorRayActor.NAMESPACE,
        "node_ip": "10.0.0.8",
        "metrics_endpoint": "http://10.0.0.8:9092/metrics",
        "prometheus_metrics_enabled": True,
        "otel_traces_enabled": True,
        "events_applied": 4,
    }
