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

"""Unit tests for OpenTelemetry trace collection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rl_insight.utils import opentelemetry_utils as otel_module


def test_init_should_disable_collection_when_endpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = MagicMock()
    monkeypatch.setattr(otel_module, "OTLPSpanExporter", exporter)

    collector = otel_module.OpenTelemetryTraceCollector(
        namespace="trainer", endpoint=None
    )
    collector.record_span("ignored", 10, 20, attributes={"step": 1})

    assert collector.enabled is False
    exporter.assert_not_called()


def test_record_span_should_export_timing_attributes_and_resource_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(otel_module, "OTLPSpanExporter", lambda endpoint: exporter)
    monkeypatch.setattr(otel_module, "BatchSpanProcessor", SimpleSpanProcessor)
    collector = otel_module.OpenTelemetryTraceCollector(
        namespace="trainer-monitor", endpoint="http://tempo:4318/v1/traces"
    )

    collector.record_span(
        "rollout", 1_000_000, 2_500_000, attributes={"step": 7, "worker": "w0"}
    )

    spans = exporter.get_finished_spans()
    assert collector.enabled is True
    assert len(spans) == 1
    assert spans[0].name == "rollout"
    assert (spans[0].start_time, spans[0].end_time) == (1_000_000, 2_500_000)
    assert spans[0].attributes == {"step": 7, "worker": "w0"}
    assert spans[0].resource.attributes[SERVICE_NAME] == "trainer-monitor"
