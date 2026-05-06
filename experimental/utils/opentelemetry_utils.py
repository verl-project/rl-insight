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

"""OpenTelemetry OTLP/HTTP trace export used by the monitor hub."""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "OpenTelemetryTraceCollector",
    "resolve_otlp_traces_endpoint",
]


def resolve_otlp_traces_endpoint(endpoint: str | None = None) -> str | None:
    """Normalize OTLP URL: strip whitespace, map empty to ``None``.

    Args:
        endpoint: Raw endpoint string from config or environment.
    """
    if not endpoint:
        return None
    return str(endpoint).strip() or None


def _normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Convert span attributes to OTLP-friendly scalars (other types stringified); drop ``None`` values."""
    normalized: dict[str, Any] = {}
    if not attributes:
        return normalized

    for key, value in attributes.items():
        key = str(key)
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


class OpenTelemetryTraceCollector:
    """Build an OTLP/HTTP exporter and record closed root spans with explicit timestamps."""

    def __init__(self, namespace: str = "", endpoint: str | None = None) -> None:
        """
        Args:
            namespace: Stored as ``service.namespace`` on the OpenTelemetry resource.
            endpoint: OTLP traces URL; if missing, collector stays disabled and trace ops no-op.
        """
        self._spans_recorded = 0
        self._enabled = False
        resolved_endpoint = resolve_otlp_traces_endpoint(endpoint)
        if not resolved_endpoint:
            logger.warning(
                "OpenTelemetry trace export is disabled because no OTLP endpoint is configured. "
                "Trainers: set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or init dict key ``otel.traces_endpoint``. "
                "Stack YAML: top-level ``otel.traces_endpoint`` (see bundled ``config/services/config.yaml``)."
            )
            return

        resource_attributes = {
            SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "rl-insight-monitor"),
        }
        if namespace:
            resource_attributes["service.namespace"] = namespace

        provider = TracerProvider(
            resource=Resource.create(resource_attributes),
        )
        exporter = OTLPSpanExporter(endpoint=resolved_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        self._provider = provider
        self._tracer = provider.get_tracer(__name__)
        self._enabled = True

    def record_span(
        self,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Create one exported span from ``start_time_ns`` to ``end_time_ns`` (no-op if disabled).

        Args:
            name: Span name.
            start_time_ns: Start time in nanoseconds.
            end_time_ns: End time in nanoseconds.
            attributes: Optional span attributes (non-scalars coerced).
        """
        if not self._enabled:
            return

        span = self._tracer.start_span(
            name=name,
            start_time=start_time_ns,
            attributes=_normalize_attributes(attributes),
        )

        span.end(end_time=end_time_ns)
        self._spans_recorded += 1

    def get_stats(self) -> dict[str, int]:
        """Return simple counters (spans successfully handed to the SDK)."""
        return {
            "trace_spans_recorded": self._spans_recorded,
        }
