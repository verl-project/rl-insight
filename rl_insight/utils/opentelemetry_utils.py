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
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

__all__ = ["OpenTelemetryTraceCollector"]

_OTEL_EXPORT_LOGGERS = (
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.sdk.trace.export",
)


def _reduce_otel_export_log_noise() -> None:
    # Suppress WARNING retry spam; keep ERROR for real export failures.
    for name in _OTEL_EXPORT_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


class OpenTelemetryTraceCollector:
    """Export closed root spans to Tempo via OTLP/HTTP."""

    def __init__(self, namespace: str = "", endpoint: str | None = None) -> None:
        self._tracer = None
        if not endpoint:
            logger.warning(
                "[rl-insight] OpenTelemetry trace export is disabled because no OTLP endpoint "
                "was returned by the RL-Insight server."
            )
            return

        _reduce_otel_export_log_noise()
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: namespace}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        self._tracer = provider.get_tracer(__name__)

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def record_span(
        self,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self._tracer is None:
            return

        span = self._tracer.start_span(
            name,
            start_time=start_time_ns,
            attributes=attributes,
        )
        span.end(end_time=end_time_ns)
