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

"""Ray named actor that collects monitor events and exposes a metrics endpoint."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import ray
from omegaconf import DictConfig

from .. import __version__
from ..server.http_api import initialize_server_connection
from ..server.network import format_host_port, service_url_from_server_url
from ..utils import (
    MetricRegistry,
    MonitorEventKind,
    OpenTelemetryTraceCollector,
    start_metrics_http_server,
)
from ..utils.constants import MonitorDefaults, MonitorRayActor
from ..utils.prometheus_utils import _register_targets
from ..utils.protocol import (
    RunContext,
    RunError,
    create_execution_suffix,
    require_version,
)
from .base import MonitorCollector

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorHubActor"]


@ray.remote
class MonitorHubActor(MonitorCollector):
    """Job-scoped actor receiving monitor events and native exporter addresses.

    Actor methods run one at a time (no ``max_concurrency``), so hub state updates are serialized.

    Collection starts when a client supplies its formal configuration.
    """

    def __init__(self) -> None:
        self._execution_suffix = create_execution_suffix()
        self._context: RunContext | None = None
        self._events_applied = 0
        self._pending_targets: list[
            tuple[list[str], str | None, list[Mapping[str, Any] | None] | None, bool]
        ] = []
        self._event_handlers = {
            MonitorEventKind.COUNTER: self._handle_counter,
            MonitorEventKind.GAUGE: self._handle_gauge,
            MonitorEventKind.HISTOGRAM: self._handle_histogram,
            MonitorEventKind.TRACE: self._handle_trace,
        }

    def connect(self, client_version: str | None = None) -> str:
        """Verify the SDK without binding task names or starting collection."""
        require_version(client_version, "Client")
        return __version__

    def init(self, conf: DictConfig) -> None:
        """Apply task configuration and start collecting with the selected names."""
        if self._context is not None:
            self.register_targets([])
            return
        metadata = conf.get("run_metadata", {})
        project = str(metadata.get("project", MonitorDefaults.PROJECT_NAME))
        experiment_name = str(
            metadata.get("experiment_name", MonitorDefaults.EXPERIMENT_NAME)
        )
        context = RunContext.create(
            project,
            experiment_name,
            str(conf.server.url),
            suffix=self._execution_suffix,
        )
        services = initialize_server_connection(str(conf.server.url))
        self._conf = conf
        self._services = services
        namespace = str(conf.server.namespace)
        self._registry = MetricRegistry(namespace=namespace)
        self._registry.value(
            "experiment_info", "Experiment execution identity.", 1, {}, context.labels()
        )
        self._trace_collector = OpenTelemetryTraceCollector(
            namespace=namespace,
            endpoint=service_url_from_server_url(
                context.server_url, services.get("otlp_port"), "/v1/traces"
            ),
        )
        self._node_ip = ray.util.get_node_ip_address()
        self._metrics_port = int(conf.prometheus.metrics_report_port)
        start_metrics_http_server(self._metrics_port, addr=self._node_ip)
        self._context = context
        self.register_targets(
            [format_host_port(self._node_ip, self._metrics_port)],
            identity_in_metrics=True,
        )
        logger.info(
            "[rl-insight] MonitorHubActor HTTP bind %s:%s, "
            "Prometheus scrape target %s:%s",
            self._node_ip,
            self._metrics_port,
            self._node_ip,
            self._metrics_port,
        )

    def register_targets(
        self,
        addresses: list[str],
        job_name: str | None = None,
        labels: list[Mapping[str, Any] | None] | None = None,
        *,
        identity_in_metrics: bool = False,
    ) -> None:
        """Save exporter addresses until formal init, then register them directly."""
        if labels is not None and len(labels) != len(addresses):
            raise RunError("labels length must match server_addresses length")
        item = (addresses, job_name, labels, identity_in_metrics)
        if addresses and item not in self._pending_targets:
            if len(self._pending_targets) >= 64:
                raise RunError("Too many targets waiting for initialization")
            self._pending_targets.append(item)
        if self._context is None:
            return
        for pending in list(self._pending_targets):
            if not _register_targets(
                *pending[:3], self._context, identity_in_metrics=pending[3]
            ):
                raise RunError(
                    "Cannot register Prometheus targets; retry init(refresh=True)", 503
                )
            self._pending_targets.remove(pending)

    def apply_event(self, event: dict[str, Any]) -> None:
        """Dispatch one event by ``kind``: counter/gauge/histogram update Prometheus registry, trace exports OTLP.

        Args:
            event: Must include ``kind``; metric kinds need ``name``/``value``; trace needs ``start_time_ns``/``end_time_ns``.
        """
        if self._context is None:
            return
        self._events_applied += 1
        event = dict(event)
        field = (
            "attributes" if event.get("kind") == MonitorEventKind.TRACE else "labels"
        )
        event[field] = self._context.labels(event.get(field))
        try:
            kind = event["kind"]
        except KeyError as e:
            raise ValueError(f"Event missing required field: {e!r}") from e

        handler = self._event_handlers.get(kind)
        if handler is None:
            raise ValueError(f"Unknown event kind: {kind!r}")
        handler(event)

    def get_context(self) -> RunContext | None:
        if self._context is None:
            return None
        return replace(
            self._context,
            actor_name=MonitorRayActor.NAME,
            namespace=MonitorRayActor.NAMESPACE,
            node_ip=self._node_ip,
            metrics_endpoint="http://"
            + format_host_port(self._node_ip, self._metrics_port)
            + "/metrics",
            otlp_endpoint=service_url_from_server_url(
                str(self._conf.server.url),
                self._services.get("otlp_port"),
                "/v1/traces",
            ),
            prometheus_metrics_enabled=True,
            otel_traces_enabled=self._trace_collector.enabled,
            events_applied=self._events_applied,
        )

    def _handle_counter(self, event: dict[str, Any]) -> None:
        """Increment a Prometheus counter from a ``counter`` event payload."""
        self._registry.count(
            event["name"],
            event.get("documentation") or "",
            float(event["value"]),
            {},
            dict(event.get("labels") or {}),
        )

    def _handle_gauge(self, event: dict[str, Any]) -> None:
        """Set a Prometheus gauge from a ``gauge`` event payload."""
        self._registry.value(
            event["name"],
            event.get("documentation") or "",
            float(event["value"]),
            {},
            dict(event.get("labels") or {}),
        )

    def _handle_histogram(self, event: dict[str, Any]) -> None:
        """Observe one sample on a Prometheus histogram from a ``histogram`` event payload."""
        self._registry.distribution(
            event["name"],
            event.get("documentation") or "",
            float(event["value"]),
            {},
            dict(event.get("labels") or {}),
            buckets=None,
        )

    def _handle_trace(self, event: dict[str, Any]) -> None:
        """Export one trace span via OTLP (no-op if collector disabled)."""
        if not self._trace_collector.enabled:
            return
        attrs = dict(event.get("attributes") or {})
        self._export_trace_span(
            event["name"],
            int(event["start_time_ns"]),
            int(event["end_time_ns"]),
            attrs,
        )

    def _export_trace_span(
        self,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: dict[str, Any],
    ) -> None:
        """Export one root span via OTLP (no-op if collector disabled)."""
        if not self._trace_collector.enabled:
            return
        self._trace_collector.record_span(
            name,
            start_time_ns,
            end_time_ns,
            attributes=attributes,
        )
