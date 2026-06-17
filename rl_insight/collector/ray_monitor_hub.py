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
from typing import Any

import ray
from omegaconf import DictConfig

from ..utils.constants import MonitorRayActor
from .base import MonitorCollector
from ..utils import (
    MetricRegistry,
    MonitorEventKind,
    OpenTelemetryTraceCollector,
    start_metrics_http_server,
    update_prometheus_config,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorHubActor"]


@ray.remote
class MonitorHubActor(MonitorCollector):
    """Ray detached actor: receives monitor events from trainers, serves ``/metrics``, optional OTLP traces.

    Actor methods run one at a time (no ``max_concurrency``), so hub state updates are serialized.

    On startup it may rewrite the local Prometheus scrape config when ``server.backend`` is ``ray``.
    """

    def __init__(self, conf: DictConfig) -> None:
        """
        Args:
            conf: Merged monitor config with ``server``, ``otel``, and ``prometheus`` sections.
        """
        self._conf = conf

        namespace = str(self._conf.server.namespace)
        service_ip = str(self._conf.server.service_ip).strip()
        trace_endpoint = (
            f"http://{service_ip}:{int(self._conf.otel.otel_port)}/v1/traces"
            if service_ip
            else ""
        )
        self._registry = MetricRegistry(namespace=namespace)
        self._trace_collector = OpenTelemetryTraceCollector(
            namespace=namespace,
            endpoint=trace_endpoint,
        )

        self._state_pending: dict[tuple[str, str], dict[str, Any]] = {}
        self._events_applied = 0
        self._node_ip = ray.util.get_node_ip_address()
        self._metrics_port = int(self._conf.prometheus.metrics_report_port)
        self._event_handlers = {
            MonitorEventKind.COUNTER: self._handle_counter,
            MonitorEventKind.GAUGE: self._handle_gauge,
            MonitorEventKind.HISTOGRAM: self._handle_histogram,
            MonitorEventKind.TRACE: self._handle_trace,
        }

        start_metrics_http_server(self._metrics_port, addr=self._node_ip)
        update_prometheus_config(self._conf, [f"{self._node_ip}:{self._metrics_port}"])
        logger.info(
            "MonitorHubActor HTTP bind %s:%s, Prometheus scrape target %s:%s",
            self._node_ip,
            self._metrics_port,
            self._node_ip,
            self._metrics_port,
        )

    def apply_event(self, event: dict[str, Any]) -> None:
        """Dispatch one event by ``kind``: counter/gauge/histogram update Prometheus registry, trace exports OTLP.

        Args:
            event: Must include ``kind``; metric kinds need ``name``/``value``; trace needs ``start_time_ns``/``end_time_ns``.
        """
        self._events_applied += 1
        try:
            kind = event["kind"]
        except KeyError as e:
            raise ValueError(f"Event missing required field: {e!r}") from e

        handler = self._event_handlers.get(kind)
        if handler is None:
            raise ValueError(f"Unknown event kind: {kind!r}")
        handler(event)

    def get_status(self) -> dict[str, Any]:
        """Return a small status dict for debugging (endpoints, counters).

        Returns:
            Dict with ``actor_name``, ``namespace`` (Ray placement namespace, not metric prefix), scrape URL, flags.
        """
        return {
            "actor_name": MonitorRayActor.NAME,
            "namespace": MonitorRayActor.NAMESPACE,
            "node_ip": self._node_ip,
            "metrics_endpoint": f"http://{self._node_ip}:{self._metrics_port}/metrics",
            "prometheus_metrics_enabled": True,
            "otel_traces_enabled": self._trace_collector.enabled,
            "events_applied": self._events_applied,
        }

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
        """Dispatch trace events; state_interval spans are merged before export."""
        if not self._trace_collector.enabled:
            return
        attrs = dict(event.get("attributes") or {})
        if attrs.get("monitor.trace_segment") == "state_interval":
            self._handle_state_interval_trace(event, attrs)
            return
        self._export_trace_span(
            event["name"],
            int(event["start_time_ns"]),
            int(event["end_time_ns"]),
            attrs,
        )

    def _handle_state_interval_trace(
        self, event: dict[str, Any], attrs: dict[str, Any]
    ) -> None:
        """Merge overlapping state intervals per (lane, name); flush on gap."""
        key = (str(attrs["state_lane_id"]), event["name"])
        start_ns = int(event["start_time_ns"])
        end_ns = int(event["end_time_ns"])

        pending = self._state_pending.get(key)
        if pending is not None and start_ns > pending["end_ns"]:
            self._export_trace_span(
                pending["name"],
                pending["start_ns"],
                pending["end_ns"],
                pending["attributes"],
            )
            pending = None

        if pending is None:
            self._state_pending[key] = {
                "name": event["name"],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "attributes": attrs,
            }
            return

        pending["start_ns"] = min(pending["start_ns"], start_ns)
        pending["end_ns"] = max(pending["end_ns"], end_ns)

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
