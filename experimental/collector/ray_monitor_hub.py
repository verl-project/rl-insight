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
from omegaconf import DictConfig, OmegaConf

from ..config import MONITOR_HUB_ACTOR_NAME, MONITOR_RAY_NAMESPACE
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
class MonitorHubActor:
    """Ray detached actor: receives monitor events from trainers, serves ``/metrics``, optional OTLP traces.

    Actor methods run one at a time (no ``max_concurrency``), so hub state updates are serialized.

    On startup it may rewrite the local Prometheus scrape config when ``prometheus.reload.mode`` is ``ray``.
    """

    def __init__(
        self,
        conf: dict[str, Any] | DictConfig,
    ) -> None:
        """
        Args:
            conf: Merged monitor config (trainer dict); expects ``namespace``, ``otel``, ``prometheus`` keys.
        """
        self._conf = conf if isinstance(conf, DictConfig) else OmegaConf.create(conf)
        namespace = str(self._conf.namespace)
        self._registry = MetricRegistry(namespace=namespace)
        te_raw = OmegaConf.select(self._conf, "otel.traces_endpoint")
        te = str(te_raw).strip() if te_raw is not None else ""
        self._trace_collector = (
            OpenTelemetryTraceCollector(namespace=namespace, endpoint=te)
            if te
            else None
        )
        self._events_applied = 0
        self._node_ip = ray.util.get_node_ip_address()
        self._metrics_port = int(self._conf.prometheus.metrics_report_port)
        self._event_handlers = {
            MonitorEventKind.COUNTER: self._handle_counter,
            MonitorEventKind.GAUGE: self._handle_gauge,
            MonitorEventKind.HISTOGRAM: self._handle_histogram,
            MonitorEventKind.TRACE: self._handle_trace,
        }

        scrape_host = self._node_ip
        start_metrics_http_server(self._metrics_port, addr=scrape_host)
        if (
            str(OmegaConf.select(self._conf, "prometheus.reload.mode") or "ray")
            .strip()
            .lower()
            == "ray"
        ):
            update_prometheus_config(
                self._conf,
                [f"{scrape_host}:{self._metrics_port}"],
            )

        listen_desc = scrape_host if scrape_host else "0.0.0.0"
        logger.info(
            "MonitorHubActor HTTP bind %s:%s, Prometheus scrape target %s:%s",
            listen_desc,
            self._metrics_port,
            scrape_host,
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
            "actor_name": MONITOR_HUB_ACTOR_NAME,
            "namespace": MONITOR_RAY_NAMESPACE,
            "node_ip": self._node_ip,
            "metrics_endpoint": f"http://{self._node_ip}:{self._metrics_port}/metrics",
            "prometheus_metrics_enabled": True,
            "otel_traces_enabled": self._trace_collector is not None,
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
        """Export one root span via OTLP if a trace collector is configured; otherwise no-op."""
        if self._trace_collector is None:
            return
        self._trace_collector.record_span(
            event["name"],
            int(event["start_time_ns"]),
            int(event["end_time_ns"]),
            attributes=dict(event.get("attributes") or {}),
        )
