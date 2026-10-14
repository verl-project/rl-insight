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
import time
from uuid import uuid4
from typing import Any

import ray
import requests
from omegaconf import DictConfig
from prometheus_client import Gauge

from ..utils.constants import MonitorRayActor
from .base import MonitorCollector
from ..server.http_api import get_server_services
from ..protocol import RunContext, RunError
from ..server.network import format_host_port, service_url_from_server_url
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

    On startup it registers its metrics endpoint with the RL-Insight server.
    """

    def __init__(self, conf: DictConfig, owner_token: str = "") -> None:
        """
        Args:
            conf: Merged monitor config with ``server`` and ``prometheus`` sections.
        """
        self._conf = conf
        self._owner_token = owner_token
        self._closing = False
        self._closed = False
        self._final_confirmed = None

        namespace = str(self._conf.server.namespace)
        services = get_server_services()
        self._context = None
        if services.get("run_context_version") == 1:
            payload = {
                "request_id": uuid4().hex,
                "owner_token": owner_token,
                "project": str(
                    conf.get("run_metadata", {}).get("project", "uncategorized")
                ),
                "experiment_name": str(
                    conf.get("run_metadata", {}).get("experiment_name", "unnamed")
                ),
                "config": {
                    "server": {"namespace": namespace, "backend": "ray"},
                    "prometheus": {
                        "metrics_report_port": int(conf.prometheus.metrics_report_port)
                    },
                },
            }
            response = requests.post(
                str(conf.server.url).rstrip("/") + "/api/v1/runs",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            self._context = RunContext.from_dict(response.json())
        trace_endpoint = service_url_from_server_url(
            str(self._conf.server.url),
            services.get("otlp_port"),
            "/v1/traces",
        )
        self._registry = MetricRegistry(namespace=namespace)
        self._flush_generation = 0
        self._flush_marker = (
            Gauge(
                "rl_insight_sdk_flush_sequence",
                "Latest SDK flush sequence offered for scrape",
                labelnames=["run_id"],
            ).labels(run_id=self._context.run_id)
            if self._context and services.get("prometheus_port")
            else None
        )
        trace_kwargs = (
            {"resource_attributes": {"rl_insight.run_id": self._context.run_id}}
            if self._context
            else {}
        )
        self._trace_collector = OpenTelemetryTraceCollector(
            namespace=namespace, endpoint=trace_endpoint, **trace_kwargs
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

        self._managed_endpoints = (
            services.get("endpoint_registration_version") == 1
            and self._context is not None
        )
        self._metrics_server = None
        if self._managed_endpoints:
            self._metrics_server, _ = start_metrics_http_server(
                self._metrics_port, addr=self._node_ip, context=self._context
            )
            self._metrics_port = self._metrics_server.server_port
        else:
            bound = start_metrics_http_server(self._metrics_port, addr=self._node_ip)
            if isinstance(bound, tuple):
                self._metrics_server = bound[0]
                self._metrics_port = self._metrics_server.server_port
            if self._context:
                update_prometheus_config(
                    [format_host_port(self._node_ip, self._metrics_port)],
                    context=self._context,
                    identity_in_metrics=True,
                )
            else:
                update_prometheus_config(
                    [format_host_port(self._node_ip, self._metrics_port)]
                )
        logger.info(
            "[rl-insight] MonitorHubActor HTTP bind %s:%s, "
            "Prometheus scrape target %s:%s",
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
        if getattr(self, "_closing", False):
            raise RunError("Run is closing; late events are rejected", 409)
        self._events_applied += 1
        context = getattr(self, "_context", None)
        if context:
            event = dict(event)
            field = (
                "attributes"
                if event.get("kind") == MonitorEventKind.TRACE
                else "labels"
            )
            event[field] = context.labels(event.get(field))
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
            "metrics_endpoint": (
                "http://"
                + format_host_port(self._node_ip, self._metrics_port)
                + "/metrics"
            ),
            "prometheus_metrics_enabled": True,
            "otel_traces_enabled": self._trace_collector.enabled,
            "events_applied": self._events_applied,
        }

    def get_context(self) -> dict[str, Any] | None:
        return self._context.to_dict() if self._context else None

    def _owner_request(self, prepare):
        if self._context is not None and self._managed_endpoints:
            response = requests.post(
                self._context.server_url
                + "/api/v1/runs/"
                + self._context.run_id
                + "/close",
                json={
                    "owner_token": self._owner_token,
                    "prepare": prepare,
                    "final_collection_confirmed": self._final_confirmed,
                },
                timeout=(2, 4),
            )
            response.raise_for_status()

    def prepare_finish(self, owner_token):
        if not self._owner_token or owner_token != self._owner_token:
            raise RunError("Only the Hub creator can finish its Run", 409)
        if self._closed:
            return
        self._owner_request(True)
        self._closing = True
        self._final_confirmed = False
        self.flush()
        self._final_confirmed = True

    def end_run(self, owner_token):
        if not self._owner_token or owner_token != self._owner_token:
            raise RunError("Only the Hub creator can finish its Run", 409)
        if self._closed:
            return
        try:
            self._owner_request(False)
        finally:
            self._closing = True
            self._closed = True
            if self._metrics_server is not None:
                try:
                    publisher = getattr(self._metrics_server, "publisher", None)
                    if publisher is not None:
                        publisher.close()
                finally:
                    self._metrics_server.shutdown()
                    self._metrics_server.server_close()

    def flush(self) -> None:
        """Confirm prior events reached OTLP and a real Prometheus scrape."""
        deadline = time.monotonic() + 30
        if not self._trace_collector.flush():
            raise TimeoutError("OTLP export did not finish within the flush budget")
        if self._context is None or self._flush_marker is None:
            return
        self._flush_generation += 1
        self._flush_marker.set(self._flush_generation)
        url = (
            self._context.server_url.rstrip("/")
            + "/api/v1/runs/"
            + self._context.run_id
            + "/prometheus/api/v1/query"
        )
        with requests.Session() as session:
            session.trust_env = False
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    response = session.get(
                        url,
                        params={"query": "rl_insight_sdk_flush_sequence"},
                        timeout=min(2, remaining),
                    )
                    response.raise_for_status()
                    samples = response.json().get("data", {}).get("result", [])
                    if any(
                        float(sample["value"][1]) >= self._flush_generation
                        for sample in samples
                    ):
                        return
                except (requests.RequestException, ValueError, KeyError, TypeError):
                    pass  # A transient query failure may recover before the deadline.
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        raise TimeoutError("Prometheus did not scrape the final SDK metrics within 30s")

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
