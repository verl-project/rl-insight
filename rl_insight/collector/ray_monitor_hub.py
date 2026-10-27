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

import json
import logging
import time
from dataclasses import replace
from email.utils import parsedate_to_datetime
from typing import Any

import ray
import requests
from omegaconf import DictConfig
from prometheus_client import Gauge

from ..server.http_api import get_server_services
from ..server.network import format_host_port, service_url_from_server_url
from ..utils import (
    MetricRegistry,
    MonitorEventKind,
    OpenTelemetryTraceCollector,
    start_metrics_http_server,
    update_prometheus_config,
)
from ..utils.constants import MonitorRayActor
from ..utils.prometheus_utils import _forget_targets
from ..utils.protocol import RunContext, RunError
from .base import MonitorCollector

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["MonitorHubActor"]


def _register_run(
    context: RunContext,
    config: dict[str, Any],
    budget: float = 10,
    *,
    owner_token: str = "",
) -> RunContext:
    """Retry transient registration failures with the same caller-owned identity."""
    deadline = time.monotonic() + budget
    delay = 0.1
    payload = {**context.metadata(), "config": config, "owner_token": owner_token}
    with requests.Session() as session:
        session.trust_env = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Run registration exceeded its initialization budget"
                )
            pause = delay
            try:
                response = session.post(
                    context.server_url + "/api/v1/runs",
                    json=payload,
                    timeout=min(2, remaining / 2),
                    allow_redirects=False,
                    stream=True,
                )
                with response:
                    body = bytearray()
                    # Registration is small. Read incrementally so a trickling
                    # response cannot reset the initialization budget forever.
                    for chunk in response.iter_content(chunk_size=1):
                        body.extend(chunk)
                        if len(body) > 65536 or time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Run registration exceeded its initialization budget"
                            )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Run registration exceeded its initialization budget"
                    )
                if response.status_code == 429 or response.status_code in {
                    500,
                    502,
                    503,
                    504,
                }:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            pause = max(pause, float(retry_after))
                        except ValueError:
                            try:
                                pause = max(
                                    pause,
                                    parsedate_to_datetime(retry_after).timestamp()
                                    - time.time(),
                                )
                            except (TypeError, ValueError, OverflowError):
                                pass
                else:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise RunError("Unexpected Run registration response")
                    registered = RunContext.from_dict(json.loads(body))
                    if (
                        registered.run_id,
                        registered.project,
                        registered.experiment_name,
                    ) != (context.run_id, context.project, context.experiment_name):
                        raise RunError("Server returned a conflicting Run identity")
                    return registered
            except (requests.ConnectionError, requests.Timeout):
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0 or pause >= remaining:
                raise TimeoutError(
                    "Run registration exceeded its initialization budget"
                )
            time.sleep(pause)
            delay = min(delay * 2, 1)


@ray.remote
class MonitorHubActor(MonitorCollector):
    """Ray job-scoped actor: receives monitor events from trainers, serves ``/metrics``, optional OTLP traces.

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
        self._final_confirmed = False

        namespace = str(self._conf.server.namespace)
        metadata = conf.get("run_metadata", {})
        self._context = RunContext.create(
            str(metadata.get("project", "uncategorized")),
            str(metadata.get("experiment_name", "unnamed")),
            str(conf.server.url),
        )
        services = get_server_services()
        self._context = _register_run(
            self._context,
            {
                "server": {"namespace": namespace, "backend": "ray"},
                "prometheus": {
                    "metrics_report_port": int(conf.prometheus.metrics_report_port)
                },
            },
            owner_token=owner_token,
        )
        self._services = services
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
        trace_kwargs = {
            "resource_attributes": {"rl_insight.run_id": self._context.run_id}
        }
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

        self._managed_endpoints = services.get("target_registration_version") == 1
        bound = start_metrics_http_server(self._metrics_port, addr=self._node_ip)
        self._metrics_server = bound[0] if isinstance(bound, tuple) else None
        if self._metrics_server is not None:
            self._metrics_port = self._metrics_server.server_port
        update_prometheus_config(
            [format_host_port(self._node_ip, self._metrics_port)],
            context=self._context,
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
            labels = dict(event.get(field) or {})
            labels.setdefault("project", context.project)
            labels.setdefault("experiment_name", context.experiment_name)
            event[field] = context.labels(labels)
        try:
            kind = event["kind"]
        except KeyError as e:
            raise ValueError(f"Event missing required field: {e!r}") from e

        handler = self._event_handlers.get(kind)
        if handler is None:
            raise ValueError(f"Unknown event kind: {kind!r}")
        handler(event)

    def get_context(self) -> RunContext:
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
            _forget_targets(self._context)
            if self._metrics_server is not None:
                self._metrics_server.shutdown()
                self._metrics_server.server_close()

    def flush(self) -> None:
        """Confirm final OTLP export and SDK scrape during creator shutdown.

        Workers use an ordered Context read to drain events, leaving this one
        collection wait until the creator has joined all workers.
        """
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
