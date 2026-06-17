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

"""Prometheus metric registry, ``/metrics`` HTTP server, and scrape-config reload helpers."""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, Mapping

import ray
import requests
import yaml
from omegaconf import DictConfig, OmegaConf
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from .constants import PrometheusScrape

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


@ray.remote(num_cpus=0)
def _write_prometheus_config_file(config_data: dict[str, Any], path: str) -> bool:
    """Ray task: write merged Prometheus YAML to ``path`` on a node."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    return True


@ray.remote(num_cpus=0)
def _reload_prometheus_on_node(port: int, reload_url: str | None = None) -> None:
    """Ray task: POST Prometheus ``/-/reload`` on a node."""
    url = str(reload_url) if reload_url else None
    if not url:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        url = f"http://{ip_address}:{int(port)}/-/reload"
    try:
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        print(f"Reloading Prometheus on node: {url}")
    except requests.RequestException as exc:
        logger.warning("Prometheus reload failed at %s: %s", url, exc)


__all__ = [
    "MetricRegistry",
    "PrometheusScrapeUpdater",
    "start_metrics_http_server",
    "update_prometheus_config",
]


def _merge_labels(
    defaults: Mapping[str, Any] | None, overrides: Mapping[str, Any] | None
) -> dict[str, str]:
    """Merge Prometheus label dicts with string keys/values; ``overrides`` wins on duplicate keys.

    Args:
        defaults: Base labels (optional).
        overrides: Labels applied after defaults (optional).
    """
    out: dict[str, str] = {}
    if defaults:
        out.update({str(k): str(v) for k, v in defaults.items()})
    if overrides:
        out.update({str(k): str(v) for k, v in overrides.items()})
    return out


def start_metrics_http_server(port: int, addr: str = "") -> None:
    """Start a background thread serving ``/metrics`` via ``prometheus_client``.

    Args:
        port: TCP port to listen on.
        addr: Bind address; empty may mean all interfaces depending on library defaults.
    """
    start_http_server(port, addr=addr)


class MetricRegistry:
    """Lazy registry of ``prometheus_client`` Counter/Gauge/Histogram objects keyed by metric name + label set."""

    def __init__(self, namespace: str = "", subsystem: str = "") -> None:
        """
        Args:
            namespace: Passed as Prometheus metric namespace (prefix segment).
            subsystem: Optional second prefix segment from ``prometheus_client`` API.
        """
        self._namespace = namespace
        self._subsystem = subsystem
        self._counters: dict[tuple[str, tuple[str, ...]], object] = {}
        self._gauges: dict[tuple[str, tuple[str, ...]], object] = {}
        self._histograms: dict[tuple[str, tuple[str, ...]], object] = {}

    def _get_or_create_counter(
        self, name: str, documentation: str, label_names: tuple[str, ...]
    ):
        """Return a cached or new ``Counter`` for ``(name, label_names)``."""
        key = (name, label_names)
        if key not in self._counters:
            self._counters[key] = Counter(
                name,
                documentation,
                labelnames=label_names,
                namespace=self._namespace,
                subsystem=self._subsystem,
            )
        return self._counters[key]

    def _get_or_create_gauge(
        self, name: str, documentation: str, label_names: tuple[str, ...]
    ):
        """Return a cached or new ``Gauge`` for ``(name, label_names)``."""
        key = (name, label_names)
        if key not in self._gauges:
            self._gauges[key] = Gauge(
                name,
                documentation,
                labelnames=label_names,
                namespace=self._namespace,
                subsystem=self._subsystem,
            )
        return self._gauges[key]

    def _get_or_create_histogram(
        self,
        name: str,
        documentation: str,
        label_names: tuple[str, ...],
        buckets: tuple[float, ...] | None,
    ):
        """Return a cached or new ``Histogram`` for ``(name, label_names)`` with optional bucket boundaries."""
        key = (name, label_names)
        if key not in self._histograms:
            kw = {}
            if buckets is not None:
                kw["buckets"] = buckets
            self._histograms[key] = Histogram(
                name,
                documentation,
                labelnames=label_names,
                namespace=self._namespace,
                subsystem=self._subsystem,
                **kw,
            )
        return self._histograms[key]

    def count(
        self,
        name: str,
        documentation: str,
        amount: float,
        defaults: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        """Increment a counter, merging ``defaults`` and ``labels`` into the label set."""
        merged = _merge_labels(defaults, labels)
        names = tuple(sorted(merged.keys()))
        counter = self._get_or_create_counter(name, documentation, names)
        if merged:
            counter.labels(**merged).inc(amount)
        else:
            counter.inc(amount)

    def value(
        self,
        name: str,
        documentation: str,
        value: float,
        defaults: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        """Set a gauge to ``value`` with merged labels."""
        merged = _merge_labels(defaults, labels)
        names = tuple(sorted(merged.keys()))
        gauge = self._get_or_create_gauge(name, documentation, names)
        if merged:
            gauge.labels(**merged).set(value)
        else:
            gauge.set(value)

    def distribution(
        self,
        name: str,
        documentation: str,
        value: float,
        defaults: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        """Observe ``value`` on a histogram with merged labels and optional ``buckets``."""
        merged = _merge_labels(defaults, labels)
        names = tuple(sorted(merged.keys()))
        histogram = self._get_or_create_histogram(name, documentation, names, buckets)
        if merged:
            histogram.labels(**merged).observe(value)
        else:
            histogram.observe(value)


class PrometheusScrapeUpdater:
    """Rewrite on-disk Prometheus scrape config and trigger ``/-/reload`` on each Ray node."""

    def __init__(
        self,
        config: Mapping[str, Any] | DictConfig | None,
        *,
        job_name: str | None = None,
    ) -> None:
        from .monitor_config_loader import load_monitor_config

        conf = load_monitor_config(config)
        self._backend = (
            str(OmegaConf.select(conf, "server.backend") or "ray").strip().lower()
        )
        self._job_name = job_name or PrometheusScrape.TRAINER_METRICS_JOB
        self._prom_file = str(conf.prometheus.config_file)
        self._reload_port = int(conf.prometheus.prometheus_port)

    def update(self, server_addresses: list[str]) -> None:
        """Rewrite on-disk Prometheus scrape config for ``server_addresses`` and POST ``/-/reload``.

        Args:
            server_addresses: ``host:port`` targets (typically the hub ``/metrics`` endpoints).

        Note:
            No-op unless ``server.backend`` is ``ray``; requires a running Ray cluster.
        """
        if not server_addresses:
            logger.warning("No server addresses available to update Prometheus config")
            return
        if self._backend != "ray":
            logger.warning(
                "server.backend is %r; only %r supports Prometheus scrape update and reload.",
                self._backend,
                "ray",
            )
            return

        try:
            with open(self._prom_file, encoding="utf-8") as f:
                prometheus_data = yaml.safe_load(f) or {}
            scrape_configs = prometheus_data.setdefault("scrape_configs", [])
            new_job = {
                "job_name": self._job_name,
                "static_configs": [{"targets": server_addresses}],
            }
            for i, sc in enumerate(scrape_configs):
                if sc.get("job_name") == self._job_name:
                    scrape_configs[i] = new_job
                    break
            else:
                scrape_configs.append(new_job)
        except Exception as e:
            logger.error(
                "Failed to read or merge Prometheus config %s: %s", self._prom_file, e
            )
            return

        msg = (
            f"Updated Prometheus configuration at {self._prom_file} with "
            f"{len(server_addresses)} targets (job_name={self._job_name})"
        )

        try:
            alive_nodes = [node for node in ray.nodes() if node["Alive"]]
            ray.get(
                [
                    _write_prometheus_config_file.options(
                        resources={"node:" + node["NodeManagerAddress"]: 0.001}
                    ).remote(prometheus_data, self._prom_file)
                    for node in alive_nodes
                ]
                + [
                    _reload_prometheus_on_node.options(
                        resources={"node:" + node["NodeManagerAddress"]: 0.001}
                    ).remote(self._reload_port, None)
                    for node in alive_nodes
                ]
            )
            print(msg)
        except Exception as e:
            logger.error("Failed to update Prometheus configuration: %s", e)


def update_prometheus_config(
    config: Mapping[str, Any] | DictConfig | None,
    server_addresses: list[str],
    job_name: str | None = None,
) -> None:
    """Rewrite on-disk Prometheus scrape config for ``server_addresses`` and POST ``/-/reload``.

    Args:
        config: Trainer monitor config (or ``None`` for defaults); used for paths and backend selection.
        server_addresses: ``host:port`` targets (typically the hub ``/metrics`` endpoints).
        job_name: Override scrape job name; default uses ``PrometheusScrape.TRAINER_METRICS_JOB``.

    Note:
        No-op unless ``server.backend`` is ``ray``; requires a running Ray cluster.
    """
    PrometheusScrapeUpdater(config, job_name=job_name).update(server_addresses)
