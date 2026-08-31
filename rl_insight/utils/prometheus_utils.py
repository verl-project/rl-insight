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

"""Prometheus metric registry, ``/metrics`` HTTP server, and target registration helpers."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml
from omegaconf import DictConfig, OmegaConf
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ..server.network import format_host_port, local_addresses
from .constants import (
    MonitorEnv,
    MonitorPaths,
    PrometheusScrape,
    prometheus_targets_file_from_config,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


__all__ = [
    "MetricRegistry",
    "PrometheusTarget",
    "PrometheusTargetStore",
    "start_metrics_http_server",
    "update_prometheus_config",
]


@dataclass(frozen=True)
class PrometheusTarget:
    """One scrape target plus optional labels for Prometheus static_configs."""

    target: str
    labels: Mapping[str, Any] = field(default_factory=dict)


class PrometheusTargetStore:
    """Maintain Prometheus scrape targets in a file_sd discovery file."""

    def __init__(
        self,
        config_file: str | Path,
        prometheus_port: int,
        targets_file: str | Path | None = None,
    ):
        self.config_file = Path(config_file).expanduser().resolve()
        self.targets_file = (
            Path(targets_file).expanduser().resolve()
            if targets_file is not None
            else self.config_file.with_name(PrometheusScrape.TARGETS_FILE_NAME)
        )
        self.prometheus_port = prometheus_port
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, conf: DictConfig) -> "PrometheusTargetStore":
        runtime_dir = OmegaConf.select(conf, "server.runtime_dir")
        base = (
            Path(str(runtime_dir)).expanduser().resolve()
            if runtime_dir
            else (MonitorPaths.STATE_ROOT / "runtime").resolve()
        )
        prometheus_port = int(OmegaConf.select(conf, "prometheus.prometheus_port"))
        return cls(
            base / "prometheus.yml",
            prometheus_port,
            prometheus_targets_file_from_config(conf),
        )

    def register(
        self, job_name: str, targets: Sequence[PrometheusTarget]
    ) -> dict[str, Any]:
        incoming = {
            str(item.target): {str(k): str(v) for k, v in item.labels.items()}
            for item in targets
        }

        with self._lock, self._file_lock():
            target_map = self._read_targets()
            target_map.update(
                {(str(job_name), target): labels for target, labels in incoming.items()}
            )
            self._write_targets(target_map)

        return {
            "job_name": job_name,
            "target_count": sum(
                1 for stored_job, _target in target_map if stored_job == job_name
            ),
            "config_file": str(self.config_file),
            "targets_file": str(self.targets_file),
        }

    @contextmanager
    def _file_lock(self):
        """Serialize read-modify-write updates made by different processes."""
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError(
                "Prometheus target persistence requires a POSIX server"
            ) from exc

        self.targets_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.targets_file.with_name(f".{self.targets_file.name}.lock")
        with lock_file.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_targets(
        self, target_map: Mapping[tuple[str, str], Mapping[str, str]]
    ) -> None:
        groups = [
            {
                "targets": [target],
                "labels": {
                    **labels,
                    PrometheusScrape.DYNAMIC_JOB_LABEL: stored_job,
                },
            }
            for (stored_job, target), labels in sorted(target_map.items())
        ]
        payload = yaml.safe_dump(groups, sort_keys=False)
        tmp_path = self.targets_file.with_name(
            f".{self.targets_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, self.targets_file)
        except BaseException:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _read_targets(self) -> dict[tuple[str, str], dict[str, str]]:
        if not self.targets_file.exists():
            return {}
        groups = yaml.safe_load(self.targets_file.read_text(encoding="utf-8")) or []
        if not isinstance(groups, list):
            raise ValueError("Prometheus file_sd targets must be a list")

        target_map: dict[tuple[str, str], dict[str, str]] = {}
        for group in groups:
            if not isinstance(group, Mapping):
                raise ValueError(
                    "Each Prometheus file_sd target group must be an object"
                )
            raw_labels = group.get("labels") or {}
            if not isinstance(raw_labels, Mapping):
                raise ValueError("Prometheus file_sd target labels must be an object")
            labels = {str(key): str(value) for key, value in raw_labels.items()}
            stored_job = labels.pop(PrometheusScrape.DYNAMIC_JOB_LABEL, "").strip()
            if not stored_job:
                raise ValueError(
                    "Prometheus file_sd target group is missing managed job label"
                )
            for target in group.get("targets") or []:
                target_map[(stored_job, str(target))] = dict(labels)
        return target_map

    def reload(self) -> bool:
        url = (
            "http://"
            + format_host_port(local_addresses()["loopback"], self.prometheus_port)
            + "/-/reload"
        )

        with requests.Session() as session:
            # bypass http_proxy for localhost Prometheus reload
            session.trust_env = False
            response = session.post(url, timeout=5)

        response.raise_for_status()
        return True


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
    addr = addr.strip("[]") if addr else addr
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


def update_prometheus_config(
    server_addresses: list[str],
    job_name: str | None = None,
    labels: list[Mapping[str, Any] | None] | None = None,
) -> None:
    """Register trainer metrics endpoints with the RL-Insight server.

    The RL-Insight server writes these targets into its file_sd discovery file;
    Prometheus observes the update automatically. ``server_addresses`` should
    contain scrape targets in ``host:port`` or ``[ipv6]:port`` form, not full URLs.

    Args:
        server_addresses: Prometheus scrape targets exposed by trainer-side
            metric HTTP servers.
        job_name: Optional Prometheus scrape job name. Defaults to the managed
            trainer metrics job.
        labels: Optional per-target labels. When provided, its length must match
            ``server_addresses``.
    """
    if not server_addresses:
        logger.warning("[rl-insight] No server addresses available to register")
        return
    if labels is not None and len(labels) != len(server_addresses):
        raise ValueError(
            "labels length must match server_addresses length: "
            f"{len(labels)} != {len(server_addresses)}"
        )

    base_url = str(os.environ.get(MonitorEnv.SERVER_URL, "")).strip().rstrip("/")
    if not base_url:
        logger.error(
            "[rl-insight] RL-Insight server URL is required; "
            "set %s to register Prometheus targets",
            MonitorEnv.SERVER_URL,
        )
        return

    payload = {
        "job_name": job_name or PrometheusScrape.TRAINER_METRICS_JOB,
        "targets": _build_target_payload(server_addresses, labels),
    }
    url = f"{base_url}/api/v1/prometheus/targets"
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(
            f"[rl-insight] Registered {len(server_addresses)} Prometheus targets "
            f"with RL-Insight server (job_name={payload['job_name']})"
        )
    except requests.RequestException as exc:
        logger.error(
            "[rl-insight] Failed to register Prometheus targets at %s: %s", url, exc
        )


def _build_target_payload(
    server_addresses: list[str],
    labels: list[Mapping[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    if labels is None:
        labels = [None] * len(server_addresses)

    targets: list[dict[str, Any]] = []
    for address, target_labels in zip(server_addresses, labels):
        item: dict[str, Any] = {"target": str(address)}
        if target_labels:
            item["labels"] = {
                str(key): str(value) for key, value in target_labels.items()
            }
        targets.append(item)
    return targets
