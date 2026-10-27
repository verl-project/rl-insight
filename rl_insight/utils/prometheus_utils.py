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
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
import yaml
from omegaconf import DictConfig, OmegaConf
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ..server.network import format_host_port, local_addresses
from .constants import (
    MonitorBackend,
    MonitorEnv,
    MonitorPaths,
    PrometheusScrape,
    prometheus_targets_file_from_config,
)
from .protocol import RunContext, RunError

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

_pending_targets: list[
    tuple[list[str], str | None, list[Mapping[str, Any] | None] | None]
] = []
_pending_lock = threading.Lock()
_pending_probe: threading.Thread | None = None


def _wait_for_context() -> None:
    """Retry pending native targets for the lifetime of their worker process."""
    global _pending_probe
    from ray.exceptions import RayActorError

    from ..client.ray_monitor_client import current_job_context

    delay = 0.1
    while True:
        with _pending_lock:
            if not _pending_targets:
                _pending_probe = None
                return
        try:
            context = current_job_context()
            if context:
                flush_pending_targets(context)
        except (RunError, RayActorError) as exc:
            logger.error("[rl-insight] Cannot resolve native target identity: %s", exc)
            with _pending_lock:
                _pending_targets.clear()
                _pending_probe = None
            return
        time.sleep(delay)
        delay = min(delay * 2, 2)


def flush_pending_targets(context: RunContext | None) -> None:
    with _pending_lock:
        pending = list(_pending_targets)
    for item in pending:
        try:
            done = _register_targets(*item, context)
        except RunError as exc:
            logger.error("[rl-insight] Native target registration rejected: %s", exc)
            done = True
        if done:
            with _pending_lock:
                if item in _pending_targets:
                    _pending_targets.remove(item)


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
        self.targets_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.targets_file.with_name(f".{self.targets_file.name}.lock")
        with lock_file.open("a+", encoding="utf-8") as handle:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if sys.platform == "win32":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
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
        self._write_discovery(
            self.targets_file, yaml.safe_dump(groups, sort_keys=False)
        )

    @staticmethod
    def _write_discovery(path: Path, payload: str) -> None:
        if path.exists() and path.read_text(encoding="utf-8") == payload:
            return
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(payload, encoding="utf-8")
            for attempt in range(6):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError as exc:
                    # Windows readers can briefly deny delete/replace sharing.
                    if getattr(exc, "winerror", None) not in {5, 32} or attempt == 5:
                        raise
                    time.sleep(0.02 * 2**attempt)
        finally:
            temporary.unlink(missing_ok=True)

    def check_ownership(self, targets, run_id):
        with self._lock, self._file_lock():
            addresses = {t.target for t in targets}
            for (_, address), labels in self._read_targets().items():
                owner = labels.get("__rl_insight_run_id", labels.get("run_id"))
                if address in addresses and owner != run_id:
                    raise RunError(
                        "Target address already has different Run ownership", 409
                    )

    def project_runs(self, targets, ended_runs=()):
        """Keep legacy registrations and project renewable targets into existing file_sd."""
        with self._lock, self._file_lock():
            direct = {
                key: labels
                for key, labels in self._read_targets().items()
                if "__rl_insight_publisher" not in labels
                and labels.get("__rl_insight_run_id", labels.get("run_id"))
                not in ended_runs
            }
            direct.update(
                {
                    (item["job_name"], item["target"]): {
                        **item["labels"],
                        "__rl_insight_publisher": item["publisher_id"],
                    }
                    for item in targets
                }
            )
            self._write_targets(direct)

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


def start_metrics_http_server(port: int, addr: str = ""):
    """Bind once and return the actual HTTP server/thread; port 0 allocates a port."""
    return start_http_server(port, addr=addr.strip("[]") if addr else addr)


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


_push_bootstrap_lock = threading.Lock()
_push_bootstrap_attempted = False


def _bootstrap_push_registry():
    """Start the push backend in a process without an explicit ``init()``.

    Some integrations (e.g. Ray rollout actors in verl) call
    :func:`update_prometheus_config` from a worker process that never ran
    ``rl_insight.init()``. For the direct-emit push backend the client needs no
    external server, so it is safe to construct it lazily here: building the
    push client also creates its :class:`PrometheusPoller` and publishes the
    process-wide scrape target registry. Returns that registry, or ``None``
    when the configured backend is not ``push`` / bootstrap is unavailable.
    """
    global _push_bootstrap_attempted
    with _push_bootstrap_lock:
        from ..client.push.poller import get_active_registry

        existing = get_active_registry()
        if existing is not None:
            return existing
        if _push_bootstrap_attempted:
            return None
        _push_bootstrap_attempted = True
        try:
            from .. import api as _api
            from .monitor_config_loader import load_monitor_config

            conf = load_monitor_config(None)
            backend = str(OmegaConf.select(conf, "server.backend") or "").strip()
            if backend != MonitorBackend.PUSH:
                return None
            _api.init()
        except Exception:  # noqa: BLE001 - monitoring must never break training
            logger.exception("[rl-insight] push backend in-process bootstrap failed")
            return None
        return get_active_registry()


def update_prometheus_config(
    server_addresses: list[str],
    job_name: str | None = None,
    labels: list[Mapping[str, Any] | None] | None = None,
    *,
    context: RunContext | None = None,
    identity_in_metrics: bool = False,
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
    global _pending_probe
    if not server_addresses:
        logger.warning("[rl-insight] No server addresses available to register")
        return
    if labels is not None and len(labels) != len(server_addresses):
        raise ValueError(
            "labels length must match server_addresses length: "
            f"{len(labels)} != {len(server_addresses)}"
        )

    # Push backend: no external RL-Insight server is involved. Register targets
    # in-process so the local rollout poller scrapes them directly.
    from ..client.push.poller import get_active_registry

    registry = get_active_registry()
    if registry is None:
        # Targets may be registered from a Ray actor process that never called
        # rl_insight.init() (e.g. the verl colocated rollout actor). For the
        # direct-emit push backend the client is self-contained, so lazily start
        # it in this process once; its poller owns the scrape target registry.
        registry = _bootstrap_push_registry()
    if registry is not None:
        registry.set_targets(server_addresses, labels)
        logger.info(
            "[rl-insight] Registered %d push scrape targets in-process",
            len(server_addresses),
        )
        return

    initialized = True
    if context is None:
        from ..api import _client_initialized, get_context

        context = get_context()
        initialized = _client_initialized()
    base_url = (
        context.server_url
        if context
        else str(os.environ.get(MonitorEnv.SERVER_URL, "")).strip().rstrip("/")
    )
    if not base_url:
        logger.error(
            "[rl-insight] RL-Insight server URL is required; "
            "set %s to register Prometheus targets",
            MonitorEnv.SERVER_URL,
        )
        return

    if context is None:
        if not initialized:
            from ..server.http_api import get_server_services

            if get_server_services():
                # Wait for logger initialization; never create a Run from an address.
                from ..client.ray_monitor_client import current_job_context

                context = current_job_context()
                item: tuple[
                    list[str], str | None, list[Mapping[str, Any] | None] | None
                ] = (
                    list(server_addresses),
                    job_name,
                    [dict(x) if x else None for x in labels] if labels else None,
                )
                if context is None:
                    with _pending_lock:
                        if item not in _pending_targets:
                            if len(_pending_targets) >= 64:
                                raise ValueError(
                                    "Too many targets waiting for Run initialization"
                                )
                            _pending_targets.append(item)
                        if _pending_probe is None or not _pending_probe.is_alive():
                            _pending_probe = threading.Thread(
                                target=_wait_for_context,
                                name="insight-context",
                                daemon=True,
                            )
                            _pending_probe.start()
                    return

    _register_targets(
        server_addresses,
        job_name,
        labels,
        context,
        identity_in_metrics=identity_in_metrics,
    )


_publisher_id = uuid4().hex
_target_publications: dict[tuple[str, str, str, str], tuple[dict[str, Any], float]] = {}
_publication_lock = threading.Lock()
_publication_thread: threading.Thread | None = None


def _forget_targets(context: RunContext | None) -> None:
    if context is not None:
        with _publication_lock:
            for key in list(_target_publications):
                if key[:2] == (context.server_url, context.run_id):
                    del _target_publications[key]


def _post_targets(base_url, payload):
    response = requests.post(
        base_url + "/api/v1/prometheus/targets", json=payload, timeout=10
    )
    response.raise_for_status()
    result = response.json()
    ttl = result.get("ttl") if isinstance(result, dict) else None
    return (
        max(0.25, float(ttl) / 3)
        if isinstance(ttl, (int, float)) and 1 <= ttl <= 3600
        else None
    )


def _registration_retry(base_url, payload, error):
    code = getattr(error.response, "status_code", None)
    if code == 409:
        try:
            response = requests.get(
                base_url + "/api/v1/runs/" + payload["run_context"]["run_id"], timeout=5
            )
            if response.status_code == 404 or (
                response.status_code == 200 and response.json().get("status") == "ended"
            ):
                return None
        except requests.RequestException:
            pass
    elif code is not None and 400 <= code < 500 and code not in {408, 429}:
        return None
    return 1.0


def _renew_targets():
    """One process-level renewal loop; no exporter-specific integration is needed."""
    global _publication_thread
    while True:
        with _publication_lock:
            if not _target_publications:
                _publication_thread = None
                return
            ready = [
                (key, value)
                for key, value in _target_publications.items()
                if value[1] <= time.monotonic()
            ]
        for key, (payload, _) in ready:
            delay = 1.0
            try:
                delay = _post_targets(key[0], payload)
            except requests.RequestException as exc:
                delay = _registration_retry(key[0], payload, exc)
                logger.warning("Target renewal failed: %s", exc)
            with _publication_lock:
                if (
                    key not in _target_publications
                    or _target_publications[key][0] is not payload
                ):
                    continue
                if delay is None:
                    _target_publications.pop(key, None)
                else:
                    _target_publications[key] = (payload, time.monotonic() + delay)
        time.sleep(0.1)


def _register_targets(
    server_addresses: list[str],
    job_name: str | None,
    labels: list[Mapping[str, Any] | None] | None,
    context: RunContext | None,
    *,
    identity_in_metrics: bool = False,
) -> bool:
    global _publication_thread
    base_url = (
        context.server_url
        if context
        else str(os.environ.get(MonitorEnv.SERVER_URL, "")).strip().rstrip("/")
    )
    if not base_url:
        return False
    payload: dict[str, Any] = {
        "job_name": job_name or PrometheusScrape.TRAINER_METRICS_JOB,
        "targets": _build_target_payload(server_addresses, labels),
    }
    if context:
        payload.update(
            run_context=context.to_dict(),
            identity_in_metrics=identity_in_metrics,
            publisher_id=_publisher_id,
        )
    success, delay = True, None
    try:
        delay = _post_targets(base_url, payload)
    except requests.RequestException as exc:
        success = False
        delay = _registration_retry(base_url, payload, exc) if context else None
        if (
            delay is None
            and exc.response is not None
            and 400 <= exc.response.status_code < 500
            and exc.response.status_code not in {408, 429}
        ):
            raise RunError(
                "Target registration rejected: " + exc.response.text[:512],
                exc.response.status_code,
            ) from exc
        logger.warning("Target registration failed: %s", exc)
    if context and delay is not None:
        with _publication_lock:
            for target in payload["targets"]:
                key = (base_url, context.run_id, payload["job_name"], target["target"])
                # Publish each target independently so later updates replace its labels.
                _target_publications[key] = (
                    {**payload, "targets": [target]},
                    time.monotonic() + delay,
                )
            if _publication_thread is None or not _publication_thread.is_alive():
                _publication_thread = threading.Thread(
                    target=_renew_targets, name="insight-targets", daemon=True
                )
                _publication_thread.start()
    return success


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
