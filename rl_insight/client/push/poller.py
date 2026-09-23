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

"""Background polling of rollout-engine Prometheus endpoints.

The active poller publishes its :class:`TargetRegistry` process-wide so
``update_prometheus_config`` can register rollout ``/metrics`` targets in
process (instead of POSTing them to a self-hosted server) whenever the push
backend is active.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from .translate import ScrapeState

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

EmitCallback = Callable[[str, str, float, Mapping[str, str]], None]
FetchCallback = Callable[[str], str]


@dataclass(frozen=True)
class Target:
    """One scrape target and its static labels (e.g. ``replica``)."""

    address: str
    labels: dict[str, str] = field(default_factory=dict)


class TargetRegistry:
    """Process-local, de-duplicated set of rollout scrape targets."""

    def __init__(self) -> None:
        self._targets: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def set_targets(
        self,
        addresses: Sequence[str],
        labels: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> None:
        """Register/replace targets keyed by address; later calls upsert."""
        per_target = list(labels) if labels is not None else [None] * len(addresses)
        with self._lock:
            for address, target_labels in zip(addresses, per_target):
                self._targets[str(address)] = {
                    str(k): str(v) for k, v in (target_labels or {}).items()
                }

    def snapshot(self) -> list[Target]:
        with self._lock:
            return [
                Target(address=address, labels=dict(labels))
                for address, labels in sorted(self._targets.items())
            ]


def _http_fetch_metrics(address: str) -> str:
    """Fetch Prometheus text exposition from ``host:port`` via localhost-safe HTTP."""
    url = address.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    url = url.rstrip("/")
    if not url.endswith("/metrics"):
        url = url + "/metrics"
    session = requests.Session()
    session.trust_env = False  # bypass proxy for engine endpoints on localhost
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    finally:
        session.close()


class PrometheusPoller:
    """Poll registered targets on a daemon thread and translate to sink ops."""

    def __init__(
        self,
        rules: Any,
        interval_seconds: float,
        emit: EmitCallback,
        fetch: FetchCallback | None = None,
    ) -> None:
        self._rules = rules
        self._interval = float(interval_seconds)
        self._emit = emit
        self._fetch = fetch or _http_fetch_metrics
        self.targets = TargetRegistry()
        self._states: dict[str, ScrapeState] = {}
        # Per-target scrape-failure counts so network problems are surfaced a
        # few times (default log level hides ``debug``) without spamming every
        # interval for the whole training run.
        self._scrape_failures: dict[str, int] = {}
        self._empty_warned: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the daemon poll loop (no-op if already running or interval <= 0)."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._interval <= 0:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="rl-insight-push-poller", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the loop to stop and wait briefly for it to finish."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll_once()

    def poll_once(self) -> None:
        """Scrape every target once and emit translated operations."""
        for target in self.targets.snapshot():
            try:
                text = self._fetch(target.address)
            except Exception as exc:  # noqa: BLE001 - network failures must not kill training
                failures = self._scrape_failures.get(target.address, 0) + 1
                self._scrape_failures[target.address] = failures
                # Debug-only hides a broken engine endpoint forever at WARNING
                # level; warn for the first few failures per target.
                if failures <= 3:
                    logger.warning(
                        "[rl-insight] rollout scrape %s failed (%d): %s",
                        target.address,
                        failures,
                        exc,
                    )
                else:
                    logger.debug(
                        "[rl-insight] rollout scrape %s failed: %s", target.address, exc
                    )
                continue
            self._scrape_failures.pop(target.address, None)
            state = self._states.setdefault(target.address, ScrapeState())
            try:
                emissions = state.translate(text, self._rules)
            except Exception as exc:  # noqa: BLE001 - parse failures are non-fatal
                logger.warning(
                    "[rl-insight] rollout translate %s failed: %s", target.address, exc
                )
                continue
            if not emissions and target.address not in self._empty_warned:
                # A reachable /metrics endpoint that yields no mapped family
                # usually means a rule source-name mismatch (vLLM renamed the
                # metric). Warn once per target instead of silently emitting 0.
                self._empty_warned.add(target.address)
                logger.warning(
                    "[rl-insight] rollout scrape %s succeeded but no configured "
                    "metric matched; check rollout.metrics source names",
                    target.address,
                )
            for emission in emissions:
                tags = {**target.labels, **emission.tags}
                try:
                    self._emit(emission.name, emission.op, emission.value, tags)
                except Exception as exc:  # noqa: BLE001 - emit failures are non-fatal
                    logger.debug("[rl-insight] rollout emit failed: %s", exc)


# Process-wide active rollout target registry; set by the push client when active.
_ACTIVE_REGISTRY: TargetRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def set_active_registry(registry: TargetRegistry | None) -> None:
    """Install (or clear with ``None``) the active push rollout registry."""
    global _ACTIVE_REGISTRY
    with _REGISTRY_LOCK:
        _ACTIVE_REGISTRY = registry


def get_active_registry() -> TargetRegistry | None:
    """Return the active push rollout registry, or ``None`` when push is inactive."""
    with _REGISTRY_LOCK:
        return _ACTIVE_REGISTRY
