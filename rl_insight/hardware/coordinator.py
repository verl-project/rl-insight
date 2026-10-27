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

"""Background hardware monitor loop started by ``rl_insight.init``."""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping

from ..utils.constants import MonitorEventKind, MonitorHardware
from .base import COLLECTOR_REGISTRY, HardwareCollector, Sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["HardwareMonitor"]

# Consecutive ``collect`` failures before a collector is deactivated for the run.
_MAX_CONSECUTIVE_FAILURES = 3
# How long ``stop`` waits for the sampling thread to exit. Must exceed the
# slowest single collector call (the ``npu-smi`` subprocess timeout is 5s) so
# a stop during sampling still joins the thread instead of returning while it
# is blocked on a subprocess.
_JOIN_TIMEOUT_SECONDS = 6.0


class _CollectorEntry:
    """Runtime bookkeeping for one resolved collector."""

    def __init__(
        self,
        name: str,
        collector: HardwareCollector | None,
        available: bool,
    ) -> None:
        self.name = name
        self.collector = collector
        self.available = available
        self.consecutive_failures = 0


class HardwareMonitor:
    """Periodically sample registered hardware collectors and emit gauge events.

    The monitor owns one daemon thread: ``start`` returns immediately and all
    capability discovery, sampling, and event forwarding happen in the
    background, so ``rl_insight.init`` never blocks on hardware probing and
    the training loop is never disturbed. Collector and client failures are
    downgraded to diagnostics (warnings plus ``rlinsight_hardware_collector_available``
    status gauges); the loop itself never raises. Samples of one round are
    batched and submitted through ``client.apply_events`` in a single call;
    batches that the transport cannot deliver are dropped (with a warning)
    instead of accumulating, since a failed transport cannot report its own
    failures either.
    """

    def __init__(
        self,
        *,
        client: Any,
        labels: Mapping[str, Any] | None = None,
        interval_seconds: float = MonitorHardware.INTERVAL_SECONDS,
        collector_names: list[str] | None = None,
        collector_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            client: Monitor client with ``apply_event``/``apply_events`` (fire-and-forget).
            labels: Base labels attached to every emitted event (project, node, ...).
            interval_seconds: Seconds between sampling rounds; callers facing user
                config should enforce the documented minimum (see
                ``start_hardware_monitor``). Non-positive values fall back to
                the default interval.
            collector_names: Collector names to resolve against the registry;
                ``None`` uses every registered collector.
            collector_options: Per-collector options (keyed by collector name)
                passed to each collector's ``configure`` before discovery.
        """
        self._client = client
        self._labels = {
            str(key): str(value) for key, value in dict(labels or {}).items()
        }
        interval = float(interval_seconds)
        if interval <= 0:
            interval = float(MonitorHardware.INTERVAL_SECONDS)
        self._interval = interval
        self._collector_names = (
            None if collector_names is None else [str(name) for name in collector_names]
        )
        self._collector_options = {
            str(name): dict(options)
            for name, options in dict(collector_options or {}).items()
        }
        self._entries: list[_CollectorEntry] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._batch: list[dict[str, Any]] = []
        self._dropped_events = 0

    @property
    def interval_seconds(self) -> float:
        """Seconds between sampling rounds."""
        return self._interval

    @property
    def is_running(self) -> bool:
        """Whether the sampling thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def dropped_events(self) -> int:
        """Number of events dropped so far because the transport failed."""
        return self._dropped_events

    def start(self) -> bool:
        """Spawn the sampling thread once.

        Returns:
            True if this call started the thread; False if it was already running.
        """
        with self._start_lock:
            if self.is_running:
                return False
            self._stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                name="rl-insight-hardware-monitor",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return True

    def stop(self, timeout: float = _JOIN_TIMEOUT_SECONDS) -> None:
        """Signal the sampling thread to stop and briefly wait for it to exit.

        Idempotent and safe to call from any thread, including before ``start``.

        Args:
            timeout: Seconds to wait for the thread to exit.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Resolve collectors, then sample forever until stopped."""
        try:
            self._resolve_and_report()
            while not self._stop_event.is_set():
                self._collect_once()
                if self._stop_event.wait(self._interval):
                    break
        except Exception:  # last resort: the loop must never raise noisily
            logger.exception(
                "[rl-insight] hardware monitor loop crashed; hardware metrics stopped."
            )

    def _resolve_and_report(self) -> None:
        """Resolve configured collectors, probe each, and emit status gauges."""
        names = self._collector_names
        if names is None:
            names = sorted(COLLECTOR_REGISTRY)
        self._entries = [self._resolve_one(name) for name in names]
        for entry in self._entries:
            self._queue_status(entry)
        self._flush()
        available = [entry.name for entry in self._entries if entry.available]
        unavailable = [entry.name for entry in self._entries if not entry.available]
        logger.info(
            "[rl-insight] hardware monitor ready (interval %.1fs): available=%s "
            "unavailable=%s",
            self._interval,
            available,
            unavailable,
        )

    def _resolve_one(self, name: str) -> _CollectorEntry:
        """Build one collector entry, degrading to unavailable on any problem."""
        factory = COLLECTOR_REGISTRY.get(name)
        if factory is None:
            logger.info(
                "[rl-insight] hardware collector %r is not registered in this "
                "build; reporting it as unavailable.",
                name,
            )
            return _CollectorEntry(name=name, collector=None, available=False)
        try:
            collector = factory()
            collector.configure(self._collector_options.get(name, {}))
            available = bool(collector.discover())
        except Exception:
            logger.warning(
                "[rl-insight] hardware collector %r discovery failed; reporting "
                "it as unavailable.",
                name,
                exc_info=True,
            )
            return _CollectorEntry(name=name, collector=None, available=False)
        return _CollectorEntry(name=name, collector=collector, available=available)

    def _collect_once(self) -> None:
        """Run one sampling round across all available collectors and flush."""
        for entry in self._entries:
            if entry.collector is None or not entry.available:
                continue
            try:
                samples = entry.collector.collect()
                for sample in samples or []:
                    self._queue_sample(sample)
            except Exception:
                self._register_failure(entry)
                continue
            entry.consecutive_failures = 0
        self._flush()

    def _register_failure(self, entry: _CollectorEntry) -> None:
        """Count one collection failure and deactivate a persistent one."""
        entry.consecutive_failures += 1
        logger.warning(
            "[rl-insight] hardware collector %r failed %d time(s) in a row.",
            entry.name,
            entry.consecutive_failures,
        )
        if entry.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            entry.available = False
            logger.warning(
                "[rl-insight] hardware collector %r keeps failing; deactivating "
                "it for this run.",
                entry.name,
            )
            self._queue_status(entry)

    # ------------------------------------------------------------------
    # Batched event emission
    # ------------------------------------------------------------------

    def _queue_sample(self, sample: Sample) -> None:
        """Append one collector sample to the pending batch."""
        try:
            value = float(sample.value)
        except (TypeError, ValueError):
            logger.warning(
                "[rl-insight] dropping hardware sample %r: value %r is not numeric.",
                sample.name,
                sample.value,
            )
            return
        labels = {
            **self._labels,
            **{str(k): str(v) for k, v in dict(sample.labels or {}).items()},
        }
        self._batch.append(
            {
                "kind": MonitorEventKind.GAUGE,
                "name": str(sample.name),
                "documentation": str(sample.documentation or f"Gauge {sample.name}"),
                "value": value,
                "labels": labels,
            }
        )

    def _queue_status(self, entry: _CollectorEntry) -> None:
        """Append one collector availability gauge to the pending batch."""
        self._batch.append(
            {
                "kind": MonitorEventKind.GAUGE,
                "name": MonitorHardware.STATUS_METRIC_NAME,
                "documentation": (
                    "Hardware collector availability (1=available, 0=unavailable)."
                ),
                "value": 1.0 if entry.available else 0.0,
                "labels": {**self._labels, "collector": entry.name},
            }
        )

    def _flush(self) -> None:
        """Submit the pending batch in one call; transport failures drop it.

        A failed batch is dropped rather than retried or buffered, so slow or
        broken transports backpressure instead of growing memory. Dropped
        events are counted locally (see :attr:`dropped_events`) and logged; no
        dropped-events metric is emitted because the transport that would carry
        it is the same one that just failed, and the fire-and-forget clients do
        not surface delivery errors at all.
        """
        if not self._batch:
            return
        batch = self._batch
        self._batch = []
        try:
            self._client.apply_events(batch)
        except Exception:
            self._register_dropped(len(batch))

    def _register_dropped(self, count: int) -> None:
        """Record dropped events as a local count and a warning."""
        self._dropped_events += count
        logger.warning(
            "[rl-insight] failed to forward %d hardware events; dropping them.",
            count,
            exc_info=True,
        )
