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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..utils.constants import MonitorEventKind, MonitorHardware
from .base import COLLECTOR_REGISTRY, HardwareCollector, Sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["HardwareMonitor"]

# Consecutive ``collect`` failures before a collector is deactivated for the run.
_MAX_CONSECUTIVE_FAILURES = 3
# How long ``stop`` waits for the sampling thread. A round already in flight,
# such as an ``npu-smi`` call, may outlive this timeout; the daemon thread then
# finishes on its own and the loop exits on the stop event.
_JOIN_TIMEOUT_SECONDS = 6.0


@dataclass
class _CollectorEntry:
    """Runtime bookkeeping for one resolved collector."""

    name: str
    collector: HardwareCollector | None
    available: bool
    consecutive_failures: int = 0


class HardwareMonitor:
    """Sample hardware collectors periodically and emit batched gauge events.

    The monitor owns one daemon thread, so ``rl_insight.init`` does not block on
    hardware probing. Collector and client failures are logged; the loop itself
    does not raise. Samples from one round are submitted through a single
    ``apply_events`` call, and undeliverable batches are dropped instead of
    accumulating.
    """

    def __init__(
        self,
        *,
        client: Any,
        labels: Mapping[str, Any] | None = None,
        interval_seconds: float = MonitorHardware.INTERVAL_SECONDS,
        collector_names: list[str] | None = None,
    ) -> None:
        """``interval_seconds <= 0`` falls back to the default interval."""
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
        self._entries: list[_CollectorEntry] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._emit_lock = threading.Lock()

    @property
    def interval_seconds(self) -> float:
        """Seconds between sampling rounds."""
        return self._interval

    @property
    def is_running(self) -> bool:
        """Whether the sampling thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the sampling thread once; return ``False`` if already running."""
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
        """Stop the sampling thread and wait up to ``timeout`` seconds.

        Idempotent and callable from any thread, including before ``start``.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        # Prevent a timed-out sampling round from emitting after the client is
        # closed by finish(). Any later flush sees the stop event and is dropped.
        with self._emit_lock:
            pass

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
        except Exception:  # last resort: keep thread failures out of the caller
            logger.exception(
                "[rl-insight] hardware monitor loop crashed; hardware metrics stopped."
            )

    def _resolve_and_report(self) -> None:
        """Resolve configured collectors, probe each, and log the result."""
        names = self._collector_names
        if names is None:
            names = list(MonitorHardware.COLLECTORS)
        self._entries = [self._resolve_one(name) for name in names]
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
        batch: list[dict[str, Any]] = []
        for entry in self._entries:
            if entry.collector is None or not entry.available:
                continue
            try:
                samples = entry.collector.collect()
                batch.extend(self._event(sample) for sample in samples)
            except Exception:
                self._register_failure(entry)
                continue
            entry.consecutive_failures = 0
        self._flush(batch)

    def _register_failure(self, entry: _CollectorEntry) -> None:
        """Count one collection failure and deactivate a persistent one."""
        entry.consecutive_failures += 1
        if entry.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            entry.available = False
            logger.warning(
                "[rl-insight] hardware collector %r failed %d time(s) in a row; "
                "deactivating it for this run.",
                entry.name,
                entry.consecutive_failures,
            )
        elif entry.consecutive_failures == 1:
            logger.warning(
                "[rl-insight] hardware collector %r failed; it will be "
                "deactivated after %d consecutive failures.",
                entry.name,
                _MAX_CONSECUTIVE_FAILURES,
            )
        else:
            logger.debug(
                "[rl-insight] hardware collector %r failed %d time(s) in a row.",
                entry.name,
                entry.consecutive_failures,
            )

    # ------------------------------------------------------------------
    # Batched event emission
    # ------------------------------------------------------------------

    def _event(self, sample: Sample) -> dict[str, Any]:
        """Convert one collector sample into a monitor gauge event."""
        return {
            "kind": MonitorEventKind.GAUGE,
            "name": str(sample.name),
            "documentation": str(sample.documentation or f"Gauge {sample.name}"),
            "value": sample.value,
            "labels": {
                **self._labels,
                **{
                    str(key): str(value)
                    for key, value in dict(sample.labels or {}).items()
                },
            },
        }

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        """Submit one round's batch in a single call.

        A failed batch is dropped instead of buffered, so a broken transport
        cannot grow memory. No metric is emitted because the failing transport
        would carry it.
        """
        with self._emit_lock:
            if self._stop_event.is_set() or not batch:
                return
            try:
                self._client.apply_events(batch)
            except Exception:
                logger.warning(
                    "[rl-insight] failed to forward %d hardware events; dropping them.",
                    len(batch),
                    exc_info=True,
                )
