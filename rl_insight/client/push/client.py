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

"""Push backend client: map monitor events to sink calls and fan out by stream."""

from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ...utils.constants import MonitorEventKind
from ..base import MonitorClient
from .driver import build_sink
from .poller import PrometheusPoller, set_active_registry
from .sinks.base import PushSink
from .tags import collect_env_tags

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

DEFAULT_STREAMS = ("metric", "trace")
DEFAULT_METRIC_PREFIX = "trainer."
DEFAULT_TRACE_PREFIX = "trace."
DEFAULT_ROLLOUT_PREFIX = "rollout."
DEFAULT_ROLLOUT_INTERVAL = 15.0

STREAM_METRIC = "metric"
STREAM_TRACE = "trace"
STREAM_ROLLOUT = "rollout"

_NS_TO_US = 1000


@dataclass
class _BoundSink:
    sink: PushSink
    streams: frozenset[str]
    rollout_prefix: str
    tags: dict[str, str]
    alive: bool = True


def _scalar_string_tags(tags: Any) -> dict[str, str]:
    """Keep only scalar tag values and stringify keys/values (drop sequences)."""
    out: dict[str, str] = {}
    for key, value in (tags or {}).items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            out[str(key)] = str(value)
    return out


class PushMonitorClient(MonitorClient):
    """Forward events in-process to configured sinks, routing by data stream."""

    def __init__(
        self,
        metric_prefix: str,
        trace_prefix: str,
        bound: list[_BoundSink],
        poller: PrometheusPoller | None,
    ) -> None:
        self._metric_prefix = metric_prefix
        self._trace_prefix = trace_prefix
        self._bound = bound
        self._poller = poller
        self._closed = False
        atexit.register(self.close)

    # -- MonitorClient -----------------------------------------------------
    def apply_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind"))
        if kind == MonitorEventKind.TRACE:
            self._route_trace(event)
        else:
            self._route_metric(kind, event)

    # -- routing -----------------------------------------------------------
    def _route_metric(self, kind: str, event: dict[str, Any]) -> None:
        suffix = f"{self._metric_prefix}{event.get('name', '')}"
        value = float(event.get("value", 0.0))
        event_tags = _scalar_string_tags(event.get("labels"))
        for bound in list(self._bound):
            if not bound.alive or STREAM_METRIC not in bound.streams:
                continue
            tags = {**bound.tags, **event_tags}
            if kind == MonitorEventKind.COUNTER:
                self._safe(bound, "emit_counter", suffix, value, tags)
            elif kind == MonitorEventKind.GAUGE:
                self._safe(bound, "emit_store", suffix, value, tags)
            else:  # histogram -> timer; producers must report microseconds
                self._safe(bound, "emit_timer", suffix, value, tags)

    def _route_trace(self, event: dict[str, Any]) -> None:
        duration_us = (
            int(event.get("end_time_ns", 0)) - int(event.get("start_time_ns", 0))
        ) // _NS_TO_US
        suffix = f"{self._trace_prefix}{event.get('name', '')}"
        event_tags = _scalar_string_tags(event.get("attributes"))
        for bound in list(self._bound):
            if not bound.alive or STREAM_TRACE not in bound.streams:
                continue
            tags = {**bound.tags, **event_tags}
            self._safe(bound, "emit_timer", suffix, float(duration_us), tags)

    def emit_rollout(
        self, name: str, op: str, value: float, tags: dict[str, str]
    ) -> None:
        """Sink callback used by the rollout poller for translated engine metrics."""
        method = {
            "counter": "emit_counter",
            "store": "emit_store",
            "timer": "emit_timer",
        }.get(op, "emit_store")
        for bound in list(self._bound):
            if not bound.alive or STREAM_ROLLOUT not in bound.streams:
                continue
            suffix = f"{bound.rollout_prefix}{name}"
            merged = {**bound.tags, **dict(tags or {})}
            self._safe(bound, method, suffix, float(value), merged)

    def _safe(self, bound: _BoundSink, method: str, *args: Any) -> None:
        try:
            getattr(bound.sink, method)(*args)
        except Exception as exc:  # noqa: BLE001 - a sink must never break training
            if bound.alive:
                logger.warning(
                    "[rl-insight] push sink %r failed once (%s); disabling it.",
                    bound.sink.__class__.__name__,
                    exc,
                )
                bound.alive = False
                self._bound = [b for b in self._bound if b.alive]

    def close(self) -> None:
        """Stop polling and flush sinks; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._poller is not None:
            self._poller.stop()
        set_active_registry(None)
        for bound in self._bound:
            try:
                bound.sink.close()
            except Exception:  # noqa: BLE001 - best-effort shutdown
                logger.debug("[rl-insight] sink close failed", exc_info=True)
        try:
            atexit.unregister(self.close)
        except Exception:  # noqa: BLE001 - pragma: no cover
            pass


def _streams(sink_conf: Any) -> frozenset[str]:
    raw = OmegaConf.select(sink_conf, "streams")
    if raw is None:
        return frozenset(DEFAULT_STREAMS)
    return frozenset(str(item) for item in raw)


def _select_or(conf: Any, key: str, default: str) -> str:
    value = OmegaConf.select(conf, key)
    return default if value is None else str(value)


def create_push_monitor_client(conf: DictConfig) -> PushMonitorClient | None:
    """Build the push client from ``push`` config; return ``None`` when disabled.

    Returns ``None`` (monitoring off) when there is no ``push`` section, no
    usable sink, or every driver fails to load.
    """
    push_conf = OmegaConf.select(conf, "push")
    sink_confs = OmegaConf.select(push_conf, "sinks") if push_conf is not None else None
    if not sink_confs:
        logger.warning("[rl-insight] push backend selected but no sinks configured.")
        return None

    bound: list[_BoundSink] = []
    for sink_conf in sink_confs:
        sink = build_sink(sink_conf)
        if sink is None:
            continue
        bound.append(
            _BoundSink(
                sink=sink,
                streams=_streams(sink_conf),
                rollout_prefix=_select_or(
                    sink_conf, "rollout_name_prefix", DEFAULT_ROLLOUT_PREFIX
                ),
                tags=collect_env_tags(OmegaConf.select(sink_conf, "tags_from_env")),
            )
        )

    if not bound:
        logger.warning("[rl-insight] push backend has no usable sinks; disabled.")
        return None

    metric_prefix = _select_or(push_conf, "metric_prefix", DEFAULT_METRIC_PREFIX)
    trace_prefix = _select_or(push_conf, "trace_prefix", DEFAULT_TRACE_PREFIX)

    rules = OmegaConf.select(push_conf, "rollout.metrics")
    interval_raw = OmegaConf.select(push_conf, "rollout.interval_seconds")
    interval = (
        float(interval_raw) if interval_raw is not None else DEFAULT_ROLLOUT_INTERVAL
    )
    poller: PrometheusPoller | None = None
    if rules and any(STREAM_ROLLOUT in b.streams for b in bound):
        poller = PrometheusPoller(
            rules=rules,
            interval_seconds=interval,
            emit=lambda name, op, value, tags: client.emit_rollout(
                name, op, value, tags
            ),
        )
        set_active_registry(poller.targets)
    else:
        set_active_registry(None)

    client = PushMonitorClient(metric_prefix, trace_prefix, bound, poller)
    if poller is not None:
        poller.start()
    return client
