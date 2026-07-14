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

"""High-level monitor API backed by a pluggable monitor client."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Mapping

from omegaconf import DictConfig

from .client import create_monitor_client
from .utils.monitor_config_loader import load_monitor_config
from .utils import MonitorEventKind

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = [
    "finish",
    "init",
    "metric_count",
    "metric_gauge",
    "metric_histogram",
    "trace_state",
    "trace_op",
]


@dataclass
class _MonitorState:
    """Per-process singleton state used by ``init`` and emit helpers.

    Attributes:
        enabled: True after ``init`` produced a non-null client.
        client: Backend object with ``apply_event`` (e.g. ``MonitorRayClient``).
        conf: Merged trainer monitor config.
        namespace: Config ``namespace`` used for metric/OTEL resource naming (not Ray actor namespace).
        process_id: String PID added to trace attributes on emit.
        labels: Process-wide labels attached to every metric and trace event.
    """

    enabled: bool = False
    client: Any | None = None
    conf: DictConfig | None = None
    namespace: str = ""
    process_id: str = field(default_factory=lambda: str(os.getpid()))
    labels: dict[str, Any] = field(default_factory=dict)


_STATE = _MonitorState()


def init(
    project: str | None = None,
    experiment_name: str | None = None,
    config: Mapping[str, Any] | DictConfig | None = None,
) -> None:
    """Load merged monitor config, create backend client, enable metric/trace helpers (once per process).

    Args:
        project: Optional project name attached to all metrics and traces as the ``project`` label/attribute.
        experiment_name: Optional experiment name attached to all metrics and traces as the
            ``experiment_name`` label/attribute.
        config: Optional dict-like or ``DictConfig`` overrides merged into training defaults; see ``load_monitor_config``.

    Note:
        Repeated calls are ignored with ``RuntimeWarning``. Ray backend requires ``ray.init()`` first.
    """
    global _STATE
    if _STATE.enabled:
        warnings.warn(
            "[rl-insight] monitor.init() called more than once; "
            "ignoring re-initialization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    monitor_conf = load_monitor_config(config)
    if not str(monitor_conf.server.url).strip():
        logger.error(
            "[rl-insight] RL-Insight server URL is required; set RL_INSIGHT_SERVER_URL "
            "or server.url in init config."
        )
        return
    client = create_monitor_client(monitor_conf)
    labels = {
        key: value
        for key, value in {
            "project": project,
            "experiment_name": experiment_name,
        }.items()
        if value is not None
    }
    _STATE = _MonitorState(
        enabled=client is not None,
        client=client,
        conf=monitor_conf,
        namespace=str(monitor_conf.server.namespace),
        labels=labels,
    )


def finish() -> None:
    """Clear in-process monitor state so further emits are no-ops.

    Does not stop the hub HTTP server or kill the detached Ray actor.
    """
    global _STATE
    _STATE = _MonitorState()
    _LANES.clear()


def metric_count(
    name: str, amount: float = 1.0, documentation: str = "", **labels: Any
) -> None:
    """Record a counter increment.

    Args:
        name: Metric name.
        amount: Increment amount (typically 1.0).
        documentation: Help string; default derived from ``name``.
        **labels: Extra label key-values attached to the event.
    """
    doc = documentation or f"Counter {name}"
    _emit(MonitorEventKind.COUNTER, name, float(amount), doc, labels)


def metric_gauge(
    name: str, value: float, documentation: str = "", **labels: Any
) -> None:
    """Record the latest value for a Prometheus gauge.

    Args:
        name: Metric name.
        value: Current value.
        documentation: Help string.
        **labels: Extra labels attached to the event.
    """
    doc = documentation or f"Gauge {name}"
    _emit(MonitorEventKind.GAUGE, name, float(value), doc, labels)


def metric_histogram(
    name: str, value: float, documentation: str = "", **labels: Any
) -> None:
    """Record one sample into a Prometheus histogram.

    Args:
        name: Metric name.
        value: Observed sample.
        documentation: Help string.
        **labels: Extra labels attached to the event.
    """
    doc = documentation or f"Histogram {name}"
    _emit(MonitorEventKind.HISTOGRAM, name, float(value), doc, labels)


@dataclass
class _LaneState:
    """Per-lane state so a lane reports one state at a time."""

    occupant: str | None = None
    count: int = 0
    start_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


# lane_id -> _LaneState, guarded by the lock for thread + coroutine safety.
_LANES: dict[str, _LaneState] = {}
_LANES_LOCK = threading.Lock()


@contextmanager
def trace_state(
    state_name: str,
    *,
    state_lane_id: str | None = None,
    **labels: Any,
) -> Generator[None, None, None]:
    """Record a named runtime state as a timeline interval on a logical lane.

    Overlapping states on the same ``state_lane_id`` are collapsed so that only
    one state is reported at any instant: same-name overlaps merge into one span,
    and a different-named state entered while the lane is occupied is dropped.

    ``state_lane_id`` groups spans into timeline columns/lanes, so Grafana can
    show one row per worker, replica, or process:

    ``time ->        t0              t1              t2              t3              t4``
    ``replica_0     | [generate responses---------------------------] [sync weights-----]``
    ``replica_1     | [generate responses---------------------------] [sync weights-----]``
    ``actor_worker_0   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_1   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_2   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_3   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_4   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_5   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_6   |       [compute logprob----] [update policy-----------------------]``
    ``actor_worker_7   |       [compute logprob----] [update policy-----------------------]``

    Args:
        state_name: Span name and human-readable state label (e.g. ``"rollout"``).
        state_lane_id: Optional id for grouping state intervals in trace UIs (swim lane).
            Defaults to the current OS process id: one lane per process unless you pass
            a custom id (e.g. Ray worker).
        **labels: Extra span attributes. Keys ``state_name``, ``state_lane_id``, and
            ``monitor.trace_segment`` cannot be overridden; they are set after merging.

    Yields:
        Control during the covered code block; the span is emitted on exit.
    """

    if not _STATE.enabled or _STATE.client is None:
        yield
        return

    lane_id = state_lane_id if state_lane_id is not None else _STATE.process_id
    attributes = {
        **labels,
        "monitor.trace_segment": "state_interval",
        "state_name": state_name,
        "state_lane_id": lane_id,
    }

    with _LANES_LOCK:
        lane = _LANES.setdefault(lane_id, _LaneState())
        if lane.occupant is None:  # idle lane: take it and open the interval
            lane.occupant = state_name
            lane.count = 1
            lane.start_ns = time.time_ns()
            lane.attributes = attributes
            counted = True
        elif state_name == lane.occupant:  # same-name overlap: keep the union open
            lane.count += 1
            counted = True
        else:  # different name while occupied: shadowed, not reported
            counted = False

    try:
        yield
    finally:
        emit_args: tuple[str, int, int, dict[str, Any]] | None = None
        with _LANES_LOCK:
            # Identity check guards against finish() replacing the lane mid-block.
            if counted and _LANES.get(lane_id) is lane:
                lane.count -= 1
                if lane.count == 0:  # occupant closed: emit merged span, free lane
                    emit_args = (
                        state_name,
                        lane.start_ns,
                        time.time_ns(),
                        lane.attributes,
                    )
                    _LANES.pop(lane_id, None)
        # Emit outside the lock so the backend submit never blocks other callers.
        if emit_args is not None:
            _emit_trace_span(
                name=emit_args[0],
                start_time_ns=emit_args[1],
                end_time_ns=emit_args[2],
                attributes=emit_args[3],
            )


def trace_op(
    name: str | None = None,
    *,
    extra_labels: Callable[[Any], dict[str, Any]] | None = None,
    **static_labels: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records one root span per synchronous call.

    Async callables are not wrapped: a :class:`RuntimeWarning` is issued and the
    function is returned unchanged.

    Args:
        name: Span name; defaults to ``func.__qualname__``.
        extra_labels: If set, ``extra_labels(first_positional_arg)`` is merged after
            ``static_labels`` when the wrapped function is called. The first positional
            is often ``self`` for bound methods; if there are no positional args, it is
            not called.
        **static_labels: Extra attributes attached to every span for this operation.

    Returns:
        Decorator that replaces sync functions with a span-wrapped version (async functions unchanged with warning).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``func`` unchanged for coroutine functions; else attach span timing wrapper."""
        if inspect.iscoroutinefunction(func):
            warnings.warn(
                "[rl-insight] trace_op does not support coroutine functions; "
                "decorator is a no-op.",
                RuntimeWarning,
                stacklevel=2,
            )
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call the wrapped function and record one duration span around it."""
            if not _STATE.enabled or _STATE.client is None:
                return func(*args, **kwargs)

            span_name = name or func.__qualname__
            merged: dict[str, Any] = dict(static_labels)
            if extra_labels is not None and args:
                merged.update(extra_labels(args[0]))

            start_time_ns = time.time_ns()
            attributes = {**merged, "monitor.trace_segment": "duration"}
            try:
                return func(*args, **kwargs)
            finally:
                _emit_trace_span(
                    name=span_name,
                    start_time_ns=start_time_ns,
                    end_time_ns=time.time_ns(),
                    attributes=attributes,
                )

        return wrapper

    return decorator


def _emit(
    kind: str,
    name: str,
    value: float,
    documentation: str,
    labels: dict[str, Any],
) -> None:
    """If monitoring is on, forward a Prometheus metric event to the hub.

    Args:
        kind: One of ``MonitorEventKind`` counter/gauge/histogram strings.
        name: Metric name.
        value: Sample or increment amount.
        documentation: Help text stored with the series.
        labels: Label dimensions for the observation.
    """
    if not _STATE.enabled or _STATE.client is None:
        return
    event = {
        "kind": kind,
        "name": name,
        "documentation": documentation,
        "value": value,
        "labels": {**_STATE.labels, **labels},
    }
    _STATE.client.apply_event(event)


def _emit_trace_span(
    *,
    name: str,
    start_time_ns: int,
    end_time_ns: int,
    attributes: dict[str, Any],
) -> None:
    """If monitoring is on, send one OTLP root span event (hub may no-op if OTLP is disabled).

    Args:
        name: Span name.
        start_time_ns: Span start (nanoseconds).
        end_time_ns: Span end (nanoseconds).
        attributes: Span attributes; ``process_id`` and init-level labels are merged in before send.
    """
    if not _STATE.enabled or _STATE.client is None:
        return

    merged_attributes: dict[str, Any] = {
        "process_id": _STATE.process_id,
        **_STATE.labels,
    }
    merged_attributes.update(attributes)

    event = {
        "kind": MonitorEventKind.TRACE,
        "name": name,
        "start_time_ns": int(start_time_ns),
        "end_time_ns": int(end_time_ns),
        "attributes": merged_attributes,
    }
    _STATE.client.apply_event(event)
