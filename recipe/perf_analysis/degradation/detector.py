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

"""Classify step values and track confirmed target degradation events."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .baseline import NormalRange, SeriesBaseline
from .data import SeriesId, StepFrame


class DetectorError(ValueError):
    """The supplied baseline, policy, or point cannot be detected safely."""


class Policy(str, Enum):
    """Directions that make a point abnormal for one metric role."""

    UP = "UP"
    DOWN = "DOWN"
    BOTH = "BOTH"


class Direction(str, Enum):
    """A valid value's position relative to all fitted normal ranges."""

    NORMAL = "NORMAL"
    UP = "UP"
    DOWN = "DOWN"
    BETWEEN_MODES = "BETWEEN_MODES"


@dataclass(frozen=True)
class DetectorParameters:
    """Evidence-window parameters for target event tracking."""

    evidence_window: int = 5
    minimum_abnormal: int = 3

    def __post_init__(self) -> None:
        for name in ("evidence_window", "minimum_abnormal"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_abnormal > self.evidence_window:
            raise ValueError("minimum_abnormal must not exceed evidence_window")


@dataclass(frozen=True)
class PointResult:
    """Point-level classification for one series in one training step."""

    identity: SeriesId
    step: int
    start_time: float
    end_time: float
    value: float | None
    policy: Policy
    direction: Direction | None
    abnormal: bool
    matched_mode: int | None

    @property
    def valid(self) -> bool:
        return self.value is not None


@dataclass
class _ActiveEvent:
    """Minimal mutable state for one confirmed target event."""

    identity: SeriesId
    direction: Direction
    start_step: int
    start_time: float
    confirmed_at_step: int
    confirmed_at_time: float
    last_abnormal_step: int
    last_abnormal_time: float
    abnormal_points: int


@dataclass(frozen=True)
class DegradationEvent:
    """Immutable event snapshot emitted at confirmation, closure, or range end."""

    identity: SeriesId
    direction: Direction
    start_step: int
    start_time: float
    confirmed_at_step: int
    confirmed_at_time: float
    end_step: int
    end_time: float
    closed_at_step: int | None
    closed_at_time: float | None
    abnormal_points: int
    evidence_points: int
    evidence_abnormal_points: int


@dataclass(frozen=True)
class EventUpdate:
    """One tracker update and any lifecycle notification it produced."""

    point: PointResult
    evidence_abnormal_points: int | None = None
    confirmed_event: DegradationEvent | None = None
    closed_event: DegradationEvent | None = None


DEFAULT_DETECTOR_PARAMETERS = DetectorParameters()


def _policy(value: Policy | str) -> Policy:
    if isinstance(value, Policy):
        return value
    if not isinstance(value, str):
        raise TypeError("policy must be UP, DOWN, or BOTH")
    try:
        return Policy(value.upper())
    except ValueError as exc:
        raise DetectorError("policy must be UP, DOWN, or BOTH") from exc


def _validated_ranges(baseline: SeriesBaseline) -> tuple[NormalRange, ...]:
    if not baseline.ranges:
        raise DetectorError("baseline must contain at least one normal range")
    for normal_range in baseline.ranges:
        if not all(
            math.isfinite(value) for value in (normal_range.lower, normal_range.upper)
        ):
            raise DetectorError("normal range bounds must be finite")
        if normal_range.lower > normal_range.upper:
            raise DetectorError("normal range lower bound must not exceed upper bound")
    return baseline.ranges


def _abnormal(direction: Direction, policy: Policy) -> bool:
    if direction is Direction.NORMAL:
        return False
    if policy is Policy.BOTH:
        return True
    return direction.value == policy.value


def classify_point(
    frame: StepFrame,
    baseline: SeriesBaseline,
    *,
    policy: Policy | str,
) -> PointResult:
    """Classify one step value without mutating baseline or runtime state."""

    selected_policy = _policy(policy)
    ranges = _validated_ranges(baseline)
    raw_value = frame.values.get(baseline.identity)
    if raw_value is None:
        return PointResult(
            identity=baseline.identity,
            step=frame.step,
            start_time=frame.start_time,
            end_time=frame.end_time,
            value=None,
            policy=selected_policy,
            direction=None,
            abnormal=False,
            matched_mode=None,
        )
    if isinstance(raw_value, bool) or not math.isfinite(raw_value):
        raise DetectorError("point value must be finite or None")

    value = float(raw_value)
    matched = next(
        (
            normal_range.mode_id
            for normal_range in ranges
            if normal_range.lower <= value <= normal_range.upper
        ),
        None,
    )
    if matched is not None:
        direction = Direction.NORMAL
    elif value > max(normal_range.upper for normal_range in ranges):
        direction = Direction.UP
    elif value < min(normal_range.lower for normal_range in ranges):
        direction = Direction.DOWN
    else:
        direction = Direction.BETWEEN_MODES

    return PointResult(
        identity=baseline.identity,
        step=frame.step,
        start_time=frame.start_time,
        end_time=frame.end_time,
        value=value,
        policy=selected_policy,
        direction=direction,
        abnormal=_abnormal(direction, selected_policy),
        matched_mode=matched,
    )


def classify_frame(
    frame: StepFrame,
    baselines: Mapping[SeriesId, SeriesBaseline],
    policies: Mapping[SeriesId, Policy | str],
) -> dict[SeriesId, PointResult]:
    """Classify every fitted series in one step using explicit policies."""

    missing = [identity for identity in baselines if identity not in policies]
    if missing:
        names = ", ".join(repr(identity) for identity in sorted(missing))
        raise DetectorError(f"missing policies for series: {names}")
    return {
        identity: classify_point(frame, baseline, policy=policies[identity])
        for identity, baseline in baselines.items()
    }


def _event_snapshot(
    event: _ActiveEvent,
    parameters: DetectorParameters,
    evidence_count: int,
    closed_by: PointResult | None = None,
) -> DegradationEvent:
    return DegradationEvent(
        identity=event.identity,
        direction=event.direction,
        start_step=event.start_step,
        start_time=event.start_time,
        confirmed_at_step=event.confirmed_at_step,
        confirmed_at_time=event.confirmed_at_time,
        end_step=event.last_abnormal_step,
        end_time=event.last_abnormal_time,
        closed_at_step=None if closed_by is None else closed_by.step,
        closed_at_time=None if closed_by is None else closed_by.end_time,
        abnormal_points=event.abnormal_points,
        evidence_points=parameters.evidence_window,
        evidence_abnormal_points=evidence_count,
    )


class EventTracker:
    """Track one UP-policy target with a symmetric valid-point window."""

    def __init__(
        self,
        identity: SeriesId,
        *,
        parameters: DetectorParameters = DEFAULT_DETECTOR_PARAMETERS,
    ) -> None:
        if not isinstance(identity, SeriesId):
            raise TypeError("identity must be a SeriesId")
        self.identity = identity
        self.parameters = parameters
        self.recent_points: deque[PointResult] = deque(
            maxlen=parameters.evidence_window
        )
        self.current_event: _ActiveEvent | None = None
        self._last_step: int | None = None

    def update(self, point: PointResult) -> EventUpdate:
        """Consume one classified target point and emit lifecycle transitions."""

        if point.identity != self.identity:
            raise DetectorError("point identity does not match event tracker")
        if point.policy is not Policy.UP:
            raise DetectorError("target event tracking requires policy UP")
        if self._last_step is not None and point.step != self._last_step + 1:
            raise DetectorError(
                f"target steps must be consecutive: expected {self._last_step + 1}, "
                f"got {point.step}"
            )

        self._last_step = point.step
        self.recent_points.append(point)
        if (
            self.current_event is not None
            and point.valid
            and point.abnormal
            and point.direction is Direction.UP
        ):
            self.current_event.last_abnormal_step = point.step
            self.current_event.last_abnormal_time = point.end_time
            self.current_event.abnormal_points += 1

        if len(self.recent_points) < self.parameters.evidence_window:
            return EventUpdate(point=point)
        if not all(item.valid for item in self.recent_points):
            return EventUpdate(point=point)

        up_points = [
            item
            for item in self.recent_points
            if item.abnormal and item.direction is Direction.UP
        ]
        evidence_count = len(up_points)
        has_evidence = evidence_count >= self.parameters.minimum_abnormal

        if self.current_event is not None:
            if has_evidence:
                return EventUpdate(
                    point=point,
                    evidence_abnormal_points=evidence_count,
                )
            closed = _event_snapshot(
                self.current_event,
                self.parameters,
                evidence_count,
                closed_by=point,
            )
            self.current_event = None
            self.recent_points.clear()
            return EventUpdate(
                point=point,
                evidence_abnormal_points=evidence_count,
                closed_event=closed,
            )

        if not has_evidence:
            return EventUpdate(
                point=point,
                evidence_abnormal_points=evidence_count,
            )

        first, last = up_points[0], up_points[-1]
        self.current_event = _ActiveEvent(
            identity=self.identity,
            direction=Direction.UP,
            start_step=first.step,
            start_time=first.start_time,
            confirmed_at_step=point.step,
            confirmed_at_time=point.end_time,
            last_abnormal_step=last.step,
            last_abnormal_time=last.end_time,
            abnormal_points=evidence_count,
        )
        confirmed = _event_snapshot(
            self.current_event,
            self.parameters,
            evidence_count,
        )
        return EventUpdate(
            point=point,
            evidence_abnormal_points=evidence_count,
            confirmed_event=confirmed,
        )

    def snapshot_open_event(self) -> DegradationEvent | None:
        """Return the current confirmed event without closing it."""

        if self.current_event is None:
            return None
        evidence_count = sum(
            point.valid and point.abnormal and point.direction is Direction.UP
            for point in self.recent_points
        )
        return _event_snapshot(
            self.current_event,
            self.parameters,
            evidence_count,
        )


__all__ = [
    "DEFAULT_DETECTOR_PARAMETERS",
    "DegradationEvent",
    "DetectorError",
    "DetectorParameters",
    "Direction",
    "EventTracker",
    "EventUpdate",
    "PointResult",
    "Policy",
    "classify_frame",
    "classify_point",
]
