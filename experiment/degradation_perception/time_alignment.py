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

"""Bounded time alignment for degradation-association analysis."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from .perception_config import SUPPORTED_SOURCE_TYPES, TimeSeries


@dataclass(frozen=True)
class AlignmentResult:
    """Matched target/candidate rows and diagnostics for one analysis window."""

    target_indices: list[int]
    timestamps: list[float]
    target_values: list[float]
    target_labels: list[bool]
    candidate_values: list[float]
    candidate_labels: list[bool]
    coverage_ratio: float
    tolerance: float


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_median_interval(timestamps: Sequence[float]) -> float:
    ordered = sorted(float(item) for item in timestamps if math.isfinite(float(item)))
    positive: list[float] = []
    for index in range(1, len(ordered)):
        difference = ordered[index] - ordered[index - 1]
        if difference > 0 and math.isfinite(difference):
            positive.append(difference)
    return float(median(positive)) if positive else 0.0


def build_analysis_window(
    raw_start: float,
    raw_end: float,
    target_timestamps: Sequence[float],
    context_ratio: float,
) -> dict[str, float | bool]:
    """Expand one raw event and clip it to the finite target-data boundaries."""

    start = _finite_float(raw_start, "raw_start")
    end = _finite_float(raw_end, "raw_end")
    ratio = _finite_float(context_ratio, "context_ratio")
    if start > end:
        raise ValueError("raw_start must not exceed raw_end")
    if ratio < 0:
        raise ValueError("context_ratio must be non-negative")

    finite_timestamps: list[float] = []
    for item in target_timestamps:
        try:
            timestamp = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(timestamp):
            finite_timestamps.append(timestamp)
    if not finite_timestamps:
        raise ValueError("target_timestamps must contain a finite value")

    lower_bound = min(finite_timestamps)
    upper_bound = max(finite_timestamps)
    duration = end - start
    used_minimum_context = duration == 0.0
    if used_minimum_context:
        context = _positive_median_interval(finite_timestamps)
    else:
        context = duration * ratio

    proposed_start = start - context
    proposed_end = end + context
    if proposed_end < lower_bound or proposed_start > upper_bound:
        raise ValueError("expanded event does not overlap target_timestamps")
    analysis_start = max(lower_bound, proposed_start)
    analysis_end = min(upper_bound, proposed_end)
    return {
        "startTime": float(analysis_start),
        "endTime": float(analysis_end),
        "usedMinimumContext": bool(used_minimum_context),
    }


def _valid_points(
    series: TimeSeries,
    labels: Sequence[bool],
    *,
    name: str,
) -> list[tuple[int, float, float, bool]]:
    if not isinstance(series, TimeSeries):
        raise TypeError(f"{name} must be a TimeSeries")
    if len(series.timestamps) != len(series.values):
        raise ValueError(f"{name} timestamps, values, and labels must align")
    try:
        label_list = list(labels)
    except TypeError as exc:
        raise ValueError(f"{name} labels must be an array") from exc
    if len(label_list) != len(series.timestamps):
        raise ValueError(f"{name} timestamps, values, and labels must align")

    points: list[tuple[int, float, float, bool]] = []
    for index in range(len(series.timestamps)):
        try:
            timestamp = float(series.timestamps[index])
            value = float(series.values[index])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(timestamp) or not math.isfinite(value):
            continue
        points.append((index, timestamp, value, bool(label_list[index])))
    return sorted(points, key=lambda item: (item[1], item[0]))


def align_candidate_series(
    target: TimeSeries,
    target_labels: Sequence[bool],
    candidate: TimeSeries,
    candidate_labels: Sequence[bool],
    start_time: float,
    end_time: float,
    source_type: str,
    max_tolerance: float | None = None,
) -> AlignmentResult:
    """Align a candidate to target timestamps without unbounded nearest fill."""

    window_start = _finite_float(start_time, "start_time")
    window_end = _finite_float(end_time, "end_time")
    if window_start > window_end:
        raise ValueError("start_time must not exceed end_time")
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"unsupported source_type: {source_type!r}")

    explicit_tolerance: float | None = None
    if max_tolerance is not None:
        explicit_tolerance = _finite_float(max_tolerance, "max_tolerance")
        if explicit_tolerance <= 0:
            raise ValueError("max_tolerance must be positive")

    target_points = _valid_points(target, target_labels, name="target")
    candidate_points = _valid_points(
        candidate,
        candidate_labels,
        name="candidate",
    )
    window_targets = [
        point for point in target_points if window_start <= point[1] <= window_end
    ]
    window_candidates = [
        point for point in candidate_points if window_start <= point[1] <= window_end
    ]

    all_timestamps = [point[1] for point in target_points]
    all_timestamps.extend(point[1] for point in candidate_points)
    exact_step_mode = source_type == "training_log" and all(
        timestamp < 10000 for timestamp in all_timestamps
    )
    if exact_step_mode:
        tolerance = 0.0
    elif explicit_tolerance is not None:
        tolerance = explicit_tolerance
    else:
        target_interval = _positive_median_interval(
            [point[1] for point in target_points]
        )
        candidate_interval = _positive_median_interval(
            [point[1] for point in candidate_points]
        )
        tolerance = max(target_interval, candidate_interval) / 2.0

    target_indices: list[int] = []
    timestamps: list[float] = []
    target_values: list[float] = []
    aligned_target_labels: list[bool] = []
    candidate_values: list[float] = []
    aligned_candidate_labels: list[bool] = []

    if exact_step_mode:
        exact_candidates: dict[float, tuple[int, float, float, bool]] = {}
        for point in window_candidates:
            exact_candidates[point[1]] = point
        matches = (
            (target_point, exact_candidates.get(target_point[1]))
            for target_point in window_targets
        )
    else:
        candidate_timestamps = [point[1] for point in window_candidates]

        def nearest(
            target_point: tuple[int, float, float, bool],
        ) -> tuple[int, float, float, bool] | None:
            if not window_candidates:
                return None
            timestamp = target_point[1]
            position = bisect_left(candidate_timestamps, timestamp)
            choices: list[tuple[int, float, float, bool]] = []
            if position > 0:
                choices.append(window_candidates[position - 1])
            if position < len(window_candidates):
                choices.append(window_candidates[position])
            selected = min(
                choices,
                key=lambda item: (
                    abs(item[1] - timestamp),
                    item[1],
                    item[0],
                ),
            )
            if abs(selected[1] - timestamp) > tolerance:
                return None
            return selected

        matches = (
            (target_point, nearest(target_point)) for target_point in window_targets
        )

    # A bounded nearest lookup is not a forward fill: one real candidate
    # observation may support at most one aligned row.  When several target
    # rows select the same observation, retain the closest target (and then
    # the earlier target for a deterministic tie).
    best_by_candidate: dict[
        int,
        tuple[
            tuple[float, float, int],
            tuple[int, float, float, bool],
            tuple[int, float, float, bool],
        ],
    ] = {}
    for target_point, candidate_point in matches:
        if candidate_point is None:
            continue
        priority = (
            abs(candidate_point[1] - target_point[1]),
            target_point[1],
            target_point[0],
        )
        current = best_by_candidate.get(candidate_point[0])
        if current is None or priority < current[0]:
            best_by_candidate[candidate_point[0]] = (
                priority,
                target_point,
                candidate_point,
            )
    unique_matches = sorted(
        ((item[1], item[2]) for item in best_by_candidate.values()),
        key=lambda item: (item[0][1], item[0][0]),
    )

    for target_point, candidate_point in unique_matches:
        if candidate_point is None:
            continue
        target_indices.append(int(target_point[0]))
        timestamps.append(float(target_point[1]))
        target_values.append(float(target_point[2]))
        aligned_target_labels.append(bool(target_point[3]))
        candidate_values.append(float(candidate_point[2]))
        aligned_candidate_labels.append(bool(candidate_point[3]))

    denominator = len(window_targets)
    coverage_ratio = len(target_indices) / denominator if denominator else 0.0
    return AlignmentResult(
        target_indices=target_indices,
        timestamps=timestamps,
        target_values=target_values,
        target_labels=aligned_target_labels,
        candidate_values=candidate_values,
        candidate_labels=aligned_candidate_labels,
        coverage_ratio=float(coverage_ratio),
        tolerance=float(tolerance),
    )
