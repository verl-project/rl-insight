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

"""Candidate interval construction and strict four-condition validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _allowed_gap(timestamps: Sequence[float], config: Mapping[str, Any]) -> float:
    explicit = config.get("maximum_time_gap")
    if explicit is not None:
        value = float(explicit)
        if value <= 0 or not math.isfinite(value):
            raise ValueError("maximum_time_gap must be finite and positive")
        return value
    differences = np.diff(np.asarray(timestamps, dtype=float))
    positive = differences[differences > 0]
    if positive.size == 0:
        return math.inf
    factor = float(config.get("time_gap_factor", 3.0))
    if factor <= 0 or not math.isfinite(factor):
        raise ValueError("time_gap_factor must be finite and positive")
    return float(np.median(positive)) * factor


def build_candidate_intervals(
    timestamps: Sequence[float],
    abnormal_flags: Sequence[bool],
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Merge nearby abnormal labels while retaining intervening normal points."""

    if len(timestamps) != len(abnormal_flags):
        raise ValueError("timestamps and abnormal_flags must have equal lengths")
    if len(timestamps) == 0:
        return []
    cfg = config or {}
    abnormal_indexes = [index for index, flag in enumerate(abnormal_flags) if flag]
    if not abnormal_indexes:
        return []
    max_normal = int(cfg.get("max_normal_points_between", 1))
    if max_normal < 0:
        raise ValueError("max_normal_points_between must be non-negative")
    allowed_gap = _allowed_gap(timestamps, cfg)

    groups: list[list[int]] = [[abnormal_indexes[0]]]
    for index in abnormal_indexes[1:]:
        previous = groups[-1][-1]
        normal_between = index - previous - 1
        underlying_gaps = [
            float(timestamps[position + 1]) - float(timestamps[position])
            for position in range(previous, index)
        ]
        continuous = all(0 < gap <= allowed_gap for gap in underlying_gaps)
        if normal_between <= max_normal and continuous:
            groups[-1].append(index)
        else:
            groups.append([index])

    intervals: list[dict[str, Any]] = []
    for indexes in groups:
        start_index = indexes[0]
        stop_index = indexes[-1] + 1
        total_count = stop_index - start_index
        abnormal_count = sum(
            bool(flag) for flag in abnormal_flags[start_index:stop_index]
        )
        rate = abnormal_count / total_count if total_count else 0.0
        gaps = np.diff(np.asarray(timestamps[start_index:stop_index], dtype=float))
        continuous = (
            bool(np.all((gaps > 0) & (gaps <= allowed_gap)))
            if gaps.size
            else False
        )
        intervals.append(
            {
                "start_index": start_index,
                "stop_index": stop_index,
                "start_time": float(timestamps[start_index]),
                "end_time": float(timestamps[stop_index - 1]),
                "duration": float(timestamps[stop_index - 1])
                - float(timestamps[start_index]),
                "total_point_count": total_count,
                "abnormal_point_count": abnormal_count,
                "abnormal_rate": rate,
                "continuous": continuous,
                "maximum_allowed_gap": allowed_gap,
            }
        )
    return intervals


def validate_candidate_interval(
    interval: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, bool]]:
    """Require all four confirmed continuous-anomaly conditions."""

    cfg = config or {}
    duration = float(interval.get("duration", 0.0))
    abnormal_points = int(interval.get("abnormal_point_count", 0))
    total_points = int(interval.get("total_point_count", 0))
    abnormal_rate = (
        abnormal_points / total_points
        if total_points > 0
        else 0.0
    )
    condition_1 = duration > float(cfg.get("minimum_duration", 0.5))
    condition_2 = abnormal_points >= int(cfg.get("minimum_abnormal_points", 5))
    condition_3 = abnormal_rate >= float(cfg.get("minimum_abnormal_rate", 0.60))
    condition_4 = bool(interval.get("continuous", False))
    details = {
        "condition_1": condition_1,
        "condition_2": condition_2,
        "condition_3": condition_3,
        "condition_4": condition_4,
    }
    return condition_1 and condition_2 and condition_3 and condition_4, details
