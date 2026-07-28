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

"""KDE-mode discovery followed by three-part stability voting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from .kde_utils import (
    KDEResult,
    adaptive_kde,
    filter_peaks,
    find_peak_influence_regions,
)


@dataclass(frozen=True)
class StableSegment:
    """One time-contiguous normal-mode candidate that passed stability voting."""

    timestamps: list[float]
    values: list[float]
    influence_region: tuple[float, float]
    flags: dict[str, bool] = field(default_factory=dict)
    means: tuple[float, float, float] = (0.0, 0.0, 0.0)
    standard_deviations: tuple[float, float, float] = (0.0, 0.0, 0.0)


def is_within_std(
    value: float,
    mean: float,
    std: float,
    *,
    std_factor: float = 2.0,
    within_std_coefficient: float = 1.05,
) -> bool:
    """Check the confirmed open interval with the explicit 1.05 coefficient.

    The approved material supplies golden boundaries but not the legacy algebra.
    This implementation therefore keeps the 1.05 coefficient in the explicit
    inward-boundary calculation instead of reducing it to a conventional
    ``mean +/- factor * std`` expression.
    """

    numeric_std = float(std)
    numeric_value = float(value)
    numeric_mean = float(mean)
    coefficient = float(within_std_coefficient)
    factor = float(std_factor)
    if not all(
        math.isfinite(item)
        for item in (numeric_std, numeric_value, numeric_mean)
    ):
        return False
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("std_factor must be finite and positive")
    if not math.isfinite(coefficient) or not 1.0 <= coefficient < 2.0:
        raise ValueError("within_std_coefficient must be in [1, 2)")
    if numeric_std < 0:
        raise ValueError("std must not be negative")
    if numeric_std == 0:
        precision = 8.0 * max(math.ulp(numeric_value), math.ulp(numeric_mean))
        return math.isclose(
            numeric_value,
            numeric_mean,
            rel_tol=0.0,
            abs_tol=precision,
        )

    nominal_margin = factor * numeric_std
    inward_adjustment = (coefficient - 1.0) * nominal_margin
    lower_bound = (numeric_mean - nominal_margin) + inward_adjustment
    upper_bound = (numeric_mean + nominal_margin) - inward_adjustment
    return lower_bound < numeric_value < upper_bound


def _maximum_time_gap(
    timestamps: Sequence[float], config: Mapping[str, Any]
) -> float:
    explicit = config.get("maximum_time_gap")
    if explicit is not None:
        gap = float(explicit)
        if gap <= 0 or not np.isfinite(gap):
            raise ValueError("maximum_time_gap must be finite and positive")
        return gap
    differences = np.diff(np.asarray(timestamps, dtype=float))
    positive = differences[differences > 0]
    if positive.size == 0:
        return math.inf
    factor = float(config.get("time_gap_factor", 3.0))
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError("time_gap_factor must be finite and positive")
    return float(np.median(positive)) * factor


def find_stable_segments_by_density(
    timestamps: Sequence[float],
    values: Sequence[float],
    influence_regions: Sequence[Mapping[str, float | int]],
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map each value region back to original, time-contiguous candidates."""

    if len(timestamps) != len(values):
        raise ValueError("timestamps and values must have equal lengths")
    cfg = config or {}
    maximum_gap = (
        _maximum_time_gap(timestamps, cfg)
        if len(timestamps) > 0
        else math.inf
    )
    regions = list(influence_regions)

    # Closed valley boundaries can be shared by adjacent modes. Assign every
    # source point to at most one region so an exact valley sample cannot create
    # duplicate normal models. Nearest peak wins; lower peak index breaks ties.
    assignments: list[int | None] = [None] * len(values)
    for value_index, raw_value in enumerate(values):
        value = float(raw_value)
        eligible: list[int] = []
        for region_index, region in enumerate(regions):
            left = float(region["left_value"])
            right = float(region["right_value"])
            if left > right:
                raise ValueError("influence region left boundary exceeds right")
            if left <= value <= right:
                eligible.append(region_index)
        if eligible:
            assignments[value_index] = min(
                eligible,
                key=lambda region_index: (
                    abs(
                        value
                        - float(
                            regions[region_index].get(
                                "peak_value",
                                (
                                    float(regions[region_index]["left_value"])
                                    + float(regions[region_index]["right_value"])
                                )
                                / 2.0,
                            )
                        )
                    ),
                    int(regions[region_index].get("peak_index", region_index)),
                    region_index,
                ),
            )

    candidates: list[dict[str, Any]] = []
    signatures: set[tuple[int, ...]] = set()
    for region_index, region in enumerate(regions):
        left = float(region["left_value"])
        right = float(region["right_value"])
        current_indexes: list[int] = []

        def flush() -> None:
            if not current_indexes:
                return
            signature = tuple(current_indexes)
            if signature not in signatures:
                signatures.add(signature)
                candidates.append(
                    {
                        "timestamps": [float(timestamps[i]) for i in current_indexes],
                        "values": [float(values[i]) for i in current_indexes],
                        "indexes": list(current_indexes),
                        "influence_region": (left, right),
                        "peak_index": int(
                            region.get("peak_index", region_index)
                        ),
                    }
                )
            current_indexes.clear()

        for index, (timestamp, value) in enumerate(zip(timestamps, values)):
            inside = assignments[index] == region_index
            gap_break = False
            if current_indexes:
                delta = float(timestamp) - float(timestamps[current_indexes[-1]])
                gap_break = delta <= 0 or delta > maximum_gap
            if not inside or gap_break:
                flush()
            if inside:
                current_indexes.append(index)
        flush()
    return candidates


def _three_parts(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    part_size = len(array) // 3
    if part_size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    return (
        array[0:part_size],
        array[part_size : 2 * part_size],
        array[2 * part_size : len(array)],
    )


def analyze_segments(
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> list[StableSegment]:
    """Apply the strict 1/3, 1/3, remainder split and six-flag vote."""

    cfg = config or {}
    factor = float(cfg.get("std_factor", 2.0))
    coefficient = float(cfg.get("within_std_coefficient", 1.05))
    minimum_flags = int(cfg.get("minimum_passed_flags", 4))
    tolerance_ratio = float(cfg.get("mean_tolerance_ratio", 0.02))
    if not 1 <= minimum_flags <= 6:
        raise ValueError("minimum_passed_flags must be between 1 and 6")
    if tolerance_ratio < 0 or not np.isfinite(tolerance_ratio):
        raise ValueError("mean_tolerance_ratio must be finite and non-negative")
    stable: list[StableSegment] = []

    for candidate in candidates:
        values = [float(value) for value in candidate["values"]]
        part_1, part_2, part_3 = _three_parts(values)
        if part_1.size == 0:
            continue
        means = tuple(float(np.mean(part)) for part in (part_1, part_2, part_3))
        stds = tuple(float(np.std(part, ddof=0)) for part in (part_1, part_2, part_3))
        effective_stds = tuple(
            max(
                local_std,
                abs(local_mean) * tolerance_ratio,
                8.0 * math.ulp(local_mean),
            )
            for local_mean, local_std in zip(means, stds)
        )

        def compatible(target_part: int, reference_part: int) -> bool:
            return is_within_std(
                means[target_part],
                means[reference_part],
                effective_stds[reference_part],
                std_factor=factor,
                within_std_coefficient=coefficient,
            )

        flags = {
            "mean_1_within_part_2": compatible(0, 1),
            "mean_2_within_part_1": compatible(1, 0),
            "mean_2_within_part_3": compatible(1, 2),
            "mean_3_within_part_2": compatible(2, 1),
            "mean_1_within_part_3": compatible(0, 2),
            "mean_3_within_part_1": compatible(2, 0),
        }
        if sum(flags.values()) >= minimum_flags:
            stable.append(
                StableSegment(
                    timestamps=[float(item) for item in candidate["timestamps"]],
                    values=values,
                    influence_region=tuple(candidate["influence_region"]),
                    flags=flags,
                    means=means,
                    standard_deviations=stds,
                )
            )
    return stable


class StableSegmentDetector:
    """Discover density modes, map them to time runs, and vote on stability."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.last_diagnostics: dict[str, Any] = {}

    def detect_stable_segments(
        self, timestamps: Sequence[float], values: Sequence[float]
    ) -> list[StableSegment]:
        """Run the required KDE → peaks/valleys → regions → thirds chain."""

        if len(timestamps) != len(values):
            raise ValueError("timestamps and values must have equal lengths")
        if len(values) < 3:
            self.last_diagnostics = {"reason": "fewer than three values"}
            return []
        kde = adaptive_kde(values, self.config)
        raw_peaks, _ = find_peaks(kde.density)
        neg_peaks, _ = find_peaks(-kde.density)
        if raw_peaks.size == 0:
            raw_peaks = np.asarray([int(np.argmax(kde.density))], dtype=int)
        filtered_peaks = filter_peaks(raw_peaks, kde.density, self.config)
        regions = find_peak_influence_regions(
            kde.grid, filtered_peaks, neg_peaks
        )
        stable_cfg = self.config.get("stable_segment", self.config)
        candidates = find_stable_segments_by_density(
            timestamps, values, regions, stable_cfg
        )
        stable = analyze_segments(candidates, stable_cfg)
        self.last_diagnostics = {
            "kde": _kde_diagnostics(kde),
            "raw_peaks": raw_peaks.tolist(),
            "filtered_peaks": filtered_peaks.tolist(),
            "neg_peaks": neg_peaks.tolist(),
            "influence_regions": regions,
            "candidate_count": len(candidates),
            "stable_count": len(stable),
        }
        return stable


def _kde_diagnostics(kde: KDEResult) -> dict[str, Any]:
    return {
        "bandwidth": kde.bandwidth,
        "data_range": kde.data_range,
        "fitted": kde.fitted,
        "zero_range_jittered": kde.zero_range_jittered,
        "jitter_scale": kde.jitter_scale,
        "adjusted_values": kde.adjusted_values,
    }
