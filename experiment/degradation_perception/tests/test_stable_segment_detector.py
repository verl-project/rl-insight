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

from __future__ import annotations

import numpy as np
import pytest

from experiment.degradation_perception import stable_segment_detector as module
from experiment.degradation_perception.kde_utils import KDEResult
from experiment.degradation_perception.stable_segment_detector import (
    StableSegmentDetector,
    analyze_segments,
    find_stable_segments_by_density,
    is_within_std,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.19, False), (1.18, True), (0.81, False), (0.82, True)],
)
def test_is_within_std_golden_boundaries(value, expected):
    assert is_within_std(value, 1.0, 0.1) is expected


def test_is_within_std_uses_coefficient_and_handles_zero_std():
    assert is_within_std(1.18, 1.0, 0.1, within_std_coefficient=1.01)
    assert not is_within_std(1.18, 1.0, 0.1, within_std_coefficient=1.20)
    assert is_within_std(1.0, 1.0, 0.0)
    assert not is_within_std(1.0001, 1.0, 0.0)


def test_is_within_std_rejects_nonfinite_and_negative_std():
    assert not is_within_std(float("nan"), 1.0, 0.1)
    assert not is_within_std(1.0, float("inf"), 0.1)
    assert not is_within_std(1.0, 1.0, float("nan"))
    with pytest.raises(ValueError, match="must not be negative"):
        is_within_std(1.0, 1.0, -0.1)


def test_detect_stable_segments_finds_the_required_two_modes():
    values = [1.00, 1.01, 1.02, 1.04, 1.05, 5.00, 5.01, 5.02]
    segments = StableSegmentDetector().detect_stable_segments(
        list(range(len(values))), values
    )
    assert len(segments) == 2
    ordered = sorted(segments, key=lambda item: np.mean(item.values))
    assert max(ordered[0].values) <= 1.05
    assert min(ordered[1].values) >= 5.00


def test_density_mapping_splits_on_an_intervening_out_of_region_point():
    candidates = find_stable_segments_by_density(
        [0, 1, 2, 3, 4],
        [1.01, 1.02, 1.03, 5.0, 1.02],
        [
            {
                "peak_index": 1,
                "peak_value": 1.0,
                "left_value": 0.9,
                "right_value": 1.1,
            }
        ],
    )
    assert [item["values"] for item in candidates] == [
        [1.01, 1.02, 1.03],
        [1.02],
    ]


def test_density_mapping_splits_on_a_large_time_gap():
    candidates = find_stable_segments_by_density(
        [0, 1, 2, 100, 101, 102],
        [1.0] * 6,
        [
            {
                "peak_index": 1,
                "peak_value": 1.0,
                "left_value": 0.9,
                "right_value": 1.1,
            }
        ],
        {"maximum_time_gap": 2},
    )
    assert [len(item["values"]) for item in candidates] == [3, 3]


def test_exact_shared_valley_value_is_assigned_to_only_one_mode():
    candidates = find_stable_segments_by_density(
        [0],
        [3.0],
        [
            {
                "peak_index": 1,
                "peak_value": 1.0,
                "left_value": 0.0,
                "right_value": 3.0,
            },
            {
                "peak_index": 5,
                "peak_value": 5.0,
                "left_value": 3.0,
                "right_value": 6.0,
            },
        ],
    )
    assert sum(len(item["values"]) for item in candidates) == 1


def test_analyze_segments_splits_eight_points_as_two_two_four_and_uses_local_std():
    values = [0.9, 1.1, 0.8, 1.2, 1.0, 1.0, 1.0, 1.0]
    candidates = [
        {
            "timestamps": list(range(8)),
            "values": values,
            "influence_region": (0.0, 2.0),
        }
    ]
    segments = analyze_segments(candidates)
    assert len(segments) == 1
    segment = segments[0]
    assert segment.means == pytest.approx((1.0, 1.0, 1.0))
    assert segment.standard_deviations == pytest.approx((0.1, 0.2, 0.0))


def test_analyze_segments_rejects_fewer_than_three_points():
    candidate = {
        "timestamps": [0, 1],
        "values": [1.0, 1.0],
        "influence_region": (0.0, 2.0),
    }
    assert analyze_segments([candidate]) == []


@pytest.mark.parametrize(
    "values",
    [list(range(1, 10)), list(range(9, 0, -1))],
)
def test_analyze_segments_rejects_persistent_trends(values):
    candidate = {
        "timestamps": list(range(len(values))),
        "values": values,
        "influence_region": (0.0, 10.0),
    }
    assert analyze_segments([candidate]) == []


def test_analyze_segments_rejects_a_large_middle_regime_jump():
    values = [1, 1, 1, 5, 5, 5, 1, 1, 1]
    candidate = {
        "timestamps": list(range(len(values))),
        "values": values,
        "influence_region": (0.0, 6.0),
    }
    assert analyze_segments([candidate]) == []


def test_detector_filters_positive_peaks_once_and_passes_raw_valleys(monkeypatch):
    fake_kde = KDEResult(
        grid=np.arange(7, dtype=float),
        density=np.asarray([0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0]),
        bandwidth=1.0,
        data_range=(0.0, 6.0),
        adjusted_values=np.arange(7, dtype=float),
        fitted=True,
        zero_range_jittered=False,
        jitter_scale=0.0,
    )
    monkeypatch.setattr(module, "adaptive_kde", lambda *_args, **_kwargs: fake_kde)
    calls = iter(
        [
            (np.asarray([1, 5]), {}),
            (np.asarray([2, 4]), {}),
        ]
    )
    monkeypatch.setattr(module, "find_peaks", lambda _density: next(calls))
    filtered_inputs = []

    def fake_filter(peaks, _density, _config):
        filtered_inputs.append(np.asarray(peaks).tolist())
        return np.asarray(peaks)

    received_valleys = []

    def fake_regions(_grid, _peaks, valleys):
        received_valleys.extend(np.asarray(valleys).tolist())
        return []

    monkeypatch.setattr(module, "filter_peaks", fake_filter)
    monkeypatch.setattr(module, "find_peak_influence_regions", fake_regions)
    detector = StableSegmentDetector()
    assert detector.detect_stable_segments(range(7), range(7)) == []
    assert filtered_inputs == [[1, 5]]
    assert received_valleys == [2, 4]
