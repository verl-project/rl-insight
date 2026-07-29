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

import math

import pytest

from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.preprocessing import preprocess_time_series
from experiment.degradation_perception.time_alignment import (
    AlignmentResult,
    align_candidate_series,
    build_analysis_window,
)


def test_exact_step_alignment_and_coverage_use_window_target_denominator():
    result = align_candidate_series(
        TimeSeries([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
        [False, True, True],
        TimeSeries([1.0, 3.0], [100.0, 300.0]),
        [False, True],
        1.0,
        3.0,
        "training_log",
    )
    assert isinstance(result, AlignmentResult)
    assert result.target_indices == [0, 2]
    assert result.timestamps == [1.0, 3.0]
    assert result.target_values == [10.0, 30.0]
    assert result.target_labels == [False, True]
    assert result.candidate_values == [100.0, 300.0]
    assert result.candidate_labels == [False, True]
    assert result.coverage_ratio == pytest.approx(2 / 3)
    assert result.tolerance == 0.0


def test_time_mode_uses_nearest_and_selects_earlier_candidate_on_tie():
    result = align_candidate_series(
        TimeSeries([10.0, 20.0], [1.0, 2.0]),
        [False, True],
        TimeSeries([9.0, 11.0, 21.0], [90.0, 110.0, 210.0]),
        [False, True, True],
        9.0,
        21.0,
        "prometheus",
        max_tolerance=1.0,
    )
    assert result.candidate_values == [90.0, 210.0]
    assert result.coverage_ratio == 1.0
    assert result.tolerance == 1.0


def test_nearest_alignment_does_not_fill_beyond_tolerance():
    result = align_candidate_series(
        TimeSeries([0.0, 10.0, 20.0], [0.0, 10.0, 20.0]),
        [False, True, True],
        TimeSeries([0.0], [100.0]),
        [True],
        0.0,
        20.0,
        "prometheus",
        max_tolerance=1.0,
    )
    assert result.target_indices == [0]
    assert result.coverage_ratio == pytest.approx(1 / 3)


def test_default_tolerance_is_half_larger_positive_median_interval():
    result = align_candidate_series(
        TimeSeries([10000.0, 10010.0, 10020.0], [1.0, 2.0, 3.0]),
        [False, False, True],
        TimeSeries([10001.0, 10021.0], [11.0, 21.0]),
        [False, True],
        10000.0,
        10021.0,
        "training_log",
    )
    assert result.tolerance == 10.0
    assert result.coverage_ratio == pytest.approx(2 / 3)


def test_sparse_candidate_points_are_not_reused_to_inflate_coverage():
    target_timestamps = [float(index) for index in range(101)]
    result = align_candidate_series(
        TimeSeries(target_timestamps, target_timestamps),
        [False] * 101,
        TimeSeries([0.0, 100.0], [1.0, 2.0]),
        [False, True],
        0.0,
        100.0,
        "prometheus",
    )
    assert len(result.timestamps) == 2
    assert len(result.candidate_values) == 2
    assert result.coverage_ratio == pytest.approx(2 / 101)
    assert result.tolerance == 50.0


def test_extreme_finite_timestamp_gap_does_not_create_infinite_tolerance():
    result = align_candidate_series(
        TimeSeries([-1.0e308, 1.0e308], [1.0, 2.0]),
        [False, True],
        TimeSeries([-1.0e308, 1.0e308], [10.0, 20.0]),
        [False, True],
        -1.0e308,
        1.0e308,
        "prometheus",
    )
    assert result.coverage_ratio == 1.0
    assert result.tolerance == 0.0
    assert math.isfinite(result.tolerance)


def test_non_finite_points_are_skipped_and_candidates_outside_window_are_unused():
    result = align_candidate_series(
        TimeSeries(
            [0.0, 1.0, math.nan, 3.0],
            [1.0, math.nan, 3.0, 4.0],
        ),
        [0, 1, 1, 1],
        TimeSeries(
            [0.0, math.nan, 3.0, 4.0],
            [10.0, 20.0, math.inf, 40.0],
        ),
        [1, 1, 1, 0],
        0.0,
        3.0,
        "prometheus",
        max_tolerance=1.1,
    )
    assert result.target_indices == [0]
    assert result.target_labels == [False]
    assert result.candidate_labels == [True]
    assert result.coverage_ratio == 0.5


def test_preprocessed_duplicate_timestamps_keep_last_values_before_alignment():
    target = preprocess_time_series([1, 1, 2], [10, 11, 20])
    candidate = preprocess_time_series([1, 1, 2], [100, 101, 200])
    result = align_candidate_series(
        target,
        [False, True],
        candidate,
        [True, False],
        1.0,
        2.0,
        "training_log",
    )
    assert result.target_values == [11.0, 20.0]
    assert result.candidate_values == [101.0, 200.0]


def test_training_log_uses_exact_alignment_only_when_all_times_are_steps():
    exact = align_candidate_series(
        TimeSeries([1.0], [1.0]),
        [False],
        TimeSeries([1.1], [2.0]),
        [True],
        1.0,
        1.1,
        "training_log",
        max_tolerance=1.0,
    )
    nearest = align_candidate_series(
        TimeSeries([10000.0], [1.0]),
        [False],
        TimeSeries([10000.1], [2.0]),
        [True],
        10000.0,
        10000.1,
        "training_log",
        max_tolerance=1.0,
    )
    assert exact.target_indices == []
    assert exact.tolerance == 0.0
    assert nearest.target_indices == [0]
    assert nearest.tolerance == 1.0


def test_empty_target_window_has_zero_coverage_without_candidate_fill():
    result = align_candidate_series(
        TimeSeries([1.0], [1.0]),
        [False],
        TimeSeries([2.0], [2.0]),
        [True],
        10.0,
        20.0,
        "prometheus",
        max_tolerance=1.0,
    )
    assert result.target_indices == []
    assert result.coverage_ratio == 0.0


def test_analysis_window_expands_by_duration_ratio_and_clips_to_data():
    result = build_analysis_window(
        4.0,
        6.0,
        list(range(0, 8)),
        context_ratio=1.0,
    )
    assert result == {
        "startTime": 2.0,
        "endTime": 7.0,
        "usedMinimumContext": False,
    }


def test_zero_duration_window_uses_positive_median_interval():
    result = build_analysis_window(
        2.0,
        2.0,
        [0.0, 1.0, 2.0, 3.0, 4.0],
        context_ratio=0.0,
    )
    assert result == {
        "startTime": 1.0,
        "endTime": 3.0,
        "usedMinimumContext": True,
    }


def test_zero_duration_window_without_positive_interval_uses_zero_context():
    result = build_analysis_window(
        2.0,
        2.0,
        [2.0, 2.0],
        context_ratio=1.0,
    )
    assert result == {
        "startTime": 2.0,
        "endTime": 2.0,
        "usedMinimumContext": True,
    }


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((2.0, 1.0, [1.0, 2.0], 1.0), "raw_start"),
        ((1.0, 2.0, [1.0, 2.0], -1.0), "context_ratio"),
        ((1.0, 2.0, [1.0, 2.0], math.nan), "context_ratio"),
        ((math.inf, 2.0, [1.0, 2.0], 1.0), "raw_start"),
        ((1.0, 2.0, [math.nan], 1.0), "target_timestamps"),
        ((10.0, 11.0, [1.0, 2.0], 1.0), "does not overlap"),
    ],
)
def test_analysis_window_rejects_invalid_parameters(args, match):
    with pytest.raises(ValueError, match=match):
        build_analysis_window(*args)


@pytest.mark.parametrize(
    ("target", "target_labels", "candidate", "candidate_labels", "kwargs", "match"),
    [
        (
            TimeSeries([1.0], []),
            [False],
            TimeSeries(),
            [],
            {},
            "target timestamps",
        ),
        (
            TimeSeries([1.0], [1.0]),
            [],
            TimeSeries(),
            [],
            {},
            "target timestamps",
        ),
        (
            TimeSeries(),
            [],
            TimeSeries([1.0], []),
            [False],
            {},
            "candidate timestamps",
        ),
        (
            TimeSeries(),
            [],
            TimeSeries([1.0], [1.0]),
            [],
            {},
            "candidate timestamps",
        ),
        (
            TimeSeries(),
            [],
            TimeSeries(),
            [],
            {"source_type": "unknown"},
            "unsupported source_type",
        ),
        (
            TimeSeries(),
            [],
            TimeSeries(),
            [],
            {"max_tolerance": 0.0},
            "max_tolerance",
        ),
        (
            TimeSeries(),
            [],
            TimeSeries(),
            [],
            {"max_tolerance": math.nan},
            "max_tolerance",
        ),
    ],
)
def test_alignment_rejects_invalid_series_and_parameters(
    target,
    target_labels,
    candidate,
    candidate_labels,
    kwargs,
    match,
):
    parameters = {
        "start_time": 0.0,
        "end_time": 2.0,
        "source_type": "prometheus",
        **kwargs,
    }
    with pytest.raises((TypeError, ValueError), match=match):
        align_candidate_series(
            target,
            target_labels,
            candidate,
            candidate_labels,
            **parameters,
        )


def test_alignment_rejects_reversed_or_non_finite_window():
    series = TimeSeries([1.0], [1.0])
    with pytest.raises(ValueError, match="start_time"):
        align_candidate_series(
            series,
            [False],
            series,
            [False],
            2.0,
            1.0,
            "prometheus",
        )
    with pytest.raises(ValueError, match="end_time"):
        align_candidate_series(
            series,
            [False],
            series,
            [False],
            1.0,
            math.inf,
            "prometheus",
        )
