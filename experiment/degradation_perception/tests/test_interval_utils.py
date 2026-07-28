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

import pytest

from experiment.degradation_perception.interval_utils import (
    build_candidate_intervals,
    validate_candidate_interval,
)


def valid_interval() -> dict:
    return {
        "duration": 1.0,
        "abnormal_point_count": 5,
        "total_point_count": 8,
        "continuous": True,
    }


def test_no_abnormal_points_produce_no_candidates():
    assert build_candidate_intervals([0, 1, 2], [False, False, False]) == []


def test_single_abnormal_point_is_only_a_candidate_not_a_formal_interval():
    candidates = build_candidate_intervals([0, 1, 2], [False, True, False])
    assert len(candidates) == 1
    valid, details = validate_candidate_interval(candidates[0])
    assert valid is False
    assert not all(details.values())


def test_candidate_merges_one_intervening_normal_and_retains_it_in_rate():
    candidates = build_candidate_intervals(
        list(range(7)),
        [True, True, True, False, True, True, True],
    )
    assert len(candidates) == 1
    assert candidates[0]["total_point_count"] == 7
    assert candidates[0]["abnormal_point_count"] == 6
    assert candidates[0]["abnormal_rate"] == pytest.approx(6 / 7)


def test_candidates_split_when_normal_gap_or_time_gap_is_too_large():
    by_normal = build_candidate_intervals(
        list(range(8)),
        [True, True, False, False, True, True, False, False],
    )
    assert len(by_normal) == 2

    by_time = build_candidate_intervals(
        [0, 1, 2, 100, 101, 102],
        [True] * 6,
        {"maximum_time_gap": 2},
    )
    assert len(by_time) == 2


def test_validation_exposes_the_required_four_condition_keys():
    valid, details = validate_candidate_interval(valid_interval())
    assert valid is True
    assert details == {
        "condition_1": True,
        "condition_2": True,
        "condition_3": True,
        "condition_4": True,
    }


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"duration": 0.5}, False),
        ({"duration": 0.500001}, True),
        ({"abnormal_point_count": 4, "total_point_count": 5}, False),
        ({"abnormal_point_count": 5, "total_point_count": 8}, True),
        ({"abnormal_point_count": 6, "total_point_count": 10}, False),
        ({"abnormal_point_count": 7, "total_point_count": 10}, True),
        ({"continuous": False}, False),
        ({"abnormal_point_count": 0, "total_point_count": 0}, False),
    ],
)
def test_formal_interval_strict_boundaries(changes, expected):
    interval = valid_interval()
    interval.update(changes)
    valid, _ = validate_candidate_interval(interval)
    assert valid is expected


def test_any_one_of_four_conditions_failing_rejects_interval():
    failures = [
        {"duration": 0.5},
        {"abnormal_point_count": 4},
        {"abnormal_point_count": 3, "total_point_count": 5},
        {"continuous": False},
    ]
    for failure in failures:
        interval = valid_interval()
        interval.update(failure)
        valid, _ = validate_candidate_interval(interval)
        assert valid is False
