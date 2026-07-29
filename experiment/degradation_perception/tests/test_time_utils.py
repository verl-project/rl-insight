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

from experiment.degradation_perception.time_utils import (
    IDENTITY_DISPLAY_MODE,
    REMOTE_MONITOR_DISPLAY_MODE,
    adjust_time_bounds,
    infer_display_time_mode,
    infer_training_time_mode,
    to_display_time,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(9999, "step"), (10000, "millisecond"), (10001, "millisecond")],
)
def test_training_time_mode_has_the_confirmed_boundary(value, expected):
    assert infer_training_time_mode(value) == expected


def test_remote_display_time_has_a_distinct_strict_boundary():
    assert to_display_time(10000, "remote_monitor") == 10000
    assert to_display_time(10001, "remote_monitor") == pytest.approx(
        10001 / 10000 / 60
    )
    assert to_display_time(10001, "training_log") == 10001
    assert to_display_time(1_710_000_000, "prometheus") == 1_710_000_000


@pytest.mark.parametrize(
    ("timestamps", "expected"),
    [
        ([9998, 9999, 10000], IDENTITY_DISPLAY_MODE),
        ([10001, 10002], REMOTE_MONITOR_DISPLAY_MODE),
        ([9999, 10000, 10001], REMOTE_MONITOR_DISPLAY_MODE),
    ],
)
def test_remote_display_mode_is_resolved_once_for_the_whole_series(
    timestamps,
    expected,
):
    mode = infer_display_time_mode(timestamps, "remote_monitor")
    converted = [
        to_display_time(value, "remote_monitor", mode=mode)
        for value in timestamps
    ]

    assert mode == expected
    assert converted == sorted(converted)


def test_explicit_source_types_take_priority_over_numeric_magnitude():
    assert infer_display_time_mode([10001], "training_log") == (
        IDENTITY_DISPLAY_MODE
    )
    assert infer_display_time_mode([1_710_000_000], "prometheus") == (
        IDENTITY_DISPLAY_MODE
    )
    assert to_display_time(
        9999,
        "remote_monitor",
        mode=REMOTE_MONITOR_DISPLAY_MODE,
    ) == pytest.approx(9999 / 10000 / 60)


def test_adjust_time_bounds_uses_exclusive_stop_and_one_raw_unit_padding():
    assert adjust_time_bounds(9, 13, [10, 11, 12], 0, 3) == (9.0, 13.0)
    assert adjust_time_bounds(10.5, 11.5, [10, 11, 12], 1, 2) == (10.0, 12.0)
    assert adjust_time_bounds(None, None, [10, 11, 12], 1, 3) == (10.0, 13.0)


@pytest.mark.parametrize(
    ("timestamps", "start", "stop", "exception"),
    [
        ([], 0, 1, ValueError),
        ([1], -1, 1, IndexError),
        ([1], 1, 1, IndexError),
        ([1], 0, 0, IndexError),
        ([1], 0, 2, IndexError),
        ([1, 2], 1, 1, ValueError),
    ],
)
def test_adjust_time_bounds_defends_indexes(timestamps, start, stop, exception):
    with pytest.raises(exception):
        adjust_time_bounds(None, None, timestamps, start, stop)


def test_adjust_time_bounds_rejects_reversed_requested_window():
    with pytest.raises(ValueError, match="must not exceed"):
        adjust_time_bounds(3, 2, [1, 2, 3], 0, 3)
