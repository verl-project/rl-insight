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

"""Centralized time-mode and interval-boundary handling."""

from __future__ import annotations

from collections.abc import Sequence


IDENTITY_DISPLAY_MODE = "identity"
REMOTE_MONITOR_DISPLAY_MODE = "remote_monitor_minutes"


def infer_training_time_mode(time_value: float) -> str:
    """Use the confirmed training-log boundary."""

    return "step" if float(time_value) < 10000 else "millisecond"


def infer_display_time_mode(
    timestamps: Sequence[float],
    source_type: str,
) -> str:
    """Choose one display conversion for an entire source series."""

    if source_type != "remote_monitor":
        return IDENTITY_DISPLAY_MODE
    return (
        REMOTE_MONITOR_DISPLAY_MODE
        if any(float(timestamp) > 10000 for timestamp in timestamps)
        else IDENTITY_DISPLAY_MODE
    )


def to_display_time(
    raw_time: float,
    source_type: str,
    *,
    mode: str | None = None,
) -> float:
    """Convert one value using an explicit series mode when supplied."""

    value = float(raw_time)
    selected_mode = (
        infer_display_time_mode([value], source_type)
        if mode is None
        else mode
    )
    supported_modes = {IDENTITY_DISPLAY_MODE, REMOTE_MONITOR_DISPLAY_MODE}
    if selected_mode not in supported_modes:
        raise ValueError(f"unsupported display time mode: {selected_mode!r}")
    if (
        selected_mode == REMOTE_MONITOR_DISPLAY_MODE
        and source_type != "remote_monitor"
    ):
        raise ValueError(
            "remote-monitor display mode requires source_type remote_monitor"
        )
    if selected_mode == REMOTE_MONITOR_DISPLAY_MODE:
        return value / 10000 / 60
    return value


def adjust_time_bounds(
    requested_start: float | None,
    requested_end: float | None,
    timestamps: Sequence[float],
    start_index: int,
    stop_index: int,
) -> tuple[float, float]:
    """Apply the confirmed one-unit padding with safe exclusive-stop indexes."""

    if len(timestamps) == 0:
        raise ValueError("timestamps must not be empty")
    if start_index < 0 or start_index >= len(timestamps):
        raise IndexError("start_index is out of range")
    if stop_index <= 0 or stop_index > len(timestamps):
        raise IndexError("stop_index is out of range")
    if start_index >= stop_index:
        raise ValueError("start_index must be less than stop_index")

    observed_start = float(timestamps[start_index])
    observed_end = float(timestamps[stop_index - 1])
    lower_request = (
        observed_start if requested_start is None else float(requested_start)
    )
    upper_request = observed_end if requested_end is None else float(requested_end)
    if lower_request > upper_request:
        raise ValueError("requested_start must not exceed requested_end")
    start_time = max(lower_request, observed_start) - 1
    end_time = min(upper_request, observed_end) + 1
    return start_time, end_time
