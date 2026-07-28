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

"""Typed values shared by the degradation-perception module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MetricState(IntEnum):
    """Per-metric detection state."""

    NORMAL = 0
    STANDARD_DATA_INSUFFICIENT = 1
    INFERENCE_DATA_INSUFFICIENT = 2


@dataclass(frozen=True)
class TimeSeries:
    """A validated time series whose timestamps and values are aligned."""

    timestamps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class ThresholdModel:
    """One normal-mode KDE and its configured thresholds."""

    mode_id: int
    segment_start_time: float
    segment_end_time: float
    point_count: int
    lower_kde_threshold: float
    upper_kde_threshold: float
    lower_threshold: float
    upper_threshold: float
    bandwidth: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


DEFAULT_METRIC = "timing_s/step"
SUPPORTED_ABNORMAL_TYPES = {"UP", "DOWN", "BOTH"}
SUPPORTED_SOURCE_TYPES = {"prometheus", "remote_monitor", "training_log"}
