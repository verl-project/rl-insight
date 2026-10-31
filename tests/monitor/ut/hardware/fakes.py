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

"""Shared test doubles for hardware monitoring unit tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest

from rl_insight.hardware.base import HardwareCollector, Sample

_WAIT_TIMEOUT_SECONDS = 5.0


class RecordingClient:
    """Small client double that preserves every submitted event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def apply_events(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)


class WorkingCollector(HardwareCollector):
    """Fake collector reporting one labeled sample per round."""

    name = "fake_working"

    def discover(self) -> bool:
        return True

    def collect(self) -> list[Sample]:
        return [
            Sample(
                name="rlinsight_fake_gauge",
                value=7.5,
                labels={"device_index": 0},
                documentation="Fake gauge",
            )
        ]


def wait_for(condition: Callable[[], bool], message: str) -> None:
    """Block until ``condition`` holds (or fail with ``message``)."""
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    pytest.fail(message)
