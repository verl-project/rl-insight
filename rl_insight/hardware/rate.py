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

"""Counter-to-rate conversion for monotonically increasing counters."""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["RateCalculator"]


class RateCalculator:
    """Convert one monotonically increasing counter into a per-second rate.

    Keeps the previous ``(timestamp, counter)`` pair and returns
    ``(counter - previous) / (timestamp - previous_timestamp)``. Counter
    resets/wraps and non-increasing elapsed time yield ``None`` so callers
    can omit the sample for that round; the baseline is always refreshed so
    the next round recovers automatically.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        """
        Args:
            clock: Monotonic clock callable; defaults to ``time.monotonic``.
        """
        self._clock = clock or time.monotonic
        self._previous: tuple[float, float] | None = None

    def rate(self, counter: float) -> float | None:
        """Return the rate since the last call, or ``None`` when not computable.

        Args:
            counter: Latest cumulative counter value.

        Returns:
            Per-second rate, or ``None`` on the first call, when the elapsed
            time is not positive, or after a counter reset/wrap.
        """
        now = self._clock()
        previous = self._previous
        self._previous = (now, float(counter))
        if previous is None:
            return None
        previous_time, previous_counter = previous
        elapsed = now - previous_time
        if elapsed <= 0.0:
            return None
        delta = float(counter) - previous_counter
        if delta < 0.0:
            return None
        return delta / elapsed
