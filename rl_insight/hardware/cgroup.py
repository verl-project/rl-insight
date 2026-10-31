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

"""Best-effort cgroup v1/v2 memory and CPU-throttling readers for containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = [
    "CgroupCpuThrottling",
    "read_cgroup_cpu_throttling",
    "read_cgroup_memory_limit",
    "read_cgroup_memory_usage",
]

_CGROUP_ROOT = Path("/sys/fs/cgroup")

# Limits at or above this are kernel "unlimited" sentinels, not real limits.
_UNLIMITED_THRESHOLD = 1 << 62

# ``cpu.stat`` exposes the throttling time under different keys and units:
# cgroup v2 uses ``throttled_usec`` (microseconds), v1 ``throttled_time``
# (nanoseconds).
_THROTTLED_USEC_FACTOR = 1e-6
_THROTTLED_TIME_FACTOR = 1e-9


def _read_first_int(path: Path) -> int | None:
    """Parse the first whitespace-separated token of ``path`` as an int."""
    try:
        return int(path.read_text(encoding="utf-8").split()[0])
    except (OSError, IndexError, ValueError):
        return None


def read_cgroup_memory_limit(cgroup_root: Path = _CGROUP_ROOT) -> int | None:
    """Return the cgroup memory limit in bytes, or ``None`` when unlimited/absent.

    Tries cgroup v2 ``memory.max`` first, then v1
    ``memory/memory.limit_in_bytes``. ``max`` and other unlimited sentinels
    map to ``None`` so hosts without container limits stay quiet.

    Args:
        cgroup_root: cgroup filesystem root to read from.

    Returns:
        The effective memory limit in bytes, or ``None``.
    """
    for relative in ("memory.max", "memory/memory.limit_in_bytes"):
        value = _read_first_int(cgroup_root / relative)
        if value is not None and 0 < value < _UNLIMITED_THRESHOLD:
            return value
    return None


def read_cgroup_memory_usage(cgroup_root: Path = _CGROUP_ROOT) -> int | None:
    """Return the cgroup memory usage in bytes, or ``None`` when unavailable.

    Args:
        cgroup_root: cgroup filesystem root to read from.

    Returns:
        The current memory usage in bytes, or ``None``.
    """
    for relative in ("memory.current", "memory/memory.usage_in_bytes"):
        value = _read_first_int(cgroup_root / relative)
        if value is not None and value >= 0:
            return value
    return None


@dataclass(frozen=True)
class CgroupCpuThrottling:
    """Accumulated CPU throttling of the container cgroup.

    Attributes:
        throttled_periods: Number of scheduling periods the cgroup was
            throttled since it started (``nr_throttled``).
        throttled_seconds: Total time the cgroup spent throttled, in seconds.
    """

    throttled_periods: int | None
    throttled_seconds: float | None


def _read_cpu_stat(path: Path) -> dict[str, int] | None:
    """Parse a ``cpu.stat`` key/value file into an int map, or ``None``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stats: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            stats[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return stats or None


def _throttled_seconds(stats: Mapping[str, int]) -> float | None:
    """Convert the throttling time field of ``cpu.stat`` into seconds."""
    for key, factor in (
        ("throttled_usec", _THROTTLED_USEC_FACTOR),  # cgroup v2
        ("throttled_time", _THROTTLED_TIME_FACTOR),  # cgroup v1
    ):
        value = stats.get(key)
        if value is not None and value >= 0:
            return value * factor
    return None


def read_cgroup_cpu_throttling(
    cgroup_root: Path = _CGROUP_ROOT,
) -> CgroupCpuThrottling | None:
    """Return the cgroup CPU throttling counters, or ``None`` when unavailable.

    Tries cgroup v2 ``cpu.stat`` first, then v1 ``cpu/cpu.stat``. Both the
    throttled-period count and the throttled time are cumulative since the
    cgroup was created, so consumers derive rates from consecutive reads.

    Args:
        cgroup_root: cgroup filesystem root to read from.

    Returns:
        The parsed throttling counters, or ``None`` when ``cpu.stat`` is
        absent or has no recognized throttling fields (e.g. on hosts without
        cgroup CPU limits).
    """
    for relative in ("cpu.stat", "cpu/cpu.stat"):
        stats = _read_cpu_stat(cgroup_root / relative)
        if stats is None:
            continue
        periods = stats.get("nr_throttled")
        seconds = _throttled_seconds(stats)
        if periods is None and seconds is None:
            continue
        return CgroupCpuThrottling(throttled_periods=periods, throttled_seconds=seconds)
    return None
