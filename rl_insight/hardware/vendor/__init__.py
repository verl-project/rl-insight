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

"""Vendor accelerator adapters for hardware monitoring.

Importing this package registers one ``device`` collector that probes every
bundled accelerator backend and samples whichever one is available on the
node. The vendor adapters probe their tooling lazily in ``discover``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from ..base import HardwareCollector, Sample, register_collector
from .ascend import AscendNpuCollector
from .nvidia import NvidiaGpuCollector

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["DeviceCollector"]


class DeviceCollector(HardwareCollector):
    """Accelerator metrics with automatic NPU/GPU detection.

    Probes each bundled vendor backend once at startup and samples every
    backend that reports itself available. Samples keep their
    ``device_vendor`` and ``device_index`` labels, so NPU and GPU series
    remain distinguishable when a node exposes both.
    """

    name = "device"

    def __init__(self, collectors: Iterable[HardwareCollector] | None = None) -> None:
        """``collectors`` is injectable for tests; defaults to bundled vendors."""
        self._candidates = (
            list(collectors)
            if collectors is not None
            else [AscendNpuCollector(), NvidiaGpuCollector()]
        )
        self._active: list[HardwareCollector] = []

    def discover(self) -> bool:
        """Return whether at least one accelerator backend is available."""
        active: list[HardwareCollector] = []
        for collector in self._candidates:
            try:
                available = bool(collector.discover())
            except Exception:
                logger.warning(
                    "[rl-insight] %s accelerator discovery failed; skipping it.",
                    collector.name,
                    exc_info=True,
                )
                continue
            if available:
                active.append(collector)
        self._active = active
        if not active:
            logger.info(
                "[rl-insight] no supported accelerator backend was detected; "
                "device metrics are unavailable on this node."
            )
        return bool(active)

    def collect(self) -> list[Sample]:
        """Return samples from every backend detected at startup."""
        samples: list[Sample] = []
        for collector in self._active:
            samples.extend(collector.collect())
        return samples


register_collector(DeviceCollector.name, DeviceCollector)
