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

"""NVIDIA GPU collector backed by NVML (optional ``nvidia-ml-py`` package)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from ..base import (
    HardwareCollector,
    Sample,
    optional_sample,
    drop_missing_samples,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["NvidiaGpuCollector"]

_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


def _safe(call: Callable[[], Any]) -> Any | None:
    """Run one NVML query; unsupported or failing fields become ``None``."""
    try:
        return call()
    except Exception:
        return None


def _text(call: Callable[[], Any]) -> str:
    """Read one string field, tolerating bytes and unsupported queries."""
    value = _safe(call)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value or "").strip()


def _visible_device_indexes() -> set[int] | None:
    """Return the integer GPU indexes allowed by ``CUDA_VISIBLE_DEVICES``.

    Only plain-integer values map to NVML indexes. UUID, MIG, and PCI-id forms
    leave the collector unrestricted. ``None`` means unrestricted.
    """
    raw = os.environ.get(_VISIBLE_DEVICES_ENV, "").strip()
    if not raw:
        return None
    indexes: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not (token.isascii() and token.isdigit()):
            return None
        indexes.add(int(token))
    return indexes or None


class NvidiaGpuCollector(HardwareCollector):
    """Per-device NVIDIA GPU metrics through the NVML library.

    ``pynvml`` is optional: when it is missing or no driver/device is visible,
    ``discover`` returns ``False`` and the collector is reported unavailable.
    Each field is queried independently, so one unsupported metric only omits
    that field. Reporting is scoped to integer ``CUDA_VISIBLE_DEVICES`` values.
    """

    name = "gpu"

    def __init__(self, pynvml_module: Any | None = None) -> None:
        """``pynvml_module`` is injectable for tests; ``None`` imports lazily."""
        self._pynvml_module = pynvml_module

    def _load_pynvml(self) -> Any | None:
        """Return the (cached) pynvml module, or ``None`` when unavailable."""
        if self._pynvml_module is None:
            try:
                import pynvml  # type: ignore[import-not-found, import-untyped]
            except ImportError:
                return None
            self._pynvml_module = pynvml
        return self._pynvml_module

    def discover(self) -> bool:
        pynvml = self._load_pynvml()
        if pynvml is None:
            logger.info(
                "[rl-insight] pynvml is not installed; NVIDIA GPU metrics are "
                "unavailable. Install with: pip install rl-insight[nvidia]"
            )
            return False
        try:
            pynvml.nvmlInit()
            return int(pynvml.nvmlDeviceGetCount()) > 0
        except Exception:
            logger.info(
                "[rl-insight] NVML initialization failed; NVIDIA GPU metrics "
                "are unavailable on this node.",
                exc_info=True,
            )
            return False

    def collect(self) -> list[Sample]:
        pynvml = self._load_pynvml()
        if pynvml is None:
            return []
        count = _safe(pynvml.nvmlDeviceGetCount)
        if not isinstance(count, int) or count <= 0:
            return []
        visible = _visible_device_indexes()
        samples: list[Sample] = []
        for index in range(count):
            if visible is not None and index not in visible:
                continue
            samples.extend(self._collect_device(pynvml, index))
        return samples

    def _collect_device(self, pynvml: Any, index: int) -> list[Sample]:
        """Collect one device; per-field failures only omit that field."""
        handle = _safe(lambda: pynvml.nvmlDeviceGetHandleByIndex(index))
        if handle is None:
            return []
        labels = {"device_vendor": "nvidia", "device_index": index}

        utilization = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        memory = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        memory_used = getattr(memory, "used", None)
        memory_total = getattr(memory, "total", None)
        memory_percent = (
            float(memory_used) / float(memory_total) * 100.0
            if memory_used is not None and memory_total
            else None
        )
        temperature = _safe(
            lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
        power_mw = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
        device_name = _text(lambda: pynvml.nvmlDeviceGetName(handle))
        driver_version = _text(pynvml.nvmlSystemGetDriverVersion)

        samples = drop_missing_samples(
            [
                optional_sample(
                    "rlinsight_gpu_utilization_percent",
                    utilization,
                    labels,
                    "NVIDIA GPU utilization percent",
                ),
                optional_sample(
                    "rlinsight_gpu_memory_used_bytes",
                    memory_used,
                    labels,
                    "NVIDIA GPU memory used in bytes",
                ),
                optional_sample(
                    "rlinsight_gpu_memory_utilization_percent",
                    memory_percent,
                    labels,
                    "NVIDIA GPU memory utilization percent",
                ),
                optional_sample(
                    "rlinsight_gpu_temperature_celsius",
                    temperature,
                    labels,
                    "NVIDIA GPU temperature in celsius",
                ),
                optional_sample(
                    "rlinsight_gpu_power_watts",
                    None if power_mw is None else float(power_mw) / 1000.0,
                    labels,
                    "NVIDIA GPU power draw in watts",
                ),
            ]
        )
        if device_name or driver_version:
            # Keep the label keys identical across devices, using empty values.
            info_labels = dict(labels)
            info_labels["device_name"] = device_name
            info_labels["driver_version"] = driver_version
            samples.append(
                Sample(
                    name="rlinsight_gpu_info",
                    value=1.0,
                    labels=info_labels,
                    documentation="NVIDIA GPU identity (constant 1; see labels)",
                )
            )
        return samples
