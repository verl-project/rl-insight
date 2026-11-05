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
from typing import Any, Callable

from ..base import (
    HardwareCollector,
    Sample,
    optional_sample,
    present_samples,
    register_collector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["NvidiaGpuCollector"]

_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


def _to_text(value: Any) -> str:
    """Decode NVML strings that are bytes on older ``pynvml`` versions."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe(call: Callable[[], Any]) -> Any | None:
    """Run one NVML query; unsupported or failing fields become ``None``."""
    try:
        return call()
    except Exception:
        return None


def _visible_device_indexes() -> set[int] | None:
    """Integer GPU indexes this process may use, or ``None`` when unrestricted.

    NVML always enumerates physical devices by integer index, so only the
    plain-integer form of ``CUDA_VISIBLE_DEVICES`` can be mapped to NVML
    indexes. Other forms (PCI bus ids, GPU UUIDs, MIG ids) and an unset or
    empty variable leave the collector unrestricted, so a job pinned to a
    subset is scoped to its cards without silently dropping every GPU.
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

    ``pynvml`` is an optional dependency: when it is not installed, or no
    driver/devices are visible, ``discover`` returns ``False`` and the
    collector is reported unavailable in diagnostics instead of failing.
    Each device and each field is queried independently so one unsupported
    metric or one unhealthy device never drops the whole round. Reporting is
    scoped to ``CUDA_VISIBLE_DEVICES`` when that variable holds plain integer
    indexes, so a job using a subset of the node's cards only publishes
    samples for those cards.
    """

    name = "gpu"

    def __init__(self, pynvml_module: Any | None = None) -> None:
        """
        Args:
            pynvml_module: Injected ``pynvml``-compatible module for tests;
                ``None`` lazily imports the real package on first use.
        """
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
        memory_percent = None
        if memory_used is not None and memory_total:
            memory_percent = float(memory_used) / float(memory_total) * 100.0
        temperature = _safe(
            lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
        power_mw = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
        device_name = _to_text(
            _safe(lambda: pynvml.nvmlDeviceGetName(handle)) or ""
        ).strip()
        driver_version = _to_text(
            _safe(pynvml.nvmlSystemGetDriverVersion) or ""
        ).strip()

        samples = present_samples(
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
                    "rlinsight_gpu_memory_total_bytes",
                    memory_total,
                    labels,
                    "NVIDIA GPU memory total in bytes",
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
            # Keep the label dimension fixed across devices so a device that
            # resolves no name/version still carries the same keys (empty
            # values) as its siblings.
            info_labels = dict(labels)
            info_labels["device_name"] = device_name
            info_labels["driver_version"] = driver_version
            samples.append(
                Sample(
                    name="rlinsight_gpu_info",
                    value=1.0,
                    labels=info_labels,
                    documentation="NVIDIA GPU identity (always 1; see labels)",
                )
            )
        return samples


register_collector(NvidiaGpuCollector.name, NvidiaGpuCollector)
