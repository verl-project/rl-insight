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

"""Shared hardware monitoring types, identity helpers, and collector registry."""

from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFactory",
    "HardwareCollector",
    "HardwareSettings",
    "Sample",
    "current_ray_job_id",
    "identity_labels",
    "optional_sample",
    "drop_missing_samples",
    "register_collector",
]


@dataclass(frozen=True)
class HardwareSettings:
    """Resolved ``hardware`` config shared by both monitor modes.

    Attributes:
        interval_seconds: Sampling interval, clamped to the documented minimum.
        collector_names: Collector names to resolve.
    """

    interval_seconds: float
    collector_names: list[str]


@dataclass
class Sample:
    """One hardware observation ready to be emitted as a gauge event.

    Attributes:
        name: Metric family name, e.g. ``rlinsight_gpu_utilization_percent``.
        value: Observed numeric value.
        labels: Extra label dimensions for this sample only (e.g. ``device_index``).
        documentation: Prometheus help text; defaults to ``Gauge {name}`` when empty.
    """

    name: str
    value: float
    labels: Mapping[str, Any] = field(default_factory=dict)
    documentation: str = ""


def optional_sample(
    name: str,
    value: float | None,
    labels: Mapping[str, Any] | None = None,
    documentation: str = "",
) -> Sample | None:
    """Build a sample, or ``None`` when ``value`` is unavailable.

    Collectors use this to omit unsupported fields instead of failing.
    """
    if value is None:
        return None
    return Sample(
        name=name,
        value=float(value),
        labels=dict(labels or {}),
        documentation=documentation,
    )


def drop_missing_samples(samples: Iterable[Sample | None]) -> list[Sample]:
    """Drop unavailable samples so missing fields are simply omitted."""
    return [sample for sample in samples if sample is not None]


def current_ray_job_id() -> str:
    """Return the current Ray job id, or an empty string outside Ray."""
    try:
        import ray

        # get_runtime_context() starts a local Ray instance; check first.
        if ray.is_initialized():
            job_id = ray.get_runtime_context().get_job_id()
            return job_id if isinstance(job_id, str) else job_id.hex()
    except Exception:
        pass
    return ""


def identity_labels() -> dict[str, str]:
    """Return the node and run-id labels for the current process.

    ``node`` is the Ray node IP inside a job and the hostname otherwise;
    ``run_id`` is the Ray job id and is omitted outside Ray.
    """
    try:
        import ray

        node = str(ray.util.get_node_ip_address()) if ray.is_initialized() else ""
    except Exception:
        node = ""
    labels = {"node": node or socket.gethostname()}
    if run_id := current_ray_job_id():
        labels["run_id"] = run_id
    return labels


class HardwareCollector(ABC):
    """One hardware sampling capability (CPU, one accelerator vendor, ...).

    ``discover`` and ``collect`` run on the monitor's background thread.
    Missing tooling belongs in ``discover`` (return ``False``); ``collect``
    raises only on unexpected errors, which the monitor counts against its
    failure policy.
    """

    name: str

    @abstractmethod
    def discover(self) -> bool:
        """Return whether this node provides the capability right now.

        The probe should be quick and avoid root privileges, for example by
        importing an optional vendor library or locating a CLI binary.
        ``False`` or an exception marks the collector unavailable, and it is
        not sampled afterwards.
        """

    @abstractmethod
    def collect(self) -> list[Sample]:
        """Return the samples for one round; may be an empty list."""


CollectorFactory = Callable[[], HardwareCollector]

# Collector name -> zero-arg factory (usually the collector class itself).
# Modules providing collectors register themselves at import time; the
# coordinator resolves the configured ``hardware.collectors`` names here.
COLLECTOR_REGISTRY: dict[str, CollectorFactory] = {}


def register_collector(name: str, factory: CollectorFactory) -> None:
    """Register a zero-arg collector factory under ``name``.

    Raises ``ValueError`` when ``name`` is empty or already registered.
    """
    if not name:
        raise ValueError("hardware collector name must be a non-empty string")
    if name in COLLECTOR_REGISTRY:
        raise ValueError(f"hardware collector {name!r} is already registered")
    COLLECTOR_REGISTRY[name] = factory
