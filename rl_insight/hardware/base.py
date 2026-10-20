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

"""Hardware collector interfaces and registry for in-process monitoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "COLLECTOR_REGISTRY",
    "CollectorFactory",
    "HardwareCollector",
    "Sample",
    "create_collector",
    "list_collector_names",
    "optional_sample",
    "present_samples",
    "register_collector",
]


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
    value: float | int | None,
    labels: Mapping[str, Any] | None = None,
    documentation: str = "",
) -> Sample | None:
    """Build one gauge sample, or ``None`` when the value is unavailable.

    Collectors use this to omit unsupported fields instead of failing.

    Args:
        name: Metric family name.
        value: Observed value; ``None`` means the field is unavailable.
        labels: Extra label dimensions for this sample.
        documentation: Prometheus help text; defaults to ``Gauge {name}``.

    Returns:
        A ready-to-emit sample, or ``None``.
    """
    if value is None:
        return None
    return Sample(
        name=name,
        value=float(value),
        labels=dict(labels or {}),
        documentation=documentation,
    )


def present_samples(samples: Iterable[Sample | None]) -> list[Sample]:
    """Drop unavailable samples so missing fields are simply omitted."""
    return [sample for sample in samples if sample is not None]


class HardwareCollector(ABC):
    """One hardware sampling capability (CPU, one accelerator vendor, ...).

    Implementations must be cheap to construct and must keep ``discover`` and
    ``collect`` fast and side-effect free: they are called from the monitor's
    background thread and must never block the training loop. Expected
    conditions such as missing vendor tooling belong in ``discover`` (return
    ``False``); ``collect`` should only raise on genuinely unexpected errors,
    which the monitor tolerates and counts towards the failure policy.
    """

    name: str

    def configure(self, options: Mapping[str, Any]) -> None:
        """Apply optional per-collector configuration before ``discover``.

        Collectors are built through zero-argument factories, so options that
        come from the ``hardware`` config section are injected here instead.
        The monitor calls this exactly once, right after construction and
        before ``discover``. The default implementation ignores ``options``;
        collectors with configurable behavior (e.g. the disk path) override it.

        Args:
            options: Options resolved for this collector, ``{}`` when none.
        """

    @abstractmethod
    def discover(self) -> bool:
        """Return whether this node provides the capability right now.

        Probing must be quick and must not require root privileges, e.g.
        importing an optional vendor library or locating a CLI binary on
        ``PATH``. Returning ``False`` (or raising) marks the collector
        unavailable in diagnostics; it is then never sampled.
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
    """Register a zero-arg collector factory (usually the class) under ``name``.

    Args:
        name: Collector name referenced by the ``hardware.collectors`` config.
        factory: Callable returning a new ``HardwareCollector`` instance.

    Raises:
        ValueError: If ``name`` is empty or already registered.
    """
    if not name:
        raise ValueError("hardware collector name must be a non-empty string")
    if name in COLLECTOR_REGISTRY:
        raise ValueError(f"hardware collector {name!r} is already registered")
    COLLECTOR_REGISTRY[name] = factory


def list_collector_names() -> list[str]:
    """Return the sorted names of all registered collectors."""
    return sorted(COLLECTOR_REGISTRY)


def create_collector(name: str) -> HardwareCollector | None:
    """Instantiate a registered collector by name.

    Args:
        name: Collector name in the registry.

    Returns:
        A new collector instance, or ``None`` if the name is not registered.
    """
    factory = COLLECTOR_REGISTRY.get(name)
    return None if factory is None else factory()
