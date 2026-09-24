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

"""Build per-series KDE baselines from step-aligned metric snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, peak_prominences  # type: ignore[import-untyped]
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

from .data import SeriesId, StepFrame


class BaselineDataError(ValueError):
    """A metric cannot produce a baseline from the supplied step frames."""


class InsufficientBaselineDataError(BaselineDataError):
    """A metric has fewer valid observations than the configured minimum."""


class NoStableSegmentError(BaselineDataError):
    """No time-contiguous density mode passed the stability test."""


@dataclass(frozen=True)
class BaselineParameters:
    """Parameters that affect fitted baseline bounds."""

    minimum_samples: int = 20
    alpha: float = 0.025
    lower_ratio: float = 1.05
    upper_ratio: float = 1.05
    bandwidth: str | float = "auto"
    grid_size: int = 1024
    padding_ratio: float = 0.10
    tail_bandwidths: float = 6.0
    zero_range_epsilon: float = 1.0e-8
    random_seed: int = 42
    peak_prominence_ratio: float = 0.01
    std_factor: float = 2.0
    within_std_coefficient: float = 1.05
    minimum_passed_flags: int = 4
    mean_tolerance_ratio: float = 0.02
    step_gap_factor: float = 3.0
    maximum_step_gap: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.minimum_samples, bool) or not isinstance(
            self.minimum_samples, int
        ):
            raise TypeError("minimum_samples must be an integer")
        if self.minimum_samples < 3:
            raise ValueError("minimum_samples must be an integer of at least 3")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5")
        for name in ("lower_ratio", "upper_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and at least 1")
        if self.bandwidth != "auto":
            _positive(self.bandwidth, "bandwidth")
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, int):
            raise TypeError("grid_size must be an integer")
        if self.grid_size < 32:
            raise ValueError("grid_size must be an integer of at least 32")
        _non_negative(self.padding_ratio, "padding_ratio")
        for name in (
            "tail_bandwidths",
            "zero_range_epsilon",
            "std_factor",
            "step_gap_factor",
        ):
            _positive(getattr(self, name), name)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        _non_negative(self.peak_prominence_ratio, "peak_prominence_ratio")
        if not 1.0 <= self.within_std_coefficient < 2.0:
            raise ValueError("within_std_coefficient must be in [1, 2)")
        if not 1 <= self.minimum_passed_flags <= 6:
            raise ValueError("minimum_passed_flags must be between 1 and 6")
        _non_negative(self.mean_tolerance_ratio, "mean_tolerance_ratio")
        if self.maximum_step_gap is not None:
            _positive(self.maximum_step_gap, "maximum_step_gap")


@dataclass(frozen=True)
class NormalRange:
    """One stable normal mode and its fitted thresholds."""

    mode_id: int
    start_step: int
    end_step: int
    sample_count: int
    kde_lower: float
    kde_upper: float
    lower: float
    upper: float
    bandwidth: float


@dataclass(frozen=True)
class SeriesBaseline:
    """All normal modes fitted for one concrete metric series."""

    identity: SeriesId
    sample_count: int
    ranges: tuple[NormalRange, ...]


@dataclass(frozen=True)
class BaselineFit:
    """Successful baselines and explicit per-series skip reasons."""

    baselines: dict[SeriesId, SeriesBaseline] = field(default_factory=dict)
    skipped: dict[SeriesId, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _KDE:
    grid: np.ndarray
    density: np.ndarray
    bandwidth: float


@dataclass(frozen=True)
class _Segment:
    steps: tuple[int, ...]
    values: tuple[float, ...]


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _fit_kde(values: tuple[float, ...], parameters: BaselineParameters) -> _KDE:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise BaselineDataError("KDE requires at least two finite values")

    adjusted = data.copy()
    if float(np.ptp(adjusted)) == 0.0:
        scale = max(abs(float(adjusted[0])), 1.0) * parameters.zero_range_epsilon
        adjusted += np.random.default_rng(parameters.random_seed).normal(
            0.0, scale, adjusted.size
        )

    try:
        method = None if parameters.bandwidth == "auto" else float(parameters.bandwidth)
        model = gaussian_kde(adjusted, bw_method=method)
        bandwidth = float(np.sqrt(model.covariance[0, 0]))
        span = float(np.ptp(adjusted))
        padding = max(
            span * parameters.padding_ratio,
            parameters.tail_bandwidths * bandwidth,
        )
        grid = np.linspace(
            float(np.min(adjusted)) - padding,
            float(np.max(adjusted)) + padding,
            parameters.grid_size,
        )
        density = np.maximum(np.asarray(model(grid), dtype=float), 0.0)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise BaselineDataError("KDE fitting failed") from exc
    if not np.all(np.isfinite(density)):
        raise BaselineDataError("KDE produced a non-finite density")
    return _KDE(grid=grid, density=density, bandwidth=bandwidth)


def _quantile(kde: _KDE, probability: float) -> float:
    increments = (kde.density[:-1] + kde.density[1:]) * 0.5 * np.diff(kde.grid)
    cdf = np.concatenate((np.asarray([0.0]), np.cumsum(increments)))
    total = float(cdf[-1])
    if not math.isfinite(total) or total <= 0.0:
        raise BaselineDataError("KDE density has no positive finite mass")
    cdf = np.maximum.accumulate(cdf / total)
    cdf[-1] = 1.0
    return float(np.interp(probability, cdf, kde.grid))


def _regions(kde: _KDE, prominence_ratio: float) -> list[tuple[float, float, float]]:
    peaks, _ = find_peaks(kde.density)
    valleys, _ = find_peaks(-kde.density)
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(kde.density))])
    minimum = float(np.max(kde.density)) * prominence_ratio
    retained = peaks[peak_prominences(kde.density, peaks)[0] >= minimum]
    if retained.size == 0:
        retained = np.asarray([peaks[int(np.argmax(kde.density[peaks]))]])
    if retained.size == 1:
        peak = int(retained[0])
        return [(float(kde.grid[0]), float(kde.grid[-1]), float(kde.grid[peak]))]

    regions: list[tuple[float, float, float]] = []
    for raw_peak in retained:
        peak = int(raw_peak)
        left = valleys[valleys < peak]
        right = valleys[valleys > peak]
        left_index = int(left[-1]) if left.size else 0
        right_index = int(right[0]) if right.size else len(kde.grid) - 1
        regions.append(
            (
                float(kde.grid[left_index]),
                float(kde.grid[right_index]),
                float(kde.grid[peak]),
            )
        )
    return regions


def _maximum_gap(steps: tuple[int, ...], parameters: BaselineParameters) -> float:
    if parameters.maximum_step_gap is not None:
        return parameters.maximum_step_gap
    differences = np.diff(np.asarray(steps, dtype=float))
    positive = differences[differences > 0.0]
    return (
        math.inf
        if positive.size == 0
        else float(np.median(positive)) * parameters.step_gap_factor
    )


def _candidates(
    steps: tuple[int, ...],
    values: tuple[float, ...],
    regions: list[tuple[float, float, float]],
    maximum_gap: float,
) -> list[_Segment]:
    assignments: list[int | None] = []
    for value in values:
        eligible = [
            index
            for index, (lower, upper, _peak) in enumerate(regions)
            if lower <= value <= upper
        ]
        assignments.append(
            min(eligible, key=lambda index: (abs(value - regions[index][2]), index))
            if eligible
            else None
        )

    candidates: list[_Segment] = []
    for region_index in range(len(regions)):
        run: list[int] = []
        for index, assignment in enumerate(assignments):
            gap = bool(run and steps[index] - steps[run[-1]] > maximum_gap)
            if (assignment != region_index or gap) and run:
                candidates.append(_segment_from_indexes(run, steps, values))
                run = []
            if assignment == region_index:
                run.append(index)
        if run:
            candidates.append(_segment_from_indexes(run, steps, values))
    return candidates


def _segment_from_indexes(
    indexes: list[int], steps: tuple[int, ...], values: tuple[float, ...]
) -> _Segment:
    return _Segment(
        steps=tuple(steps[index] for index in indexes),
        values=tuple(values[index] for index in indexes),
    )


def _within(
    value: float, mean: float, std: float, parameters: BaselineParameters
) -> bool:
    if std == 0.0:
        precision = 8.0 * max(math.ulp(value), math.ulp(mean))
        return math.isclose(value, mean, rel_tol=0.0, abs_tol=precision)
    margin = parameters.std_factor * std
    adjustment = (parameters.within_std_coefficient - 1.0) * margin
    return mean - margin + adjustment < value < mean + margin - adjustment


def _stable(segment: _Segment, parameters: BaselineParameters) -> bool:
    size = len(segment.values) // 3
    if size == 0:
        return False
    values = np.asarray(segment.values)
    parts = (values[:size], values[size : 2 * size], values[2 * size :])
    means = tuple(float(np.mean(part)) for part in parts)
    deviations = tuple(float(np.std(part, ddof=0)) for part in parts)
    effective = tuple(
        max(std, abs(mean) * parameters.mean_tolerance_ratio, 8.0 * math.ulp(mean))
        for mean, std in zip(means, deviations)
    )
    comparisons = ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0))
    passed = sum(
        _within(means[target], means[reference], effective[reference], parameters)
        for target, reference in comparisons
    )
    return passed >= parameters.minimum_passed_flags


def _outward_lower(value: float, ratio: float) -> float:
    return value / ratio if value >= 0.0 else value * ratio


def _outward_upper(value: float, ratio: float) -> float:
    return value * ratio if value >= 0.0 else value / ratio


DEFAULT_BASELINE_PARAMETERS = BaselineParameters()


def fit_series_baseline(
    frames: list[StepFrame] | tuple[StepFrame, ...],
    identity: SeriesId,
    *,
    parameters: BaselineParameters = DEFAULT_BASELINE_PARAMETERS,
) -> SeriesBaseline:
    """Fit independent KDE bounds for every stable mode of one series."""

    observations = tuple(
        (frame.step, float(value))
        for frame in frames
        if (value := frame.values.get(identity)) is not None and math.isfinite(value)
    )
    if len(observations) < parameters.minimum_samples:
        raise InsufficientBaselineDataError(
            f"{identity.name!r} has {len(observations)} valid samples; "
            f"at least {parameters.minimum_samples} are required"
        )

    steps = tuple(step for step, _value in observations)
    values = tuple(value for _step, value in observations)
    history_kde = _fit_kde(values, parameters)
    candidates = _candidates(
        steps,
        values,
        _regions(history_kde, parameters.peak_prominence_ratio),
        _maximum_gap(steps, parameters),
    )
    stable = [segment for segment in candidates if _stable(segment, parameters)]
    if not stable:
        raise NoStableSegmentError(f"{identity.name!r} has no stable normal segment")

    ranges: list[NormalRange] = []
    for mode_id, segment in enumerate(stable):
        kde = _fit_kde(segment.values, parameters)
        kde_lower = _quantile(kde, parameters.alpha)
        kde_upper = _quantile(kde, 1.0 - parameters.alpha)
        ranges.append(
            NormalRange(
                mode_id=mode_id,
                start_step=segment.steps[0],
                end_step=segment.steps[-1],
                sample_count=len(segment.values),
                kde_lower=kde_lower,
                kde_upper=kde_upper,
                lower=_outward_lower(kde_lower, parameters.lower_ratio),
                upper=_outward_upper(kde_upper, parameters.upper_ratio),
                bandwidth=kde.bandwidth,
            )
        )
    return SeriesBaseline(
        identity=identity, sample_count=len(values), ranges=tuple(ranges)
    )


def fit_baselines(
    frames: list[StepFrame] | tuple[StepFrame, ...],
    *,
    parameters: BaselineParameters = DEFAULT_BASELINE_PARAMETERS,
) -> BaselineFit:
    """Fit every series present in the frames without hiding sparse inputs."""

    identities = sorted({identity for frame in frames for identity in frame.values})
    baselines: dict[SeriesId, SeriesBaseline] = {}
    skipped: dict[SeriesId, str] = {}
    for identity in identities:
        try:
            baselines[identity] = fit_series_baseline(
                frames, identity, parameters=parameters
            )
        except BaselineDataError as exc:
            skipped[identity] = str(exc)
    return BaselineFit(baselines=baselines, skipped=skipped)


__all__ = [
    "DEFAULT_BASELINE_PARAMETERS",
    "BaselineDataError",
    "BaselineFit",
    "BaselineParameters",
    "InsufficientBaselineDataError",
    "NoStableSegmentError",
    "NormalRange",
    "SeriesBaseline",
    "fit_baselines",
    "fit_series_baseline",
]
