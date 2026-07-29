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

"""KDE fitting, peak filtering, valley partitioning, and quantiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import peak_prominences
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class KDEResult:
    """A fitted one-dimensional Gaussian KDE and diagnostic metadata."""

    grid: np.ndarray
    density: np.ndarray
    bandwidth: float
    data_range: tuple[float, float]
    adjusted_values: np.ndarray
    fitted: bool
    zero_range_jittered: bool
    jitter_scale: float


def _kde_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = config or {}
    nested = raw.get("kde")
    return nested if isinstance(nested, Mapping) else raw


def adaptive_kde(
    values: Sequence[float] | np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> KDEResult:
    """Fit the full-history or stable-segment KDE deterministically."""

    data = np.asarray(values, dtype=float).copy()
    if data.ndim != 1 or data.size < 2:
        raise ValueError("KDE requires at least two one-dimensional values")
    if not np.all(np.isfinite(data)):
        raise ValueError("KDE values must all be finite")

    cfg = _kde_config(config)
    if str(cfg.get("kernel", "gaussian")).lower() != "gaussian":
        raise ValueError("Only the gaussian KDE kernel is supported")
    epsilon = float(cfg.get("zero_range_epsilon", 1e-8))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("zero_range_epsilon must be finite and positive")
    seed = cfg.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("random_seed must be an integer")

    original_min = float(np.min(data))
    original_max = float(np.max(data))
    original_range = original_max - original_min
    adjusted = data.copy()
    jittered = original_range == 0
    jitter_scale = 0.0
    if jittered:
        jitter_scale = max(abs(float(data[0])), 1.0) * epsilon
        rng = np.random.default_rng(int(seed))
        adjusted = adjusted + rng.normal(
            loc=0.0, scale=jitter_scale, size=adjusted.size
        )

    bandwidth_setting = cfg.get("bandwidth", "auto")
    if bandwidth_setting in {None, "auto"}:
        bw_method: str | float | None = None
    else:
        bw_method = float(bandwidth_setting)
        if not np.isfinite(bw_method) or bw_method <= 0:
            raise ValueError("KDE bandwidth must be finite and positive")

    model = gaussian_kde(adjusted, bw_method=bw_method)
    bandwidth = float(np.sqrt(model.covariance[0, 0]))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE produced a non-finite or non-positive bandwidth")

    adjusted_min = float(np.min(adjusted))
    adjusted_max = float(np.max(adjusted))
    adjusted_range = adjusted_max - adjusted_min
    padding_ratio = float(cfg.get("padding_ratio", 0.10))
    if padding_ratio < 0 or not np.isfinite(padding_ratio):
        raise ValueError("KDE padding_ratio must be finite and non-negative")
    tail_bandwidths = float(cfg.get("tail_bandwidths", 6.0))
    if tail_bandwidths <= 0 or not np.isfinite(tail_bandwidths):
        raise ValueError("KDE tail_bandwidths must be finite and positive")
    # Data-range-only padding truncates material Gaussian mass whenever Scott's
    # bandwidth is large. Six effective bandwidths makes the numerical CDF
    # accurate enough for the configured alpha=0.01 tail quantiles.
    padding = max(
        adjusted_range * padding_ratio,
        tail_bandwidths * bandwidth,
    )
    grid_size = int(cfg.get("grid_size", 1024))
    if grid_size < 32:
        raise ValueError("KDE grid_size must be at least 32")
    grid = np.linspace(adjusted_min - padding, adjusted_max + padding, grid_size)
    density = np.asarray(model(grid), dtype=float)
    if density.shape != grid.shape or not np.all(np.isfinite(density)):
        raise ValueError("KDE produced an invalid density curve")
    density = np.maximum(density, 0.0)
    return KDEResult(
        grid=grid,
        density=density,
        bandwidth=bandwidth,
        data_range=(original_min, original_max),
        adjusted_values=adjusted,
        fitted=True,
        zero_range_jittered=jittered,
        jitter_scale=jitter_scale,
    )


def filter_peaks(
    peaks: Sequence[int] | np.ndarray,
    density: Sequence[float] | np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Conservatively filter density peaks; valleys never call this function."""

    peak_array = np.sort(np.unique(np.asarray(peaks, dtype=int)))
    density_array = np.asarray(density, dtype=float)
    if density_array.ndim != 1 or density_array.size == 0:
        raise ValueError("density must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(density_array)):
        raise ValueError("density must contain only finite values")
    if peak_array.size == 0:
        return peak_array
    if np.any(peak_array < 0) or np.any(peak_array >= density_array.size):
        raise IndexError("peak index is outside density")
    cfg = _kde_config(config)
    ratio = float(cfg.get("peak_prominence_ratio", 0.01))
    if ratio < 0 or not np.isfinite(ratio):
        raise ValueError("peak_prominence_ratio must be finite and non-negative")
    minimum_prominence = max(float(np.max(density_array)), 0.0) * ratio
    prominences = peak_prominences(density_array, peak_array)[0]
    filtered = peak_array[prominences >= minimum_prominence]
    if filtered.size == 0:
        return np.asarray([peak_array[int(np.argmax(density_array[peak_array]))]])
    return filtered


def find_peak_influence_regions(
    grid: Sequence[float] | np.ndarray,
    peaks: Sequence[int] | np.ndarray,
    neg_peaks: Sequence[int] | np.ndarray,
) -> list[dict[str, float | int]]:
    """Partition the KDE value axis with unfiltered negative-density peaks."""

    grid_array = np.asarray(grid, dtype=float)
    peak_array = np.sort(np.unique(np.asarray(peaks, dtype=int)))
    valleys = np.sort(np.unique(np.asarray(neg_peaks, dtype=int)))
    if grid_array.ndim != 1 or grid_array.size == 0:
        raise ValueError("KDE grid must be a non-empty one-dimensional array")
    if np.any(peak_array < 0) or np.any(peak_array >= grid_array.size):
        raise IndexError("peak index is outside KDE grid")
    if np.any(valleys < 0) or np.any(valleys >= grid_array.size):
        raise IndexError("valley index is outside KDE grid")

    if peak_array.size == 0:
        return []

    # A single retained mode is both the first and last mode. Give it both
    # outer boundaries instead of truncating it with valleys belonging to
    # positive peaks removed by filtering.
    if peak_array.size == 1:
        peak = int(peak_array[0])
        return [
            {
                "peak_index": peak,
                "peak_value": float(grid_array[peak]),
                "left_index": 0,
                "right_index": int(grid_array.size - 1),
                "left_value": float(grid_array[0]),
                "right_value": float(grid_array[-1]),
            }
        ]

    regions: list[dict[str, float | int]] = []
    for peak in peak_array:
        left_candidates = valleys[valleys < peak]
        right_candidates = valleys[valleys > peak]
        left_index = int(left_candidates[-1]) if left_candidates.size else 0
        right_index = (
            int(right_candidates[0])
            if right_candidates.size
            else grid_array.size - 1
        )
        regions.append(
            {
                "peak_index": int(peak),
                "peak_value": float(grid_array[peak]),
                "left_index": left_index,
                "right_index": right_index,
                "left_value": float(grid_array[left_index]),
                "right_value": float(grid_array[right_index]),
            }
        )
    return regions


def kde_cdf(result: KDEResult) -> np.ndarray:
    """Numerically integrate and normalize a KDE density curve."""

    cdf = cumulative_trapezoid(result.density, result.grid, initial=0.0)
    total = float(cdf[-1])
    if total <= 0 or not np.isfinite(total):
        raise ValueError("KDE density has no finite positive mass")
    normalized = np.maximum.accumulate(cdf / total)
    normalized[0] = 0.0
    normalized[-1] = 1.0
    return normalized


def kde_quantile(result: KDEResult, quantile: float) -> float:
    """Return a KDE-CDF quantile rather than treating density as probability."""

    q = float(quantile)
    if not 0 <= q <= 1:
        raise ValueError("quantile must be between 0 and 1")
    return float(np.interp(q, kde_cdf(result), result.grid))
