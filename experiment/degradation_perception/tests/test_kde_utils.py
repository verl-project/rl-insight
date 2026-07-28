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

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from experiment.degradation_perception.kde_utils import (
    KDEResult,
    adaptive_kde,
    filter_peaks,
    find_peak_influence_regions,
    kde_cdf,
    kde_quantile,
)


def test_adaptive_kde_fits_a_finite_single_mode():
    result = adaptive_kde([0.9, 1.0, 1.0, 1.1, 1.05])
    assert result.fitted is True
    assert result.grid.shape == result.density.shape
    assert np.all(np.isfinite(result.density))
    assert len(find_peaks(result.density)[0]) >= 1


def test_adaptive_kde_retains_two_clearly_separated_modes():
    low = np.linspace(-5.1, -4.9, 40)
    high = np.linspace(4.9, 5.1, 40)
    result = adaptive_kde(np.concatenate([low, high]))
    peaks = find_peaks(result.density)[0]
    assert len(peaks) >= 2
    peak_values = result.grid[peaks]
    assert np.any(peak_values < 0)
    assert np.any(peak_values > 0)


def test_zero_range_jitter_is_deterministic_and_does_not_mutate_input():
    original = np.asarray([3.0] * 12)
    snapshot = original.copy()
    first = adaptive_kde(original)
    second = adaptive_kde(original)
    assert np.array_equal(original, snapshot)
    assert first.zero_range_jittered is True
    assert first.jitter_scale == pytest.approx(3.0e-8)
    assert np.array_equal(first.adjusted_values, second.adjusted_values)
    assert not np.array_equal(first.adjusted_values, original)
    assert first.data_range == (3.0, 3.0)


def test_zero_range_jitter_does_not_modify_numpy_global_rng_state():
    np.random.seed(2026)
    expected = np.random.random(5)
    np.random.seed(2026)
    adaptive_kde([1.0] * 8)
    actual = np.random.random(5)
    assert np.array_equal(actual, expected)


def test_kde_grid_covers_gaussian_tails_by_absolute_bandwidth():
    result = adaptive_kde([1.0, 1.01, 1.02])
    assert 1.0 - result.grid[0] >= 5.9 * result.bandwidth
    assert result.grid[-1] - 1.02 >= 5.9 * result.bandwidth


def test_kde_cdf_is_monotonic_and_quantiles_use_probability_mass():
    result = adaptive_kde(np.linspace(0.9, 1.1, 50))
    cdf = kde_cdf(result)
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= 0)
    lower = kde_quantile(result, 0.01)
    upper = kde_quantile(result, 1 - 0.01)
    assert lower < upper


def test_kde_quantile_uses_one_minus_alpha_not_density_height():
    grid = np.linspace(0.0, 1.0, 1001)
    result = KDEResult(
        grid=grid,
        density=np.ones_like(grid),
        bandwidth=0.1,
        data_range=(0.0, 1.0),
        adjusted_values=np.asarray([0.25, 0.75]),
        fitted=True,
        zero_range_jittered=False,
        jitter_scale=0.0,
    )
    assert kde_quantile(result, 0.99) == pytest.approx(0.99, abs=1e-3)


def test_filter_peaks_removes_only_low_prominence_positive_peaks():
    density = np.asarray([0.0, 10.0, 0.0, 0.01, 0.0])
    filtered = filter_peaks(
        np.asarray([1, 3]),
        density,
        {"peak_prominence_ratio": 0.1},
    )
    assert filtered.tolist() == [1]


def test_valleys_define_two_peak_influence_regions():
    grid = np.arange(7, dtype=float)
    regions = find_peak_influence_regions(grid, [1, 5], [3])
    assert [(item["left_index"], item["right_index"]) for item in regions] == [
        (0, 3),
        (3, 6),
    ]


def test_single_peak_uses_both_outer_grid_boundaries():
    regions = find_peak_influence_regions(np.arange(7), [3], [1, 5])
    assert len(regions) == 1
    assert regions[0]["left_index"] == 0
    assert regions[0]["right_index"] == 6


@pytest.mark.parametrize("quantile", [-0.1, 1.1, float("nan")])
def test_kde_quantile_rejects_invalid_probability(quantile):
    result = adaptive_kde([1.0, 1.1, 1.2])
    with pytest.raises(ValueError, match="quantile"):
        kde_quantile(result, quantile)
