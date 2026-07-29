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

import json
import math

import numpy as np
import pytest

from experiment.degradation_perception.prometheus_matrix_adapter import (
    convert_simulation_package,
    validate_simulation_package,
)
from experiment.degradation_perception.simulated_prometheus import (
    ASSOCIATED_METRICS,
    DEFAULT_SEED,
    METRICS,
    TARGET_METRIC,
    generate_simulation_package,
    save_simulation_package,
)


def _converted(package=None):
    generated = generate_simulation_package() if package is None else package
    return convert_simulation_package(generated)[0]


def test_fixed_seed_is_exactly_repeatable_and_changes_with_seed():
    first = generate_simulation_package(seed=DEFAULT_SEED)
    second = generate_simulation_package(seed=DEFAULT_SEED)
    changed = generate_simulation_package(seed=DEFAULT_SEED + 1)

    assert first == second
    first_values = first["inference"][TARGET_METRIC]["response"]["data"][
        "result"
    ][0]["values"]
    changed_values = changed["inference"][TARGET_METRIC]["response"]["data"][
        "result"
    ][0]["values"]
    assert first_values != changed_values


def test_generated_package_validates_and_converts_without_mocking_adapter():
    package = generate_simulation_package()
    normalized = validate_simulation_package(package)
    dataset, diagnostics = convert_simulation_package(package)

    assert normalized == package
    assert set(dataset) == {"standard", "inference"}
    assert tuple(dataset["standard"]) == METRICS
    assert tuple(dataset["inference"]) == METRICS
    assert all(
        set(series) == {"timestamps", "values"}
        for phase in dataset.values()
        for series in phase.values()
    )
    assert set(diagnostics["phases"]) == {"standard", "inference"}
    assert all(
        isinstance(value, float)
        for phase in dataset.values()
        for series in phase.values()
        for value in series["values"]
    )
    json.dumps(
        {"package": package, "dataset": dataset, "diagnostics": diagnostics},
        allow_nan=False,
    )


def test_dense_series_counts_intervals_and_prometheus_sample_types():
    package = generate_simulation_package()
    for phase, minimum_points in (("standard", 120), ("inference", 180)):
        for metric in METRICS:
            response = package[phase][metric]["response"]
            series = response["data"]["result"][0]
            samples = series["values"]
            assert response["status"] == "success"
            assert response["data"]["resultType"] == "matrix"
            assert all(
                isinstance(timestamp, (int, float))
                and not isinstance(timestamp, bool)
                and math.isfinite(float(timestamp))
                and 1.0e9 < float(timestamp) < 1.0e10
                and isinstance(value, str)
                and math.isfinite(float(value))
                for timestamp, value in samples
            )
            if metric == "sparse_metric":
                assert 5 <= len(samples) < minimum_points * 0.2
            else:
                assert len(samples) >= minimum_points
                assert all(
                    samples[index][0] - samples[index - 1][0]
                    == pytest.approx(10.0)
                    for index in range(1, len(samples))
                )
    json.dumps(package, allow_nan=False)


def test_target_has_normal_anomaly_and_recovery_regions():
    values = np.asarray(
        _converted()["inference"][TARGET_METRIC]["values"],
        dtype=float,
    )
    before = values[:70]
    abnormal = values[70:120]
    recovery = values[120:]

    assert len(before) == 70
    assert len(abnormal) == 50
    assert len(recovery) == 60
    assert float(np.mean(abnormal)) > float(np.mean(before)) + 0.6
    assert float(np.min(abnormal)) > max(float(np.max(before)), float(np.max(recovery)))
    assert float(np.mean(recovery)) == pytest.approx(
        float(np.mean(before)),
        abs=0.01,
    )


@pytest.mark.parametrize(
    ("metric", "abnormal_slice", "minimum_correlation", "expected_coverage"),
    [
        ("kv_cache_usage_perc", slice(70, 120), 0.99, 1.00),
        ("response_length_mean", slice(71, 120), 0.97, 0.98),
        ("num_requests_swapped", slice(73, 118), 0.90, 0.90),
        ("e2e_request_latency", slice(76, 115), 0.80, 0.78),
        ("global_seqlen_minimax_diff", slice(80, 112), 0.70, 0.64),
    ],
)
def test_fine_grained_candidates_have_ordered_signal_strength(
    metric,
    abnormal_slice,
    minimum_correlation,
    expected_coverage,
):
    dataset = _converted()
    target = np.asarray(dataset["inference"][TARGET_METRIC]["values"])
    candidate = np.asarray(dataset["inference"][metric]["values"])
    abnormal = candidate[abnormal_slice]
    threshold = (float(np.mean(candidate[:70])) + float(np.mean(abnormal))) / 2

    assert metric in ASSOCIATED_METRICS
    assert float(np.corrcoef(target, candidate)[0, 1]) > minimum_correlation
    assert float(np.mean(candidate[70:120] > threshold)) == pytest.approx(
        expected_coverage,
        abs=0.01,
    )


def test_swapped_request_metric_remains_integer_valued():
    values = _converted()["inference"]["num_requests_swapped"]["values"]

    assert all(float(value).is_integer() and value >= 0 for value in values)


def test_unrelated_metric_remains_normal():
    dataset = _converted()
    target = np.asarray(dataset["inference"][TARGET_METRIC]["values"])
    unrelated = np.asarray(dataset["inference"]["unrelated_metric"]["values"])
    mean_shift = abs(
        float(np.mean(unrelated[70:120]) - np.mean(unrelated[:70]))
    )

    assert abs(float(np.corrcoef(target, unrelated)[0, 1])) < 0.2
    assert mean_shift < float(np.std(unrelated)) * 0.5


def test_constant_and_sparse_metrics_have_expected_shapes():
    dataset = _converted()
    constant_values: set[float] = set()
    for phase in ("standard", "inference"):
        constant = dataset[phase]["constant_metric"]["values"]
        assert max(constant) - min(constant) == 0.0
        constant_values.update(constant)

        sparse = dataset[phase]["sparse_metric"]
        target = dataset[phase][TARGET_METRIC]
        assert 0 < len(sparse["timestamps"]) < len(target["timestamps"]) * 0.2
        assert set(sparse["timestamps"]) < set(target["timestamps"])
    assert constant_values == {5.0}


def test_save_simulation_package_round_trips_unicode_and_spaces(tmp_path):
    package = generate_simulation_package()
    path = tmp_path / "含 空格" / "simulated prometheus matrix.json"

    saved = save_simulation_package(package, path)
    loaded = json.loads(saved.read_text(encoding="utf-8"))

    assert saved == path
    assert loaded == package
    assert validate_simulation_package(loaded) == package
    json.dumps(loaded, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"standard_points": 119},
        {"inference_points": 179},
        {"query_step_seconds": 0},
        {"base_timestamp": float("nan")},
    ],
)
def test_generator_rejects_invalid_contract_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_simulation_package(**kwargs)
