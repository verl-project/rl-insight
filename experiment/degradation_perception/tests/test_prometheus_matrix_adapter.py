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

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiment.degradation_perception.prometheus_matrix_adapter import (
    PrometheusMatrixError,
    convert_matrix_response,
    convert_simulation_package,
    load_simulation_package,
    validate_simulation_package,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prometheus"
LOGICAL_METRIC = "timing_s/step"
QUERY = (
    'rl_insight_monitor_timing_s_step{experiment_name="mock_inference"}'
)
EXPECTED_LABELS = {
    "__name__": "rl_insight_monitor_timing_s_step",
    "project": "verl",
    "experiment_name": "mock_inference",
    "job": "trainer_metrics",
    "instance": "127.0.0.1:9092",
    "worker": "trainer_0",
}


def fixture_response(name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def convert_response(
    response: Any,
    *,
    phase: str = "inference",
    query: str = QUERY,
    series_policy: str = "exactly_one",
    select_labels: dict[str, str] | None = None,
    query_window: dict[str, Any] | None = None,
):
    return convert_matrix_response(
        response,
        LOGICAL_METRIC,
        phase,
        query,
        series_policy,
        select_labels,
        query_window,
    )


def simulation_package(
    standard_response: dict[str, Any] | None = None,
    inference_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    standard = (
        fixture_response("valid_single_series.json")
        if standard_response is None
        else standard_response
    )
    inference = (
        fixture_response("valid_single_series.json")
        if inference_response is None
        else inference_response
    )
    return {
        "formatVersion": 1,
        "source": "simulated_prometheus_query_range",
        "queryStepSeconds": 10,
        "standard": {
            LOGICAL_METRIC: {
                "query": QUERY.replace("mock_inference", "mock_standard"),
                "response": standard,
            }
        },
        "inference": {
            LOGICAL_METRIC: {
                "query": QUERY,
                "response": inference,
            }
        },
    }


def error_from(response: Any, **kwargs) -> PrometheusMatrixError:
    with pytest.raises(PrometheusMatrixError) as caught:
        convert_response(response, **kwargs)
    return caught.value


def test_valid_single_series_converts_strings_and_preserves_unix_seconds():
    series, diagnostics = convert_response(
        fixture_response("valid_single_series.json")
    )

    assert series == {
        "timestamps": [1785301200.0, 1785301210.0],
        "values": [1.002, 0.998],
    }
    assert diagnostics["selectedLabels"] == EXPECTED_LABELS
    assert diagnostics["returnedSeriesLabels"] == [EXPECTED_LABELS]
    assert diagnostics["filteredNonFiniteCount"] == 0
    assert diagnostics["duplicateTimestampCount"] == 0
    assert diagnostics["nativeHistogramSamplesPresent"] is False


def test_validator_returns_a_normalized_deep_copy_with_default_policy():
    payload = simulation_package()
    original = copy.deepcopy(payload)

    normalized = validate_simulation_package(payload)

    assert normalized == {
        **original,
        "standard": {
            LOGICAL_METRIC: {
                **original["standard"][LOGICAL_METRIC],
                "seriesPolicy": "exactly_one",
            }
        },
        "inference": {
            LOGICAL_METRIC: {
                **original["inference"][LOGICAL_METRIC],
                "seriesPolicy": "exactly_one",
            }
        },
    }
    assert normalized is not payload
    assert normalized["standard"] is not payload["standard"]
    assert payload == original


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {
            "formatVersion": 2,
            "source": "simulated_prometheus_query_range",
            "queryStepSeconds": 10,
            "standard": {},
            "inference": {},
        },
        {
            "formatVersion": 1,
            "source": "not_a_simulation_package",
            "queryStepSeconds": 10,
            "standard": {},
            "inference": {},
        },
        {
            "formatVersion": 1,
            "source": "simulated_prometheus_query_range",
            "queryStepSeconds": 0,
            "standard": {},
            "inference": {},
        },
    ],
)
def test_validator_rejects_invalid_package_roots(payload):
    with pytest.raises(PrometheusMatrixError):
        validate_simulation_package(payload)


def test_validator_rejects_unknown_metric_entry_fields():
    payload = simulation_package()
    payload["inference"][LOGICAL_METRIC]["responseAlias"] = payload[
        "inference"
    ][LOGICAL_METRIC].pop("response")

    with pytest.raises(PrometheusMatrixError):
        validate_simulation_package(payload)


@pytest.mark.parametrize("missing_phase", ["standard", "inference"])
def test_validator_requires_both_explicit_phases(missing_phase):
    payload = simulation_package()
    payload.pop(missing_phase)

    with pytest.raises(PrometheusMatrixError) as caught:
        validate_simulation_package(payload)

    assert caught.value.code == "invalid_simulation_package"


def test_validator_rejects_unknown_root_field_and_non_mapping_phase():
    unknown = simulation_package()
    unknown["diagnostics"] = {}
    with pytest.raises(PrometheusMatrixError) as caught:
        validate_simulation_package(unknown)
    assert caught.value.code == "invalid_simulation_package"

    invalid_phase = simulation_package()
    invalid_phase["standard"] = []
    with pytest.raises(PrometheusMatrixError) as caught:
        validate_simulation_package(invalid_phase)
    assert caught.value.code == "invalid_simulation_package"


@pytest.mark.parametrize("missing_field", ["query", "response"])
def test_validator_requires_query_and_response_per_metric(missing_field):
    payload = simulation_package()
    payload["inference"][LOGICAL_METRIC].pop(missing_field)

    with pytest.raises(PrometheusMatrixError) as caught:
        validate_simulation_package(payload)

    assert caught.value.code == "invalid_simulation_package"


@pytest.mark.parametrize("invalid_step", [True, "10", float("nan"), float("inf")])
def test_validator_rejects_non_numeric_or_nonfinite_query_step(invalid_step):
    payload = simulation_package()
    payload["queryStepSeconds"] = invalid_step

    with pytest.raises(PrometheusMatrixError) as caught:
        validate_simulation_package(payload)

    assert caught.value.code == "invalid_simulation_package"


def test_package_conversion_keeps_algorithm_input_root_strict():
    payload = simulation_package(
        inference_response=fixture_response("duplicate_timestamps.json")
    )
    original = copy.deepcopy(payload)

    dataset, diagnostics = convert_simulation_package(payload)

    assert dataset == {
        "standard": {
            LOGICAL_METRIC: {
                "timestamps": [1785301200.0, 1785301210.0],
                "values": [1.002, 0.998],
            }
        },
        "inference": {
            LOGICAL_METRIC: {
                "timestamps": [1785301200.0, 1785301210.0],
                "values": [1.004, 0.998],
            }
        },
    }
    assert set(dataset) == {"standard", "inference"}
    assert set(dataset["standard"][LOGICAL_METRIC]) == {
        "timestamps",
        "values",
    }
    assert (
        diagnostics["phases"]["inference"][LOGICAL_METRIC][
            "duplicateTimestampCount"
        ]
        == 1
    )
    assert payload == original


def test_zero_series_has_contextual_error_code_and_json_details():
    query_window = {"start": 1785301200, "end": 1785302390}
    error = error_from(
        fixture_response("zero_series.json"),
        phase="standard",
        query="standard_query",
        query_window=query_window,
    )

    assert error.code == "no_series"
    assert error.message
    assert error.details["logicalMetric"] == LOGICAL_METRIC
    assert error.details["phase"] == "standard"
    assert error.details["query"] == "standard_query"
    assert error.details["queryWindow"] == query_window
    assert error.details["returnedSeriesCount"] == 0
    assert error.details["returnedSeriesLabels"] == []
    json.dumps(error.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "fixture_name",
    ["multiple_instances.json", "multiple_workers.json"],
)
def test_multiple_series_never_silently_selects_result_zero(fixture_name):
    response = fixture_response(fixture_name)

    error = error_from(response)

    assert error.code == "multiple_series"
    assert error.details["logicalMetric"] == LOGICAL_METRIC
    assert error.details["phase"] == "inference"
    assert error.details["query"] == QUERY
    assert error.details["returnedSeriesCount"] == len(
        response["data"]["result"]
    )
    assert error.details["returnedSeriesLabels"] == [
        item["metric"] for item in response["data"]["result"]
    ]
    assert error.details["seriesLabels"] == [
        item["metric"] for item in response["data"]["result"]
    ]


def test_label_selection_can_select_exactly_one_series():
    series, diagnostics = convert_response(
        fixture_response("select_unique.json"),
        series_policy="select_by_labels",
        select_labels={
            "instance": "127.0.0.1:9093",
            "worker": "trainer_1",
        },
    )

    assert series == {
        "timestamps": [1785301200.0, 1785301210.0],
        "values": [1.012, 1.008],
    }
    assert diagnostics["selectedLabels"]["instance"] == "127.0.0.1:9093"
    assert diagnostics["selectedLabels"]["worker"] == "trainer_1"


@pytest.mark.parametrize(
    ("label", "selected_value", "other_value"),
    [
        ("project", "project_a", "project_b"),
        ("experiment_name", "standard_run", "inference_run"),
        ("instance", "127.0.0.1:9092", "127.0.0.1:9093"),
        ("worker", "trainer_0", "trainer_1"),
    ],
)
def test_task_identity_labels_each_support_unique_selection(
    label,
    selected_value,
    other_value,
):
    response = fixture_response("select_unique.json")
    response["data"]["result"][0]["metric"][label] = selected_value
    response["data"]["result"][1]["metric"][label] = other_value

    _, diagnostics = convert_response(
        response,
        series_policy="select_by_labels",
        select_labels={label: selected_value},
    )

    assert diagnostics["selectedLabels"][label] == selected_value


def test_label_selection_that_remains_ambiguous_is_an_explicit_error():
    response = fixture_response("select_non_unique.json")
    select_labels = {
        "instance": "127.0.0.1:9092",
        "worker": "trainer_0",
    }

    error = error_from(
        response,
        series_policy="select_by_labels",
        select_labels=select_labels,
    )

    assert error.code == "multiple_matching_series"
    assert error.details["selectLabels"] == select_labels
    assert error.details["seriesLabels"] == [
        item["metric"] for item in response["data"]["result"]
    ]


def test_label_selection_with_no_match_is_explicit_and_never_falls_back():
    select_labels = {"worker": "missing_worker"}

    error = error_from(
        fixture_response("select_unique.json"),
        series_policy="select_by_labels",
        select_labels=select_labels,
    )

    assert error.code == "no_matching_series"
    assert error.details["selectLabels"] == select_labels
    assert error.details["matchingSeriesLabels"] == []
    assert error.details["returnedSeriesLabels"]


@pytest.mark.parametrize(
    ("series_policy", "select_labels"),
    [
        ("first", None),
        ([], None),
        ({}, None),
        (float("nan"), None),
        ("select_by_labels", None),
        ("select_by_labels", {}),
        ("select_by_labels", []),
        ("select_by_labels", {"worker": 0}),
        ("select_by_labels", {1: "trainer_0"}),
    ],
)
def test_invalid_series_policy_or_selector_is_rejected(
    series_policy,
    select_labels,
):
    error = error_from(
        fixture_response("select_unique.json"),
        series_policy=series_policy,
        select_labels=select_labels,
    )

    assert error.code == "invalid_series_policy"
    json.dumps(error.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("fixture_name", "error_code"),
    [
        ("status_error.json", "query_failed"),
        ("vector_result.json", "invalid_matrix_response"),
        ("missing_values.json", "invalid_series"),
        ("invalid_sample_length.json", "invalid_sample"),
        ("invalid_value_string.json", "invalid_sample"),
    ],
)
def test_invalid_responses_have_structured_errors(fixture_name, error_code):
    error = error_from(fixture_response(fixture_name))

    assert error.code == error_code
    assert error.details["logicalMetric"] == LOGICAL_METRIC
    assert error.details["phase"] == "inference"
    assert error.details["query"] == QUERY


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"status": "success", "data": []},
        {
            "status": "success",
            "data": {"resultType": "matrix", "result": {}},
        },
    ],
)
def test_invalid_matrix_response_shapes_are_not_partially_parsed(response):
    error = error_from(response)

    assert error.code == "invalid_matrix_response"


@pytest.mark.parametrize(
    "fixture_name",
    ["nan_value.json", "positive_inf_value.json"],
)
def test_nonfinite_values_are_filtered_with_diagnostics(fixture_name):
    series, diagnostics = convert_response(fixture_response(fixture_name))

    assert series == {
        "timestamps": [1785301210.0],
        "values": [0.998],
    }
    assert diagnostics["filteredNonFiniteCount"] == 1
    json.dumps(
        {"series": series, "diagnostics": diagnostics},
        allow_nan=False,
    )


def test_negative_infinity_is_filtered_too():
    response = fixture_response("positive_inf_value.json")
    response["data"]["result"][0]["values"][0][1] = "-Inf"

    series, diagnostics = convert_response(response)

    assert series["timestamps"] == [1785301210.0]
    assert series["values"] == [0.998]
    assert diagnostics["filteredNonFiniteCount"] == 1


@pytest.mark.parametrize("invalid_timestamp", ["not-a-timestamp", "1785301200"])
def test_non_numeric_timestamp_is_an_invalid_sample(invalid_timestamp):
    response = fixture_response("valid_single_series.json")
    response["data"]["result"][0]["values"][0][0] = invalid_timestamp

    error = error_from(response)

    assert error.code == "invalid_sample"


def test_boolean_timestamp_is_an_invalid_sample():
    response = fixture_response("valid_single_series.json")
    response["data"]["result"][0]["values"][0][0] = True

    assert error_from(response).code == "invalid_sample"


@pytest.mark.parametrize("invalid_value", [1.002, True])
def test_sample_value_must_use_prometheus_string_format(invalid_value):
    response = fixture_response("valid_single_series.json")
    response["data"]["result"][0]["values"][0][1] = invalid_value

    assert error_from(response).code == "invalid_sample"


@pytest.mark.parametrize(
    "invalid_labels",
    [
        None,
        [],
        {"worker": 0},
        {"worker": True},
        {"worker": None},
        {1: "trainer_0"},
    ],
)
def test_metric_labels_must_be_a_string_dictionary(invalid_labels):
    response = fixture_response("valid_single_series.json")
    response["data"]["result"][0]["metric"] = invalid_labels

    assert error_from(response).code == "invalid_series"


def test_metric_labels_are_required():
    response = fixture_response("valid_single_series.json")
    response["data"]["result"][0].pop("metric")

    assert error_from(response).code == "invalid_series"


def test_unordered_timestamps_are_sorted_without_unit_conversion():
    series, diagnostics = convert_response(
        fixture_response("unordered_timestamps.json")
    )

    assert series == {
        "timestamps": [
            1785301200.0,
            1785301210.0,
            1785301220.0,
        ],
        "values": [1.002, 0.998, 1.004],
    }
    assert diagnostics["duplicateTimestampCount"] == 0


def test_duplicate_timestamp_keeps_the_last_valid_value():
    series, diagnostics = convert_response(
        fixture_response("duplicate_timestamps.json")
    )

    assert series == {
        "timestamps": [1785301200.0, 1785301210.0],
        "values": [1.004, 0.998],
    }
    assert diagnostics["duplicateTimestampCount"] == 1


def test_filtered_duplicate_does_not_replace_the_last_valid_value():
    response = fixture_response("valid_single_series.json")
    response["data"]["result"][0]["values"] = [
        [1785301200.0, "1.002"],
        [1785301200.0, "NaN"],
    ]

    series, diagnostics = convert_response(response)

    assert series == {
        "timestamps": [1785301200.0],
        "values": [1.002],
    }
    assert diagnostics["filteredNonFiniteCount"] == 1


def test_empty_scalar_series_is_an_explicit_error():
    error = error_from(fixture_response("empty_values.json"))

    assert error.code == "empty_series"
    assert error.details["logicalMetric"] == LOGICAL_METRIC


def test_all_filtered_samples_are_an_explicit_empty_series_error():
    response = fixture_response("nan_value.json")
    response["data"]["result"][0]["values"] = [
        [1785301200.0, "NaN"],
        [1785301210.0, "+Inf"],
        [1785301220.0, "-Inf"],
    ]

    error = error_from(response)

    assert error.code == "empty_series"
    assert error.details["filteredNonFiniteCount"] == 3


def test_histogram_only_series_is_rejected_explicitly():
    error = error_from(fixture_response("native_histogram_only.json"))

    assert error.code == "unsupported_native_histogram"
    assert error.details["nativeHistogramSamplesPresent"] is True


def test_values_take_precedence_with_explicit_histogram_diagnostics():
    series, diagnostics = convert_response(
        fixture_response("values_and_histograms.json")
    )

    assert series == {
        "timestamps": [1785301200.0, 1785301210.0],
        "values": [1.002, 0.998],
    }
    assert diagnostics["nativeHistogramSamplesPresent"] is True
    assert diagnostics["selectedLabels"] == EXPECTED_LABELS
    json.dumps(diagnostics, allow_nan=False)


def test_converter_outputs_are_strict_json_serializable():
    dataset, diagnostics = convert_simulation_package(simulation_package())

    serialized = json.dumps(
        {"dataset": dataset, "diagnostics": diagnostics},
        allow_nan=False,
        sort_keys=True,
    )

    assert json.loads(serialized)["dataset"] == dataset


def test_load_simulation_package_reads_utf8_json_and_normalizes(tmp_path):
    payload = simulation_package()
    path = tmp_path / "simulated_prometheus_matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_simulation_package(path)

    assert loaded == validate_simulation_package(payload)
    assert (
        loaded["inference"][LOGICAL_METRIC]["seriesPolicy"]
        == "exactly_one"
    )


def test_load_simulation_package_rejects_invalid_json(tmp_path):
    path = tmp_path / "simulated_prometheus_matrix.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(PrometheusMatrixError):
        load_simulation_package(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_simulation_package_rejects_nonstandard_json_constants(
    tmp_path,
    constant,
):
    path = tmp_path / "simulated_prometheus_matrix.json"
    text = json.dumps(simulation_package()).replace(
        '"queryStepSeconds": 10',
        f'"queryStepSeconds": {constant}',
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(PrometheusMatrixError) as caught:
        load_simulation_package(path)

    assert caught.value.code == "invalid_simulation_package"
    json.dumps(caught.value.to_dict(), allow_nan=False)
