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
from datetime import datetime, timezone

import numpy as np
import pytest

from experiment.degradation_perception.preprocessing import (
    DataValidationError,
    extract_metric_series,
    load_dataset,
    parse_dataset_text,
    preprocess_time_series,
    prometheus_query_range_to_series,
    validate_canonical_dataset,
)


METRIC = "timing_s/step"


def canonical_payload() -> dict:
    return {
        "standard": {
            METRIC: {"timestamps": [1, 2, 3], "values": [1.0, 1.1, 1.0]}
        },
        "inference": {
            METRIC: {"timestamps": [9, 10, 11, 12], "values": [0.9, 1.0, 1.1, 1.2]}
        },
    }


def prometheus_payload(samples, *, result_type="matrix") -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": result_type,
            "result": [
                {
                    "metric": {"__name__": "rl_insight_monitor_timing_s_step"},
                    "values": samples,
                }
            ],
        },
    }


@pytest.mark.parametrize(
    ("timestamps", "values"),
    [
        (None, [1.0]),
        ([1], None),
        (None, None),
    ],
)
def test_preprocess_rejects_none_inputs(timestamps, values):
    with pytest.raises(DataValidationError, match="must not be None"):
        preprocess_time_series(timestamps, values)


def test_preprocess_accepts_two_empty_arrays():
    series = preprocess_time_series([], [])
    assert series.timestamps == []
    assert series.values == []


@pytest.mark.parametrize(
    ("timestamps", "values"),
    [([], [1.0]), ([1], []), ([1, 2], [1.0])],
)
def test_preprocess_rejects_unequal_lengths_before_pair_iteration(
    timestamps, values
):
    with pytest.raises(DataValidationError, match="equal lengths"):
        preprocess_time_series(timestamps, values)


def test_preprocess_filters_invalid_pairs_together():
    series = preprocess_time_series(
        [1, 2, 3, 4, float("inf"), "bad", True],
        [10, float("nan"), float("inf"), 40, 50, 60, 70],
    )
    assert series.timestamps == [1.0, 4.0]
    assert series.values == [10.0, 40.0]


def test_preprocess_stable_sorts_pairs_and_keeps_last_duplicate_value():
    series = preprocess_time_series([3, 1, 2, 1], [30, 10, 20, 11])
    assert series.timestamps == [1.0, 2.0, 3.0]
    assert series.values == [11.0, 20.0, 30.0]


def test_preprocess_supports_numpy_arrays_and_scalars():
    series = preprocess_time_series(
        np.asarray([np.int64(2), np.int64(1)]),
        np.asarray([np.float64(2.5), np.float32(1.5)]),
    )
    assert series.timestamps == [1.0, 2.0]
    assert series.values == pytest.approx([1.5, 2.5])


def test_preprocess_supports_zero_dimensional_numpy_arrays_as_one_point():
    series = preprocess_time_series(np.asarray(1), np.asarray(2.5))
    assert series.timestamps == [1.0]
    assert series.values == [2.5]


def test_preprocess_normalizes_datetime_to_utc_epoch_seconds():
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    naive_same_instant = datetime(2026, 1, 1)
    series = preprocess_time_series([instant, naive_same_instant], [1.0, 2.0])
    assert series.timestamps == [instant.timestamp()]
    assert series.values == [2.0]


def test_preprocess_drops_nat_and_boolean_values():
    series = preprocess_time_series(
        [np.datetime64("NaT"), 2, 3],
        [1.0, True, 3.0],
    )
    assert series.timestamps == [3.0]
    assert series.values == [3.0]


def test_validate_canonical_dataset_preserves_metric_name_with_slash():
    validated = validate_canonical_dataset(canonical_payload())
    assert METRIC in validated["standard"]
    assert validated["standard"][METRIC]["values"] == [1.0, 1.1, 1.0]


def test_prometheus_matrix_converts_strings_and_filters_nonfinite_pairs():
    payload = prometheus_payload(
        [[3, "1.0"], [1, "NaN"], [2, "+Inf"], [4, "1.2"]]
    )
    converted = prometheus_query_range_to_series(payload)
    assert converted == {
        "timestamps": [3, 1, 2, 4],
        "values": ["1.0", "NaN", "+Inf", "1.2"],
    }
    dataset = validate_canonical_dataset(
        {
            "standard": {METRIC: payload},
            "inference": {METRIC: prometheus_payload([])},
        }
    )
    series = extract_metric_series(dataset, "standard", METRIC)
    assert series.timestamps == [3.0, 4.0]
    assert series.values == [1.0, 1.2]


def test_prometheus_matrix_accepts_empty_result():
    payload = {
        "status": "success",
        "data": {"resultType": "matrix", "result": []},
    }
    assert prometheus_query_range_to_series(payload) == {
        "timestamps": [],
        "values": [],
    }


def test_prometheus_matrix_rejects_error_multiple_series_and_histograms():
    with pytest.raises(DataValidationError, match="bad_data: invalid query"):
        prometheus_query_range_to_series(
            {
                "status": "error",
                "errorType": "bad_data",
                "error": "invalid query",
            }
        )

    multiple = prometheus_payload([[1, "1"]])
    multiple["data"]["result"].append(
        {"metric": {"worker": "trainer_1"}, "values": [[1, "2"]]}
    )
    with pytest.raises(DataValidationError, match="at most one"):
        prometheus_query_range_to_series(multiple)

    histogram = prometheus_payload([])
    histogram["data"]["result"][0]["histograms"] = [
        [1, {"count": "1", "sum": "2", "buckets": []}]
    ]
    with pytest.raises(DataValidationError, match="native histogram"):
        prometheus_query_range_to_series(histogram)


def test_prometheus_adapter_requires_range_matrix_and_explicit_phases():
    with pytest.raises(DataValidationError, match="resultType"):
        prometheus_query_range_to_series(
            prometheus_payload([[1, "1"]], result_type="vector")
        )
    with pytest.raises(DataValidationError, match="missing phases"):
        validate_canonical_dataset(prometheus_payload([[1, "1"]]))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"standard": {}},
        {"inference": {}},
        {"standard": {}, "inference": {}, "unknown": {}},
        {"standard": [], "inference": {}},
        {"standard": {METRIC: []}, "inference": {}},
        {
            "standard": {METRIC: {"timestamps": [1]}},
            "inference": {},
        },
        {
            "standard": {
                METRIC: {"timestamps": [1], "values": [1], "extra": []}
            },
            "inference": {},
        },
    ],
)
def test_validate_canonical_dataset_rejects_ambiguous_shapes(payload):
    with pytest.raises(DataValidationError):
        validate_canonical_dataset(payload)


def test_missing_metric_is_an_empty_series_not_a_schema_error():
    dataset = {"standard": {}, "inference": {}}
    assert extract_metric_series(dataset, "standard", "missing").timestamps == []


def test_extract_metric_series_rejects_mismatch_without_zip_truncation():
    dataset = canonical_payload()
    dataset["inference"][METRIC] = {"timestamps": [1, 2], "values": [1.0]}
    with pytest.raises(DataValidationError, match="equal lengths"):
        extract_metric_series(dataset, "inference", METRIC)


def test_local_loader_applies_closed_window_to_inference_only(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(canonical_payload()), encoding="utf-8")
    dataset = load_dataset(path, [METRIC], start_time=10, end_time=11)
    assert dataset["standard"][METRIC]["timestamps"] == [1, 2, 3]
    assert dataset["inference"][METRIC]["timestamps"] == [10, 11]
    assert dataset["inference"][METRIC]["values"] == [1.0, 1.1]


def test_local_loader_requires_a_json_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json", [METRIC])
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path, [METRIC])

    jsonl = tmp_path / "input.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="canonical"):
        load_dataset(jsonl, [METRIC])


def test_local_loader_rejects_non_utf8_json(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(DataValidationError, match="UTF-8"):
        load_dataset(path, [METRIC])


def test_parse_canonical_json_rejects_bad_json_and_invalid_bounds():
    with pytest.raises(DataValidationError, match="invalid canonical JSON"):
        parse_dataset_text("{", [METRIC], suffix=".json")
    with pytest.raises(DataValidationError, match="must not exceed"):
        parse_dataset_text(
            json.dumps(canonical_payload()),
            [METRIC],
            suffix=".json",
            start_time=5,
            end_time=4,
        )


def test_remote_jsonl_requires_explicit_phase_and_metrics_object():
    no_phase = json.dumps({"timestamp": 1, "metrics": {METRIC: 1.0}})
    with pytest.raises(DataValidationError, match="must set phase"):
        parse_dataset_text(no_phase, [METRIC], suffix=".jsonl")

    no_metrics = json.dumps({"phase": "standard", "timestamp": 1})
    with pytest.raises(DataValidationError, match="metrics must be an object"):
        parse_dataset_text(no_metrics, [METRIC], suffix=".jsonl")


def test_remote_jsonl_does_not_infer_or_window_standard_phase():
    lines = "\n".join(
        [
            json.dumps(
                {"phase": "standard", "timestamp": 100, "metrics": {METRIC: 1.0}}
            ),
            json.dumps(
                {"phase": "inference", "timestamp": 9, "metrics": {METRIC: 0.9}}
            ),
            json.dumps(
                {"phase": "inference", "timestamp": 10, "metrics": {METRIC: 1.0}}
            ),
            json.dumps(
                {"phase": "inference", "timestamp": 11, "metrics": {METRIC: 1.1}}
            ),
            json.dumps(
                {"phase": "inference", "timestamp": 12, "metrics": {METRIC: 1.2}}
            ),
        ]
    )
    dataset = parse_dataset_text(
        lines,
        [METRIC],
        suffix=".jsonl",
        start_time=10,
        end_time=11,
    )
    assert dataset["standard"][METRIC]["timestamps"] == [100]
    assert dataset["inference"][METRIC]["timestamps"] == [10, 11]


def test_remote_jsonl_reports_the_malformed_line_number():
    with pytest.raises(DataValidationError, match="line 2"):
        parse_dataset_text(
            '{"phase":"standard","timestamp":1,"metrics":{}}\n{',
            [METRIC],
            suffix=".jsonl",
        )


def test_parse_dataset_rejects_unapproved_format():
    with pytest.raises(DataValidationError, match="unsupported input format"):
        parse_dataset_text("", [METRIC], suffix=".csv")
