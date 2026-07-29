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
from pathlib import Path

import pytest
import yaml

from experiment.degradation_perception.algorithm import (
    DegradationPerception,
    _classify_value,
    build_standard_data,
    get_standard_data,
)
from experiment.degradation_perception.config_loader import (
    ensure_metric_config,
    load_metric_config,
)
from experiment.degradation_perception.perception_config import ThresholdModel


METRIC = "timing_s/step"
STANDARD_VALUES = [
    1.00,
    1.01,
    0.99,
    1.02,
    1.00,
    0.98,
    1.01,
    1.00,
    0.99,
    1.02,
] * 3
NORMAL_INFERENCE = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]
UP_INFERENCE = [1.00, 1.01, 0.99, 1.00, 1.50, 1.51, 1.49, 1.50, 1.52, 1.50]


def make_dataset(
    standard_values=STANDARD_VALUES,
    inference_values=NORMAL_INFERENCE,
    *,
    metric=METRIC,
):
    return {
        "standard": {
            metric: {
                "timestamps": list(range(1, len(standard_values) + 1)),
                "values": list(standard_values),
            }
        },
        "inference": {
            metric: {
                "timestamps": list(range(100, 100 + len(inference_values))),
                "values": list(inference_values),
            }
        },
    }


def prometheus_range_payload(values, *, start=1_710_000_000, step=15):
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": METRIC, "worker": "trainer_0"},
                    "values": [
                        [start + index * step, str(value)]
                        for index, value in enumerate(values)
                    ],
                }
            ],
        },
    }


def detector(tmp_path, dataset, **kwargs):
    return DegradationPerception(
        dataset=dataset,
        metrics=kwargs.pop("metrics", [METRIC]),
        config_dir=tmp_path / "config",
        **kwargs,
    )


def set_metric_abnormal_type(config_dir: Path, metric: str, abnormal_type: str):
    path = ensure_metric_config(metric, config_dir=config_dir)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["abnormal_type"] = abnormal_type
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_get_standard_data_returns_aligned_sorted_series():
    dataset = make_dataset([1.0, 1.1, 1.2])
    dataset["standard"][METRIC] = {
        "timestamps": [3, 1, 2, 1],
        "values": [3.0, 1.0, 2.0, 1.1],
    }
    series = get_standard_data(dataset, METRIC)
    assert series.timestamps == [1.0, 2.0, 3.0]
    assert series.values == [1.1, 2.0, 3.0]


def test_build_standard_data_applies_one_minus_alpha_and_outward_ratios(tmp_path):
    config = load_metric_config(METRIC, config_dir=tmp_path / "config")
    models = build_standard_data(
        list(range(len(STANDARD_VALUES))),
        STANDARD_VALUES,
        config,
    )
    assert models
    for model in models:
        assert model.upper_threshold == pytest.approx(
            model.upper_kde_threshold * config["upper_ratio"]
        )
        assert model.lower_threshold == pytest.approx(
            model.lower_kde_threshold / config["lower_ratio"]
        )
        assert (
            model.lower_threshold
            <= model.lower_kde_threshold
            < model.upper_kde_threshold
            <= model.upper_threshold
        )


def test_task_id_none_and_normal_response_are_json_serializable(tmp_path):
    response = detector(tmp_path, make_dataset(), task_id=None).detect()
    assert response["taskId"] == "default"
    assert response["states"] == {METRIC: 0}
    assert response["abnormalTimeRange"][METRIC] == []
    json.dumps(response, allow_nan=False)


def test_standard_insufficient_has_state_one_and_no_fabricated_detection(tmp_path):
    response = detector(tmp_path, make_dataset([1.0, 1.1], UP_INFERENCE)).detect()
    assert response["states"][METRIC] == 1
    assert response["results"][METRIC]["thresholds"] == []
    assert response["results"][METRIC]["abnormalTimeRange"] == []


def test_standard_state_one_has_priority_when_both_phases_are_insufficient(tmp_path):
    response = detector(tmp_path, make_dataset([1.0], [1.0])).detect()
    assert response["states"][METRIC] == 1


def test_inference_insufficient_has_state_two_and_no_degradation(tmp_path):
    response = detector(tmp_path, make_dataset(STANDARD_VALUES, [1.5] * 4)).detect()
    assert response["states"][METRIC] == 2
    assert response["results"][METRIC]["thresholds"] == []
    assert response["abnormalTimeRange"][METRIC] == []


def test_sustained_up_degradation_produces_a_formal_interval(tmp_path):
    response = detector(tmp_path, make_dataset(STANDARD_VALUES, UP_INFERENCE)).detect()
    assert response["states"][METRIC] == 0
    intervals = response["abnormalTimeRange"][METRIC]
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval["abnormalPointCount"] >= 5
    assert interval["abnormalRate"] > 0.60
    assert interval["duration"] > 0.5
    assert interval["abnormalType"] == "UP"
    assert interval["validationDetail"] == {
        "condition_1": True,
        "condition_2": True,
        "condition_3": True,
        "condition_4": True,
    }


@pytest.mark.parametrize(
    ("abnormal_type", "inference_values"),
    [
        ("UP", [1.5] * 6),
        ("DOWN", [0.5] * 6),
        ("BOTH", [1.5] * 6),
    ],
)
def test_configured_abnormal_types_are_not_inferred_from_metric_name(
    tmp_path, abnormal_type, inference_values
):
    config_dir = tmp_path / "config"
    set_metric_abnormal_type(config_dir, METRIC, abnormal_type)
    response = DegradationPerception(
        dataset=make_dataset(STANDARD_VALUES, inference_values),
        metrics=[METRIC],
        config_dir=config_dir,
    ).detect()
    assert response["states"][METRIC] == 0
    assert response["abnormalTimeRange"][METRIC][0]["abnormalType"] == abnormal_type


@pytest.mark.parametrize(
    ("standard_values", "inference_values", "abnormal_type"),
    [
        (STANDARD_VALUES, [1.0] * 6, "DOWN"),
        (STANDARD_VALUES, [1.0] * 6, "BOTH"),
        ([-value for value in STANDARD_VALUES], [-1.0] * 6, "UP"),
        ([-value for value in STANDARD_VALUES], [-1.0] * 6, "BOTH"),
    ],
)
def test_outward_thresholds_do_not_flag_normal_positive_or_negative_data(
    tmp_path, standard_values, inference_values, abnormal_type
):
    config_dir = tmp_path / "config"
    set_metric_abnormal_type(config_dir, METRIC, abnormal_type)
    response = DegradationPerception(
        dataset=make_dataset(standard_values, inference_values),
        metrics=[METRIC],
        config_dir=config_dir,
    ).detect()
    assert response["states"][METRIC] == 0
    assert response["abnormalTimeRange"][METRIC] == []
    assert not any(
        item["abnormal"] for item in response["results"][METRIC]["pointDiagnostics"]
    )


def test_prometheus_matrix_runs_end_to_end_and_preserves_epoch_seconds(tmp_path):
    metric = "rl_insight_monitor_timing_s_step"
    standard_payload = prometheus_range_payload(STANDARD_VALUES)
    standard_payload["data"]["result"][0]["metric"]["__name__"] = metric
    inference_payload = prometheus_range_payload(UP_INFERENCE, start=1_710_001_000)
    inference_payload["data"]["result"][0]["metric"]["__name__"] = metric
    response = DegradationPerception(
        dataset={
            "standard": {metric: standard_payload},
            "inference": {metric: inference_payload},
        },
        metrics=[metric],
        source_type="prometheus",
        config_dir=tmp_path / "config",
    ).detect()
    assert response["states"][metric] == 0
    interval = response["abnormalTimeRange"][metric][0]
    assert interval["startTime"] > 1_700_000_000
    assert interval["endTime"] > interval["startTime"]


def test_remote_monitor_interval_crossing_10000_uses_one_time_mode(tmp_path):
    dataset = make_dataset(STANDARD_VALUES, UP_INFERENCE)
    dataset["inference"][METRIC]["timestamps"] = list(range(9996, 10006))

    response = detector(
        tmp_path,
        dataset,
        source_type="remote_monitor",
    ).detect()

    interval = response["abnormalTimeRange"][METRIC][0]
    assert interval["startTime"] == pytest.approx(9999 / 10000 / 60)
    assert interval["endTime"] == pytest.approx(10006 / 10000 / 60)
    assert interval["startTime"] <= interval["endTime"]


def test_known_high_normal_mode_is_not_flagged_by_up_detection(tmp_path):
    standard = [1.00, 1.01, 1.02, 1.04, 1.05, 5.00, 5.01, 5.02]
    response = detector(
        tmp_path,
        make_dataset(standard, [5.00, 5.01, 5.02, 5.01, 5.00, 5.02]),
    ).detect()
    assert response["states"][METRIC] == 0
    assert len(response["results"][METRIC]["thresholds"]) == 2
    assert response["abnormalTimeRange"][METRIC] == []


def test_directional_multi_mode_compatibility_is_any_mode_not_average():
    models = [
        ThresholdModel(0, 0, 1, 3, 0.8, 1.2, 0.8, 1.2, 0.1),
        ThresholdModel(1, 2, 3, 3, 4.8, 5.2, 4.8, 5.2, 0.1),
    ]
    assert _classify_value(0, 5.0, models, "DOWN")[0] is False
    assert _classify_value(0, 0.5, models, "DOWN")[0] is True
    assert _classify_value(0, 1.0, models, "BOTH")[0] is False
    assert _classify_value(0, 5.0, models, "BOTH")[0] is False
    assert _classify_value(0, 3.0, models, "BOTH")[0] is True


def test_both_detection_rejects_a_model_with_lower_above_upper():
    invalid = ThresholdModel(0, 0, 1, 3, 2.0, 1.0, 2.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="lower threshold above upper"):
        _classify_value(0, 1.5, [invalid], "BOTH")


def test_multi_metric_failure_is_isolated(tmp_path):
    bad_metric = "broken/metric"
    dataset = make_dataset()
    dataset["standard"][bad_metric] = {
        "timestamps": [1, 2, 3],
        "values": [1.0],
    }
    dataset["inference"][bad_metric] = {
        "timestamps": list(range(100, 106)),
        "values": [1.0] * 6,
    }
    response = detector(
        tmp_path,
        dataset,
        metrics=[bad_metric, METRIC],
    ).detect()
    assert bad_metric not in response["states"]
    assert response["metricErrors"][bad_metric] == {
        "code": "metric_input_error",
        "type": "DataValidationError",
        "message": "metric input could not be validated",
    }
    assert "state" not in response["results"][bad_metric]
    assert response["states"][METRIC] == 0


def test_metric_config_error_is_not_reported_as_business_state(tmp_path):
    bad_metric = "bad-config/metric"
    config_dir = tmp_path / "config"
    config_path = ensure_metric_config(bad_metric, config_dir=config_dir)
    config_path.write_text(
        "minimum_standard_points: 0\n",
        encoding="utf-8",
    )
    response = detector(
        tmp_path,
        make_dataset(),
        metrics=[bad_metric, METRIC],
    )
    response.config_dir = config_dir
    output = response.detect()

    assert bad_metric not in output["states"]
    assert output["metricErrors"][bad_metric]["code"] == "metric_config_error"
    assert output["states"][METRIC] == 0


def test_internal_metric_error_is_serializable_and_redacted(
    tmp_path,
    monkeypatch,
):
    instance = detector(tmp_path, make_dataset())

    def fail_detection(*args, **kwargs):
        raise RuntimeError(
            "token=do-not-expose http://sensitive.internal/full/address"
        )

    monkeypatch.setattr(instance, "build_standard_data", fail_detection)
    response = instance.detect()

    assert METRIC not in response["states"]
    assert response["metricErrors"][METRIC] == {
        "code": "metric_detection_error",
        "type": "RuntimeError",
        "message": "metric detection raised an internal error",
    }
    serialized = json.dumps(response, allow_nan=False)
    assert "do-not-expose" not in serialized
    assert "sensitive.internal" not in serialized


def test_history_is_independent_and_requires_configured_number_of_abnormal_runs(
    tmp_path,
):
    common = tmp_path / "common.yaml"
    common.write_text("n_keep_result: 3\nn_keep_abnormal: 2\n", encoding="utf-8")
    instance = DegradationPerception(
        dataset=make_dataset(STANDARD_VALUES, UP_INFERENCE),
        metrics=[METRIC],
        task_id="task-a",
        config_dir=tmp_path / "config",
        common_config_path=common,
    )
    first = instance.detect()
    second = instance.detect()
    assert first["abnormalTimeRange"][METRIC] == []
    assert first["results"][METRIC]["currentAbnormalTimeRange"]
    assert second["abnormalTimeRange"][METRIC]
    assert instance.history[("task-a", METRIC)].maxlen == 3


def test_insufficient_run_does_not_advance_valid_history(tmp_path):
    common = tmp_path / "common.yaml"
    common.write_text("n_keep_result: 2\nn_keep_abnormal: 2\n", encoding="utf-8")
    instance = DegradationPerception(
        dataset=make_dataset(STANDARD_VALUES, UP_INFERENCE),
        metrics=[METRIC],
        config_dir=tmp_path / "config",
        common_config_path=common,
    )
    first = instance.detect()
    assert first["abnormalTimeRange"][METRIC] == []
    insufficient = instance.detect_dataset(
        make_dataset(STANDARD_VALUES, [1.5] * 4)
    )
    assert insufficient["states"][METRIC] == 2
    third = instance.detect_dataset(make_dataset(STANDARD_VALUES, UP_INFERENCE))
    assert third["abnormalTimeRange"][METRIC]


def test_detect_dataset_uses_programmatic_data_without_a_path(tmp_path):
    instance = DegradationPerception(
        metrics=[METRIC],
        config_dir=tmp_path / "config",
    )
    response = instance.detect_dataset(make_dataset())
    assert response["states"][METRIC] == 0


def test_cached_standard_model_supports_inference_only_followup(tmp_path):
    instance = detector(
        tmp_path,
        make_dataset(STANDARD_VALUES, NORMAL_INFERENCE),
    )
    assert instance.detect()["states"][METRIC] == 0

    inference_only = {
        "standard": {},
        "inference": make_dataset(
            STANDARD_VALUES,
            UP_INFERENCE,
        )["inference"],
    }
    followup = instance.detect_dataset(inference_only)
    assert followup["states"][METRIC] == 0
    assert followup["abnormalTimeRange"][METRIC]


def test_config_change_invalidates_cached_standard_model(tmp_path):
    config_dir = tmp_path / "config"
    instance = DegradationPerception(
        dataset=make_dataset(),
        metrics=[METRIC],
        config_dir=config_dir,
    )
    assert instance.detect()["states"][METRIC] == 0

    target = ensure_metric_config(METRIC, config_dir=config_dir)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    data["upper_ratio"] = 1.20
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    inference_only = {
        "standard": {},
        "inference": make_dataset()["inference"],
    }
    response = instance.detect_dataset(inference_only)
    assert response["states"][METRIC] == 1
    assert response["results"][METRIC]["thresholds"] == []
