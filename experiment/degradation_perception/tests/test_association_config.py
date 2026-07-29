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

from collections.abc import Callable
from typing import Any

import pytest
import yaml

from experiment.degradation_perception.config_loader import (
    ensure_metric_config,
    load_metric_config,
)


METRIC = "timing_s/step"


def _rewrite(
    tmp_path,
    change: Callable[[dict[str, Any]], None],
) -> None:
    path = ensure_metric_config(METRIC, config_dir=tmp_path / "config")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_default_association_config_is_disabled_and_complete(tmp_path):
    config = load_metric_config(METRIC, config_dir=tmp_path / "config")
    association = config["association"]
    assert association == {
        "enabled": False,
        "target_metrics": [METRIC],
        "candidate_mode": "abnormal_lower_metrics",
        "weights": {"correlation": 0.5, "random_forest": 0.5},
        "top_k": 5,
        "context_ratio": 1.0,
        "min_aligned_points": 10,
        "min_rf_samples": 30,
        "min_coverage_ratio": 0.6,
        "alignment_tolerance": None,
        "random_forest": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
            "importance_method": "permutation",
        },
    }


def test_old_metric_yaml_without_association_is_deep_merged(tmp_path):
    _rewrite(tmp_path, lambda data: data.pop("association"))
    config = load_metric_config(METRIC, config_dir=tmp_path / "config")
    assert config["association"]["enabled"] is False
    assert config["association"]["weights"]["correlation"] == 0.5


def test_partial_association_override_keeps_nested_defaults(tmp_path):
    def change(data):
        data["association"] = {
            "enabled": True,
            "target_metrics": ["top/a", "top/b"],
            "weights": {"correlation": 0.7, "random_forest": 0.3},
        }

    _rewrite(tmp_path, change)
    association = load_metric_config(METRIC, config_dir=tmp_path / "config")[
        "association"
    ]
    assert association["enabled"] is True
    assert association["target_metrics"] == ["top/a", "top/b"]
    assert association["weights"] == {
        "correlation": 0.7,
        "random_forest": 0.3,
    }
    assert association["random_forest"]["n_estimators"] == 200


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("association", "enabled"), 1, "enabled"),
        (("association", "target_metrics"), METRIC, "target_metrics"),
        (("association", "target_metrics"), [""], "target_metrics"),
        (("association", "target_metrics"), [1], "target_metrics"),
        (("association", "candidate_mode"), "all", "candidate_mode"),
        (("association", "weights", "correlation"), -0.1, "correlation"),
        (("association", "weights", "correlation"), "0.5", "correlation"),
        (("association", "weights", "random_forest"), 0.6, "sum to 1"),
        (("association", "top_k"), 0, "top_k"),
        (("association", "top_k"), True, "top_k"),
        (("association", "context_ratio"), -1.0, "context_ratio"),
        (("association", "min_aligned_points"), 0, "min_aligned_points"),
        (("association", "min_rf_samples"), 0, "min_rf_samples"),
        (("association", "min_coverage_ratio"), 1.1, "min_coverage_ratio"),
        (("association", "alignment_tolerance"), 0.0, "alignment_tolerance"),
        (
            ("association", "random_forest", "n_estimators"),
            0,
            "n_estimators",
        ),
        (
            ("association", "random_forest", "class_weight"),
            None,
            "class_weight",
        ),
        (
            ("association", "random_forest", "random_state"),
            -1,
            "random_state",
        ),
        (
            ("association", "random_forest", "importance_method"),
            "impurity",
            "importance_method",
        ),
    ],
)
def test_invalid_association_values_are_rejected(
    tmp_path,
    path,
    value,
    match,
):
    _rewrite(tmp_path, lambda data: _set_path(data, path, value))
    with pytest.raises(ValueError, match=match):
        load_metric_config(METRIC, config_dir=tmp_path / "config")


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (("association",), "unknown"),
        (("association", "weights"), "unknown"),
        (("association", "random_forest"), "unknown"),
    ],
)
def test_unknown_association_keys_are_rejected(tmp_path, section, key):
    def change(data):
        target = data
        for part in section:
            target = target[part]
        target[key] = 1

    _rewrite(tmp_path, change)
    with pytest.raises(ValueError, match="unknown keys"):
        load_metric_config(METRIC, config_dir=tmp_path / "config")
