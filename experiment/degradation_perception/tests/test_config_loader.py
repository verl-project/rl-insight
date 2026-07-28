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

import hashlib

import pytest
import yaml

from experiment.degradation_perception.config_loader import (
    ConfigCollisionError,
    ensure_metric_config,
    load_common_config,
    load_metric_config,
    metric_to_safe_filename,
)


def test_metric_to_safe_filename_preserves_confirmed_mapping():
    assert metric_to_safe_filename("timing_s/step") == "timing_s__step.yaml"
    assert metric_to_safe_filename(r"timing_s\step") == "timing_s__step.yaml"


@pytest.mark.parametrize(
    "metric",
    [
        "",
        " ",
        " leading",
        "trailing ",
        "/absolute",
        r"C:\absolute",
        r"\\server\share",
        "../escape",
        "a/../escape",
        "a//b",
        "a\x00b",
        "a\nb",
    ],
)
def test_metric_to_safe_filename_rejects_path_and_control_input(metric):
    with pytest.raises(ValueError):
        metric_to_safe_filename(metric)


def test_metric_to_safe_filename_defends_windows_reserved_names():
    assert metric_to_safe_filename("CON") == "_CON.yaml"
    assert metric_to_safe_filename("lpt1.metric") == "_lpt1.metric.yaml"


def test_long_metric_name_has_a_stable_hash_suffix():
    metric = "metric_" + "x" * 300
    filename = metric_to_safe_filename(metric)
    digest = hashlib.sha256(metric.encode("utf-8")).hexdigest()[:16]
    assert filename.endswith(f"__{digest}.yaml")
    assert len(filename.removesuffix(".yaml")) <= 160


def test_default_config_is_copied_once_and_binds_original_metric(tmp_path):
    target = ensure_metric_config("timing_s/step", config_dir=tmp_path)
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert target.parent == tmp_path.resolve()
    assert target.name == "timing_s__step.yaml"
    assert loaded["metric"] == "timing_s/step"
    assert loaded["upper_ratio"] == 1.15

    target.write_text(
        target.read_text(encoding="utf-8").replace("upper_ratio: 1.15", "upper_ratio: 1.25"),
        encoding="utf-8",
    )
    same = ensure_metric_config("timing_s/step", config_dir=tmp_path)
    assert same == target
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["upper_ratio"] == 1.25


def test_missing_default_template_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Default metric config"):
        ensure_metric_config(
            "metric",
            config_dir=tmp_path / "config",
            default_config_path=tmp_path / "missing.yaml",
        )


def test_safe_filename_collision_is_detected_by_embedded_metric(tmp_path):
    ensure_metric_config("a/b", config_dir=tmp_path)
    with pytest.raises(ConfigCollisionError, match="already bound"):
        ensure_metric_config(r"a\b", config_dir=tmp_path)


def test_loading_valid_config_returns_original_metric_and_defaults(tmp_path):
    config = load_metric_config("timing_s/step", config_dir=tmp_path)
    assert config["metric"] == "timing_s/step"
    assert config["abnormal_type"] == "UP"
    assert config["alpha"] == 0.01
    assert config["normalization"]["type"] == "identity"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unknown",), 1),
        (("alpha",), 0.0),
        (("alpha",), 0.5),
        (("upper_ratio",), float("inf")),
        (("upper_ratio",), 0.99),
        (("lower_ratio",), 0.0),
        (("lower_ratio",), 0.99),
        (("kde", "random_seed"), True),
        (("kde", "zero_range_epsilon"), 0.0),
        (("stable_segment", "minimum_passed_flags"), 7),
        (("abnormal_interval", "minimum_abnormal_points"), 0),
        (("abnormal_interval", "minimum_abnormal_rate"), 1.1),
    ],
)
def test_metric_config_rejects_invalid_and_unknown_values(tmp_path, path, value):
    metric = "metric"
    target = ensure_metric_config(metric, config_dir=tmp_path)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    cursor = data
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_metric_config(metric, config_dir=tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "n_keep_result: 0\nn_keep_abnormal: 1\n",
        "n_keep_result: 2\nn_keep_abnormal: 3\n",
        "n_keep_result: true\nn_keep_abnormal: 1\n",
        "n_keep_result: 2\nn_keep_abnormal: 1\nunknown: 3\n",
    ],
)
def test_common_config_validates_history_bounds(tmp_path, text):
    path = tmp_path / "common.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_common_config(path)


def test_common_config_accepts_valid_history_window(tmp_path):
    path = tmp_path / "common.yaml"
    path.write_text("n_keep_result: 3\nn_keep_abnormal: 2\n", encoding="utf-8")
    assert load_common_config(path) == {
        "n_keep_result": 3,
        "n_keep_abnormal": 2,
    }
