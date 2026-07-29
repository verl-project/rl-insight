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
import math

import pytest

from experiment.degradation_perception.association_analysis import (
    AssociationAnalyzer,
    DEFAULT_ASSOCIATION_CONFIG,
    _random_forest_importances,
    combine_evidence,
    compute_correlations,
    resolve_association_config,
)
from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.time_alignment import AlignmentResult


def test_strong_positive_linear_correlation_prefers_pearson_on_numeric_tie():
    result = compute_correlations(
        [1, 2, 3, 4, 5, 6],
        [2, 4, 6, 8, 10, 12],
    )
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["selectedCorrelationMethod"] == "pearson"
    assert result["correlationDirection"] == "positive"


def test_strong_negative_correlation_uses_absolute_strength_but_keeps_direction():
    result = compute_correlations(
        [1, 2, 3, 4, 5, 6],
        [12, 10, 8, 6, 4, 2],
    )
    assert result["selectedCorrelation"] == pytest.approx(-1.0)
    assert result["correlationStrength"] == pytest.approx(1.0)
    assert result["correlationDirection"] == "negative"


def test_monotonic_nonlinear_relation_prefers_spearman():
    result = compute_correlations(
        [1, 2, 3, 4, 5, 6],
        [1, 4, 9, 16, 25, 36],
    )
    assert result["spearman"] == pytest.approx(1.0)
    assert abs(result["pearson"]) < abs(result["spearman"])
    assert result["selectedCorrelationMethod"] == "spearman"


def test_nearly_linear_rank_inversion_prefers_pearson():
    result = compute_correlations(
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 6, 5.9],
    )
    assert abs(result["pearson"]) > abs(result["spearman"])
    assert result["selectedCorrelationMethod"] == "pearson"


def test_absolute_selection_does_not_use_plain_max_for_negative_values():
    result = compute_correlations(
        [1, 2, 3, 4, 5, 6, 7],
        [20, 18, 15, 14, 10, 8, 5],
    )
    expected = (
        result["pearson"]
        if abs(result["pearson"]) >= abs(result["spearman"])
        else result["spearman"]
    )
    assert expected < 0
    assert result["selectedCorrelation"] == pytest.approx(expected)


def test_spearman_handles_tied_ranks():
    result = compute_correlations(
        [1, 2, 3, 4, 5, 6],
        [1, 1, 2, 2, 3, 3],
    )
    assert result["valid"] is True
    assert math.isfinite(result["spearman"])


@pytest.mark.parametrize(
    ("target", "candidate", "reason"),
    [
        ([1, 2, 3], [5, 5, 5], "constant_candidate_series"),
        ([5, 5, 5], [1, 2, 3], "constant_target_series"),
        ([1, math.nan], [2, math.nan], "insufficient_finite_points"),
    ],
)
def test_invalid_correlation_inputs_degrade_without_nan(target, candidate, reason):
    result = compute_correlations(target, candidate)
    assert result["valid"] is False
    assert result["reason"] == reason
    assert result["pearson"] is None
    assert result["spearman"] is None


def test_pairwise_finite_deletion_preserves_valid_correlation():
    result = compute_correlations(
        [1, 2, math.nan, 4],
        [2, 4, 100, 8],
    )
    assert result["valid"] is True
    assert result["correlationStrength"] == pytest.approx(1.0)


def test_correlation_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="equal lengths"):
        compute_correlations([1, 2], [1])


def test_fifty_fifty_scoring_normalizes_each_source_before_weighting():
    combined = combine_evidence(
        ["corr", "forest"],
        {"corr": 3.0, "forest": 1.0},
        {"corr": 1.0, "forest": 3.0},
        {"correlation": 0.5, "random_forest": 0.5},
    )
    assert combined["status"] == "success"
    assert combined["scores"]["corr"]["score"] == pytest.approx(0.5)
    assert combined["scores"]["forest"]["score"] == pytest.approx(0.5)
    assert combined["scores"]["corr"][
        "normalizedCorrelationContribution"
    ] == pytest.approx(0.75)
    assert combined["scores"]["corr"][
        "normalizedRandomForestContribution"
    ] == pytest.approx(0.25)


def test_configured_seventy_thirty_weights_change_final_scores():
    combined = combine_evidence(
        ["corr", "forest"],
        {"corr": 3.0, "forest": 1.0},
        {"corr": 1.0, "forest": 3.0},
        {"correlation": 0.7, "random_forest": 0.3},
    )
    assert combined["scores"]["corr"]["score"] == pytest.approx(0.6)
    assert combined["scores"]["forest"]["score"] == pytest.approx(0.4)


def test_correlation_only_degradation_renormalizes_to_full_contribution():
    combined = combine_evidence(
        ["a", "b"],
        {"a": 3.0, "b": 1.0},
        {},
        {"correlation": 0.5, "random_forest": 0.5},
    )
    assert combined["status"] == "partial_success"
    assert combined["scores"]["a"]["score"] == pytest.approx(0.75)
    assert combined["scores"]["b"]["score"] == pytest.approx(0.25)


def test_random_forest_only_degradation_renormalizes_to_full_contribution():
    combined = combine_evidence(
        ["a", "b"],
        {},
        {"a": 1.0, "b": 3.0},
        {"correlation": 0.5, "random_forest": 0.5},
    )
    assert combined["status"] == "partial_success"
    assert combined["scores"]["a"]["score"] == pytest.approx(0.25)
    assert combined["scores"]["b"]["score"] == pytest.approx(0.75)


def test_no_valid_evidence_returns_insufficient_data():
    combined = combine_evidence(
        ["a"],
        {"a": math.nan},
        {"a": -1.0},
        {"correlation": 0.5, "random_forest": 0.5},
    )
    assert combined["status"] == "insufficient_data"
    assert combined["scores"]["a"]["score"] == 0.0


def test_all_candidate_scores_sum_to_one_and_names_are_stable():
    combined = combine_evidence(
        ["z", "a", "m", "a"],
        {"z": 2.0, "a": 2.0, "m": 2.0},
        {},
        {"correlation": 0.5, "random_forest": 0.5},
    )
    assert list(combined["scores"]) == ["a", "m", "z"]
    assert sum(item["score"] for item in combined["scores"].values()) == (
        pytest.approx(1.0)
    )


@pytest.mark.parametrize("candidate_count", [3, 5, 8])
def test_evidence_normalizes_over_every_candidate_before_top_k(candidate_count):
    candidates = [f"candidate_{index}" for index in range(candidate_count)]
    combined = combine_evidence(
        candidates,
        {
            metric: float(index + 1)
            for index, metric in enumerate(candidates)
        },
        {
            metric: float(candidate_count - index)
            for index, metric in enumerate(candidates)
        },
        {"correlation": 0.5, "random_forest": 0.5},
    )

    assert set(combined["scores"]) == set(candidates)
    assert sum(
        details["score"] for details in combined["scores"].values()
    ) == pytest.approx(1.0)


def test_disabled_yaml_does_not_request_association_analysis():
    configs = {"target": {"association": copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)}}
    assert resolve_association_config(["target"], configs, None) is None


def test_enabled_yaml_and_cli_override_resolve_explicit_targets():
    configured = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)
    configured["enabled"] = True
    configured["target_metrics"] = ["configured"]
    configs = {"configured": {"association": configured}}
    assert resolve_association_config(["configured"], configs, None)[
        "target_metrics"
    ] == ["configured"]
    cli = resolve_association_config(
        ["configured"],
        configs,
        ["cli", "cli", "other"],
    )
    assert cli["enabled"] is True
    assert cli["target_metrics"] == ["cli", "other"]


def _alignment(
    target_labels,
    candidate_labels,
    *,
    indices=None,
) -> AlignmentResult:
    if indices is None:
        indices = list(range(len(target_labels)))
    size = len(indices)
    return AlignmentResult(
        target_indices=list(indices),
        timestamps=[float(index) for index in indices],
        target_values=[float(index) for index in range(size)],
        target_labels=list(target_labels),
        candidate_values=[float(index) for index in range(size)],
        candidate_labels=list(candidate_labels),
        coverage_ratio=1.0,
        tolerance=0.0,
    )


def _rf_config(*, min_samples=20):
    config = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)
    config["min_rf_samples"] = min_samples
    config["random_forest"]["n_estimators"] = 64
    return config


def test_random_forest_strong_feature_ranks_above_weaker_feature():
    target = [bool(index % 4 >= 2) for index in range(80)]
    strong = list(target)
    weaker = [bool(index % 7 == 0) for index in range(80)]
    result = _random_forest_importances(
        {
            "strong": _alignment(target, strong),
            "weaker": _alignment(target, weaker),
        },
        _rf_config(),
    )
    assert result["status"] == "success"
    assert result["importanceMethod"] == "permutation"
    assert result["importances"]["strong"] > result["importances"]["weaker"]
    assert set(result["rawImportances"]) == {"strong", "weaker"}
    assert all(
        value is None or math.isfinite(value)
        for value in result["rawImportances"].values()
    )


def test_random_forest_handles_class_imbalance_with_balanced_configuration():
    target = [bool(index % 10 == 0) for index in range(100)]
    result = _random_forest_importances(
        {"strong": _alignment(target, target)},
        _rf_config(),
    )
    assert result["status"] == "success"
    assert result["importances"]["strong"] > 0


def test_random_forest_single_target_class_degrades_safely():
    labels = [False] * 40
    result = _random_forest_importances(
        {"candidate": _alignment(labels, [bool(i % 2) for i in range(40)])},
        _rf_config(),
    )
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "single_target_class"


def test_random_forest_insufficient_samples_skips_model():
    target = [False, True] * 5
    result = _random_forest_importances(
        {"candidate": _alignment(target, target)},
        _rf_config(min_samples=30),
    )
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "insufficient_common_samples"


def test_random_forest_all_constant_features_degrade_safely():
    target = [False, True] * 20
    result = _random_forest_importances(
        {"constant": _alignment(target, [False] * 40)},
        _rf_config(),
    )
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "all_candidate_features_constant"


def test_random_forest_time_split_single_class_uses_explicit_impurity_fallback():
    target = [False] * 30 + [True] * 10
    result = _random_forest_importances(
        {"candidate": _alignment(target, target)},
        _rf_config(),
    )
    assert result["status"] == "partial_success"
    assert result["importanceMethod"] == "impurity_fallback"
    assert result["reason"] == "time_split_lacks_both_classes"


def test_random_forest_fixed_seed_is_repeatable():
    target = [bool(index % 4 >= 2) for index in range(80)]
    alignments = {
        "a": _alignment(target, target),
        "b": _alignment(target, [bool(index % 5 == 0) for index in range(80)]),
    }
    first = _random_forest_importances(alignments, _rf_config())
    second = _random_forest_importances(alignments, _rf_config())
    assert first["importanceMethod"] == "permutation"
    assert first["importances"] == pytest.approx(second["importances"])
    assert first["validationScore"] == pytest.approx(second["validationScore"])
    assert first["rawImportances"] == pytest.approx(second["rawImportances"])


def test_random_forest_permutation_failure_has_specific_diagnostic(monkeypatch):
    import sklearn.inspection

    def fail_permutation(*args, **kwargs):
        raise ValueError("synthetic permutation failure")

    monkeypatch.setattr(
        sklearn.inspection,
        "permutation_importance",
        fail_permutation,
    )
    target = [bool(index % 4 >= 2) for index in range(80)]
    result = _random_forest_importances(
        {"candidate": _alignment(target, target)},
        _rf_config(),
    )
    assert result["status"] == "insufficient_data"
    assert result["reason"] == "permutation_importance_failed"
    assert result["validationScore"] is not None


def test_mixed_unsuccessful_targets_do_not_report_partial_success():
    config = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)
    config["enabled"] = True
    config["target_metrics"] = ["missing", "normal"]
    result = AssociationAnalyzer(
        config,
        source_type="training_log",
    ).analyze(
        {},
        {
            "states": {"normal": 0},
            "results": {"normal": {}},
            "abnormalTimeRange": {"normal": []},
        },
    )
    assert result["status"] == "insufficient_data"
    assert result["targets"]["missing"] == {
        "status": "target_metric_missing",
        "events": [],
        "topAssociations": [],
    }
    assert result["targets"]["normal"] == {
        "status": "target_not_abnormal",
        "events": [],
        "topAssociations": [],
    }


def test_published_event_without_raw_context_is_explicitly_insufficient():
    config = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)
    config["enabled"] = True
    config["target_metrics"] = ["target"]
    public_range = {
        "startTime": 10.0,
        "endTime": 12.0,
        "duration": 0.0,
    }
    result = AssociationAnalyzer(
        config,
        source_type="training_log",
    ).analyze(
        {
            "target": {
                "input_present": True,
                "series": TimeSeries([10.0, 11.0], [1.0, 2.0]),
                "raw_events": [
                    {"targetAbnormalRange": public_range},
                ],
            },
        },
        {
            "states": {"target": 0},
            "results": {
                "target": {
                    "pointDiagnostics": [
                        {"timestamp": 10.0, "abnormal": False},
                        {"timestamp": 11.0, "abnormal": True},
                    ]
                }
            },
            "abnormalTimeRange": {"target": [public_range]},
        },
    )
    assert result["status"] == "insufficient_data"
    target = result["targets"]["target"]
    assert target["status"] == "insufficient_data"
    assert len(target["events"]) == 1
    event = target["events"][0]
    assert event["status"] == "insufficient_data"
    assert event["reason"] == "raw_event_context_unavailable"
    assert event["targetAbnormalRange"] == public_range
    assert event["rawTargetAbnormalRange"] is None
    assert event["analysisWindow"] is None
    assert event["topAssociations"] == []
