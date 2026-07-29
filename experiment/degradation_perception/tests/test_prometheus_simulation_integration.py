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

from experiment.degradation_perception.algorithm import DegradationPerception
from experiment.degradation_perception.prometheus_matrix_adapter import (
    convert_simulation_package,
    load_simulation_package,
)
from experiment.degradation_perception.simulated_prometheus import (
    ASSOCIATED_METRICS,
    METRICS,
    TARGET_METRIC,
    generate_simulation_package,
    main,
    prepare_simulation_configs,
    run_simulation,
)


EXAMPLE_PACKAGE = (
    Path(__file__).parents[1] / "examples" / "simulated_prometheus_matrix.json"
)
REQUIRED_ASSOCIATION_FIELDS = {
    "rank",
    "metric",
    "abnormalContribution",
    "pearson",
    "spearman",
    "selectedCorrelation",
    "selectedCorrelationMethod",
    "correlationDirection",
    "randomForestImportance",
    "coverageRatio",
    "alignedSampleCount",
}


def _strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_committed_matrix_fixture_runs_through_real_kde_and_association(
    tmp_path,
):
    package = load_simulation_package(EXAMPLE_PACKAGE)
    assert package == generate_simulation_package()
    dataset, adapter_diagnostics = convert_simulation_package(package)
    config_dir = tmp_path / "config"
    prepare_simulation_configs(config_dir)

    response = DegradationPerception(
        dataset=dataset,
        metrics=list(METRICS),
        association_targets=[TARGET_METRIC],
        source_type="prometheus",
        config_dir=config_dir,
    ).detect()

    assert response["states"] == {metric: 0 for metric in METRICS}
    assert all(
        diagnostics["filteredNonFiniteCount"] == 0
        for phase in adapter_diagnostics["phases"].values()
        for diagnostics in phase.values()
    )
    target_ranges = response["abnormalTimeRange"][TARGET_METRIC]
    assert len(target_ranges) == 1
    assert target_ranges[0]["abnormalPointCount"] == 50
    target = response["associationAnalysis"]["targets"][TARGET_METRIC]
    assert target["status"] == "success"
    assert len(target["events"]) == 1
    event = target["events"][0]
    assert event["rawTargetAbnormalRange"] == {
        "startTime": 1785303700.0,
        "endTime": 1785304190.0,
    }
    assert [item["metric"] for item in event["topAssociations"]] == [
        *ASSOCIATED_METRICS,
    ]
    assert all(
        REQUIRED_ASSOCIATION_FIELDS <= set(item)
        for item in event["topAssociations"]
    )
    assert all(
        item["correlationDirection"] == "positive"
        for item in event["topAssociations"]
    )
    assert event["randomForestStatus"] == "success"
    assert event["randomForestDiagnostics"]["importanceMethod"] == "permutation"
    assert sum(
        item["abnormalContribution"] for item in event["allAssociations"]
    ) == pytest.approx(100.0)
    assert {
        item["metric"]: item["reason"] for item in event["excludedMetrics"]
    } == {
        "constant_metric": "constant_candidate_series",
        "sparse_metric": "insufficient_coverage",
        "unrelated_metric": "not_abnormal_in_target_window",
    }
    json.dumps(response, allow_nan=False)


def test_run_simulation_writes_all_acceptance_outputs_with_strict_json(tmp_path):
    output_dir = tmp_path / "output with spaces"
    summary = run_simulation(
        output_dir,
        run_analysis=True,
        config_dir=tmp_path / "配置 dir",
    )

    expected = {
        "simulated_prometheus_matrix.json",
        "converted_algorithm_input.json",
        "adapter_diagnostics.json",
        "analysis_result.json",
        "top5_result.json",
        "validation_summary.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    loaded = {name: _strict_load(output_dir / name) for name in expected}
    assert loaded["validation_summary.json"] == summary
    assert summary["matrixFormatValid"] is True
    assert summary["strictJsonSerializable"] is True
    assert summary["targetAbnormalEventCount"] == 1
    assert summary["candidateMetrics"] == [
        *ASSOCIATED_METRICS,
    ]
    assert [
        item["metric"]
        for item in loaded["top5_result.json"]["events"][0]["top5"]
    ] == list(ASSOCIATED_METRICS)
    assert summary["allCandidateContributionTotal"] == pytest.approx(100.0)
    assert summary["randomForestStatus"] == "success"
    assert summary["randomForestMethod"] == "permutation"
    assert all(summary["checks"].values())
    json.dumps(loaded, allow_nan=False)


def test_cli_supports_cross_platform_path_objects_and_emits_one_json(
    tmp_path,
    capsys,
):
    output_dir = tmp_path / "cli output"
    code = main(
        [
            "--output-dir",
            str(output_dir),
            "--run-analysis",
            "--association-target",
            TARGET_METRIC,
            "--config-dir",
            str(tmp_path / "cli config"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert payload["ok"] is True
    assert all(payload["summary"]["checks"].values())
    assert (output_dir / "validation_summary.json").is_file()
