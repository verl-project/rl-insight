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

"""Deterministic high-fidelity Prometheus matrix simulation data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .algorithm import DegradationPerception
from .config_loader import ensure_metric_config
from .prometheus_matrix_adapter import (
    convert_simulation_package,
    validate_simulation_package,
)
from .result_presentation import build_top5_result


DEFAULT_SEED = 20260729
TARGET_METRIC = "timing_s/step"
ASSOCIATED_METRICS = (
    "kv_cache_usage_perc",
    "response_length_mean",
    "num_requests_swapped",
    "e2e_request_latency",
    "global_seqlen_minimax_diff",
)
METRICS = (
    TARGET_METRIC,
    *ASSOCIATED_METRICS,
    "unrelated_metric",
    "constant_metric",
    "sparse_metric",
)
_METRIC_SPECS = {
    TARGET_METRIC: "rl_insight_monitor_timing_s_step",
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
    "response_length_mean": "rl_insight_monitor_response_length_mean",
    "num_requests_swapped": "vllm:num_requests_swapped",
    "e2e_request_latency": "simulated_e2e_request_latency_seconds",
    "global_seqlen_minimax_diff": (
        "rl_insight_monitor_global_seqlen_minmax_diff"
    ),
    "unrelated_metric": "rl_insight_monitor_unrelated_metric",
    "constant_metric": "rl_insight_monitor_constant_metric",
    "sparse_metric": "rl_insight_monitor_sparse_metric",
}


def _validated_generation_parameters(
    seed: int,
    standard_points: int,
    inference_points: int,
    query_step_seconds: float,
    base_timestamp: float,
) -> tuple[int, int, int, float, float]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(standard_points, bool)
        or not isinstance(standard_points, int)
        or standard_points < 120
    ):
        raise ValueError("standard_points must be an integer of at least 120")
    if (
        isinstance(inference_points, bool)
        or not isinstance(inference_points, int)
        or inference_points < 180
    ):
        raise ValueError("inference_points must be an integer of at least 180")
    if (
        isinstance(query_step_seconds, bool)
        or not isinstance(query_step_seconds, (int, float))
        or not math.isfinite(float(query_step_seconds))
        or float(query_step_seconds) <= 0
    ):
        raise ValueError("query_step_seconds must be a positive finite number")
    if (
        isinstance(base_timestamp, bool)
        or not isinstance(base_timestamp, (int, float))
        or not math.isfinite(float(base_timestamp))
    ):
        raise ValueError("base_timestamp must be a finite Unix-seconds number")
    return (
        seed,
        standard_points,
        inference_points,
        float(query_step_seconds),
        float(base_timestamp),
    )


def _rng(seed: int, metric_index: int, phase_index: int) -> np.random.Generator:
    return np.random.default_rng(seed + metric_index * 1009 + phase_index * 65537)


def _generate_values(
    seed: int,
    standard_points: int,
    inference_points: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    standard: dict[str, np.ndarray] = {}
    inference: dict[str, np.ndarray] = {}
    for metric_index, metric in enumerate(METRICS):
        standard_rng = _rng(seed, metric_index, 0)
        inference_rng = _rng(seed, metric_index, 1)
        if metric == TARGET_METRIC:
            baseline = 1.0 + standard_rng.normal(0.0, 0.006, standard_points)
            observed = 1.0 + inference_rng.normal(0.0, 0.008, inference_points)
            observed[70:120] = 1.92 + inference_rng.normal(0.0, 0.035, 50)
        elif metric == "kv_cache_usage_perc":
            baseline = 46.0 + standard_rng.normal(0.0, 0.45, standard_points)
            observed = 46.0 + inference_rng.normal(0.0, 0.65, inference_points)
            observed[70:120] = 91.0 + inference_rng.normal(0.0, 1.1, 50)
        elif metric == "response_length_mean":
            baseline = 820.0 + standard_rng.normal(0.0, 8.0, standard_points)
            observed = 820.0 + inference_rng.normal(0.0, 12.0, inference_points)
            observed[71:120] = 1580.0 + inference_rng.normal(0.0, 35.0, 49)
        elif metric == "num_requests_swapped":
            baseline = standard_rng.poisson(0.2, standard_points).astype(float)
            observed = inference_rng.poisson(0.2, inference_points).astype(float)
            observed[73:118] = inference_rng.integers(8, 15, 45)
        elif metric == "e2e_request_latency":
            baseline = 1.45 + standard_rng.normal(0.0, 0.01, standard_points)
            observed = 1.45 + inference_rng.normal(0.0, 0.014, inference_points)
            observed[76:115] = 3.05 + inference_rng.normal(0.0, 0.045, 39)
        elif metric == "global_seqlen_minimax_diff":
            baseline = 65.0 + standard_rng.normal(0.0, 3.0, standard_points)
            observed = 65.0 + inference_rng.normal(0.0, 4.0, inference_points)
            observed[80:112] = 430.0 + inference_rng.normal(0.0, 55.0, 32)
        elif metric == "unrelated_metric":
            baseline = 20.0 + standard_rng.normal(0.0, 0.3, standard_points)
            observed = 20.0 + inference_rng.normal(0.0, 0.3, inference_points)
        elif metric == "constant_metric":
            baseline = np.full(standard_points, 5.0, dtype=float)
            observed = np.full(inference_points, 5.0, dtype=float)
        else:
            baseline = 3.0 + standard_rng.normal(0.0, 0.04, standard_points)
            observed = 3.0 + inference_rng.normal(0.0, 0.05, inference_points)
            observed[70:120] = 7.5 + inference_rng.normal(0.0, 0.2, 50)
        standard[metric] = np.asarray(baseline, dtype=float)
        inference[metric] = np.asarray(observed, dtype=float)
    return standard, inference


def _matrix_response(
    prometheus_metric: str,
    phase: str,
    timestamps: list[float],
    values: np.ndarray,
) -> dict[str, Any]:
    experiment_name = f"mock_{phase}"
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {
                        "__name__": prometheus_metric,
                        "project": "verl",
                        "experiment_name": experiment_name,
                        "job": "trainer_metrics",
                        "instance": "127.0.0.1:9092",
                        "worker": "trainer_0",
                    },
                    "values": [
                        [float(timestamp), f"{float(value):.6f}"]
                        for timestamp, value in zip(timestamps, values)
                    ],
                }
            ],
        },
    }


def generate_simulation_package(
    seed: int = DEFAULT_SEED,
    standard_points: int = 120,
    inference_points: int = 180,
    query_step_seconds: float = 10,
    base_timestamp: float = 1785301200.0,
) -> dict[str, Any]:
    """Return a reproducible offline package of full matrix responses."""

    (
        seed,
        standard_points,
        inference_points,
        step,
        base,
    ) = _validated_generation_parameters(
        seed,
        standard_points,
        inference_points,
        query_step_seconds,
        base_timestamp,
    )
    standard_values, inference_values = _generate_values(
        seed,
        standard_points,
        inference_points,
    )
    standard_timestamps = [
        base + index * step for index in range(standard_points)
    ]
    inference_base = base + (standard_points + 60) * step
    inference_timestamps = [
        inference_base + index * step for index in range(inference_points)
    ]

    phases: dict[str, dict[str, dict[str, Any]]] = {
        "standard": {},
        "inference": {},
    }
    for metric in METRICS:
        prometheus_metric = _METRIC_SPECS[metric]
        for phase, all_timestamps, all_values in (
            ("standard", standard_timestamps, standard_values[metric]),
            ("inference", inference_timestamps, inference_values[metric]),
        ):
            if metric == "sparse_metric":
                stride = 10 if phase == "standard" else 15
                indexes = list(range(0, len(all_timestamps), stride))
                timestamps = [all_timestamps[index] for index in indexes]
                values = all_values[indexes]
            else:
                timestamps = list(all_timestamps)
                values = all_values
            experiment_name = f"mock_{phase}"
            phases[phase][metric] = {
                "query": (
                    f'{prometheus_metric}{{experiment_name="{experiment_name}"}}'
                ),
                "seriesPolicy": "exactly_one",
                "response": _matrix_response(
                    prometheus_metric,
                    phase,
                    timestamps,
                    values,
                ),
            }

    package = {
        "formatVersion": 1,
        "source": "simulated_prometheus_query_range",
        "queryStepSeconds": step,
        "standard": phases["standard"],
        "inference": phases["inference"],
    }
    normalized = validate_simulation_package(package)
    json.dumps(normalized, allow_nan=False)
    return normalized


def save_simulation_package(
    package: Any,
    path: str | os.PathLike[str],
) -> Path:
    """Validate and save one package as deterministic strict UTF-8 JSON."""

    normalized = validate_simulation_package(package)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    output.write_text(serialized + "\n", encoding="utf-8")
    return output


def _write_strict_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    path.write_text(serialized + "\n", encoding="utf-8")
    return path


def prepare_simulation_configs(
    config_dir: str | os.PathLike[str],
    *,
    association_target: str = TARGET_METRIC,
) -> dict[str, Path]:
    """Create deterministic configs in a caller-designated simulation directory."""

    if association_target != TARGET_METRIC:
        raise ValueError(
            f"this simulation is designed for target {TARGET_METRIC!r}"
        )
    directory = Path(config_dir).expanduser()
    paths: dict[str, Path] = {}
    for metric in METRICS:
        path = ensure_metric_config(metric, config_dir=directory)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"simulation metric config must be an object: {path}")
        config = dict(raw)
        config["abnormal_type"] = "UP"
        association = dict(config.get("association") or {})
        if metric == association_target:
            association.update(
                {
                    "enabled": True,
                    "target_metrics": [association_target],
                    "candidate_mode": "abnormal_lower_metrics",
                    "weights": {
                        "correlation": 0.5,
                        "random_forest": 0.5,
                    },
                    "top_k": 5,
                    "context_ratio": 0.2,
                    "min_aligned_points": 10,
                    "min_rf_samples": 30,
                    "min_coverage_ratio": 0.6,
                    "alignment_tolerance": None,
                }
            )
            random_forest = dict(association.get("random_forest") or {})
            random_forest.update(
                {
                    "n_estimators": 128,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "importance_method": "permutation",
                }
            )
            association["random_forest"] = random_forest
        else:
            association["enabled"] = False
        config["association"] = association
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        paths[metric] = path
    return paths


def _strict_json_serializable(*values: Any) -> bool:
    try:
        json.dumps(values, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _build_validation_summary(
    package: Mapping[str, Any],
    dataset: Mapping[str, Any],
    adapter_diagnostics: Mapping[str, Any],
    analysis_result: Mapping[str, Any],
    *,
    association_target: str,
) -> dict[str, Any]:
    association = analysis_result.get("associationAnalysis", {})
    target_result = association.get("targets", {}).get(association_target, {})
    events = target_result.get("events", [])
    event = events[0] if events else {}
    all_associations = list(event.get("allAssociations", []))
    top_associations = list(event.get("topAssociations", []))
    excluded_metrics = list(event.get("excludedMetrics", []))
    excluded_by_metric = {
        item["metric"]: item.get("reason")
        for item in excluded_metrics
        if isinstance(item, Mapping) and isinstance(item.get("metric"), str)
    }
    contributions = [
        float(item["abnormalContribution"])
        for item in all_associations
        if isinstance(item, Mapping) and "abnormalContribution" in item
    ]
    contribution_total = sum(contributions)
    contributions_valid = (
        len(contributions) == len(all_associations)
        and bool(contributions)
        and all(math.isfinite(value) and value >= 0 for value in contributions)
        and math.isclose(
            contribution_total,
            100.0,
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        )
    )
    target_ranges = analysis_result.get("abnormalTimeRange", {}).get(
        association_target,
        [],
    )
    rf_diagnostics = event.get("randomForestDiagnostics", {})
    point_abnormal_counts = {
        metric: sum(
            bool(item.get("abnormal"))
            for item in analysis_result.get("results", {})
            .get(metric, {})
            .get("pointDiagnostics", [])
            if isinstance(item, Mapping)
        )
        for metric in METRICS
    }
    ranked_metrics = [item.get("metric") for item in top_associations]
    simple_result = build_top5_result(analysis_result, association_target)
    simple_top5 = (
        simple_result["events"][0]["top5"]
        if simple_result.get("events")
        else []
    )
    checks = {
        "targetAbnormalDetected": len(target_ranges) == 1 and len(events) == 1,
        "fineGrainedTopFiveReturned": (
            ranked_metrics == list(ASSOCIATED_METRICS)
        ),
        "allTopFiveDirectionsPositive": all(
            item.get("correlationDirection") == "positive"
            for item in top_associations
        ),
        "unrelatedMetricExcluded": (
            excluded_by_metric.get("unrelated_metric")
            == "not_abnormal_in_target_window"
        ),
        "constantMetricExcluded": (
            excluded_by_metric.get("constant_metric")
            == "constant_candidate_series"
        ),
        "sparseMetricExcluded": excluded_by_metric.get("sparse_metric")
        in {"insufficient_coverage", "insufficient_aligned_points"},
        "contributionsValid": contributions_valid,
        "randomForestPermutationSucceeded": (
            event.get("randomForestStatus") == "success"
            and rf_diagnostics.get("importanceMethod") == "permutation"
        ),
        "simplifiedTopFiveMatches": (
            [item.get("metric") for item in simple_top5] == ranked_metrics
            and all(
                isinstance(item.get("abnormalContribution"), float)
                for item in simple_top5
            )
        ),
        "allStatesCompleted": all(
            int(value) == 0 for value in analysis_result.get("states", {}).values()
        ),
    }
    summary = {
        "matrixFormatValid": True,
        "strictJsonSerializable": _strict_json_serializable(
            package,
            dataset,
            adapter_diagnostics,
            analysis_result,
            simple_result,
        ),
        "standardPointCounts": {
            metric: len(dataset["standard"][metric]["timestamps"])
            for metric in METRICS
        },
        "inferencePointCounts": {
            metric: len(dataset["inference"][metric]["timestamps"])
            for metric in METRICS
        },
        "states": dict(analysis_result.get("states", {})),
        "pointAbnormalCounts": point_abnormal_counts,
        "targetMetric": association_target,
        "targetAbnormalEventCount": len(events),
        "targetAbnormalRanges": list(target_ranges),
        "rawTargetAbnormalRange": event.get("rawTargetAbnormalRange"),
        "candidateMetrics": [item["metric"] for item in all_associations],
        "excludedMetrics": excluded_metrics,
        "topAssociations": top_associations,
        "testerTop5": simple_top5,
        "allCandidateContributionTotal": contribution_total,
        "randomForestStatus": event.get("randomForestStatus"),
        "randomForestMethod": rf_diagnostics.get("importanceMethod"),
        "randomForestReason": rf_diagnostics.get("reason"),
        "checks": checks,
    }
    summary["strictJsonSerializable"] = summary[
        "strictJsonSerializable"
    ] and _strict_json_serializable(summary)
    return summary


def run_simulation(
    output_dir: str | os.PathLike[str],
    *,
    run_analysis: bool = False,
    association_target: str = TARGET_METRIC,
    config_dir: str | os.PathLike[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Generate, convert, and optionally analyze one deterministic package."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    package = generate_simulation_package(seed=seed)
    package_path = save_simulation_package(
        package,
        output / "simulated_prometheus_matrix.json",
    )
    dataset, adapter_diagnostics = convert_simulation_package(package)
    converted_path = _write_strict_json(
        output / "converted_algorithm_input.json",
        dataset,
    )
    diagnostics_path = _write_strict_json(
        output / "adapter_diagnostics.json",
        adapter_diagnostics,
    )
    if not run_analysis:
        return {
            "matrixFormatValid": True,
            "strictJsonSerializable": _strict_json_serializable(
                package,
                dataset,
                adapter_diagnostics,
            ),
            "outputFiles": {
                "matrix": str(package_path),
                "convertedInput": str(converted_path),
                "adapterDiagnostics": str(diagnostics_path),
            },
        }

    selected_config_dir = (
        output / "config" if config_dir is None else Path(config_dir).expanduser()
    )
    prepare_simulation_configs(
        selected_config_dir,
        association_target=association_target,
    )
    analysis_result = DegradationPerception(
        dataset=dataset,
        metrics=list(METRICS),
        association_targets=[association_target],
        source_type="prometheus",
        config_dir=selected_config_dir,
    ).detect()
    analysis_path = _write_strict_json(
        output / "analysis_result.json",
        analysis_result,
    )
    top5_result = build_top5_result(analysis_result, association_target)
    top5_path = _write_strict_json(
        output / "top5_result.json",
        top5_result,
    )
    summary = _build_validation_summary(
        package,
        dataset,
        adapter_diagnostics,
        analysis_result,
        association_target=association_target,
    )
    summary["outputFiles"] = {
        "matrix": str(package_path),
        "convertedInput": str(converted_path),
        "adapterDiagnostics": str(diagnostics_path),
        "analysisResult": str(analysis_path),
        "top5Result": str(top5_path),
        "validationSummary": str(output / "validation_summary.json"),
    }
    summary_path = _write_strict_json(
        output / "validation_summary.json",
        summary,
    )
    summary["outputFiles"]["validationSummary"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation_perception.simulated_prometheus",
        description=(
            "Generate an offline Prometheus query_range matrix package and "
            "optionally run the real degradation-perception pipeline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated package, converted input, and results.",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run the real KDE and association pipeline after conversion.",
    )
    parser.add_argument(
        "--association-target",
        default=TARGET_METRIC,
        help=f"Simulation target metric (currently {TARGET_METRIC!r}).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Dedicated simulation config directory; defaults to "
            "<output-dir>/config."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Non-negative deterministic random seed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the simulation CLI and emit exactly one strict JSON object."""

    args = _build_parser().parse_args(argv)
    try:
        summary = run_simulation(
            args.output_dir,
            run_analysis=bool(args.run_analysis),
            association_target=str(args.association_target),
            config_dir=args.config_dir,
            seed=int(args.seed),
        )
        if args.run_analysis:
            failed_checks = sorted(
                name
                for name, passed in summary.get("checks", {}).items()
                if passed is not True
            )
            if failed_checks:
                raise RuntimeError(
                    "simulation validation failed: " + ", ".join(failed_checks)
                )
        output = {"ok": True, "summary": summary}
        exit_code = 0
    except Exception as exc:  # CLI boundary: failures remain one JSON object.
        output = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 1
    sys.stdout.write(
        json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
