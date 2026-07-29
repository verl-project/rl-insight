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

"""Post-KDE association ranking for confirmed target-metric anomaly events."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .perception_config import TimeSeries
from .time_alignment import (
    AlignmentResult,
    align_candidate_series,
    build_analysis_window,
)


DEFAULT_ASSOCIATION_CONFIG: dict[str, Any] = {
    "enabled": False,
    "target_metrics": ["timing_s/step"],
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


def _ordered_unique_strings(values: Sequence[Any], *, name: str) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        if value not in result:
            result.append(value)
    return result


def resolve_association_config(
    metrics: Sequence[str],
    metric_configs: Mapping[str, Mapping[str, Any]],
    cli_targets: Sequence[str] | None,
) -> dict[str, Any] | None:
    """Resolve CLI-over-YAML association settings after all KDE configs load."""

    if cli_targets is not None:
        targets = _ordered_unique_strings(list(cli_targets), name="association targets")
        target_config = metric_configs.get(targets[0], {}) if targets else {}
        raw = target_config.get("association")
        if not isinstance(raw, Mapping):
            raw = next(
                (
                    config.get("association")
                    for metric in metrics
                    if isinstance(
                        (config := metric_configs.get(metric, {})).get("association"),
                        Mapping,
                    )
                ),
                DEFAULT_ASSOCIATION_CONFIG,
            )
        resolved = copy.deepcopy(dict(raw))
        resolved["enabled"] = True
        resolved["target_metrics"] = targets
        return resolved

    for metric in metrics:
        raw = metric_configs.get(metric, {}).get("association")
        if isinstance(raw, Mapping) and raw.get("enabled") is True:
            return copy.deepcopy(dict(raw))
    return None


def compute_correlations(
    target_values: Sequence[float],
    candidate_values: Sequence[float],
) -> dict[str, Any]:
    """Compute finite Pearson/Spearman evidence and select by absolute value."""

    target = np.asarray(target_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    if target.shape != candidate.shape:
        raise ValueError("target and candidate values must have equal lengths")
    finite = np.isfinite(target) & np.isfinite(candidate)
    target = target[finite]
    candidate = candidate[finite]
    if target.size < 2:
        return _invalid_correlation("insufficient_finite_points")
    if np.ptp(target) == 0:
        return _invalid_correlation("constant_target_series")
    if np.ptp(candidate) == 0:
        return _invalid_correlation("constant_candidate_series")

    try:
        pearson = float(pearsonr(target, candidate).statistic)
    except ValueError:
        pearson = math.nan
    try:
        spearman = float(spearmanr(target, candidate).statistic)
    except ValueError:
        spearman = math.nan

    valid_values = [
        (method, value)
        for method, value in (("pearson", pearson), ("spearman", spearman))
        if math.isfinite(value)
    ]
    if not valid_values:
        return _invalid_correlation("undefined_correlation")
    by_method = dict(valid_values)
    if len(by_method) == 2 and math.isclose(
        abs(by_method["pearson"]),
        abs(by_method["spearman"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        selected_method, selected = "pearson", by_method["pearson"]
    else:
        selected_method, selected = max(valid_values, key=lambda item: abs(item[1]))
    if selected > 0:
        direction = "positive"
    elif selected < 0:
        direction = "negative"
    else:
        direction = "none"
    return {
        "valid": True,
        "reason": None,
        "pearson": pearson if math.isfinite(pearson) else None,
        "spearman": spearman if math.isfinite(spearman) else None,
        "selectedCorrelation": selected,
        "selectedCorrelationMethod": selected_method,
        "correlationDirection": direction,
        "correlationStrength": abs(selected),
    }


def _invalid_correlation(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "pearson": None,
        "spearman": None,
        "selectedCorrelation": None,
        "selectedCorrelationMethod": None,
        "correlationDirection": "none",
        "correlationStrength": None,
    }


def combine_evidence(
    candidates: Sequence[str],
    correlation_scores: Mapping[str, float],
    random_forest_importances: Mapping[str, float],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Normalize both evidence sources independently, then apply configured weights."""

    names = sorted(dict.fromkeys(str(name) for name in candidates))
    correlations = {
        name: _finite_nonnegative(correlation_scores.get(name, 0.0)) for name in names
    }
    importances = {
        name: _finite_nonnegative(random_forest_importances.get(name, 0.0))
        for name in names
    }
    correlation_total = sum(correlations.values())
    importance_total = sum(importances.values())
    correlation_valid = correlation_total > 0
    random_forest_valid = importance_total > 0

    if correlation_valid and random_forest_valid:
        correlation_weight = float(weights["correlation"])
        random_forest_weight = float(weights["random_forest"])
        status = "success"
    elif correlation_valid:
        correlation_weight = 1.0
        random_forest_weight = 0.0
        status = "partial_success"
    elif random_forest_valid:
        correlation_weight = 0.0
        random_forest_weight = 1.0
        status = "partial_success"
    else:
        correlation_weight = 0.0
        random_forest_weight = 0.0
        status = "insufficient_data"

    scores: dict[str, dict[str, float]] = {}
    for name in names:
        normalized_correlation = (
            correlations[name] / correlation_total if correlation_valid else 0.0
        )
        normalized_importance = (
            importances[name] / importance_total if random_forest_valid else 0.0
        )
        score = (
            correlation_weight * normalized_correlation
            + random_forest_weight * normalized_importance
        )
        scores[name] = {
            "normalizedCorrelationContribution": normalized_correlation,
            "normalizedRandomForestContribution": normalized_importance,
            "score": score,
        }
    return {
        "status": status,
        "correlationValid": correlation_valid,
        "randomForestValid": random_forest_valid,
        "scores": scores,
    }


def _finite_nonnegative(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return max(0.0, result)


def _diagnostic_labels(
    series: TimeSeries,
    metric_result: Mapping[str, Any],
) -> list[bool] | None:
    diagnostics = metric_result.get("pointDiagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        return None
    by_timestamp: dict[float, bool] = {}
    for item in diagnostics:
        if not isinstance(item, Mapping):
            continue
        try:
            timestamp = float(item["timestamp"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        abnormal = item.get("abnormal")
        if not math.isfinite(timestamp) or not isinstance(abnormal, bool):
            continue
        by_timestamp[timestamp] = abnormal
    if any(float(timestamp) not in by_timestamp for timestamp in series.timestamps):
        return None
    return [by_timestamp[float(timestamp)] for timestamp in series.timestamps]


def _random_forest_importances(
    alignments: Mapping[str, AlignmentResult],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not alignments:
        return _rf_failure("no_candidate_metrics")

    min_samples = int(config["min_rf_samples"])
    active = sorted(alignments)
    label_maps = {
        metric: {
            index: bool(label)
            for index, label in zip(
                alignment.target_indices,
                alignment.candidate_labels,
            )
        }
        for metric, alignment in alignments.items()
    }
    target_label_maps = {
        metric: {
            index: bool(label)
            for index, label in zip(
                alignment.target_indices,
                alignment.target_labels,
            )
        }
        for metric, alignment in alignments.items()
    }
    removed_for_common_rows: list[str] = []
    common_indexes: set[int] = set()
    while active:
        common_indexes = set(label_maps[active[0]])
        for metric in active[1:]:
            common_indexes.intersection_update(label_maps[metric])
        if len(common_indexes) >= min_samples or len(active) == 1:
            break
        drop = min(
            active,
            key=lambda metric: (
                len(label_maps[metric]),
                alignments[metric].coverage_ratio,
                metric,
            ),
        )
        active.remove(drop)
        removed_for_common_rows.append(drop)

    if len(common_indexes) < min_samples:
        return _rf_failure(
            "insufficient_common_samples",
            sampleCount=len(common_indexes),
            excludedForCommonRows=removed_for_common_rows,
        )

    ordered_indexes = sorted(common_indexes)
    first_metric = active[0]
    target_labels = np.asarray(
        [target_label_maps[first_metric][index] for index in ordered_indexes],
        dtype=int,
    )
    if np.unique(target_labels).size < 2:
        return _rf_failure(
            "single_target_class",
            sampleCount=len(ordered_indexes),
            excludedForCommonRows=removed_for_common_rows,
        )

    matrix = np.asarray(
        [[label_maps[metric][index] for metric in active] for index in ordered_indexes],
        dtype=float,
    )
    variable_columns = [
        index
        for index in range(matrix.shape[1])
        if np.unique(matrix[:, index]).size > 1
    ]
    if not variable_columns:
        return _rf_failure(
            "all_candidate_features_constant",
            sampleCount=len(ordered_indexes),
            excludedForCommonRows=removed_for_common_rows,
        )
    model_metrics = [active[index] for index in variable_columns]
    matrix = matrix[:, variable_columns]

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        return _rf_failure(
            "scikit_learn_unavailable",
            status="dependency_unavailable",
            sampleCount=len(ordered_indexes),
        )

    rf_config = config["random_forest"]
    model_kwargs = {
        "n_estimators": int(rf_config["n_estimators"]),
        "class_weight": rf_config["class_weight"],
        "random_state": int(rf_config["random_state"]),
        "n_jobs": 1,
    }
    split_index = max(1, min(len(target_labels) - 1, int(len(target_labels) * 0.7)))
    training_labels = target_labels[:split_index]
    validation_labels = target_labels[split_index:]
    can_use_permutation = (
        np.unique(training_labels).size == 2 and np.unique(validation_labels).size == 2
    )

    try:
        model = RandomForestClassifier(**model_kwargs)
        if can_use_permutation:
            model.fit(matrix[:split_index], training_labels)
            prediction = model.predict(matrix[split_index:])
            validation_score = float(
                balanced_accuracy_score(validation_labels, prediction)
            )
            try:
                permutation = permutation_importance(
                    model,
                    matrix[split_index:],
                    validation_labels,
                    scoring="balanced_accuracy",
                    n_repeats=10,
                    random_state=int(rf_config["random_state"]),
                    n_jobs=1,
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                return _rf_failure(
                    "permutation_importance_failed",
                    detail=str(exc),
                    sampleCount=len(ordered_indexes),
                    validationScore=validation_score,
                )
            raw_importances = np.asarray(permutation.importances_mean, dtype=float)
            method = "permutation"
            status = "success"
            reason = None
        else:
            model.fit(matrix, target_labels)
            prediction = model.predict(matrix)
            validation_score = float(balanced_accuracy_score(target_labels, prediction))
            raw_importances = np.asarray(model.feature_importances_, dtype=float)
            method = "impurity_fallback"
            status = "partial_success"
            reason = "time_split_lacks_both_classes"
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return _rf_failure(
            "model_training_failed",
            detail=str(exc),
            sampleCount=len(ordered_indexes),
        )

    raw_importances_by_metric: dict[str, float | None] = {
        metric: 0.0 for metric in active
    }
    raw_importances_by_metric.update(
        {
            metric: float(raw_importances[index])
            if math.isfinite(float(raw_importances[index]))
            else None
            for index, metric in enumerate(model_metrics)
        }
    )
    clipped = np.where(np.isfinite(raw_importances), raw_importances, 0.0)
    clipped = np.maximum(clipped, 0.0)
    if float(np.sum(clipped)) <= 0:
        return _rf_failure(
            "all_importances_zero",
            sampleCount=len(ordered_indexes),
            validationScore=validation_score,
            importanceMethod=method,
        )

    importances = {metric: 0.0 for metric in active}
    importances.update(
        {metric: float(clipped[index]) for index, metric in enumerate(model_metrics)}
    )
    return {
        "status": status,
        "reason": reason,
        "importanceMethod": method,
        "importances": importances,
        "rawImportances": raw_importances_by_metric,
        "sampleCount": len(ordered_indexes),
        "validationScore": validation_score,
        "excludedForCommonRows": removed_for_common_rows,
    }


def _rf_failure(
    reason: str,
    *,
    status: str = "insufficient_data",
    **diagnostics: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "importanceMethod": None,
        "importances": {},
        "rawImportances": diagnostics.pop("rawImportances", {}),
        "sampleCount": int(diagnostics.pop("sampleCount", 0)),
        "validationScore": diagnostics.pop("validationScore", None),
        **diagnostics,
    }


class AssociationAnalyzer:
    """Rank lower metrics independently for each confirmed target event."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        source_type: str,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.source_type = source_type
        self.targets = _ordered_unique_strings(
            list(self.config.get("target_metrics", [])),
            name="association.target_metrics",
        )

    def analyze(
        self,
        metric_context: Mapping[str, Mapping[str, Any]],
        detection_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a JSON-safe association section without mutating KDE output."""

        weights = self.config["weights"]
        if not self.targets:
            return {
                "enabled": True,
                "status": "target_not_configured",
                "weights": {
                    "correlation": float(weights["correlation"]),
                    "randomForest": float(weights["random_forest"]),
                },
                "targets": {},
            }

        target_results = {
            target: self._analyze_target(
                target,
                metric_context,
                detection_output,
            )
            for target in self.targets
        }
        statuses = [result["status"] for result in target_results.values()]
        if statuses and all(status == "success" for status in statuses):
            overall_status = "success"
        elif any(status in {"success", "partial_success"} for status in statuses):
            overall_status = "partial_success"
        elif len(set(statuses)) == 1:
            overall_status = statuses[0]
        else:
            overall_status = "insufficient_data"
        return {
            "enabled": True,
            "status": overall_status,
            "weights": {
                "correlation": float(weights["correlation"]),
                "randomForest": float(weights["random_forest"]),
            },
            "targets": target_results,
        }

    def _analyze_target(
        self,
        target: str,
        metric_context: Mapping[str, Mapping[str, Any]],
        detection_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        states = detection_output.get("states", {})
        results = detection_output.get("results", {})
        metric_errors = detection_output.get("metricErrors", {})
        if target in metric_errors:
            return _target_status("target_detection_failed")
        if target not in states or target not in results:
            return _target_status("target_metric_missing")
        context = metric_context.get(target)
        if isinstance(context, Mapping) and context.get("input_present") is False:
            return _target_status("target_metric_missing")
        if int(states[target]) != 0:
            return _target_status("target_detection_failed")
        published = detection_output.get("abnormalTimeRange", {}).get(target, [])
        if not published:
            return _target_status("target_not_abnormal")
        if not isinstance(context, Mapping):
            return _target_status(
                "insufficient_data", "target inference context is unavailable"
            )
        series = context.get("series")
        raw_events = context.get("raw_events")
        if not isinstance(series, TimeSeries):
            return _target_status(
                "insufficient_data", "target inference series is unavailable"
            )
        target_labels = _diagnostic_labels(series, results[target])
        if target_labels is None:
            return _target_status(
                "insufficient_data", "target point diagnostics are unavailable"
            )
        if not isinstance(raw_events, Sequence) or not raw_events:
            return _target_status(
                "insufficient_data",
                "confirmed ranges have no raw event in the current inference batch",
            )

        events = [
            self._analyze_event(
                target,
                event,
                series,
                target_labels,
                metric_context,
                detection_output,
            )
            for event in raw_events
            if isinstance(event, Mapping)
        ]
        if not events:
            return _target_status(
                "insufficient_data", "no analyzable target event is available"
            )
        statuses = [event["status"] for event in events]
        if all(status == "success" for status in statuses):
            status = "success"
        elif any(status in {"success", "partial_success"} for status in statuses):
            status = "partial_success"
        elif all(status == "no_candidate_metrics" for status in statuses):
            status = "no_candidate_metrics"
        else:
            status = "insufficient_data"
        return {"status": status, "events": events}

    def _analyze_event(
        self,
        target: str,
        event: Mapping[str, Any],
        target_series: TimeSeries,
        target_labels: Sequence[bool],
        metric_context: Mapping[str, Mapping[str, Any]],
        detection_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        states = detection_output.get("states", {})
        metric_errors = detection_output.get("metricErrors", {})
        all_metrics = list(dict.fromkeys([*states, *metric_errors]))
        lower_metrics = [
            metric for metric in all_metrics if metric not in set(self.targets)
        ]
        if "rawStartTime" not in event or "rawEndTime" not in event:
            return {
                "targetAbnormalRange": copy.deepcopy(
                    dict(event["targetAbnormalRange"])
                ),
                "rawTargetAbnormalRange": None,
                "analysisWindow": None,
                "candidateMetricCount": len(lower_metrics),
                "validCandidateMetricCount": 0,
                "excludedMetrics": [],
                "candidateDiagnostics": {},
                "status": "insufficient_data",
                "reason": "raw_event_context_unavailable",
                "alignedSampleCount": 0,
                "randomForestStatus": "insufficient_data",
                "randomForestDiagnostics": {"reason": "raw_event_context_unavailable"},
                "topAssociations": [],
                "allAssociations": [],
            }
        raw_start = float(event["rawStartTime"])
        raw_end = float(event["rawEndTime"])
        analysis_window = build_analysis_window(
            raw_start,
            raw_end,
            target_series.timestamps,
            float(self.config["context_ratio"]),
        )
        results = detection_output.get("results", {})
        excluded: list[dict[str, str]] = [
            {
                "metric": metric,
                "reason": "configured_target_metric",
            }
            for metric in all_metrics
            if metric != target and metric in set(self.targets)
        ]
        candidate_diagnostics: dict[str, dict[str, Any]] = {}
        alignments: dict[str, AlignmentResult] = {}
        correlations: dict[str, dict[str, Any]] = {}

        for metric in lower_metrics:
            context = metric_context.get(metric)
            if metric in metric_errors:
                excluded.append(
                    {"metric": metric, "reason": "candidate_detection_failed"}
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "candidate_detection_failed",
                    "errorCode": metric_errors[metric].get("code"),
                }
                continue
            if isinstance(context, Mapping) and context.get("input_present") is False:
                excluded.append(
                    {"metric": metric, "reason": "candidate_inference_missing"}
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "candidate_inference_missing",
                }
                continue
            if int(states.get(metric, 1)) != 0:
                excluded.append(
                    {"metric": metric, "reason": "candidate_detection_failed"}
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "candidate_detection_failed",
                }
                continue
            candidate_series = (
                context.get("series") if isinstance(context, Mapping) else None
            )
            if not isinstance(candidate_series, TimeSeries):
                excluded.append(
                    {"metric": metric, "reason": "candidate_inference_missing"}
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "candidate_inference_missing",
                }
                continue
            candidate_result = results.get(metric, {})
            candidate_labels = _diagnostic_labels(candidate_series, candidate_result)
            if candidate_labels is None:
                excluded.append(
                    {"metric": metric, "reason": "candidate_labels_unavailable"}
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "candidate_labels_unavailable",
                }
                continue
            if np.ptp(np.asarray(candidate_series.values, dtype=float)) == 0:
                excluded.append(
                    {"metric": metric, "reason": "constant_candidate_series"}
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "constant_candidate_series",
                }
                continue
            abnormal_in_window = any(
                bool(label)
                and float(analysis_window["startTime"])
                <= float(timestamp)
                <= float(analysis_window["endTime"])
                for timestamp, label in zip(
                    candidate_series.timestamps, candidate_labels
                )
            )
            if not abnormal_in_window:
                excluded.append(
                    {
                        "metric": metric,
                        "reason": "not_abnormal_in_target_window",
                    }
                )
                candidate_diagnostics[metric] = {
                    "status": "excluded",
                    "exclusionReason": "not_abnormal_in_target_window",
                }
                continue
            alignment = align_candidate_series(
                target_series,
                target_labels,
                candidate_series,
                candidate_labels,
                start_time=float(analysis_window["startTime"]),
                end_time=float(analysis_window["endTime"]),
                source_type=self.source_type,
                max_tolerance=self.config.get("alignment_tolerance"),
            )
            candidate_diagnostics[metric] = {
                "coverageRatio": alignment.coverage_ratio,
                "alignedSampleCount": len(alignment.timestamps),
                "alignmentTolerance": alignment.tolerance,
            }
            if alignment.coverage_ratio < float(self.config["min_coverage_ratio"]):
                excluded.append({"metric": metric, "reason": "insufficient_coverage"})
                candidate_diagnostics[metric].update(
                    {
                        "status": "excluded",
                        "exclusionReason": "insufficient_coverage",
                    }
                )
                continue
            if len(alignment.timestamps) < int(self.config["min_aligned_points"]):
                excluded.append(
                    {"metric": metric, "reason": "insufficient_aligned_points"}
                )
                candidate_diagnostics[metric].update(
                    {
                        "status": "excluded",
                        "exclusionReason": "insufficient_aligned_points",
                    }
                )
                continue
            if np.ptp(np.asarray(alignment.candidate_values, dtype=float)) == 0:
                excluded.append(
                    {"metric": metric, "reason": "constant_candidate_series"}
                )
                candidate_diagnostics[metric].update(
                    {
                        "status": "excluded",
                        "exclusionReason": "constant_candidate_series",
                    }
                )
                continue
            correlation = compute_correlations(
                alignment.target_values,
                alignment.candidate_values,
            )
            alignments[metric] = alignment
            correlations[metric] = correlation
            candidate_diagnostics[metric] = {
                "status": "included",
                "coverageRatio": alignment.coverage_ratio,
                "alignedSampleCount": len(alignment.timestamps),
                "alignmentTolerance": alignment.tolerance,
                "correlationStatus": (
                    "success" if correlation["valid"] else "insufficient_data"
                ),
                "correlationReason": correlation["reason"],
            }

        base_event = {
            "targetAbnormalRange": copy.deepcopy(dict(event["targetAbnormalRange"])),
            "rawTargetAbnormalRange": {
                "startTime": raw_start,
                "endTime": raw_end,
            },
            "analysisWindow": analysis_window,
            "candidateMetricCount": len(lower_metrics),
            "validCandidateMetricCount": len(alignments),
            "excludedMetrics": sorted(
                excluded, key=lambda item: (item["metric"], item["reason"])
            ),
            "candidateDiagnostics": candidate_diagnostics,
        }
        if not alignments:
            return {
                **base_event,
                "status": "no_candidate_metrics",
                "alignedSampleCount": 0,
                "randomForestStatus": "insufficient_data",
                "randomForestDiagnostics": {"reason": "no_candidate_metrics"},
                "topAssociations": [],
                "allAssociations": [],
            }

        rf_result = _random_forest_importances(alignments, self.config)
        valid_correlations = {
            metric: float(correlation["correlationStrength"])
            for metric, correlation in correlations.items()
            if correlation["valid"]
        }
        combined = combine_evidence(
            list(alignments),
            valid_correlations,
            rf_result["importances"],
            self.config["weights"],
        )
        if combined["status"] == "insufficient_data":
            return {
                **base_event,
                "status": "insufficient_data",
                "alignedSampleCount": int(rf_result["sampleCount"]),
                "randomForestStatus": rf_result["status"],
                "randomForestDiagnostics": _rf_diagnostics(rf_result),
                "topAssociations": [],
                "allAssociations": [],
            }
        event_status = combined["status"]
        if event_status == "success" and rf_result["status"] != "success":
            event_status = "partial_success"

        importance_method = rf_result["importanceMethod"]
        associations: list[dict[str, Any]] = []
        for metric, score_details in combined["scores"].items():
            correlation = correlations[metric]
            alignment = alignments[metric]
            associations.append(
                {
                    "metric": metric,
                    "abnormalContribution": float(score_details["score"] * 100.0),
                    "pearson": correlation["pearson"],
                    "spearman": correlation["spearman"],
                    "selectedCorrelation": correlation["selectedCorrelation"],
                    "selectedCorrelationMethod": correlation[
                        "selectedCorrelationMethod"
                    ],
                    "correlationDirection": correlation["correlationDirection"],
                    "correlationStrength": correlation["correlationStrength"],
                    "normalizedCorrelationContribution": float(
                        score_details["normalizedCorrelationContribution"]
                    ),
                    "randomForestImportance": (
                        float(rf_result["importances"][metric])
                        if metric in rf_result["importances"]
                        else None
                    ),
                    "normalizedRandomForestContribution": float(
                        score_details["normalizedRandomForestContribution"]
                    ),
                    "randomForestImportanceMethod": (
                        importance_method
                        if metric in rf_result["importances"]
                        else None
                    ),
                    "coverageRatio": float(alignment.coverage_ratio),
                    "alignedSampleCount": len(alignment.timestamps),
                }
            )
        associations.sort(
            key=lambda item: (-item["abnormalContribution"], item["metric"])
        )
        for rank, association in enumerate(associations, start=1):
            association["rank"] = rank
        top_k = int(self.config["top_k"])
        return {
            **base_event,
            "status": event_status,
            "alignedSampleCount": int(rf_result["sampleCount"]),
            "randomForestStatus": rf_result["status"],
            "randomForestDiagnostics": _rf_diagnostics(rf_result),
            "topAssociations": copy.deepcopy(associations[:top_k]),
            "allAssociations": associations,
        }


def _rf_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if key not in {"importances", "status"}
    }


def _target_status(status: str, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "events": [],
        "topAssociations": [],
    }
    if reason is not None:
        result["reason"] = reason
    return result
