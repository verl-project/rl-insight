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

"""Rank metrics associated with one confirmed target degradation event."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]
from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-untyped]

from .detector import PointResult, Policy
from .data import SeriesId


class AssociationError(ValueError):
    """The supplied event or point history cannot be analyzed safely."""


@dataclass(frozen=True)
class AssociationParameters:
    """Configurable association and random-forest parameters."""

    correlation_weight: float = 0.85
    random_forest_weight: float = 0.15
    min_aligned_points: int = 10
    min_rf_samples: int = 30
    min_coverage_ratio: float = 0.6
    n_estimators: int = 200
    class_weight: str | None = "balanced"
    random_state: int = 42
    train_fraction: float = 0.7
    permutation_repeats: int = 10
    n_jobs: int = 1

    def __post_init__(self) -> None:
        nonnegative_integers = ("random_state",)
        positive_integers = (
            "min_aligned_points",
            "min_rf_samples",
            "n_estimators",
            "permutation_repeats",
        )
        for name in (*nonnegative_integers, *positive_integers, "n_jobs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in nonnegative_integers:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        for name in positive_integers:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must not be zero")

        numeric = (
            "correlation_weight",
            "random_forest_weight",
            "min_coverage_ratio",
            "train_fraction",
        )
        for name in numeric:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.correlation_weight < 0 or self.random_forest_weight < 0:
            raise ValueError("association weights must not be negative")
        if not math.isclose(
            self.correlation_weight + self.random_forest_weight,
            1.0,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError("association weights must sum to 1")
        if not 0.0 <= self.min_coverage_ratio <= 1.0:
            raise ValueError("min_coverage_ratio must be between 0 and 1")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.class_weight is not None and not isinstance(self.class_weight, str):
            raise TypeError("class_weight must be a string or None")


@dataclass(frozen=True)
class CandidateExclusion:
    """A candidate omitted before evidence scoring."""

    identity: SeriesId
    reason: str


@dataclass(frozen=True)
class AssociationItem:
    """Correlation and random-forest evidence for one ranked candidate."""

    identity: SeriesId
    association_score: float
    pearson: float | None
    spearman: float | None
    selected_correlation: float | None
    selected_correlation_method: str | None
    correlation_share: float
    random_forest_eligible: bool
    random_forest_importance: float | None
    random_forest_share: float
    aligned_sample_count: int
    coverage_ratio: float


@dataclass(frozen=True)
class AssociationResult:
    """Complete candidate ranking for one runtime-selected event window."""

    status: str
    effective_correlation_weight: float
    effective_random_forest_weight: float
    random_forest_status: str
    random_forest_reason: str | None
    random_forest_sample_count: int
    random_forest_validation_score: float | None
    skipped_candidates: tuple[CandidateExclusion, ...]
    associations: tuple[AssociationItem, ...]


@dataclass(frozen=True)
class _CorrelationEvidence:
    valid: bool
    pearson: float | None
    spearman: float | None
    selected: float | None
    method: str | None
    strength: float


@dataclass(frozen=True)
class _CandidateEvidence:
    identity: SeriesId
    steps: tuple[int, ...]
    target_values: tuple[float, ...]
    target_labels: tuple[bool, ...]
    candidate_values: tuple[float, ...]
    candidate_labels: tuple[bool, ...]
    coverage_ratio: float
    abnormal_in_event: bool
    correlation: _CorrelationEvidence


@dataclass(frozen=True)
class _RandomForestEvidence:
    status: str
    reason: str | None
    importances: Mapping[SeriesId, float]
    sample_count: int
    validation_score: float | None


DEFAULT_ASSOCIATION_PARAMETERS = AssociationParameters()


def _point_map(
    points: Sequence[PointResult],
    identity: SeriesId,
    policy: Policy,
) -> dict[int, PointResult]:
    result: dict[int, PointResult] = {}
    for point in points:
        if not isinstance(point, PointResult):
            raise TypeError("point histories must contain PointResult values")
        if point.identity != identity:
            raise AssociationError("point identity does not match its history")
        if point.policy is not policy:
            raise AssociationError(
                f"{identity.name!r} association points require policy {policy.value}"
            )
        if point.step in result:
            raise AssociationError(f"duplicate point for step {point.step}")
        result[point.step] = point
    return result


def _valid_value(point: PointResult | None) -> float | None:
    if point is None or point.value is None:
        return None
    value = float(point.value)
    return value if math.isfinite(value) else None


def _invalid_correlation() -> _CorrelationEvidence:
    return _CorrelationEvidence(
        valid=False,
        pearson=None,
        spearman=None,
        selected=None,
        method=None,
        strength=0.0,
    )


def _compute_correlation(
    target_values: Sequence[float],
    candidate_values: Sequence[float],
) -> _CorrelationEvidence:
    target = np.asarray(target_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    if target.shape != candidate.shape:
        raise AssociationError("aligned target and candidate shapes must match")
    finite = np.isfinite(target) & np.isfinite(candidate)
    target = target[finite]
    candidate = candidate[finite]
    if target.size < 2:
        return _invalid_correlation()
    if np.ptp(target) == 0:
        return _invalid_correlation()
    if np.ptp(candidate) == 0:
        return _invalid_correlation()

    try:
        pearson = float(pearsonr(target, candidate).statistic)
    except ValueError:
        pearson = math.nan
    try:
        spearman = float(spearmanr(target, candidate).statistic)
    except ValueError:
        spearman = math.nan
    valid = [
        (method, value)
        for method, value in (("pearson", pearson), ("spearman", spearman))
        if math.isfinite(value)
    ]
    if not valid:
        return _invalid_correlation()
    values = dict(valid)
    if len(values) == 2 and math.isclose(
        abs(values["pearson"]),
        abs(values["spearman"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        method, selected = "pearson", values["pearson"]
    else:
        method, selected = max(valid, key=lambda item: abs(item[1]))
    return _CorrelationEvidence(
        valid=True,
        pearson=pearson if math.isfinite(pearson) else None,
        spearman=spearman if math.isfinite(spearman) else None,
        selected=selected,
        method=method,
        strength=abs(selected),
    )


def _candidate_evidence(
    identity: SeriesId,
    points: Mapping[int, PointResult],
    target_points: Mapping[int, PointResult],
    *,
    analysis_start: int,
    analysis_end: int,
    event_start: int,
    valid_target_count: int,
    parameters: AssociationParameters,
) -> tuple[_CandidateEvidence | None, CandidateExclusion | None]:
    common_steps: list[int] = []
    target_values: list[float] = []
    target_labels: list[bool] = []
    candidate_values: list[float] = []
    candidate_labels: list[bool] = []
    for step in range(analysis_start, analysis_end + 1):
        target_point = target_points.get(step)
        candidate_point = points.get(step)
        target_value = _valid_value(target_point)
        candidate_value = _valid_value(candidate_point)
        if target_value is None or candidate_value is None:
            continue
        common_steps.append(step)
        target_values.append(target_value)
        target_labels.append(bool(target_point and target_point.abnormal))
        candidate_values.append(candidate_value)
        candidate_labels.append(bool(candidate_point and candidate_point.abnormal))

    coverage = len(common_steps) / valid_target_count if valid_target_count else 0.0
    if coverage < parameters.min_coverage_ratio:
        return None, CandidateExclusion(identity, "insufficient_coverage")
    if len(common_steps) < parameters.min_aligned_points:
        return None, CandidateExclusion(identity, "insufficient_aligned_points")
    if np.ptp(np.asarray(candidate_values, dtype=float)) == 0:
        return None, CandidateExclusion(identity, "constant_candidate_series")

    abnormal_in_event = any(
        event_start <= step <= analysis_end
        and (point := points.get(step)) is not None
        and point.valid
        and point.abnormal
        for step in range(event_start, analysis_end + 1)
    )
    return (
        _CandidateEvidence(
            identity=identity,
            steps=tuple(common_steps),
            target_values=tuple(target_values),
            target_labels=tuple(target_labels),
            candidate_values=tuple(candidate_values),
            candidate_labels=tuple(candidate_labels),
            coverage_ratio=float(coverage),
            abnormal_in_event=abnormal_in_event,
            correlation=_compute_correlation(target_values, candidate_values),
        ),
        None,
    )


def _rf_failure(
    reason: str,
    *,
    sample_count: int = 0,
) -> _RandomForestEvidence:
    return _RandomForestEvidence(
        status="insufficient_data",
        reason=reason,
        importances={},
        sample_count=sample_count,
        validation_score=None,
    )


def _random_forest_evidence(
    candidates: Sequence[_CandidateEvidence],
    parameters: AssociationParameters,
) -> _RandomForestEvidence:
    active = sorted(
        (candidate for candidate in candidates if candidate.abnormal_in_event),
        key=lambda candidate: candidate.identity,
    )
    if not active:
        return _rf_failure("no_abnormal_candidates")

    candidate_maps = {
        candidate.identity: dict(zip(candidate.steps, candidate.candidate_labels))
        for candidate in active
    }
    target_maps = {
        candidate.identity: dict(zip(candidate.steps, candidate.target_labels))
        for candidate in active
    }
    common_steps = set(candidate_maps[active[0].identity])
    for candidate in active[1:]:
        common_steps.intersection_update(candidate_maps[candidate.identity])

    if len(common_steps) < parameters.min_rf_samples:
        return _rf_failure(
            "insufficient_common_samples",
            sample_count=len(common_steps),
        )

    ordered_steps = sorted(common_steps)
    target_labels = np.asarray(
        [target_maps[active[0].identity][step] for step in ordered_steps],
        dtype=int,
    )
    if np.unique(target_labels).size < 2:
        return _rf_failure(
            "single_target_class",
            sample_count=len(ordered_steps),
        )

    matrix = np.asarray(
        [
            [candidate_maps[candidate.identity][step] for candidate in active]
            for step in ordered_steps
        ],
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
            sample_count=len(ordered_steps),
        )

    model_candidates = [active[index] for index in variable_columns]
    matrix = matrix[:, variable_columns]
    split = max(
        1,
        min(
            len(target_labels) - 1, int(len(target_labels) * parameters.train_fraction)
        ),
    )
    training_labels = target_labels[:split]
    validation_labels = target_labels[split:]
    if np.unique(training_labels).size < 2 or np.unique(validation_labels).size < 2:
        return _rf_failure(
            "time_split_lacks_both_classes",
            sample_count=len(ordered_steps),
        )

    try:
        model = RandomForestClassifier(
            n_estimators=parameters.n_estimators,
            class_weight=parameters.class_weight,
            random_state=parameters.random_state,
            n_jobs=parameters.n_jobs,
        )
        model.fit(matrix[:split], training_labels)
        prediction = model.predict(matrix[split:])
        validation_score = float(balanced_accuracy_score(validation_labels, prediction))
        permutation = permutation_importance(
            model,
            matrix[split:],
            validation_labels,
            scoring="balanced_accuracy",
            n_repeats=parameters.permutation_repeats,
            random_state=parameters.random_state,
            n_jobs=parameters.n_jobs,
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return _RandomForestEvidence(
            status="insufficient_data",
            reason=f"random_forest_failed: {exc}",
            importances={},
            sample_count=len(ordered_steps),
            validation_score=None,
        )

    raw = np.asarray(permutation.importances_mean, dtype=float)
    clipped = np.maximum(np.where(np.isfinite(raw), raw, 0.0), 0.0)
    if float(np.sum(clipped)) <= 0:
        return _rf_failure(
            "all_importances_zero",
            sample_count=len(ordered_steps),
        )
    importances = {candidate.identity: 0.0 for candidate in active}
    for index, candidate in enumerate(model_candidates):
        importances[candidate.identity] = float(clipped[index])
    return _RandomForestEvidence(
        status="success",
        reason=None,
        importances=importances,
        sample_count=len(ordered_steps),
        validation_score=validation_score,
    )


def _rank_associations(
    candidates: Sequence[_CandidateEvidence],
    random_forest: _RandomForestEvidence,
    parameters: AssociationParameters,
) -> tuple[str, float, float, tuple[AssociationItem, ...]]:
    correlation_total = sum(
        candidate.correlation.strength
        for candidate in candidates
        if candidate.correlation.valid
    )
    random_forest_total = sum(random_forest.importances.values())
    correlation_valid = correlation_total > 0
    random_forest_valid = random_forest_total > 0
    if correlation_valid and random_forest_valid:
        status = "success"
        correlation_weight = parameters.correlation_weight
        random_forest_weight = parameters.random_forest_weight
    elif correlation_valid:
        status = "partial_success"
        correlation_weight = 1.0
        random_forest_weight = 0.0
    elif random_forest_valid:
        status = "partial_success"
        correlation_weight = 0.0
        random_forest_weight = 1.0
    else:
        return "insufficient_data", 0.0, 0.0, ()

    rows: list[tuple[_CandidateEvidence, float, float, float]] = []
    for candidate in candidates:
        correlation_share = (
            candidate.correlation.strength / correlation_total
            if correlation_valid and candidate.correlation.valid
            else 0.0
        )
        random_forest_share = (
            random_forest.importances.get(candidate.identity, 0.0) / random_forest_total
            if random_forest_valid
            else 0.0
        )
        score = (
            correlation_weight * correlation_share
            + random_forest_weight * random_forest_share
        )
        rows.append((candidate, correlation_share, random_forest_share, score))
    rows.sort(key=lambda row: (-row[3], row[0].identity))

    associations = tuple(
        AssociationItem(
            identity=candidate.identity,
            association_score=float(score),
            pearson=candidate.correlation.pearson,
            spearman=candidate.correlation.spearman,
            selected_correlation=candidate.correlation.selected,
            selected_correlation_method=candidate.correlation.method,
            correlation_share=float(correlation_share),
            random_forest_eligible=candidate.abnormal_in_event,
            random_forest_importance=random_forest.importances.get(candidate.identity),
            random_forest_share=float(random_forest_share),
            aligned_sample_count=len(candidate.steps),
            coverage_ratio=candidate.coverage_ratio,
        )
        for (
            candidate,
            correlation_share,
            random_forest_share,
            score,
        ) in rows
    )
    return status, correlation_weight, random_forest_weight, associations


def analyze_association(
    target_points: Sequence[PointResult],
    candidate_points: Mapping[SeriesId, Sequence[PointResult]],
    *,
    event_start_step: int,
    parameters: AssociationParameters = DEFAULT_ASSOCIATION_PARAMETERS,
) -> AssociationResult:
    """Rank candidates in a runtime-selected target-event window."""

    if not isinstance(parameters, AssociationParameters):
        raise TypeError("parameters must be AssociationParameters")
    if isinstance(event_start_step, bool) or not isinstance(event_start_step, int):
        raise TypeError("event_start_step must be an integer")
    if not target_points:
        raise AssociationError("target history must not be empty")
    target_identity = target_points[0].identity
    target_map = _point_map(target_points, target_identity, Policy.UP)
    target_steps = sorted(target_map)
    if target_steps != list(range(target_steps[0], target_steps[-1] + 1)):
        raise AssociationError("target history must contain consecutive steps")
    analysis_start, analysis_end = target_steps[0], target_steps[-1]
    if not analysis_start <= event_start_step <= analysis_end:
        raise AssociationError("event_start_step must be inside the analysis window")
    valid_target_count = sum(
        _valid_value(target_map[step]) is not None for step in target_steps
    )

    evidence: list[_CandidateEvidence] = []
    exclusions: list[CandidateExclusion] = []
    for identity in sorted(candidate_points):
        if not isinstance(identity, SeriesId):
            raise TypeError("candidate identities must be SeriesId values")
        if identity == target_identity:
            raise AssociationError("target series must not be a candidate")
        points = _point_map(candidate_points[identity], identity, Policy.BOTH)
        candidate, exclusion = _candidate_evidence(
            identity,
            points,
            target_map,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            event_start=event_start_step,
            valid_target_count=valid_target_count,
            parameters=parameters,
        )
        if candidate is not None:
            evidence.append(candidate)
        if exclusion is not None:
            exclusions.append(exclusion)

    random_forest = _random_forest_evidence(evidence, parameters)
    status, correlation_weight, random_forest_weight, associations = _rank_associations(
        evidence, random_forest, parameters
    )
    return AssociationResult(
        status=status,
        effective_correlation_weight=correlation_weight,
        effective_random_forest_weight=random_forest_weight,
        random_forest_status=random_forest.status,
        random_forest_reason=random_forest.reason,
        random_forest_sample_count=random_forest.sample_count,
        random_forest_validation_score=random_forest.validation_score,
        skipped_candidates=tuple(sorted(exclusions, key=lambda item: item.identity)),
        associations=associations,
    )


__all__ = [
    "DEFAULT_ASSOCIATION_PARAMETERS",
    "AssociationError",
    "AssociationItem",
    "AssociationParameters",
    "AssociationResult",
    "CandidateExclusion",
    "analyze_association",
]
