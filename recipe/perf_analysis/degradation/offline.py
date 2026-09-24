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

"""Run the existing degradation algorithm once over offline data, plus its JSON persistence."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .association import (
    AssociationItem,
    AssociationParameters,
    AssociationResult,
    analyze_association,
)
from .baseline import (
    BaselineParameters,
    NormalRange,
    SeriesBaseline,
    fit_baselines,
)
from .detector import (
    DegradationEvent,
    Direction,
    EventTracker,
    PointResult,
    Policy,
    classify_frame,
)
from .data import (
    SeriesId,
    StepFrame,
    TimeSeries,
    build_step_frames,
    load_time_series,
)
from .metrics import (
    CANDIDATE_CATEGORIES,
    CANDIDATE_METRICS,
    CATEGORY_BY_METRIC,
    GLOBAL_STEP_METRIC,
    TARGET_METRICS,
)

BASELINE_STEPS = 30
PRE_CONTEXT_STEPS = 30
TOP_K = 25


class OfflineAnalysisError(RuntimeError):
    """The supplied offline batch cannot complete the analysis."""


def _step_number(value: float) -> int:
    step = round(value)
    if not math.isclose(value, step, rel_tol=0.0, abs_tol=1.0e-9):
        raise OfflineAnalysisError("global-step values must be integers")
    return step


def _global_step_series(series: Sequence[TimeSeries]) -> TimeSeries:
    matches = [item for item in series if item.identity.name == GLOBAL_STEP_METRIC]
    if len(matches) != 1:
        raise OfflineAnalysisError(
            f"{GLOBAL_STEP_METRIC!r} must have exactly one concrete series, "
            f"got {len(matches)}"
        )
    return matches[0]


def _step_span(global_steps: TimeSeries) -> tuple[int, int]:
    """Return the first complete step after range start and the final boundary."""

    if not global_steps.samples:
        raise OfflineAnalysisError("global-step series is empty")
    steps = [_step_number(sample.value) for sample in global_steps.samples]
    return min(steps) + 1, max(steps)


def _dominant_direction(
    context: Sequence[Mapping[SeriesId, PointResult]],
    identity: SeriesId,
    *,
    start_step: int,
    end_step: int,
) -> str:
    counts = Counter(
        point.direction.value
        for row in context
        if (point := row[identity]).valid
        and point.abnormal
        and point.direction is not None
        and start_step <= point.step <= end_step
    )
    if not counts:
        return Direction.NORMAL.value
    maximum = max(counts.values())
    leaders = [direction for direction, count in counts.items() if count == maximum]
    return leaders[0] if len(leaders) == 1 else "MIXED"


class OfflineAnalyzer:
    """Stateful event tracker used only while one offline batch is processed."""

    def __init__(
        self,
        baselines: Mapping[SeriesId, SeriesBaseline],
        *,
        association_parameters: AssociationParameters | None = None,
    ) -> None:
        targets = frozenset(TARGET_METRICS)
        candidates = frozenset(CANDIDATE_METRICS)
        self.baselines = {
            identity: baseline
            for identity, baseline in baselines.items()
            if identity.name in targets or identity.name in candidates
        }
        self.policies = {
            identity: Policy.UP if identity.name in targets else Policy.BOTH
            for identity in self.baselines
        }
        target_ids = [
            identity
            for identity, policy in self.policies.items()
            if policy is Policy.UP
        ]
        candidate_ids = [
            identity
            for identity, policy in self.policies.items()
            if policy is Policy.BOTH
        ]
        if not target_ids:
            raise OfflineAnalysisError("the baseline contains no configured target")
        if not candidate_ids:
            raise OfflineAnalysisError("the baseline contains no configured candidate")
        self.trackers = {identity: EventTracker(identity) for identity in target_ids}
        self.association_parameters = association_parameters or AssociationParameters()
        self.recent: deque[dict[SeriesId, PointResult]] = deque(
            maxlen=PRE_CONTEXT_STEPS + 5
        )
        self.active_contexts: dict[SeriesId, list[dict[SeriesId, PointResult]]] = {}
        self.events: list[dict[str, Any]] = []
        self._last_step: int | None = None

    def _association(
        self,
        event: DegradationEvent,
        context: Sequence[Mapping[SeriesId, PointResult]],
    ) -> dict[str, Any]:
        result = analyze_association(
            [row[event.identity] for row in context],
            {
                identity: [row[identity] for row in context]
                for identity, policy in self.policies.items()
                if policy is Policy.BOTH
            },
            event_start_step=event.start_step,
            parameters=self.association_parameters,
        )
        top = result.associations[:TOP_K]
        directions = {
            item.identity: _dominant_direction(
                context,
                item.identity,
                start_step=event.start_step,
                end_step=event.end_step,
            )
            for item in top
        }
        return association_json(result, top, directions)

    def process(self, frames: Sequence[StepFrame]) -> None:
        for frame in frames:
            if self._last_step is not None and frame.step != self._last_step + 1:
                raise OfflineAnalysisError(
                    f"detection frames must be consecutive: expected "
                    f"{self._last_step + 1}, got {frame.step}"
                )
            results = classify_frame(frame, self.baselines, self.policies)
            for context in self.active_contexts.values():
                context.append(results)
            self.recent.append(results)

            for identity, tracker in sorted(self.trackers.items()):
                update = tracker.update(results[identity])
                if update.confirmed_event is not None:
                    event = update.confirmed_event
                    first_step = event.start_step - PRE_CONTEXT_STEPS
                    context = [
                        row for row in self.recent if row[identity].step >= first_step
                    ]
                    self.active_contexts[identity] = context
                if update.closed_event is not None:
                    event = update.closed_event
                    context = self.active_contexts.pop(identity)
                    self.events.append(
                        event_json(
                            event,
                            self._association(event, context),
                            phase="closed",
                        )
                    )
            self._last_step = frame.step

    def finalize_range_end(self) -> None:
        """Analyze confirmed events that remain open at the selected range end."""

        for identity, tracker in sorted(self.trackers.items()):
            event = tracker.snapshot_open_event()
            if event is None:
                continue
            context = self.active_contexts.pop(identity)
            self.events.append(
                event_json(
                    event,
                    self._association(event, context),
                    phase="open_at_range_end",
                )
            )


def analyze_offline(
    data_dir: Path,
    *,
    start_time: float,
    end_time: float,
    baseline_path: Path,
    output_path: Path,
    promtool_path: Path | None = None,
) -> dict[str, Any]:
    """Read one TSDB range and analyze each final event once."""

    configured_names = {GLOBAL_STEP_METRIC, *TARGET_METRICS, *CANDIDATE_METRICS}
    all_series = load_time_series(
        data_dir,
        start_time=start_time,
        end_time=end_time,
        metric_names=frozenset(configured_names),
        promtool_path=promtool_path,
    )
    selected_series = tuple(
        item for item in all_series if item.identity.name in configured_names
    )
    global_steps = _global_step_series(selected_series)
    first_step, final_boundary = _step_span(global_steps)
    metrics = [
        item for item in selected_series if item.identity != global_steps.identity
    ]

    if baseline_path.exists():
        snapshot = load_baseline(baseline_path)
        baseline_action = "loaded"
    else:
        if final_boundary - first_step < BASELINE_STEPS:
            raise OfflineAnalysisError(
                f"training a baseline requires {BASELINE_STEPS} complete steps; "
                f"the batch contains {final_boundary - first_step}"
            )
        baseline_frames = build_step_frames(
            global_steps,
            metrics,
            start_step=first_step,
            step_count=BASELINE_STEPS,
        )
        fitted = fit_baselines(baseline_frames, parameters=BaselineParameters())
        target_count = sum(
            identity.name in TARGET_METRICS for identity in fitted.baselines
        )
        candidate_count = sum(
            identity.name in CANDIDATE_METRICS for identity in fitted.baselines
        )
        if not target_count or not candidate_count:
            raise OfflineAnalysisError(
                "baseline fitting requires at least one usable configured target "
                "and one usable configured candidate"
            )
        snapshot = BaselineSnapshot(
            start_step=baseline_frames[0].step,
            end_step=baseline_frames[-1].step,
            global_step_identity=global_steps.identity,
            parameters=BaselineParameters(),
            baselines=fitted.baselines,
        )
        save_baseline(baseline_path, snapshot)
        baseline_action = "trained"

    analyzer = OfflineAnalyzer(snapshot.baselines)
    detection_start = max(first_step, snapshot.end_step + 1)
    detection_count = max(0, final_boundary - detection_start)
    if detection_count:
        analysis_metrics = [
            item for item in metrics if item.identity in analyzer.baselines
        ]
        analyzer.process(
            build_step_frames(
                global_steps,
                analysis_metrics,
                start_step=detection_start,
                step_count=detection_count,
            )
        )
        analyzer.finalize_range_end()
    save_events(output_path, analyzer.events)
    return {
        "status": "ok",
        "baseline": {
            "action": baseline_action,
            "path": str(baseline_path.resolve()),
            "start_step": snapshot.start_step,
            "end_step": snapshot.end_step,
            "series_count": len(snapshot.baselines),
        },
        "analysis": {
            "data_dir": str(data_dir.expanduser().resolve()),
            "start_time": start_time,
            "end_time": end_time,
            "detection_start_step": detection_start if detection_count else None,
            "detection_end_step": final_boundary - 1 if detection_count else None,
            "processed_step_count": detection_count,
            "event_count": len(analyzer.events),
            "result_path": str(output_path.resolve()),
        },
        "events": present_events(analyzer.events),
    }


# --- JSON persistence: Minimal JSON persistence for an offline baseline and analysis result. ---

SCHEMA_VERSION = 1


class StateError(RuntimeError):
    """A baseline or result file cannot be read or written."""


@dataclass(frozen=True)
class BaselineSnapshot:
    start_step: int
    end_step: int
    global_step_identity: SeriesId
    parameters: BaselineParameters
    baselines: dict[SeriesId, SeriesBaseline]


def _identity_json(identity: SeriesId) -> dict[str, Any]:
    return {"metric": identity.name, "labels": dict(identity.labels)}


def _identity_from_json(value: object) -> SeriesId:
    if not isinstance(value, Mapping):
        raise StateError("series identity must be an object")
    metric = value.get("metric")
    labels = value.get("labels", {})
    if not isinstance(metric, str) or not metric or not isinstance(labels, Mapping):
        raise StateError("series identity is invalid")
    label_set: dict[str, str] = {"__name__": metric}
    for key, label_value in labels.items():
        if not isinstance(key, str) or not isinstance(label_value, str):
            raise StateError("series labels must be strings")
        label_set[key] = label_value
    return SeriesId.from_label_set(label_set)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise StateError(f"cannot write {path}: {exc}") from exc


def save_baseline(path: Path, snapshot: BaselineSnapshot) -> None:
    series = []
    for identity, baseline in sorted(snapshot.baselines.items()):
        series.append(
            {
                **_identity_json(identity),
                "sample_count": baseline.sample_count,
                "ranges": [asdict(normal_range) for normal_range in baseline.ranges],
            }
        )
    _write(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "baseline": {
                "start_step": snapshot.start_step,
                "end_step": snapshot.end_step,
                "global_step": _identity_json(snapshot.global_step_identity),
                "parameters": asdict(snapshot.parameters),
                "series": series,
            },
        },
    )


def load_baseline(path: Path) -> BaselineSnapshot:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload["baseline"]
        parameters = BaselineParameters(**raw["parameters"])
        baselines: dict[SeriesId, SeriesBaseline] = {}
        for item in raw["series"]:
            identity = _identity_from_json(item)
            baselines[identity] = SeriesBaseline(
                identity=identity,
                sample_count=int(item["sample_count"]),
                ranges=tuple(NormalRange(**value) for value in item["ranges"]),
            )
        snapshot = BaselineSnapshot(
            start_step=int(raw["start_step"]),
            end_step=int(raw["end_step"]),
            global_step_identity=_identity_from_json(raw["global_step"]),
            parameters=parameters,
            baselines=baselines,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise StateError(f"cannot load baseline {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION or not snapshot.baselines:
        raise StateError(f"baseline {path} has an unsupported or empty schema")
    return snapshot


def association_json(
    result: AssociationResult,
    top_associations: Sequence[AssociationItem],
    directions: Mapping[SeriesId, str],
) -> dict[str, Any]:
    return {
        "status": result.status,
        "top_k": [
            {
                **_identity_json(item.identity),
                "direction": directions[item.identity],
                "association_percent": item.association_score * 100.0,
                "pearson": item.pearson,
                "spearman": item.spearman,
                "selected_correlation": item.selected_correlation,
                "selected_correlation_method": item.selected_correlation_method,
                "correlation_share": item.correlation_share,
                "random_forest_eligible": item.random_forest_eligible,
                "random_forest_importance": item.random_forest_importance,
                "random_forest_share": item.random_forest_share,
                "aligned_sample_count": item.aligned_sample_count,
                "coverage_ratio": item.coverage_ratio,
            }
            for item in top_associations
        ],
        "effective_correlation_weight": result.effective_correlation_weight,
        "effective_random_forest_weight": result.effective_random_forest_weight,
        "random_forest_status": result.random_forest_status,
        "random_forest_reason": result.random_forest_reason,
        "random_forest_sample_count": result.random_forest_sample_count,
        "random_forest_validation_score": result.random_forest_validation_score,
        "skipped_candidates": [
            {**_identity_json(item.identity), "reason": item.reason}
            for item in result.skipped_candidates
        ],
    }


def event_json(
    event: DegradationEvent,
    association: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Serialize one final event view and its single whole-event analysis."""

    if phase not in {"closed", "open_at_range_end"}:
        raise ValueError("phase must be closed or open_at_range_end")

    return {
        "target": event.identity.name,
        "target_labels": dict(event.identity.labels),
        "direction": event.direction.value,
        "start_step": event.start_step,
        "start_time": event.start_time,
        "end_step": event.end_step,
        "end_time": event.end_time,
        "confirmed_at_step": event.confirmed_at_step,
        "confirmed_at_time": event.confirmed_at_time,
        "closed_at_step": event.closed_at_step,
        "closed_at_time": event.closed_at_time,
        "abnormal_points": event.abnormal_points,
        "event_state": phase,
        "association": {phase: dict(association)},
    }


def save_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    _write(path, {"schema_version": SCHEMA_VERSION, "events": list(events)})


def _group_top_k(top_k: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    category_order = {name: index for index, name in enumerate(CANDIDATE_CATEGORIES)}
    groups: dict[str, list[dict[str, Any]]] = {}
    for rank, raw in enumerate(top_k, start=1):
        item = dict(raw)
        item["global_rank"] = rank
        category = CATEGORY_BY_METRIC.get(str(item["metric"]), "uncategorized")
        groups.setdefault(category, []).append(item)
    ordered = sorted(
        groups.items(),
        key=lambda pair: (
            pair[1][0]["global_rank"],
            category_order.get(pair[0], len(category_order)),
        ),
    )
    return [
        {
            "category": category,
            "distinct_metric_count": len({item["metric"] for item in metrics}),
            "best_global_rank": metrics[0]["global_rank"],
            "metrics": metrics,
        }
        for category, metrics in ordered
    ]


def present_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Present closed and range-end-open event association results."""

    presented = []
    for event in json.loads(json.dumps(list(events))):
        phase = str(event.get("event_state", "closed"))
        association = event["association"].get(phase)
        if association is None:
            continue
        association["top_k_by_category"] = _group_top_k(association.pop("top_k"))
        event["association"] = {phase: association}
        presented.append(event)
    return presented


__all__ = [
    "BaselineSnapshot",
    "OfflineAnalysisError",
    "StateError",
    "analyze_offline",
    "association_json",
    "event_json",
    "load_baseline",
    "present_events",
    "save_baseline",
    "save_events",
]
