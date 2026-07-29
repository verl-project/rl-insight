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

"""Orchestration for inference-performance degradation perception."""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .association_analysis import AssociationAnalyzer, resolve_association_config
from .config_loader import (
    COMMON_CONFIG_PATH,
    get_default_config_dir,
    load_common_config,
    load_metric_config,
)
from .interval_utils import build_candidate_intervals, validate_candidate_interval
from .kde_utils import adaptive_kde, kde_quantile
from .normalization import normalize_data
from .perception_config import (
    DEFAULT_METRIC,
    SUPPORTED_ABNORMAL_TYPES,
    SUPPORTED_SOURCE_TYPES,
    MetricState,
    ThresholdModel,
    TimeSeries,
)
from .preprocessing import extract_metric_series, load_dataset
from .serialization import to_json_serializable
from .stable_segment_detector import StableSegmentDetector
from .time_utils import (
    adjust_time_bounds,
    infer_display_time_mode,
    to_display_time,
)


def get_standard_data(dataset: Mapping[str, Any], metric: str) -> TimeSeries:
    """Extract one aligned, finite, sorted, and deduplicated baseline series."""

    return extract_metric_series(dataset, "standard", metric)


def _validate_threshold_config(config: Mapping[str, Any]) -> tuple[float, float, float]:
    alpha = float(config.get("alpha", 0.01))
    upper_ratio = float(config.get("upper_ratio", 1.15))
    lower_ratio = float(config.get("lower_ratio", 1.15))
    if not math.isfinite(alpha) or not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be finite and between 0 and 0.5")
    if not math.isfinite(upper_ratio) or upper_ratio < 1:
        raise ValueError("upper_ratio must be finite and at least 1")
    if not math.isfinite(lower_ratio) or lower_ratio < 1:
        raise ValueError("lower_ratio must be finite and at least 1")
    return alpha, upper_ratio, lower_ratio


def build_standard_data(
    timestamps: Sequence[float],
    values: Sequence[float],
    config: Mapping[str, Any],
) -> list[ThresholdModel]:
    """Build one independent KDE threshold model per trusted stable segment."""

    if len(timestamps) != len(values):
        raise ValueError("standard timestamps and values must have equal lengths")
    if len(values) < 3:
        return []
    alpha, upper_ratio, lower_ratio = _validate_threshold_config(config)
    normalized_values = normalize_data(values, config.get("normalization"))
    detector = StableSegmentDetector(config)
    segments = detector.detect_stable_segments(timestamps, normalized_values)
    models: list[ThresholdModel] = []

    for mode_id, segment in enumerate(segments):
        if len(segment.values) < 2 or not segment.timestamps:
            continue
        result = adaptive_kde(segment.values, config)
        lower_base = kde_quantile(result, alpha)
        upper_base = kde_quantile(result, 1.0 - alpha)
        # Ratios are outward-expansion factors. Multiplying both bounds moves a
        # positive lower bound inward (and a negative upper bound inward), which
        # makes normal data fail DOWN/BOTH or UP/BOTH classification. Apply the
        # factor according to the bound's sign so both sides always move away
        # from the fitted KDE interval.
        lower_threshold = (
            lower_base / lower_ratio
            if lower_base >= 0
            else lower_base * lower_ratio
        )
        upper_threshold = (
            upper_base * upper_ratio
            if upper_base >= 0
            else upper_base / upper_ratio
        )
        if not all(
            math.isfinite(item)
            for item in (
                lower_base,
                upper_base,
                lower_threshold,
                upper_threshold,
            )
        ):
            raise ValueError("stable-segment KDE produced a non-finite threshold")
        if lower_threshold > upper_threshold:
            raise ValueError(
                "configured ratios invert the stable-segment threshold interval"
            )
        models.append(
            ThresholdModel(
                mode_id=mode_id,
                segment_start_time=float(segment.timestamps[0]),
                segment_end_time=float(segment.timestamps[-1]),
                point_count=len(segment.values),
                lower_kde_threshold=lower_base,
                upper_kde_threshold=upper_base,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                bandwidth=result.bandwidth,
                diagnostics={
                    "flags": segment.flags,
                    "means": segment.means,
                    "standardDeviations": segment.standard_deviations,
                    "influenceRegion": segment.influence_region,
                    "zeroRangeJittered": result.zero_range_jittered,
                    "jitterScale": result.jitter_scale,
                },
            )
        )
    return models


def _clip_series(
    series: TimeSeries,
    start_time: float | None,
    end_time: float | None,
) -> TimeSeries:
    timestamps: list[float] = []
    values: list[float] = []
    for index in range(len(series.timestamps)):
        timestamp = float(series.timestamps[index])
        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            continue
        timestamps.append(timestamp)
        values.append(float(series.values[index]))
    return TimeSeries(timestamps=timestamps, values=values)


def _classify_value(
    timestamp: float,
    value: float,
    models: Sequence[ThresholdModel],
    abnormal_type: str,
) -> tuple[bool, dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    matched_mode: int | None = None
    for model in models:
        if abnormal_type == "UP":
            compatible = value <= model.upper_threshold
        elif abnormal_type == "DOWN":
            compatible = value >= model.lower_threshold
        else:
            if model.lower_threshold > model.upper_threshold:
                raise ValueError(
                    f"mode {model.mode_id} has lower threshold above upper threshold"
                )
            compatible = model.lower_threshold <= value <= model.upper_threshold
        evaluations.append(
            {
                "modeId": model.mode_id,
                "compatible": bool(compatible),
                "lowerThreshold": model.lower_threshold,
                "upperThreshold": model.upper_threshold,
            }
        )
        if compatible and matched_mode is None:
            matched_mode = model.mode_id

    abnormal = matched_mode is None
    if abnormal_type == "UP":
        effective_threshold: Any = max(model.upper_threshold for model in models)
    elif abnormal_type == "DOWN":
        effective_threshold = min(model.lower_threshold for model in models)
    else:
        effective_threshold = [
            {
                "modeId": model.mode_id,
                "lower": model.lower_threshold,
                "upper": model.upper_threshold,
            }
            for model in models
        ]
    return abnormal, {
        "timestamp": timestamp,
        "value": value,
        "matchedNormalMode": matched_mode,
        "threshold": effective_threshold,
        "abnormalType": abnormal_type,
        "abnormal": abnormal,
        "modeEvaluations": evaluations,
    }


def _deduplicate_history_ranges(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[float, float, str], dict[str, Any]] = {}
    for record in records:
        for interval in record.get("abnormalTimeRange", []):
            key = (
                float(interval["startTime"]),
                float(interval["endTime"]),
                str(interval.get("abnormalType", "")),
            )
            unique[key] = dict(interval)
    return sorted(
        unique.values(),
        key=lambda item: (float(item["startTime"]), float(item["endTime"])),
    )


class DegradationPerception:
    """Coordinate per-metric baseline modelling and current-window detection."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        metrics: Sequence[str] | str | None = None,
        task_id: str | None = None,
        source_type: str = "training_log",
        config_dir: str | os.PathLike[str] | None = None,
        common_config_path: str | os.PathLike[str] | None = COMMON_CONFIG_PATH,
        dataset: Mapping[str, Any] | None = None,
        association_targets: Sequence[str] | str | None = None,
    ) -> None:
        self.path = None if path is None else os.fspath(path)
        self.start_time = None if start_time is None else float(start_time)
        self.end_time = None if end_time is None else float(end_time)
        if self.start_time is not None and not math.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if self.end_time is not None and not math.isfinite(self.end_time):
            raise ValueError("end_time must be finite")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time > self.end_time
        ):
            raise ValueError("start_time must not exceed end_time")

        raw_metrics = (
            [metrics]
            if isinstance(metrics, str)
            else list(metrics or [DEFAULT_METRIC])
        )
        self.metrics = list(dict.fromkeys(str(metric) for metric in raw_metrics))
        if not self.metrics or any(not metric.strip() for metric in self.metrics):
            raise ValueError("metrics must contain at least one non-empty name")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"unsupported source_type: {source_type!r}")
        if dataset is not None and not isinstance(dataset, Mapping):
            raise TypeError("dataset must be a mapping")
        if isinstance(association_targets, str):
            raw_association_targets: Sequence[str] | None = [
                association_targets
            ]
        elif association_targets is None:
            raw_association_targets = None
        else:
            raw_association_targets = list(association_targets)
        if raw_association_targets is None:
            self.association_targets: list[str] | None = None
        else:
            self.association_targets = list(
                dict.fromkeys(str(item) for item in raw_association_targets)
            )
            if not self.association_targets or any(
                not item.strip() for item in self.association_targets
            ):
                raise ValueError(
                    "association_targets must contain non-empty metric names"
                )

        self.task_id = "default" if task_id is None else str(task_id)
        self.source_type = source_type
        self.config_dir = Path(
            get_default_config_dir() if config_dir is None else config_dir
        )
        self.common_config_path = Path(
            COMMON_CONFIG_PATH
            if common_config_path is None
            else common_config_path
        )
        self.dataset = dataset
        self.states: dict[str, int] = {}
        self.config_dict: dict[str, dict[str, Any]] = {}
        self.history: dict[tuple[str, str], deque[dict[str, Any]]] = {}
        self.standard_models: dict[str, list[ThresholdModel]] = {}
        self._standard_model_configs: dict[str, dict[str, Any]] = {}
        self.common_config = load_common_config(self.common_config_path)
        self._validate_history_config()

    def _validate_history_config(self) -> None:
        try:
            n_keep_result = int(self.common_config.get("n_keep_result", 1))
            n_keep_abnormal = int(self.common_config.get("n_keep_abnormal", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("history counts must be integers") from exc
        if n_keep_result <= 0:
            raise ValueError("n_keep_result must be positive")
        if not 1 <= n_keep_abnormal <= n_keep_result:
            raise ValueError(
                "n_keep_abnormal must be between 1 and n_keep_result"
            )
        self.n_keep_result = n_keep_result
        self.n_keep_abnormal = n_keep_abnormal

    def get_standard_data(
        self, dataset: Mapping[str, Any], metric: str
    ) -> TimeSeries:
        """Programmatic counterpart of the required get_standard_data step."""

        return get_standard_data(dataset, metric)

    def build_standard_data(
        self,
        timestamps: Sequence[float],
        values: Sequence[float],
        config: Mapping[str, Any],
    ) -> list[ThresholdModel]:
        """Programmatic counterpart of the required build_standard_data step."""

        return build_standard_data(timestamps, values, config)

    def detect(self) -> dict[str, Any]:
        """Load the configured source, detect every metric, and return JSON data."""

        if self.dataset is not None:
            dataset = self.dataset
        elif self.path is not None:
            dataset = load_dataset(
                self.path,
                self.metrics,
                start_time=self.start_time,
                end_time=self.end_time,
            )
        else:
            raise ValueError("either path or dataset must be provided")
        return self._detect_loaded_dataset(
            dataset,
            metrics=self.metrics,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    def detect_dataset(
        self,
        dataset: Mapping[str, Any],
        *,
        metrics: Sequence[str] | str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, Any]:
        """Detect an already fetched dataset without invoking the CLI entry point."""

        if not isinstance(dataset, Mapping):
            raise TypeError("dataset must be a mapping")
        if isinstance(metrics, str):
            selected_metrics = [metrics]
        elif metrics is None:
            selected_metrics = self.metrics
        else:
            selected_metrics = list(metrics)
        selected_metrics = list(
            dict.fromkeys(str(metric) for metric in selected_metrics)
        )
        if not selected_metrics or any(not item.strip() for item in selected_metrics):
            raise ValueError("metrics must contain at least one non-empty name")
        selected_start = self.start_time if start_time is None else float(start_time)
        selected_end = self.end_time if end_time is None else float(end_time)
        if selected_start is not None and not math.isfinite(selected_start):
            raise ValueError("start_time must be finite")
        if selected_end is not None and not math.isfinite(selected_end):
            raise ValueError("end_time must be finite")
        if (
            selected_start is not None
            and selected_end is not None
            and selected_start > selected_end
        ):
            raise ValueError("start_time must not exceed end_time")
        return self._detect_loaded_dataset(
            dataset,
            metrics=selected_metrics,
            start_time=selected_start,
            end_time=selected_end,
        )

    def _detect_loaded_dataset(
        self,
        dataset: Mapping[str, Any],
        *,
        metrics: Sequence[str],
        start_time: float | None,
        end_time: float | None,
    ) -> dict[str, Any]:
        states: dict[str, int] = {}
        results: dict[str, dict[str, Any]] = {}
        confirmed_ranges: dict[str, list[dict[str, Any]]] = {}
        metric_errors: dict[str, dict[str, str]] = {}
        metric_configs: dict[str, dict[str, Any]] = {}
        metric_context: dict[str, dict[str, Any]] = {}

        for metric in metrics:
            error_stage = "configuration"
            try:
                config = load_metric_config(metric, config_dir=self.config_dir)
                self.config_dict[metric] = config
                metric_configs[metric] = config
                abnormal_type = str(config.get("abnormal_type", "UP")).upper()
                if abnormal_type not in SUPPORTED_ABNORMAL_TYPES:
                    raise ValueError(
                        f"unsupported abnormal_type for {metric!r}: {abnormal_type!r}"
                    )
                minimum_standard = int(config.get("minimum_standard_points", 3))
                minimum_inference = int(config.get("minimum_inference_points", 5))
                if minimum_standard <= 0 or minimum_inference <= 0:
                    raise ValueError("minimum data counts must be positive")

                inference_section = dataset.get("inference")
                metric_context[metric] = {
                    "input_present": (
                        isinstance(inference_section, Mapping)
                        and metric in inference_section
                    ),
                    "series": None,
                    "raw_events": [],
                }
                error_stage = "input"
                standard = self.get_standard_data(dataset, metric)
                inference = _clip_series(
                    extract_metric_series(dataset, "inference", metric),
                    start_time,
                    end_time,
                )
                metric_context[metric]["series"] = inference
                error_stage = "detection"
                detection_config = {
                    key: value for key, value in config.items()
                    if key != "association"
                }
                cached_models = self.standard_models.get(metric, [])
                cache_is_valid = bool(cached_models) and (
                    self._standard_model_configs.get(metric) == detection_config
                )
                if len(standard) < minimum_standard and not cache_is_valid:
                    state = MetricState.STANDARD_DATA_INSUFFICIENT
                    states[metric] = int(state)
                    results[metric] = self._state_result(
                        state,
                        "standard data is insufficient",
                    )
                    confirmed_ranges[metric] = []
                    continue
                if len(standard) >= minimum_standard:
                    models = self.build_standard_data(
                        standard.timestamps,
                        standard.values,
                        config,
                    )
                else:
                    models = list(cached_models)
                if not models:
                    state = MetricState.STANDARD_DATA_INSUFFICIENT
                    states[metric] = int(state)
                    results[metric] = self._state_result(
                        state,
                        "standard data contains no trusted stable segment",
                    )
                    confirmed_ranges[metric] = []
                    continue
                if len(standard) >= minimum_standard:
                    self.standard_models[metric] = list(models)
                    self._standard_model_configs[metric] = detection_config
                if len(inference) < minimum_inference:
                    state = MetricState.INFERENCE_DATA_INSUFFICIENT
                    states[metric] = int(state)
                    results[metric] = self._state_result(
                        state,
                        "inference data is insufficient",
                    )
                    confirmed_ranges[metric] = []
                    continue

                normalized_inference = normalize_data(
                    inference.values,
                    config.get("normalization"),
                )
                abnormal_flags: list[bool] = []
                point_diagnostics: list[dict[str, Any]] = []
                for index in range(len(inference.timestamps)):
                    abnormal, diagnostic = _classify_value(
                        float(inference.timestamps[index]),
                        float(normalized_inference[index]),
                        models,
                        abnormal_type,
                    )
                    abnormal_flags.append(abnormal)
                    point_diagnostics.append(diagnostic)

                current_ranges, raw_event_records = self._formal_interval_records(
                    inference,
                    abnormal_flags,
                    config,
                    abnormal_type,
                    start_time,
                    end_time,
                )
                published_ranges, history_confirmed, abnormal_history_count = (
                    self._update_history(metric, current_ranges)
                )
                contextual_events: list[dict[str, Any]] = []
                for published_range in published_ranges:
                    current_record = next(
                        (
                            record
                            for record in raw_event_records
                            if record["targetAbnormalRange"] == published_range
                        ),
                        None,
                    )
                    contextual_events.append(
                        current_record
                        if current_record is not None
                        else {"targetAbnormalRange": dict(published_range)}
                    )
                metric_context[metric]["raw_events"] = contextual_events
                metric_context[metric]["abnormal_flags"] = abnormal_flags
                state = MetricState.NORMAL
                states[metric] = int(state)
                confirmed_ranges[metric] = published_ranges
                if published_ranges:
                    message = "performance degradation detected"
                elif current_ranges:
                    message = "degradation interval is awaiting history confirmation"
                else:
                    message = ""
                results[metric] = {
                    "state": int(state),
                    "message": message,
                    "thresholds": models,
                    "abnormalTimeRange": published_ranges,
                    "currentAbnormalTimeRange": current_ranges,
                    "historyConfirmed": history_confirmed,
                    "abnormalHistoryCount": abnormal_history_count,
                    "pointDiagnostics": point_diagnostics,
                }
            except Exception as exc:  # Per-metric isolation is a required behavior.
                error_codes = {
                    "configuration": "metric_config_error",
                    "input": "metric_input_error",
                    "detection": "metric_detection_error",
                }
                error_messages = {
                    "configuration": (
                        "metric configuration could not be loaded or validated"
                    ),
                    "input": "metric input could not be validated",
                    "detection": "metric detection raised an internal error",
                }
                error = {
                    "code": error_codes[error_stage],
                    "type": type(exc).__name__,
                    "message": error_messages[error_stage],
                }
                metric_errors[metric] = error
                confirmed_ranges[metric] = []
                results[metric] = {
                    "message": error["message"],
                    "thresholds": [],
                    "abnormalTimeRange": [],
                    "error": error,
                }

        for metric in metric_errors:
            self.states.pop(metric, None)
        self.states.update(states)
        response = {
            "taskId": self.task_id,
            "states": states,
            "results": results,
            "abnormalTimeRange": confirmed_ranges,
        }
        if metric_errors:
            response["metricErrors"] = metric_errors
        try:
            association_config = resolve_association_config(
                metrics,
                metric_configs,
                self.association_targets,
            )
            if association_config is not None:
                response["associationAnalysis"] = AssociationAnalyzer(
                    association_config,
                    source_type=self.source_type,
                ).analyze(metric_context, response)
        except Exception as exc:
            # Association is post-processing: expose its failure without
            # changing any completed per-metric KDE state or interval.
            response["associationAnalysis"] = {
                "enabled": True,
                "status": "analysis_error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "targets": {},
            }
        return to_json_serializable(response)

    @staticmethod
    def _state_result(state: MetricState, message: str) -> dict[str, Any]:
        return {
            "state": int(state),
            "message": message,
            "thresholds": [],
            "abnormalTimeRange": [],
        }

    def _formal_intervals(
        self,
        inference: TimeSeries,
        abnormal_flags: Sequence[bool],
        config: Mapping[str, Any],
        abnormal_type: str,
        requested_start: float | None,
        requested_end: float | None,
    ) -> list[dict[str, Any]]:
        formal, _ = self._formal_interval_records(
            inference,
            abnormal_flags,
            config,
            abnormal_type,
            requested_start,
            requested_end,
        )
        return formal

    def _formal_interval_records(
        self,
        inference: TimeSeries,
        abnormal_flags: Sequence[bool],
        config: Mapping[str, Any],
        abnormal_type: str,
        requested_start: float | None,
        requested_end: float | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build public ranges plus private raw boundaries from the same labels."""

        interval_config = config.get("abnormal_interval", {})
        candidates = build_candidate_intervals(
            inference.timestamps,
            abnormal_flags,
            interval_config,
        )
        display_time_mode = infer_display_time_mode(
            inference.timestamps,
            self.source_type,
        )
        formal: list[dict[str, Any]] = []
        raw_events: list[dict[str, Any]] = []
        for candidate in candidates:
            valid, details = validate_candidate_interval(candidate, interval_config)
            if not valid:
                continue
            adjusted_start, adjusted_end = adjust_time_bounds(
                requested_start,
                requested_end,
                inference.timestamps,
                int(candidate["start_index"]),
                int(candidate["stop_index"]),
            )
            display_start = to_display_time(
                adjusted_start,
                self.source_type,
                mode=display_time_mode,
            )
            display_end = to_display_time(
                adjusted_end,
                self.source_type,
                mode=display_time_mode,
            )
            if display_start > display_end:
                raise ValueError("converted display interval is reversed")
            public_range = {
                "startTime": display_start,
                "endTime": display_end,
                "duration": float(candidate["duration"]),
                "totalPointCount": int(candidate["total_point_count"]),
                "abnormalPointCount": int(candidate["abnormal_point_count"]),
                "abnormalRate": float(candidate["abnormal_rate"]),
                "abnormalType": abnormal_type,
                "validationDetail": details,
                "maximumAllowedGap": float(candidate["maximum_allowed_gap"]),
            }
            formal.append(public_range)
            raw_events.append(
                {
                    "rawStartTime": float(candidate["start_time"]),
                    "rawEndTime": float(candidate["end_time"]),
                    "startIndex": int(candidate["start_index"]),
                    "stopIndex": int(candidate["stop_index"]),
                    "targetAbnormalRange": public_range,
                }
            )
        return formal, raw_events

    def _update_history(
        self,
        metric: str,
        current_ranges: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool, int]:
        key = (self.task_id, metric)
        history = self.history.get(key)
        if history is None or history.maxlen != self.n_keep_result:
            previous = [] if history is None else list(history)
            history = deque(previous[-self.n_keep_result :], maxlen=self.n_keep_result)
            self.history[key] = history
        history.append(
            {
                "detectionTime": datetime.now(timezone.utc),
                "hasAbnormal": bool(current_ranges),
                "abnormalTimeRange": [dict(item) for item in current_ranges],
            }
        )
        abnormal_count = sum(bool(record["hasAbnormal"]) for record in history)
        confirmed = abnormal_count >= self.n_keep_abnormal
        ranges = _deduplicate_history_ranges(list(history)) if confirmed else []
        return ranges, confirmed, abnormal_count
