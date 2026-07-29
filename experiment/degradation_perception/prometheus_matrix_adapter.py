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

"""Strict adapter for offline Prometheus ``query_range`` matrix packages.

The package format handled here is an experiment-only container around full
Prometheus responses. This module performs structural validation, explicit
label-series selection, and scalar sample cleaning. It deliberately contains
no KDE, interval, correlation, or random-forest logic.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
SIMULATION_SOURCE = "simulated_prometheus_query_range"
PHASES = ("standard", "inference")
SERIES_POLICIES = frozenset({"exactly_one", "select_by_labels"})
_PACKAGE_FIELDS = frozenset(
    {"formatVersion", "source", "queryStepSeconds", *PHASES}
)
_ENTRY_FIELDS = frozenset(
    {"query", "seriesPolicy", "selectLabels", "response"}
)


class PrometheusMatrixError(ValueError):
    """Structured validation failure for one package, response, or series."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.details = _json_safe_detail(dict(details or {}))
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible public representation."""

        return {
            "code": self.code,
            "message": self.message,
            "details": copy.deepcopy(self.details),
        }


def _json_safe_detail(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_detail(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_detail(item) for item in value]
    return repr(value)


def _raise(
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    merged = dict(details or {})
    merged.update(extra)
    raise PrometheusMatrixError(code, message, details=merged)


def _context(
    logical_metric: str,
    phase: str,
    query: str,
    query_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "logicalMetric": logical_metric,
        "phase": phase,
        "query": query,
    }
    if query_window is not None:
        context["queryWindow"] = copy.deepcopy(dict(query_window))
    return context


def _validate_string_labels(
    value: Any,
    *,
    context: Mapping[str, Any],
    field_name: str,
    code: str,
    require_nonempty: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _raise(code, f"{field_name} must be an object", context)
    labels = dict(value)
    if require_nonempty and not labels:
        _raise(code, f"{field_name} must not be empty", context)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or (require_nonempty and not item)
        for key, item in labels.items()
    ):
        _raise(
            code,
            f"{field_name} must be a dictionary of strings",
            context,
        )
    return labels


def _validate_selection(
    series_policy: Any,
    select_labels: Any,
    *,
    context: Mapping[str, Any],
) -> tuple[str, dict[str, str] | None]:
    if not isinstance(series_policy, str):
        _raise(
            "invalid_series_policy",
            f"seriesPolicy must be one of {sorted(SERIES_POLICIES)}",
            context,
            seriesPolicyType=type(series_policy).__name__,
        )
    if series_policy not in SERIES_POLICIES:
        _raise(
            "invalid_series_policy",
            f"seriesPolicy must be one of {sorted(SERIES_POLICIES)}",
            context,
            seriesPolicy=series_policy,
        )
    policy = str(series_policy)
    if policy == "exactly_one":
        if select_labels is not None:
            _raise(
                "invalid_series_policy",
                "selectLabels is only valid with select_by_labels",
                context,
                seriesPolicy=policy,
            )
        return policy, None
    labels = _validate_string_labels(
        select_labels,
        context=context,
        field_name="selectLabels",
        code="invalid_series_policy",
        require_nonempty=True,
    )
    return policy, labels


def validate_simulation_package(payload: Any) -> dict[str, Any]:
    """Validate and return a normalized deep copy of an offline package."""

    if not isinstance(payload, Mapping):
        _raise(
            "invalid_simulation_package",
            "simulation package root must be an object",
        )
    unknown = set(payload) - _PACKAGE_FIELDS
    missing = _PACKAGE_FIELDS - set(payload)
    if missing:
        _raise(
            "invalid_simulation_package",
            f"simulation package is missing fields: {sorted(missing)}",
        )
    if unknown:
        _raise(
            "invalid_simulation_package",
            f"simulation package contains unsupported fields: {sorted(unknown)}",
        )
    version = payload["formatVersion"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        _raise(
            "invalid_simulation_package",
            f"formatVersion must be {FORMAT_VERSION}",
        )
    if payload["source"] != SIMULATION_SOURCE:
        _raise(
            "invalid_simulation_package",
            f"source must be {SIMULATION_SOURCE!r}",
        )
    step = payload["queryStepSeconds"]
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(float(step))
        or float(step) <= 0
    ):
        _raise(
            "invalid_simulation_package",
            "queryStepSeconds must be a positive finite number",
        )

    normalized = copy.deepcopy(dict(payload))
    for phase in PHASES:
        section = normalized[phase]
        if not isinstance(section, Mapping):
            _raise(
                "invalid_simulation_package",
                f"{phase} must be an object keyed by logical metric",
            )
        phase_data: dict[str, dict[str, Any]] = {}
        for logical_metric, raw_entry in section.items():
            if not isinstance(logical_metric, str) or not logical_metric:
                _raise(
                    "invalid_simulation_package",
                    f"{phase} metric keys must be non-empty strings",
                )
            entry_context = {
                "logicalMetric": logical_metric,
                "phase": phase,
            }
            if not isinstance(raw_entry, Mapping):
                _raise(
                    "invalid_simulation_package",
                    "metric package entry must be an object",
                    entry_context,
                )
            unknown_entry = set(raw_entry) - _ENTRY_FIELDS
            missing_entry = {"query", "response"} - set(raw_entry)
            if missing_entry:
                _raise(
                    "invalid_simulation_package",
                    f"metric package entry is missing fields: {sorted(missing_entry)}",
                    entry_context,
                )
            if unknown_entry:
                _raise(
                    "invalid_simulation_package",
                    "metric package entry contains unsupported fields: "
                    f"{sorted(unknown_entry)}",
                    entry_context,
                )
            query = raw_entry["query"]
            if not isinstance(query, str) or not query.strip():
                _raise(
                    "invalid_simulation_package",
                    "query must be a non-empty string",
                    entry_context,
                )
            context = _context(logical_metric, phase, query)
            policy, select_labels = _validate_selection(
                raw_entry.get("seriesPolicy", "exactly_one"),
                raw_entry.get("selectLabels"),
                context=context,
            )
            if not isinstance(raw_entry["response"], Mapping):
                _raise(
                    "invalid_simulation_package",
                    "response must be a complete Prometheus response object",
                    context,
                )
            entry = copy.deepcopy(dict(raw_entry))
            entry["seriesPolicy"] = policy
            if select_labels is not None:
                entry["selectLabels"] = select_labels
            phase_data[logical_metric] = entry
        normalized[phase] = phase_data
    return normalized


def _validate_response_envelope(
    response: Any,
    *,
    context: Mapping[str, Any],
) -> list[Any]:
    if not isinstance(response, Mapping):
        _raise(
            "invalid_matrix_response",
            "Prometheus response must be an object",
            context,
        )
    if "status" not in response:
        _raise(
            "invalid_matrix_response",
            "Prometheus response is missing status",
            context,
        )
    status = response.get("status")
    if status != "success":
        _raise(
            "query_failed",
            "Prometheus response status must be 'success'",
            context,
            status=status,
            errorType=response.get("errorType"),
            error=response.get("error"),
        )
    data = response.get("data")
    if not isinstance(data, Mapping):
        _raise(
            "invalid_matrix_response",
            "Prometheus response data must be an object",
            context,
        )
    if data.get("resultType") != "matrix":
        _raise(
            "invalid_matrix_response",
            "Prometheus data.resultType must be 'matrix'",
            context,
            resultType=data.get("resultType"),
        )
    result = data.get("result")
    if not isinstance(result, list):
        _raise(
            "invalid_matrix_response",
            "Prometheus data.result must be a list",
            context,
        )
    return result


def _validate_series_headers(
    result: list[Any],
    *,
    context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    series_list: list[dict[str, Any]] = []
    labels_list: list[dict[str, str]] = []
    for index, raw_series in enumerate(result):
        series_context = {**context, "seriesIndex": index}
        if not isinstance(raw_series, Mapping):
            _raise(
                "invalid_series",
                "each Prometheus result series must be an object",
                series_context,
            )
        if "metric" not in raw_series:
            _raise(
                "invalid_series",
                "Prometheus result series is missing metric labels",
                series_context,
            )
        labels = _validate_string_labels(
            raw_series["metric"],
            context=series_context,
            field_name="metric",
            code="invalid_series",
        )
        if "values" not in raw_series and "histograms" not in raw_series:
            _raise(
                "invalid_series",
                "Prometheus result series is missing scalar values",
                series_context,
            )
        series_list.append(dict(raw_series))
        labels_list.append(labels)
    return series_list, labels_list


def convert_matrix_response(
    response: Any,
    logical_metric: str,
    phase: str,
    query: str,
    series_policy: str = "exactly_one",
    select_labels: Mapping[str, str] | None = None,
    query_window: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Select and clean exactly one scalar series from a matrix response."""

    if not isinstance(logical_metric, str) or not logical_metric:
        _raise("invalid_matrix_response", "logical metric must be non-empty")
    if phase not in PHASES:
        _raise(
            "invalid_matrix_response",
            f"phase must be one of {list(PHASES)}",
            {"logicalMetric": logical_metric, "phase": phase},
        )
    if not isinstance(query, str) or not query.strip():
        _raise(
            "invalid_matrix_response",
            "query must be a non-empty string",
            {"logicalMetric": logical_metric, "phase": phase},
        )
    context = _context(logical_metric, phase, query, query_window)
    policy, selected_filter = _validate_selection(
        series_policy,
        select_labels,
        context=context,
    )
    result = _validate_response_envelope(response, context=context)
    series_list, labels_list = _validate_series_headers(
        result,
        context=context,
    )

    selected_indexes = list(range(len(series_list)))
    if policy == "select_by_labels":
        assert selected_filter is not None
        selected_indexes = [
            index
            for index, labels in enumerate(labels_list)
            if all(labels.get(key) == value for key, value in selected_filter.items())
        ]
    selected_labels = [labels_list[index] for index in selected_indexes]
    selection_details: dict[str, Any] = {
        **context,
        "seriesPolicy": policy,
        "returnedSeriesCount": len(series_list),
        "returnedSeriesLabels": copy.deepcopy(labels_list),
        "matchingSeriesLabels": copy.deepcopy(selected_labels),
        "seriesLabels": copy.deepcopy(selected_labels),
    }
    if selected_filter is not None:
        selection_details["selectLabels"] = copy.deepcopy(selected_filter)
    if not selected_indexes:
        code = (
            "no_matching_series"
            if policy == "select_by_labels"
            else "no_series"
        )
        _raise(
            code,
            "Prometheus query returned no series matching the configured "
            "labels"
            if policy == "select_by_labels"
            else "Prometheus query returned no series",
            selection_details,
        )
    if len(selected_indexes) > 1:
        code = (
            "multiple_matching_series"
            if policy == "select_by_labels"
            else "multiple_series"
        )
        _raise(
            code,
            "configured labels matched multiple Prometheus series"
            if policy == "select_by_labels"
            else "Prometheus query returned multiple series",
            selection_details,
        )

    selected_index = selected_indexes[0]
    series = series_list[selected_index]
    labels = labels_list[selected_index]
    histogram_samples_present = False
    if "histograms" in series:
        histograms = series["histograms"]
        if not isinstance(histograms, list):
            _raise(
                "invalid_series",
                "native histogram samples must be a list",
                {**context, "seriesIndex": selected_index},
            )
        histogram_samples_present = bool(histograms)
    if "values" not in series:
        _raise(
            "unsupported_native_histogram",
            "native histogram-only series is unsupported; query a scalar series",
            {
                **context,
                "seriesIndex": selected_index,
                "selectedLabels": labels,
                "nativeHistogramSamplesPresent": True,
            },
        )
    samples = series["values"]
    if not isinstance(samples, list):
        _raise(
            "invalid_series",
            "Prometheus scalar values must be a list",
            {**context, "seriesIndex": selected_index},
        )

    last_values: dict[float, float] = {}
    valid_sample_count = 0
    filtered_nonfinite_count = 0
    for sample_index, sample in enumerate(samples):
        sample_context = {
            **context,
            "seriesIndex": selected_index,
            "sampleIndex": sample_index,
        }
        if not isinstance(sample, list) or len(sample) != 2:
            _raise(
                "invalid_sample",
                "each scalar sample must be [timestamp, value]",
                sample_context,
            )
        raw_timestamp, raw_value = sample
        if (
            isinstance(raw_timestamp, bool)
            or not isinstance(raw_timestamp, (int, float))
        ):
            _raise(
                "invalid_sample",
                "sample timestamp must be a Unix-seconds number",
                sample_context,
            )
        timestamp = float(raw_timestamp)
        if not math.isfinite(timestamp):
            _raise(
                "invalid_sample",
                "sample timestamp must be finite",
                sample_context,
            )
        if not isinstance(raw_value, str):
            _raise(
                "invalid_sample",
                "sample value must be a numeric string",
                sample_context,
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            _raise(
                "invalid_sample",
                "sample value must be a numeric string",
                sample_context,
                value=raw_value,
            )
        if not math.isfinite(value):
            filtered_nonfinite_count += 1
            continue
        valid_sample_count += 1
        last_values[timestamp] = value

    if not last_values:
        _raise(
            "empty_series",
            "scalar series has no finite samples after cleaning",
            {
                **context,
                "selectedLabels": labels,
                "inputSampleCount": len(samples),
                "filteredNonFiniteCount": filtered_nonfinite_count,
            },
        )
    ordered = sorted(last_values.items(), key=lambda item: item[0])
    converted = {
        "timestamps": [timestamp for timestamp, _ in ordered],
        "values": [value for _, value in ordered],
    }
    diagnostics = {
        **context,
        "seriesPolicy": policy,
        "selectLabels": copy.deepcopy(selected_filter),
        "returnedSeriesCount": len(series_list),
        "returnedSeriesLabels": copy.deepcopy(labels_list),
        "selectedLabels": copy.deepcopy(labels),
        "inputSampleCount": len(samples),
        "validSampleCount": valid_sample_count,
        "outputPointCount": len(ordered),
        "filteredNonFiniteCount": filtered_nonfinite_count,
        "duplicateTimestampCount": valid_sample_count - len(ordered),
        "nativeHistogramSamplesPresent": histogram_samples_present,
    }
    json.dumps(
        {"series": converted, "diagnostics": diagnostics},
        allow_nan=False,
    )
    return converted, diagnostics


def convert_simulation_package(
    payload: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert every package entry to the existing two-phase algorithm input."""

    package = validate_simulation_package(payload)
    dataset: dict[str, dict[str, dict[str, list[float]]]] = {
        phase: {} for phase in PHASES
    }
    phase_diagnostics: dict[str, dict[str, dict[str, Any]]] = {
        phase: {} for phase in PHASES
    }
    for phase in PHASES:
        for logical_metric, entry in package[phase].items():
            converted, diagnostics = convert_matrix_response(
                entry["response"],
                logical_metric,
                phase,
                entry["query"],
                entry["seriesPolicy"],
                entry.get("selectLabels"),
            )
            dataset[phase][logical_metric] = converted
            phase_diagnostics[phase][logical_metric] = diagnostics
    diagnostics = {
        "formatVersion": FORMAT_VERSION,
        "source": SIMULATION_SOURCE,
        "queryStepSeconds": float(package["queryStepSeconds"]),
        "phases": phase_diagnostics,
    }
    json.dumps(
        {"dataset": dataset, "diagnostics": diagnostics},
        allow_nan=False,
    )
    return dataset, diagnostics


def load_simulation_package(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read one UTF-8 JSON simulation package and return its normalized copy."""

    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _raise(
            "simulation_package_io_error",
            f"failed to read simulation package: {exc}",
            {"path": str(source)},
        )
    try:
        payload = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        _raise(
            "invalid_simulation_package",
            f"simulation package is not valid JSON: {exc.msg}",
            {"path": str(source), "line": exc.lineno, "column": exc.colno},
        )
    except ValueError as exc:
        _raise(
            "invalid_simulation_package",
            f"simulation package is not strict JSON: {exc}",
            {"path": str(source)},
        )
    return validate_simulation_package(payload)
