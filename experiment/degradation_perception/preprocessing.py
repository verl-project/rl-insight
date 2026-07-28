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

"""Deterministic input contracts and aligned time-series preprocessing.

Local CLI input is one UTF-8 JSON object with explicit ``standard`` and
``inference`` sections. A metric entry may use the canonical timestamps/values
shape or one complete Prometheus ``query_range`` response. Incremental remote
input may use JSON Lines, but every row must explicitly identify its phase. No
phase is inferred from a filename or from the presence or absence of a time
bound.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .perception_config import TimeSeries

CANONICAL_PHASES = ("standard", "inference")
CANONICAL_SERIES_FIELDS = frozenset({"timestamps", "values"})


class DataValidationError(ValueError):
    """Raised when input violates the deterministic dataset contract."""


def _sequence_to_list(value: Any, *, field_name: str) -> list[Any]:
    """Convert supported array-like values without testing ndarray truthiness."""

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return value.tolist()
    if isinstance(value, (str, bytes, bytearray)) or isinstance(value, Mapping):
        raise DataValidationError(f"{field_name} must be an array")
    if not isinstance(value, Sequence):
        raise DataValidationError(f"{field_name} must be an array")
    return list(value)


def prometheus_query_range_to_series(
    payload: Any,
    *,
    location: str = "Prometheus query_range response",
) -> dict[str, list[Any]]:
    """Convert one scalar Prometheus range-query result to a canonical series.

    A caller must provide separate responses for the explicit ``standard`` and
    ``inference`` phases. At most one label series is accepted: silently merging
    multiple series would make equal timestamps overwrite each other and would
    give the KDE an undefined business meaning. Use PromQL aggregation or label
    selection before calling this adapter.
    """

    if not isinstance(payload, Mapping):
        raise DataValidationError(f"{location} must be an object")
    status = payload.get("status")
    if status != "success":
        error_type = payload.get("errorType")
        error = payload.get("error")
        detail = ": ".join(
            str(item) for item in (error_type, error) if item not in (None, "")
        )
        suffix = f": {detail}" if detail else ""
        raise DataValidationError(
            f"{location} status must be 'success'{suffix}"
        )

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise DataValidationError(f"{location}.data must be an object")
    if data.get("resultType") != "matrix":
        raise DataValidationError(
            f"{location}.data.resultType must be 'matrix'"
        )
    result = _sequence_to_list(
        data.get("result"), field_name=f"{location}.data.result"
    )
    if len(result) > 1:
        raise DataValidationError(
            f"{location} returned {len(result)} label series; expected at most "
            "one after PromQL aggregation or label selection"
        )
    if not result:
        return {"timestamps": [], "values": []}

    series = result[0]
    if not isinstance(series, Mapping):
        raise DataValidationError(f"{location}.data.result[0] must be an object")
    if "metric" in series and not isinstance(series["metric"], Mapping):
        raise DataValidationError(
            f"{location}.data.result[0].metric must be an object"
        )
    if "histograms" in series:
        histograms = _sequence_to_list(
            series["histograms"],
            field_name=f"{location}.data.result[0].histograms",
        )
        if histograms:
            raise DataValidationError(
                f"{location} contains native histogram samples; query a scalar "
                "series before KDE detection"
            )
    if "values" not in series:
        raise DataValidationError(
            f"{location}.data.result[0] is missing scalar values"
        )

    samples = _sequence_to_list(
        series["values"],
        field_name=f"{location}.data.result[0].values",
    )
    timestamps: list[Any] = []
    values: list[Any] = []
    for index, sample in enumerate(samples):
        pair = _sequence_to_list(
            sample,
            field_name=f"{location}.data.result[0].values[{index}]",
        )
        if len(pair) != 2:
            raise DataValidationError(
                f"{location}.data.result[0].values[{index}] must contain "
                "timestamp and value"
            )
        timestamps.append(pair[0])
        values.append(pair[1])
    return {"timestamps": timestamps, "values": values}


def _numeric_timestamp(value: Any) -> float:
    """Convert supported timestamps to one sortable numeric representation.

    Numeric timestamps retain their supplied unit. Datetimes are converted to
    Unix seconds; naive datetimes are treated as UTC to avoid host-timezone
    dependent results.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean is not a timestamp")
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("timestamp is NaT")
        return float(value.astype("datetime64[ns]").astype(np.int64)) / 1_000_000_000
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return float(aware.timestamp())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp is empty")
        try:
            return float(text)
        except ValueError:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            aware = (
                parsed
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=timezone.utc)
            )
            return float(aware.timestamp())
    return float(value)


def _numeric_value(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean is not a metric value")
    return float(value)


def deduplicate_by_timestamp(
    timestamps: Sequence[float], values: Sequence[float]
) -> TimeSeries:
    """Stable-sort aligned pairs and keep the last value for each timestamp."""

    if len(timestamps) != len(values):
        raise DataValidationError(
            "timestamps and values must have equal lengths before deduplication"
        )
    # Python sorting is stable. For equal timestamps, assignment below therefore
    # retains the last value from the original input order.
    last_values: dict[float, float] = {}
    for timestamp, value in sorted(
        zip(timestamps, values), key=lambda item: item[0]
    ):
        last_values[float(timestamp)] = float(value)
    ordered = sorted(last_values.items(), key=lambda item: item[0])
    return TimeSeries(
        timestamps=[item[0] for item in ordered],
        values=[item[1] for item in ordered],
    )


def preprocess_time_series(
    timestamps: Sequence[Any] | np.ndarray | None,
    values: Sequence[Any] | np.ndarray | None,
) -> TimeSeries:
    """Validate, coerce, filter, sort, and deduplicate one aligned series.

    Length equality is checked before any paired iteration; invalid or
    non-finite timestamp/value pairs are then removed together.
    """

    if timestamps is None or values is None:
        raise DataValidationError("timestamps and values must not be None")
    raw_timestamps = _sequence_to_list(timestamps, field_name="timestamps")
    raw_values = _sequence_to_list(values, field_name="values")
    if len(raw_timestamps) != len(raw_values):
        raise DataValidationError(
            "timestamps and values must have equal lengths: "
            f"{len(raw_timestamps)} != {len(raw_values)}"
        )
    if not raw_timestamps:
        return TimeSeries()

    valid_timestamps: list[float] = []
    valid_values: list[float] = []
    for index in range(len(raw_timestamps)):
        try:
            timestamp = _numeric_timestamp(raw_timestamps[index])
            value = _numeric_value(raw_values[index])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(timestamp) or not math.isfinite(value):
            continue
        valid_timestamps.append(timestamp)
        valid_values.append(value)
    return deduplicate_by_timestamp(valid_timestamps, valid_values)


def _validate_metric_entry(entry: Any, *, location: str) -> dict[str, list[Any]]:
    if not isinstance(entry, Mapping):
        raise DataValidationError(f"{location} must be an object")
    if "status" in entry or "data" in entry:
        return prometheus_query_range_to_series(entry, location=location)
    unknown = set(entry) - CANONICAL_SERIES_FIELDS
    missing = CANONICAL_SERIES_FIELDS - set(entry)
    if missing:
        raise DataValidationError(
            f"{location} is missing required fields: {sorted(missing)}"
        )
    if unknown:
        raise DataValidationError(
            f"{location} contains unsupported fields: {sorted(unknown)}"
        )
    return {
        "timestamps": _sequence_to_list(
            entry["timestamps"], field_name=f"{location}.timestamps"
        ),
        "values": _sequence_to_list(
            entry["values"], field_name=f"{location}.values"
        ),
    }


def validate_canonical_dataset(payload: Any) -> dict[str, Any]:
    """Validate and copy the canonical two-phase dataset structure.

    Missing metrics are allowed so the algorithm can assign state 1 or 2 per
    metric. Both phase containers themselves are mandatory.
    """

    if not isinstance(payload, Mapping):
        raise DataValidationError("canonical dataset root must be an object")
    unknown_phases = set(payload) - set(CANONICAL_PHASES)
    missing_phases = set(CANONICAL_PHASES) - set(payload)
    if missing_phases:
        raise DataValidationError(
            f"canonical dataset is missing phases: {sorted(missing_phases)}"
        )
    if unknown_phases:
        raise DataValidationError(
            "canonical dataset contains unsupported root fields: "
            f"{sorted(unknown_phases)}"
        )

    validated: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for phase in CANONICAL_PHASES:
        section = payload[phase]
        if not isinstance(section, Mapping):
            raise DataValidationError(f"{phase} must be an object keyed by metric")
        phase_data: dict[str, dict[str, list[Any]]] = {}
        for metric, entry in section.items():
            if not isinstance(metric, str) or not metric:
                raise DataValidationError(
                    f"{phase} metric keys must be non-empty strings"
                )
            phase_data[metric] = _validate_metric_entry(
                entry, location=f"{phase}[{metric!r}]"
            )
        validated[phase] = phase_data
    return validated


def _validated_time_bounds(
    start_time: float | None, end_time: float | None
) -> tuple[float | None, float | None]:
    start = None if start_time is None else float(start_time)
    end = None if end_time is None else float(end_time)
    if start is not None and not math.isfinite(start):
        raise DataValidationError("start_time must be finite")
    if end is not None and not math.isfinite(end):
        raise DataValidationError("end_time must be finite")
    if start is not None and end is not None and start > end:
        raise DataValidationError("start_time must not exceed end_time")
    return start, end


def _filter_canonical_inference_window(
    dataset: dict[str, Any],
    start_time: float | None,
    end_time: float | None,
) -> dict[str, Any]:
    """Filter valid inference pairs only; standard data is never windowed.

    Length-mismatched metric data is deliberately left intact so the algorithm
    can report state 2 for that metric without blocking other metrics.
    """

    start, end = _validated_time_bounds(start_time, end_time)
    if start is None and end is None:
        return dataset

    for entry in dataset["inference"].values():
        timestamps = entry["timestamps"]
        values = entry["values"]
        if len(timestamps) != len(values):
            continue
        kept_timestamps: list[Any] = []
        kept_values: list[Any] = []
        for index in range(len(timestamps)):
            try:
                numeric_time = _numeric_timestamp(timestamps[index])
            except (TypeError, ValueError, OverflowError):
                # Preserve invalid pairs for the normal preprocessing filter.
                kept_timestamps.append(timestamps[index])
                kept_values.append(values[index])
                continue
            if (start is None or numeric_time >= start) and (
                end is None or numeric_time <= end
            ):
                kept_timestamps.append(timestamps[index])
                kept_values.append(values[index])
        entry["timestamps"] = kept_timestamps
        entry["values"] = kept_values
    return dataset


def _parse_remote_jsonl(
    text: str,
    metrics: Sequence[str],
    *,
    start_time: float | None,
    end_time: float | None,
) -> dict[str, Any]:
    """Parse explicit-phase incremental rows into the canonical structure."""

    start, end = _validated_time_bounds(start_time, end_time)
    result: dict[str, dict[str, dict[str, list[Any]]]] = {
        phase: {
            metric: {"timestamps": [], "values": []} for metric in metrics
        }
        for phase in CANONICAL_PHASES
    }
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(
                f"invalid JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, Mapping):
            raise DataValidationError(
                f"JSONL line {line_number} must contain an object"
            )
        phase = row.get("phase")
        if phase not in CANONICAL_PHASES:
            raise DataValidationError(
                f"JSONL line {line_number} must set phase to "
                "'standard' or 'inference'"
            )
        if "timestamp" not in row:
            raise DataValidationError(
                f"JSONL line {line_number} is missing timestamp"
            )
        values = row.get("metrics")
        if not isinstance(values, Mapping):
            raise DataValidationError(
                f"JSONL line {line_number} metrics must be an object"
            )
        try:
            numeric_time = _numeric_timestamp(row["timestamp"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError(
                f"JSONL line {line_number} has an invalid timestamp"
            ) from exc
        if not math.isfinite(numeric_time):
            raise DataValidationError(
                f"JSONL line {line_number} has a non-finite timestamp"
            )
        if phase == "inference" and (
            (start is not None and numeric_time < start)
            or (end is not None and numeric_time > end)
        ):
            continue
        for metric in metrics:
            if metric not in values:
                continue
            result[phase][metric]["timestamps"].append(row["timestamp"])
            result[phase][metric]["values"].append(values[metric])
    return result


def parse_dataset_text(
    text: str,
    metrics: Sequence[str],
    *,
    suffix: str = ".json",
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    """Parse local canonical JSON or explicit-phase remote JSON Lines."""

    normalized_suffix = suffix.lower()
    if normalized_suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"invalid canonical JSON: {exc.msg}") from exc
        dataset = validate_canonical_dataset(payload)
        return _filter_canonical_inference_window(dataset, start_time, end_time)
    if normalized_suffix == ".jsonl":
        return _parse_remote_jsonl(
            text,
            metrics,
            start_time=start_time,
            end_time=end_time,
        )
    raise DataValidationError(
        f"unsupported input format {suffix!r}; expected canonical .json "
        "or explicit-phase remote .jsonl"
    )


def load_dataset(
    path: str | os.PathLike[str],
    metrics: Sequence[str],
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    """Load one canonical local JSON file.

    JSON Lines is intentionally reserved for the Remote Monitor adapter; the
    local ``--path`` contract accepts only one explicit two-phase JSON file.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input data file not found: {source}")
    if source.suffix.lower() != ".json":
        raise DataValidationError(
            "local input must be one canonical UTF-8 .json file"
        )
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DataValidationError("input JSON must be UTF-8") from exc
    return parse_dataset_text(
        text,
        metrics,
        suffix=".json",
        start_time=start_time,
        end_time=end_time,
    )


def extract_metric_series(
    dataset: Mapping[str, Any], phase: str, metric: str
) -> TimeSeries:
    """Extract and preprocess one metric from a canonical dataset phase."""

    if phase not in CANONICAL_PHASES:
        raise DataValidationError(f"unsupported dataset phase: {phase!r}")
    section = dataset.get(phase)
    if not isinstance(section, Mapping):
        raise DataValidationError(f"{phase} must be an object keyed by metric")
    if metric not in section:
        return TimeSeries()
    entry = _validate_metric_entry(
        section[metric], location=f"{phase}[{metric!r}]"
    )
    return preprocess_time_series(entry["timestamps"], entry["values"])
