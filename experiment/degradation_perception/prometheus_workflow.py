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

"""One-command Prometheus ``query_range`` to Top-5 association workflow."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .algorithm import DegradationPerception
from .association_analysis import DEFAULT_ASSOCIATION_CONFIG
from .config_loader import (
    ensure_metric_config,
    load_metric_config,
    metric_to_safe_filename,
)
from .perception_config import SUPPORTED_ABNORMAL_TYPES
from .prometheus_matrix_adapter import (
    PHASES,
    SERIES_POLICIES,
    PrometheusMatrixError,
    convert_matrix_response,
)
from .result_presentation import build_top5_result


_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_EXPECTED_POINTS_PER_SERIES = 1_000_000
_CONFIG_KEYS = {
    "prometheus",
    "query_step_seconds",
    "windows",
    "association_target",
    "metrics",
    "association",
}
_PROMETHEUS_KEYS = {
    "base_url",
    "timeout_seconds",
    "bearer_token_env",
    "use_environment_proxy",
}
_WINDOW_KEYS = {"start", "end"}
_METRIC_KEYS = {
    "standard_query",
    "inference_query",
    "abnormal_type",
    "series_policy",
    "select_labels",
}
_ASSOCIATION_KEYS = {
    "weights",
    "top_k",
    "context_ratio",
    "min_aligned_points",
    "min_rf_samples",
    "min_coverage_ratio",
    "alignment_tolerance",
    "random_forest",
}
_RANDOM_FOREST_KEYS = {
    "n_estimators",
    "class_weight",
    "random_state",
    "importance_method",
}

QueryFetcher = Callable[..., Mapping[str, Any]]


class PrometheusWorkflowError(ValueError):
    """Structured configuration, HTTP, or orchestration failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.details = copy.deepcopy(dict(details or {}))
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": copy.deepcopy(self.details),
        }


def _fail(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise PrometheusWorkflowError(code, message, details=details)


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        _fail(
            "invalid_workflow_config",
            f"{context} contains unknown keys: {unknown}",
        )


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("invalid_workflow_config", f"{context} must be an object")
    return dict(value)


def _positive_number(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        _fail(
            "invalid_workflow_config",
            f"{context} must be a positive finite number",
        )
    return float(value)


def _integer(value: Any, context: str, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        _fail(
            "invalid_workflow_config",
            f"{context} must be an integer >= {minimum}",
        )
    return value


def _bounded_number(
    value: Any,
    context: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(
            "invalid_workflow_config",
            f"{context} must be a finite number",
        )
    number = float(value)
    below = number < minimum if minimum_inclusive else number <= minimum
    if below or (maximum is not None and number > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        operator = ">=" if minimum_inclusive else ">"
        _fail(
            "invalid_workflow_config",
            f"{context} must be {operator} {minimum}{suffix}",
        )
    return number


def _time_value(value: Any, context: str) -> tuple[str, float]:
    if isinstance(value, bool) or value is None:
        _fail(
            "invalid_workflow_config",
            f"{context} must be RFC3339 text or finite Unix seconds",
        )
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            _fail(
                "invalid_workflow_config",
                f"{context} must be RFC3339 text or finite Unix seconds",
            )
        return str(float(value)), float(value)
    if not isinstance(value, str) or not value.strip():
        _fail(
            "invalid_workflow_config",
            f"{context} must be RFC3339 text or finite Unix seconds",
        )
    text = value.strip()
    try:
        timestamp = float(text)
        if not math.isfinite(timestamp):
            raise ValueError
        return text, timestamp
    except ValueError:
        pass
    iso_text = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
        if parsed.tzinfo is None:
            raise ValueError("timezone is required")
        timestamp = parsed.timestamp()
    except (OverflowError, ValueError) as exc:
        _fail(
            "invalid_workflow_config",
            f"{context} must be RFC3339 text or finite Unix seconds: {exc}",
        )
    return text, timestamp


def _normalize_association(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    association = _mapping(value, "association")
    _reject_unknown_keys(association, _ASSOCIATION_KEYS, "association")
    normalized: dict[str, Any] = {}
    if "weights" in association:
        weights = _mapping(association["weights"], "association.weights")
        _reject_unknown_keys(
            weights,
            {"correlation", "random_forest"},
            "association.weights",
        )
        merged_weights = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG["weights"])
        merged_weights.update(weights)
        correlation = _bounded_number(
            merged_weights["correlation"],
            "association.weights.correlation",
            minimum=0.0,
        )
        random_forest_weight = _bounded_number(
            merged_weights["random_forest"],
            "association.weights.random_forest",
            minimum=0.0,
        )
        if not math.isclose(
            correlation + random_forest_weight,
            1.0,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            _fail(
                "invalid_workflow_config",
                "association weights must sum to 1",
            )
        normalized["weights"] = {
            "correlation": correlation,
            "random_forest": random_forest_weight,
        }
    integer_fields = {
        "top_k": 1,
        "min_aligned_points": 1,
        "min_rf_samples": 1,
    }
    for key, minimum in integer_fields.items():
        if key in association:
            normalized[key] = _integer(
                association[key],
                f"association.{key}",
                minimum=minimum,
            )
    if "context_ratio" in association:
        normalized["context_ratio"] = _bounded_number(
            association["context_ratio"],
            "association.context_ratio",
            minimum=0.0,
        )
    if "min_coverage_ratio" in association:
        normalized["min_coverage_ratio"] = _bounded_number(
            association["min_coverage_ratio"],
            "association.min_coverage_ratio",
            minimum=0.0,
            maximum=1.0,
        )
    if "alignment_tolerance" in association:
        tolerance = association["alignment_tolerance"]
        normalized["alignment_tolerance"] = (
            None
            if tolerance is None
            else _bounded_number(
                tolerance,
                "association.alignment_tolerance",
                minimum=0.0,
                minimum_inclusive=False,
            )
        )
    if "random_forest" in association:
        forest = _mapping(
            association["random_forest"],
            "association.random_forest",
        )
        _reject_unknown_keys(
            forest,
            _RANDOM_FOREST_KEYS,
            "association.random_forest",
        )
        merged_forest = copy.deepcopy(
            DEFAULT_ASSOCIATION_CONFIG["random_forest"]
        )
        merged_forest.update(forest)
        merged_forest["n_estimators"] = _integer(
            merged_forest["n_estimators"],
            "association.random_forest.n_estimators",
            minimum=1,
        )
        merged_forest["random_state"] = _integer(
            merged_forest["random_state"],
            "association.random_forest.random_state",
            minimum=0,
        )
        if merged_forest["class_weight"] != "balanced":
            _fail(
                "invalid_workflow_config",
                "association.random_forest.class_weight must be balanced",
            )
        if merged_forest["importance_method"] != "permutation":
            _fail(
                "invalid_workflow_config",
                "association.random_forest.importance_method must be permutation",
            )
        normalized["random_forest"] = merged_forest
    return normalized


def normalize_workflow_config(payload: Any) -> dict[str, Any]:
    """Validate and normalize the tester-facing YAML configuration."""

    config = _mapping(payload, "workflow config")
    _reject_unknown_keys(config, _CONFIG_KEYS, "workflow config")
    missing = _CONFIG_KEYS - {"association"} - set(config)
    if missing:
        _fail(
            "invalid_workflow_config",
            f"workflow config is missing keys: {sorted(missing)}",
        )

    prometheus = _mapping(config["prometheus"], "prometheus")
    _reject_unknown_keys(prometheus, _PROMETHEUS_KEYS, "prometheus")
    base_url = prometheus.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        _fail(
            "invalid_workflow_config",
            "prometheus.base_url must be a non-empty HTTP(S) URL",
        )
    base_url = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        _fail(
            "invalid_workflow_config",
            "prometheus.base_url must be a valid HTTP(S) URL",
        )
    if parsed.username is not None or parsed.password is not None:
        _fail(
            "invalid_workflow_config",
            "prometheus.base_url must not contain embedded credentials",
        )
    if parsed.query or parsed.fragment:
        _fail(
            "invalid_workflow_config",
            "prometheus.base_url must not contain a query or fragment",
        )
    timeout_seconds = _positive_number(
        prometheus.get("timeout_seconds", 30),
        "prometheus.timeout_seconds",
    )
    token_env = prometheus.get("bearer_token_env")
    if token_env is not None and (
        not isinstance(token_env, str) or not token_env.strip()
    ):
        _fail(
            "invalid_workflow_config",
            "prometheus.bearer_token_env must be null or an environment name",
        )
    token_env = token_env.strip() if isinstance(token_env, str) else None
    use_environment_proxy = prometheus.get("use_environment_proxy", False)
    if not isinstance(use_environment_proxy, bool):
        _fail(
            "invalid_workflow_config",
            "prometheus.use_environment_proxy must be a boolean",
        )

    step = _positive_number(
        config["query_step_seconds"],
        "query_step_seconds",
    )
    windows = _mapping(config["windows"], "windows")
    if set(windows) != set(PHASES):
        _fail(
            "invalid_workflow_config",
            "windows must contain exactly standard and inference",
        )
    normalized_windows: dict[str, dict[str, str]] = {}
    for phase in PHASES:
        window = _mapping(windows[phase], f"windows.{phase}")
        _reject_unknown_keys(window, _WINDOW_KEYS, f"windows.{phase}")
        if set(window) != _WINDOW_KEYS:
            _fail(
                "invalid_workflow_config",
                f"windows.{phase} must contain start and end",
            )
        start, start_epoch = _time_value(
            window["start"], f"windows.{phase}.start"
        )
        end, end_epoch = _time_value(
            window["end"], f"windows.{phase}.end"
        )
        if start_epoch >= end_epoch:
            _fail(
                "invalid_workflow_config",
                f"windows.{phase}.start must be earlier than end",
            )
        expected_points = math.floor((end_epoch - start_epoch) / step) + 1
        if expected_points > _MAX_EXPECTED_POINTS_PER_SERIES:
            _fail(
                "invalid_workflow_config",
                f"windows.{phase} would request too many points per series",
                details={
                    "expectedPoints": expected_points,
                    "maximum": _MAX_EXPECTED_POINTS_PER_SERIES,
                },
            )
        normalized_windows[phase] = {"start": start, "end": end}

    target = config["association_target"]
    if not isinstance(target, str) or not target.strip():
        _fail(
            "invalid_workflow_config",
            "association_target must be a non-empty logical metric name",
        )
    target = target.strip()
    metrics = _mapping(config["metrics"], "metrics")
    if not metrics:
        _fail("invalid_workflow_config", "metrics must not be empty")
    normalized_metrics: dict[str, dict[str, Any]] = {}
    for logical_metric, raw_spec in metrics.items():
        if not isinstance(logical_metric, str) or not logical_metric.strip():
            _fail(
                "invalid_workflow_config",
                "metrics keys must be non-empty logical metric names",
            )
        logical_metric = logical_metric.strip()
        spec = _mapping(raw_spec, f"metrics.{logical_metric}")
        _reject_unknown_keys(spec, _METRIC_KEYS, f"metrics.{logical_metric}")
        queries: dict[str, str] = {}
        for phase in PHASES:
            query_key = f"{phase}_query"
            query = spec.get(query_key)
            if not isinstance(query, str) or not query.strip():
                _fail(
                    "invalid_workflow_config",
                    f"metrics.{logical_metric}.{query_key} must be "
                    "non-empty PromQL",
                )
            queries[query_key] = query.strip()
        abnormal_type = spec.get("abnormal_type", "UP")
        if abnormal_type not in SUPPORTED_ABNORMAL_TYPES:
            _fail(
                "invalid_workflow_config",
                f"metrics.{logical_metric}.abnormal_type must be UP, DOWN, or BOTH",
            )
        policy = spec.get("series_policy", "exactly_one")
        if policy not in SERIES_POLICIES:
            _fail(
                "invalid_workflow_config",
                f"metrics.{logical_metric}.series_policy is unsupported",
            )
        select_labels = spec.get("select_labels")
        if policy == "exactly_one" and select_labels is not None:
            _fail(
                "invalid_workflow_config",
                f"metrics.{logical_metric}.select_labels requires "
                "series_policy: select_by_labels",
            )
        if policy == "select_by_labels":
            labels = _mapping(
                select_labels,
                f"metrics.{logical_metric}.select_labels",
            )
            if not labels or any(
                not isinstance(key, str)
                or not key
                or not isinstance(item, str)
                or not item
                for key, item in labels.items()
            ):
                _fail(
                    "invalid_workflow_config",
                    f"metrics.{logical_metric}.select_labels must contain "
                    "non-empty string pairs",
                )
            select_labels = labels
        normalized_metrics[logical_metric] = {
            **queries,
            "abnormal_type": abnormal_type,
            "series_policy": policy,
            "select_labels": copy.deepcopy(select_labels),
        }
    if target not in normalized_metrics:
        _fail(
            "invalid_workflow_config",
            "association_target must also be declared under metrics",
            details={"associationTarget": target},
        )

    normalized = {
        "prometheus": {
            "base_url": base_url,
            "timeout_seconds": timeout_seconds,
            "bearer_token_env": token_env,
            "use_environment_proxy": use_environment_proxy,
        },
        "query_step_seconds": step,
        "windows": normalized_windows,
        "association_target": target,
        "metrics": normalized_metrics,
        "association": _normalize_association(config.get("association")),
    }
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return normalized


def load_workflow_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one small UTF-8 YAML file with safe YAML semantics."""

    source = Path(path).expanduser()
    if not source.is_file():
        _fail(
            "invalid_workflow_config",
            f"workflow config file does not exist: {source}",
        )
    if source.stat().st_size > _MAX_CONFIG_BYTES:
        _fail(
            "invalid_workflow_config",
            f"workflow config exceeds {_MAX_CONFIG_BYTES} bytes",
        )
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _fail(
            "invalid_workflow_config",
            f"failed to read workflow config: {exc}",
        )
    return normalize_workflow_config(payload)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def fetch_query_range(
    *,
    base_url: str,
    query: str,
    start: str,
    end: str,
    step: float,
    timeout_seconds: float,
    bearer_token: str | None = None,
    use_environment_proxy: bool = False,
) -> dict[str, Any]:
    """Issue one real Prometheus HTTP GET and decode its full JSON response."""

    parameters = urllib.parse.urlencode(
        {
            "query": query,
            "start": start,
            "end": end,
            "step": f"{step:g}",
        }
    )
    url = f"{base_url.rstrip('/')}/api/v1/query_range?{parameters}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "rl-insight-degradation-perception/1",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        if use_environment_proxy:
            opened = urllib.request.urlopen(request, timeout=timeout_seconds)
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            opened = opener.open(request, timeout=timeout_seconds)
        with opened as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            body_excerpt = exc.read(4096).decode("utf-8", errors="replace")
        except OSError:
            body_excerpt = ""
        _fail(
            "prometheus_http_error",
            f"Prometheus returned HTTP {exc.code}",
            details={"status": exc.code, "bodyExcerpt": body_excerpt},
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _fail(
            "prometheus_connection_error",
            f"failed to contact Prometheus: {exc}",
        )
    if len(body) > _MAX_RESPONSE_BYTES:
        _fail(
            "prometheus_response_too_large",
            f"Prometheus response exceeds {_MAX_RESPONSE_BYTES} bytes",
        )
    try:
        decoded = body.decode("utf-8")
        payload = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(
            "invalid_prometheus_json",
            f"Prometheus response is not strict UTF-8 JSON: {exc}",
        )
    if not isinstance(payload, Mapping):
        _fail(
            "invalid_prometheus_json",
            "Prometheus response root must be an object",
        )
    return dict(payload)


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


def _validate_converted_window(
    series: Mapping[str, Any],
    *,
    phase: str,
    logical_metric: str,
    start: str,
    end: str,
    step: float,
) -> None:
    _, start_epoch = _time_value(start, f"windows.{phase}.start")
    _, end_epoch = _time_value(end, f"windows.{phase}.end")
    tolerance = max(1.0e-6, step * 1.0e-9)
    outside = [
        float(timestamp)
        for timestamp in series.get("timestamps", [])
        if float(timestamp) < start_epoch - tolerance
        or float(timestamp) > end_epoch + tolerance
    ]
    if outside:
        _fail(
            "sample_outside_requested_window",
            "Prometheus returned samples outside the requested range",
            details={
                "phase": phase,
                "logicalMetric": logical_metric,
                "outsideSampleCount": len(outside),
                "firstOutsideTimestamp": outside[0],
                "requestedStart": start,
                "requestedEnd": end,
            },
        )


def _prepare_metric_configs(
    config_dir: Path,
    metrics: Mapping[str, Mapping[str, Any]],
    target_metric: str,
    association_overrides: Mapping[str, Any],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for metric, spec in metrics.items():
        expected_path = (
            config_dir.expanduser().resolve() / metric_to_safe_filename(metric)
        )
        existed_before = expected_path.exists()
        path = ensure_metric_config(metric, config_dir=config_dir)
        current_config = load_metric_config(metric, config_dir=config_dir)
        metric_config = copy.deepcopy(current_config)
        metric_config["abnormal_type"] = spec["abnormal_type"]
        association = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)
        if metric == target_metric:
            association.update(copy.deepcopy(dict(association_overrides)))
            if "weights" in association_overrides:
                association["weights"] = copy.deepcopy(
                    association_overrides["weights"]
                )
            if "random_forest" in association_overrides:
                forest = copy.deepcopy(
                    DEFAULT_ASSOCIATION_CONFIG["random_forest"]
                )
                forest.update(association_overrides["random_forest"])
                association["random_forest"] = forest
            association.update(
                {
                    "enabled": True,
                    "target_metrics": [target_metric],
                    "candidate_mode": "abnormal_lower_metrics",
                }
            )
        else:
            association["enabled"] = False
        metric_config["association"] = association
        if existed_before:
            conflict_fields = [
                field
                for field in ("abnormal_type", "association")
                if current_config[field] != metric_config[field]
            ]
            if conflict_fields:
                _fail(
                    "existing_metric_config_conflict",
                    "existing metric config conflicts with workflow settings "
                    "and was not overwritten",
                    details={
                        "logicalMetric": metric,
                        "conflictFields": conflict_fields,
                    },
                )
        else:
            path.write_text(
                yaml.safe_dump(
                    metric_config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            load_metric_config(metric, config_dir=config_dir)
        paths[metric] = path
    return paths


def _ordered_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    target_metric: str,
) -> list[str]:
    return [target_metric, *(metric for metric in metrics if metric != target_metric)]


def run_prometheus_workflow(
    config: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    config_dir: str | os.PathLike[str] | None = None,
    fetcher: QueryFetcher = fetch_query_range,
) -> dict[str, Any]:
    """Fetch both windows, run KDE/association, and write full plus simple output."""

    normalized = normalize_workflow_config(config)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    prometheus = normalized["prometheus"]
    token_env = prometheus["bearer_token_env"]
    bearer_token = os.environ.get(token_env) if token_env else None
    if token_env and not bearer_token:
        _fail(
            "missing_bearer_token",
            f"environment variable {token_env!r} is not set",
        )

    raw_package: dict[str, Any] = {
        "formatVersion": 1,
        "source": "prometheus_query_range",
        "baseUrl": prometheus["base_url"],
        "queryStepSeconds": normalized["query_step_seconds"],
        "windows": copy.deepcopy(normalized["windows"]),
        "standard": {},
        "inference": {},
    }
    dataset: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    diagnostics: dict[str, Any] = {
        "formatVersion": 1,
        "source": "prometheus_query_range",
        "queryStepSeconds": normalized["query_step_seconds"],
        "phases": {phase: {} for phase in PHASES},
    }
    for phase in PHASES:
        window = normalized["windows"][phase]
        for logical_metric, spec in normalized["metrics"].items():
            query = spec[f"{phase}_query"]
            try:
                response = fetcher(
                    base_url=prometheus["base_url"],
                    query=query,
                    start=window["start"],
                    end=window["end"],
                    step=normalized["query_step_seconds"],
                    timeout_seconds=prometheus["timeout_seconds"],
                    bearer_token=bearer_token,
                    use_environment_proxy=prometheus[
                        "use_environment_proxy"
                    ],
                )
                converted, metric_diagnostics = convert_matrix_response(
                    response,
                    logical_metric,
                    phase,
                    query,
                    spec["series_policy"],
                    spec["select_labels"],
                    query_window=window,
                )
                _validate_converted_window(
                    converted,
                    phase=phase,
                    logical_metric=logical_metric,
                    start=window["start"],
                    end=window["end"],
                    step=normalized["query_step_seconds"],
                )
            except (PrometheusWorkflowError, PrometheusMatrixError):
                raise
            except Exception as exc:
                _fail(
                    "prometheus_query_failed",
                    f"unexpected query failure: {exc}",
                    details={
                        "phase": phase,
                        "logicalMetric": logical_metric,
                        "query": query,
                        "queryWindow": copy.deepcopy(window),
                    },
                )
            raw_entry = {
                "query": query,
                "seriesPolicy": spec["series_policy"],
                "response": copy.deepcopy(response),
            }
            if spec["select_labels"] is not None:
                raw_entry["selectLabels"] = copy.deepcopy(spec["select_labels"])
            raw_package[phase][logical_metric] = raw_entry
            dataset[phase][logical_metric] = converted
            diagnostics["phases"][phase][logical_metric] = metric_diagnostics

    raw_path = _write_strict_json(
        output / "prometheus_query_responses.json",
        raw_package,
    )
    input_path = _write_strict_json(
        output / "converted_algorithm_input.json",
        dataset,
    )
    diagnostics_path = _write_strict_json(
        output / "adapter_diagnostics.json",
        diagnostics,
    )
    target_metric = normalized["association_target"]
    runtime_config_dir = (
        output / "runtime_config"
        if config_dir is None
        else Path(config_dir).expanduser()
    )
    _prepare_metric_configs(
        runtime_config_dir,
        normalized["metrics"],
        target_metric,
        normalized["association"],
    )
    metric_order = _ordered_metrics(normalized["metrics"], target_metric)
    analysis_result = DegradationPerception(
        dataset=dataset,
        metrics=metric_order,
        association_targets=[target_metric],
        source_type="prometheus",
        config_dir=runtime_config_dir,
    ).detect()
    analysis_path = _write_strict_json(
        output / "analysis_result.json",
        analysis_result,
    )
    top5_result = build_top5_result(analysis_result, target_metric)
    top5_path = _write_strict_json(
        output / "top5_result.json",
        top5_result,
    )
    result = {
        "ok": True,
        "targetMetric": target_metric,
        "metricCount": len(metric_order),
        "top5Result": top5_result,
        "outputFiles": {
            "prometheusResponses": str(raw_path),
            "convertedInput": str(input_path),
            "adapterDiagnostics": str(diagnostics_path),
            "analysisResult": str(analysis_path),
            "top5Result": str(top5_path),
            "runtimeConfigDir": str(runtime_config_dir),
        },
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation_perception.prometheus_workflow",
        description=(
            "Query real Prometheus standard/inference windows, run KDE plus "
            "correlation/random-forest ranking, and write top5_result.json."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Tester-facing Prometheus workflow YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for raw responses, diagnostics, and results.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Dedicated generated metric config directory (optional); "
            "conflicting existing files are never overwritten."
        ),
    )
    return parser


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, (PrometheusWorkflowError, PrometheusMatrixError)):
        details = exc.to_dict()
    else:
        details = {
            "code": type(exc).__name__,
            "message": str(exc),
            "details": {},
        }
    return {"ok": False, "error": details}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the workflow and always emit one strict JSON object."""

    args = _build_parser().parse_args(argv)
    try:
        config = load_workflow_config(args.config)
        output = run_prometheus_workflow(
            config,
            args.output_dir,
            config_dir=args.config_dir,
        )
        exit_code = 0
    except Exception as exc:  # CLI boundary returns structured tester feedback.
        output = _error_payload(exc)
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
