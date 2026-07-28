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

"""Safe YAML configuration loading and per-metric template creation."""

from __future__ import annotations

import copy
import hashlib
import math
import ntpath
import os
import posixpath
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "default_config.yaml"
COMMON_CONFIG_PATH = MODULE_DIR / "common_config.yaml"
DEFAULT_CONFIG_DIR = MODULE_DIR / "config"
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_FILENAME_STEM_LENGTH = 160
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

_METRIC_CONFIG_KEYS = {
    "metric",
    "abnormal_type",
    "alpha",
    "upper_ratio",
    "lower_ratio",
    "minimum_standard_points",
    "minimum_inference_points",
    "normalization",
    "kde",
    "stable_segment",
    "abnormal_interval",
}
_NORMALIZATION_KEYS = {"type"}
_KDE_KEYS = {
    "kernel",
    "bandwidth",
    "grid_size",
    "padding_ratio",
    "tail_bandwidths",
    "zero_range_epsilon",
    "random_seed",
    "peak_prominence_ratio",
}
_STABLE_SEGMENT_KEYS = {
    "std_factor",
    "within_std_coefficient",
    "minimum_passed_flags",
    "mean_tolerance_ratio",
    "time_gap_factor",
    "maximum_time_gap",
}
_ABNORMAL_INTERVAL_KEYS = {
    "minimum_duration",
    "minimum_abnormal_points",
    "minimum_abnormal_rate",
    "max_normal_points_between",
    "time_gap_factor",
    "maximum_time_gap",
}
_COMMON_CONFIG_KEYS = {"n_keep_result", "n_keep_abnormal"}


class ConfigCollisionError(ValueError):
    """A safe filename already belongs to a different raw metric name."""


def metric_to_safe_filename(metric: str) -> str:
    """Convert an arbitrary metric name to one non-traversing YAML filename."""

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string")
    if metric != metric.strip():
        raise ValueError("metric must not have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in metric):
        raise ValueError("metric must not contain control characters")
    if (
        posixpath.isabs(metric)
        or ntpath.isabs(metric)
        or bool(ntpath.splitdrive(metric)[0])
    ):
        raise ValueError("metric must not be an absolute path or contain a drive")
    components = re.split(r"[\\/]", metric)
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("metric contains an unsafe path component")

    safe = metric.replace("/", "__").replace("\\", "__")
    while ".." in safe:
        safe = safe.replace("..", "__")
    safe = re.sub(r'[<>:"|?*]', "_", safe)
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", safe)
    safe = safe.strip(" .")
    if not safe:
        raise ValueError("metric does not contain a safe filename component")
    if safe.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        safe = f"_{safe}"
    if len(safe) > _MAX_FILENAME_STEM_LENGTH:
        digest = hashlib.sha256(metric.encode("utf-8")).hexdigest()[:16]
        prefix_length = _MAX_FILENAME_STEM_LENGTH - len(digest) - 2
        safe = f"{safe[:prefix_length]}__{digest}"
    return f"{safe}.yaml"


def _safe_target(config_dir: str | os.PathLike[str], metric: str) -> Path:
    base = Path(config_dir).expanduser().resolve()
    target = (base / metric_to_safe_filename(metric)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("metric configuration path escapes config_dir") from exc
    return target


def ensure_metric_config(
    metric: str,
    *,
    config_dir: str | os.PathLike[str] = DEFAULT_CONFIG_DIR,
    default_config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
) -> Path:
    """Copy the complete default template once and bind it to ``metric``.

    Existing files are never modified. Their embedded raw metric name is used
    to detect collisions caused by filename sanitization.
    """

    template = Path(default_config_path).expanduser().resolve()
    if not template.is_file():
        raise FileNotFoundError(f"Default metric config not found: {template}")
    target = _safe_target(config_dir, metric)
    if target.exists():
        if not target.is_file():
            raise OSError(f"Metric config path is not a file: {target}")
        _verify_metric_binding(target, metric)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    # Reserve the target exclusively before copyfile. This closes the
    # exists/copy race without ever overwriting a concurrently created config.
    try:
        with target.open("xb"):
            pass
    except FileExistsError:
        if not target.is_file():
            raise OSError(f"Metric config path is not a file: {target}")
        _verify_metric_binding(target, metric)
        return target
    try:
        shutil.copyfile(template, target)
        raw = _read_yaml(target)
        raw["metric"] = metric
        _write_yaml_atomic(target, raw)
    except OSError as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise OSError(f"Failed to copy metric config to {target}: {exc}") from exc
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.stat().st_size > _MAX_CONFIG_BYTES:
            raise ValueError(
                f"YAML config exceeds {_MAX_CONFIG_BYTES} bytes: {path}"
            )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ValueError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to load YAML config {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML config must contain an object: {path}")
    return dict(loaded)


def _write_yaml_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write one small YAML mapping beside its destination."""

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            yaml.safe_dump(
                dict(data),
                stream,
                allow_unicode=True,
                sort_keys=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _verify_metric_binding(path: Path, metric: str) -> None:
    raw_metric = _read_yaml(path).get("metric")
    if raw_metric != metric:
        if raw_metric is None or raw_metric == "":
            detail = "does not contain its original metric name"
        else:
            detail = f"is already bound to metric {raw_metric!r}"
        raise ConfigCollisionError(
            f"Metric config {path} {detail}; requested metric is {metric!r}"
        )


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_metric_config(
    metric: str,
    *,
    config_dir: str | os.PathLike[str] = DEFAULT_CONFIG_DIR,
    default_config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Explicitly initialize and load one per-metric YAML config."""

    target = ensure_metric_config(
        metric,
        config_dir=config_dir,
        default_config_path=default_config_path,
    )
    defaults = _read_yaml(Path(default_config_path).expanduser().resolve())
    config = _deep_merge(defaults, _read_yaml(target))
    _validate_metric_config(config, metric)
    return config


def load_common_config(
    path: str | os.PathLike[str] = COMMON_CONFIG_PATH,
) -> dict[str, Any]:
    """Load history aggregation settings."""

    config = _read_yaml(Path(path).expanduser().resolve())
    _reject_unknown_keys(config, _COMMON_CONFIG_KEYS, "common config")
    n_keep_result = _require_integer(
        config, "n_keep_result", minimum=1, context="common config"
    )
    n_keep_abnormal = _require_integer(
        config, "n_keep_abnormal", minimum=1, context="common config"
    )
    if n_keep_abnormal > n_keep_result:
        raise ValueError(
            "common config n_keep_abnormal must not exceed n_keep_result"
        )
    return config


def _validate_metric_config(config: Mapping[str, Any], metric: str) -> None:
    _reject_unknown_keys(config, _METRIC_CONFIG_KEYS, "metric config")
    if config.get("metric") != metric:
        raise ConfigCollisionError(
            f"metric config is bound to {config.get('metric')!r}, expected {metric!r}"
        )
    abnormal_type = config.get("abnormal_type")
    if abnormal_type not in {"UP", "DOWN", "BOTH"}:
        raise ValueError("metric config abnormal_type must be UP, DOWN, or BOTH")
    _require_number(
        config,
        "alpha",
        minimum=0.0,
        maximum=0.5,
        minimum_inclusive=False,
        maximum_inclusive=False,
        context="metric config",
    )
    _require_number(
        config,
        "upper_ratio",
        minimum=1.0,
        context="metric config",
    )
    _require_number(
        config,
        "lower_ratio",
        minimum=1.0,
        context="metric config",
    )
    _require_integer(
        config,
        "minimum_standard_points",
        minimum=3,
        context="metric config",
    )
    _require_integer(
        config,
        "minimum_inference_points",
        minimum=1,
        context="metric config",
    )

    normalization = _require_mapping(config, "normalization", "metric config")
    _reject_unknown_keys(normalization, _NORMALIZATION_KEYS, "normalization")
    if normalization.get("type") != "identity":
        raise ValueError("normalization.type must be identity")

    kde = _require_mapping(config, "kde", "metric config")
    _reject_unknown_keys(kde, _KDE_KEYS, "kde")
    if kde.get("kernel") != "gaussian":
        raise ValueError("kde.kernel must be gaussian")
    bandwidth = kde.get("bandwidth")
    if bandwidth != "auto":
        _require_finite_number(
            bandwidth,
            "kde.bandwidth",
            minimum=0.0,
            minimum_inclusive=False,
        )
    _require_integer(kde, "grid_size", minimum=32, context="kde")
    _require_number(kde, "padding_ratio", minimum=0.0, context="kde")
    _require_number(
        kde,
        "tail_bandwidths",
        minimum=0.0,
        minimum_inclusive=False,
        context="kde",
    )
    _require_number(
        kde,
        "zero_range_epsilon",
        minimum=0.0,
        minimum_inclusive=False,
        context="kde",
    )
    _require_integer(kde, "random_seed", minimum=0, context="kde")
    _require_number(kde, "peak_prominence_ratio", minimum=0.0, context="kde")

    stable = _require_mapping(config, "stable_segment", "metric config")
    _reject_unknown_keys(stable, _STABLE_SEGMENT_KEYS, "stable_segment")
    _require_number(
        stable,
        "std_factor",
        minimum=0.0,
        minimum_inclusive=False,
        context="stable_segment",
    )
    _require_number(
        stable,
        "within_std_coefficient",
        minimum=1.0,
        maximum=2.0,
        maximum_inclusive=False,
        context="stable_segment",
    )
    _require_integer(
        stable,
        "minimum_passed_flags",
        minimum=1,
        maximum=6,
        context="stable_segment",
    )
    _require_number(
        stable,
        "mean_tolerance_ratio",
        minimum=0.0,
        context="stable_segment",
    )
    _require_number(
        stable,
        "time_gap_factor",
        minimum=0.0,
        minimum_inclusive=False,
        context="stable_segment",
    )
    _require_optional_positive_number(
        stable.get("maximum_time_gap"), "stable_segment.maximum_time_gap"
    )

    interval = _require_mapping(config, "abnormal_interval", "metric config")
    _reject_unknown_keys(interval, _ABNORMAL_INTERVAL_KEYS, "abnormal_interval")
    _require_number(
        interval,
        "minimum_duration",
        minimum=0.0,
        context="abnormal_interval",
    )
    _require_integer(
        interval,
        "minimum_abnormal_points",
        minimum=1,
        context="abnormal_interval",
    )
    _require_number(
        interval,
        "minimum_abnormal_rate",
        minimum=0.0,
        maximum=1.0,
        context="abnormal_interval",
    )
    _require_integer(
        interval,
        "max_normal_points_between",
        minimum=0,
        context="abnormal_interval",
    )
    _require_number(
        interval,
        "time_gap_factor",
        minimum=0.0,
        minimum_inclusive=False,
        context="abnormal_interval",
    )
    _require_optional_positive_number(
        interval.get("maximum_time_gap"), "abnormal_interval.maximum_time_gap"
    )


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(repr(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {', '.join(unknown)}")


def _require_mapping(
    value: Mapping[str, Any], key: str, context: str
) -> dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{context}.{key} must be an object")
    return dict(nested)


def _require_integer(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    context: str,
) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{context}.{key} must be an integer")
    if minimum is not None and raw < minimum:
        raise ValueError(f"{context}.{key} must be >= {minimum}")
    if maximum is not None and raw > maximum:
        raise ValueError(f"{context}.{key} must be <= {maximum}")
    return raw


def _require_number(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    context: str,
) -> float:
    return _require_finite_number(
        value.get(key),
        f"{context}.{key}",
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )


def _require_finite_number(
    raw: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}")
    return result


def _require_optional_positive_number(raw: Any, name: str) -> None:
    if raw is None:
        return
    _require_finite_number(
        raw,
        name,
        minimum=0.0,
        minimum_inclusive=False,
    )
