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

"""Trainer vs observability-stack paths and loaders for RL-Insight monitoring."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

MONITOR_CONFIG_DIR = Path(__file__).resolve().parent
MONITOR_SERVICE_CONFIG_DIR = MONITOR_CONFIG_DIR / "services"
MONITOR_CONFIG_FILE = MONITOR_SERVICE_CONFIG_DIR / "config.yaml"

OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"

MONITOR_HUB_ACTOR_NAME = "RLInsightMonitorHub"
MONITOR_RAY_NAMESPACE = "rl-insight-monitor"

_DEFAULT_PROM_FILE = str((MONITOR_SERVICE_CONFIG_DIR / "prometheus.yml").resolve())

_TRAINING_MONITOR_DEFAULTS = OmegaConf.create(
    {
        "namespace": "rl_insight_monitor",
        "backend": {"type": "ray"},
        "prometheus": {
            "metrics_report_port": 9092,
            "prometheus_port": 9090,
            "config_file": _DEFAULT_PROM_FILE,
            "reload": {"mode": "ray"},
        },
        "otel": {"traces_endpoint": "http://127.0.0.1:4318/v1/traces"},
    }
)

__all__ = [
    "MONITOR_CONFIG_FILE",
    "MONITOR_CONFIG_DIR",
    "MONITOR_SERVICE_CONFIG_DIR",
    "MONITOR_HUB_ACTOR_NAME",
    "MONITOR_RAY_NAMESPACE",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "load_monitor_config",
    "load_server_config_file",
    "resolve_monitor_stack_paths",
]


def load_monitor_config(
    config: Mapping[str, Any] | DictConfig | None = None,
) -> DictConfig:
    """Merge trainer monitor defaults with optional user config and resolve OTLP trace endpoint.

    Args:
        config: Partial mapping or ``DictConfig`` merged on top of built-in training defaults; may be ``None``.

    Returns:
        Fully merged config; ``otel.traces_endpoint`` prefers non-empty ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``.
    """
    base = OmegaConf.create(
        OmegaConf.to_container(_TRAINING_MONITOR_DEFAULTS, resolve=True)
    )
    if config is None:
        merged = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    else:
        user = (
            OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            if OmegaConf.is_config(config)
            else OmegaConf.create(dict(config))
        )
        merged = OmegaConf.merge(base, user)

    env_ep = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, "").strip()
    dict_ep = str(OmegaConf.select(merged, "otel.traces_endpoint") or "").strip()
    final_ep = env_ep if env_ep else dict_ep
    if not final_ep:
        logger.warning(
            "No OTLP traces endpoint: set %s or ``otel.traces_endpoint`` in the monitor config dict. Trace export disabled.",
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
        )
    if merged.get("otel") is None:
        merged.otel = OmegaConf.create({})
    merged.otel.traces_endpoint = final_ep
    return merged


def load_server_config_file(config_path: str | Path | None = None) -> DictConfig:
    """Load the observability stack YAML used by ``rl-insight server start/stop`` and absolutize relative paths.

    Args:
        config_path: YAML file path; default is the bundled ``config/services/config.yaml``.

    Returns:
        Loaded config with ``config_file`` / ``compose_file`` paths resolved against the YAML directory.
    """
    yaml_path = (
        MONITOR_CONFIG_FILE.resolve()
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    conf = OmegaConf.load(str(yaml_path))
    resolve_monitor_stack_paths(conf, yaml_path.parent)
    return conf


def resolve_monitor_stack_paths(conf: DictConfig, config_root: Path) -> None:
    """Mutate ``conf`` so stack file/directory paths become absolute.

    Args:
        conf: Stack config as loaded from YAML.
        config_root: Directory used to resolve relative paths (typically the YAML parent folder).
    """
    root = Path(config_root).expanduser().resolve()
    filenames = {
        "prometheus": "prometheus.yml",
        "tempo": "tempo.yaml",
        "grafana": "grafana.ini",
    }
    for section, filename in filenames.items():
        section_conf = conf.get(section)
        if section_conf is None:
            continue
        if not section_conf.get("config_file"):
            section_conf.config_file = str(
                (MONITOR_SERVICE_CONFIG_DIR / filename).resolve()
            )
            continue
        path = Path(str(section_conf.config_file)).expanduser()
        if not path.is_absolute():
            path = root / path
        section_conf.config_file = str(path.resolve())

    grafana = conf.get("grafana")
    if grafana is not None:
        for key in ("provisioning_dir", "dashboards_dir"):
            path_value = grafana.get(key)
            if not path_value:
                continue
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute():
                path = root / path
            grafana[key] = str(path.resolve())

    server = conf.get("server")
    if server is not None and server.get("compose_file"):
        path = Path(str(server.compose_file)).expanduser()
        if not path.is_absolute():
            path = root / path
        server.compose_file = str(path.resolve())
