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

"""Load and merge RL-Insight monitor configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .constants import MonitorBackend, MonitorDefaults, MonitorEnv, MonitorPaths

logger = logging.getLogger(__name__)

_TRAINING_MONITOR_DEFAULTS = OmegaConf.create(
    {
        "server": {
            "namespace": MonitorDefaults.NAMESPACE,
            "backend": MonitorBackend.RAY,
            "url": "",
        },
        "prometheus": {
            "metrics_report_port": MonitorDefaults.METRICS_REPORT_PORT,
        },
    }
)
__all__ = [
    "load_monitor_config",
    "load_server_config_file",
]


def load_monitor_config(
    config: Mapping[str, Any] | DictConfig | None = None,
) -> DictConfig:
    """Merge trainer monitor defaults with optional user config.

    Args:
        config: Partial mapping or ``DictConfig`` merged on top of built-in training defaults; may be ``None``.

    Returns:
        Fully merged config with environment variable overrides applied.
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

    if url := os.environ.get(MonitorEnv.SERVER_URL):
        merged.server.url = str(url).strip()
    return merged


def load_server_config_file(config_path: str | Path | None = None) -> DictConfig:
    """Load server YAML used by ``rl-insight server start/stop``.

    Args:
        config_path: YAML file path; default is the bundled ``config/config.yaml``.

    Returns:
        Loaded server config.
    """
    yaml_path = (
        MonitorPaths.CONFIG_FILE.resolve()
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    if config_path is None:
        user_conf = OmegaConf.load(str(yaml_path))
    else:
        default_conf = OmegaConf.load(str(MonitorPaths.CONFIG_FILE.resolve()))
        user_conf = OmegaConf.merge(default_conf, OmegaConf.load(str(yaml_path)))
    conf = OmegaConf.merge(
        OmegaConf.create({"service_root": str(yaml_path.parent)}),
        user_conf,
    )
    conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    del conf.service_root
    return conf
