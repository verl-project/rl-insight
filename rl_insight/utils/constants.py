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

"""Shared constants for RL-Insight online monitoring."""

from __future__ import annotations

from pathlib import Path

_MONITOR_DIR = Path(__file__).resolve().parents[1]


class MonitorPaths:
    """Bundled monitor config and service file locations."""

    CONFIG_DIR = _MONITOR_DIR / "config"
    CONFIG_FILE = CONFIG_DIR / "config.yaml"
    SERVICES_DIR = CONFIG_DIR / "services"
    PROMETHEUS_CONFIG_FILE = SERVICES_DIR / "prometheus" / "prometheus.yml"
    TEMPO_CONFIG_FILE = SERVICES_DIR / "tempo" / "tempo.yaml"
    GRAFANA_CONFIG_FILE = SERVICES_DIR / "grafana" / "grafana.ini"
    GRAFANA_PROVISIONING_DIR = SERVICES_DIR / "grafana" / "provisioning"
    GRAFANA_DASHBOARDS_DIR = SERVICES_DIR / "grafana" / "dashboards"


class MonitorRayActor:
    """Ray placement metadata for the detached monitor hub actor."""

    NAME = "RLInsightMonitorHub"
    NAMESPACE = "rl-insight-monitor"


class MonitorEnv:
    """Environment variable names used by trainer-side monitor config overrides."""

    SERVICE_IP = "RL_INSIGHT_SERVICE_IP"
    OTEL_PORT = "RL_INSIGHT_OTEL_PORT"
    PROMETHEUS_PORT = "RL_INSIGHT_PROMETHEUS_PORT"
    PROMETHEUS_CONFIG_FILE = "RL_INSIGHT_PROMETHEUS_CONFIG_FILE"


class MonitorDefaults:
    """Default trainer monitor config values."""

    NAMESPACE = "rl_insight_monitor"
    METRICS_REPORT_PORT = 9092
    PROMETHEUS_PORT = 9090
    OTEL_PORT = 4318


class MonitorBackend:
    """Supported trainer-side monitor client backends (``server.backend`` registry keys)."""

    RAY = "ray"


class MonitorEventKind:
    """String ``kind`` field on events sent through monitor collectors."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TRACE = "trace"


class PrometheusScrape:
    """Prometheus scrape job names managed by the monitor hub."""

    TRAINER_METRICS_JOB = "trainer_metrics"
