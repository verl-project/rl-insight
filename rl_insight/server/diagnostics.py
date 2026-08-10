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

"""Best-effort health checks appended to ``rl-insight server start`` output."""

from __future__ import annotations

import socket
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests
from omegaconf import DictConfig, OmegaConf

from .network import format_host_port, local_addresses
from .runtime import StartedService

REQUEST_TIMEOUT = (1.0, 2.0)
STARTUP_READINESS_ATTEMPTS = 12
STARTUP_READINESS_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class ServiceDiagnostic:
    service: str
    pid: str
    port: str
    process: str
    readiness: str
    status: str
    detail: str = ""
    log_file: str = "-"


@dataclass
class StartupDiagnostics:
    services: list[ServiceDiagnostic] = field(default_factory=list)
    target_summaries: list[str] = field(default_factory=list)
    target_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


HttpGet = Callable[..., Any]
SocketConnect = Callable[..., Any]
Sleep = Callable[[float], None]


def run_startup_diagnostics(
    conf: DictConfig,
    services: Sequence[StartedService],
    *,
    http_get: HttpGet = requests.get,
    socket_connect: SocketConnect = socket.create_connection,
    sleep: Sleep = time.sleep,
) -> StartupDiagnostics:
    """Check local service endpoints without changing stack state or return codes."""
    report = StartupDiagnostics()
    service_by_name = {service.name: service for service in services}
    host = local_addresses()["loopback"]

    checks = [
        (
            "rl-insight-server",
            "RL-Insight server",
            bool(OmegaConf.select(conf, "server.enable", default=True)),
            int(OmegaConf.select(conf, "server.port", default=18080)),
            "/healthz",
            _validate_server_health,
        ),
        (
            "prometheus",
            "Prometheus",
            bool(OmegaConf.select(conf, "prometheus.enable", default=True)),
            int(OmegaConf.select(conf, "prometheus.prometheus_port", default=9090)),
            "/-/ready",
            None,
        ),
        (
            "tempo",
            "Tempo",
            bool(OmegaConf.select(conf, "tempo.enable", default=True)),
            int(OmegaConf.select(conf, "tempo.query_port", default=3200)),
            "/ready",
            None,
        ),
        (
            "grafana",
            "Grafana",
            bool(OmegaConf.select(conf, "grafana.enable", default=True)),
            int(OmegaConf.select(conf, "grafana.port", default=3000)),
            "/api/health",
            _validate_grafana_health,
        ),
    ]

    readiness_by_name: dict[str, str] = {}
    for name, display_name, enabled, port, path, validator in checks:
        result = _check_http_service(
            name=name,
            display_name=display_name,
            enabled=enabled,
            port=port,
            path=path,
            validator=validator,
            service=service_by_name.get(name),
            host=host,
            http_get=http_get,
            sleep=sleep,
        )
        report.services.append(result)
        readiness_by_name[name] = result.status

    tempo_enabled = bool(OmegaConf.select(conf, "tempo.enable", default=True))
    otlp_port = int(OmegaConf.select(conf, "otel.otel_port", default=4318))
    report.services.append(
        _check_otlp_listener(
            enabled=tempo_enabled,
            port=otlp_port,
            host=host,
            socket_connect=socket_connect,
            tempo_service=service_by_name.get("tempo"),
        )
    )

    if bool(OmegaConf.select(conf, "prometheus.enable", default=True)) and readiness_by_name.get("prometheus") == "OK":
        summaries, failures, detail = _check_prometheus_targets(
            host=host,
            port=int(OmegaConf.select(conf, "prometheus.prometheus_port", default=9090)),
            http_get=http_get,
        )
        report.target_summaries.extend(summaries)
        report.target_failures.extend(failures)
        if detail:
            report.notes.append(detail)

    detected_host = local_addresses()["host"]
    if detected_host:
        server_port = int(OmegaConf.select(conf, "server.port", default=18080))
        report.notes.append(
            f"Verify that training workers can reach http://{format_host_port(detected_host, server_port)}."
        )
    report.notes.append(
        "Local health checks do not verify firewall, security-group, or routing access from remote training workers."
    )
    return report


def _check_http_service(
    *,
    name: str,
    display_name: str,
    enabled: bool,
    port: int,
    path: str,
    validator: Callable[[Any], str | None] | None,
    service: StartedService | None,
    host: str,
    http_get: HttpGet,
    sleep: Sleep,
) -> ServiceDiagnostic:
    if not enabled:
        return ServiceDiagnostic(display_name, "-", str(port), "disabled", "disabled", "OK")

    pid, process, log_file = _process_fields(service)
    url = f"http://{format_host_port(host, port)}{path}"
    result: ServiceDiagnostic | None = None
    for attempt in range(STARTUP_READINESS_ATTEMPTS):
        result = _request_http_service(
            name=name,
            display_name=display_name,
            pid=pid,
            port=port,
            process=process,
            url=url,
            validator=validator,
            log_file=log_file,
            http_get=http_get,
        )
        if result.status == "OK":
            return result
        if attempt + 1 < STARTUP_READINESS_ATTEMPTS:
            sleep(STARTUP_READINESS_RETRY_DELAY_SECONDS)
    return result


def _request_http_service(
    *,
    name: str,
    display_name: str,
    pid: str,
    port: int,
    process: str,
    url: str,
    validator: Callable[[Any], str | None] | None,
    log_file: str,
    http_get: HttpGet,
) -> ServiceDiagnostic:
    try:
        response = http_get(url, timeout=REQUEST_TIMEOUT)
        if not 200 <= int(response.status_code) < 300:
            return ServiceDiagnostic(
                display_name,
                pid,
                str(port),
                process,
                f"HTTP {response.status_code}",
                "WARNING",
                f"{url} returned HTTP {response.status_code}",
                log_file,
            )
        validation_error = validator(response) if validator else None
        if validation_error:
            return ServiceDiagnostic(
                display_name,
                pid,
                str(port),
                process,
                "invalid response",
                "WARNING",
                validation_error,
                log_file,
            )
        readiness = "healthy" if name in {"rl-insight-server", "grafana"} else "ready"
        return ServiceDiagnostic(display_name, pid, str(port), process, readiness, "OK", log_file=log_file)
    except requests.Timeout as exc:
        return _warning(display_name, pid, port, process, "timeout", exc, log_file)
    except requests.ConnectionError as exc:
        reason = _connection_error_reason(exc)
        return _warning(display_name, pid, port, process, reason, exc, log_file)
    except requests.RequestException as exc:
        return _warning(display_name, pid, port, process, "request failed", exc, log_file)
    except (TypeError, ValueError) as exc:
        return _warning(display_name, pid, port, process, "invalid response", exc, log_file)


def _check_otlp_listener(
    *,
    enabled: bool,
    port: int,
    host: str,
    socket_connect: SocketConnect,
    tempo_service: StartedService | None,
) -> ServiceDiagnostic:
    if not enabled:
        return ServiceDiagnostic("OTLP traces", "-", str(port), "disabled", "disabled", "OK")

    _, process, log_file = _process_fields(tempo_service)
    try:
        connection = socket_connect((host, port), timeout=REQUEST_TIMEOUT[0])
        close = getattr(connection, "close", None)
        if callable(close):
            close()
        return ServiceDiagnostic("OTLP traces", "-", str(port), process, "listening", "OK", log_file=log_file)
    except TimeoutError as exc:
        return _warning("OTLP traces", "-", port, process, "timeout", exc, log_file)
    except OSError as exc:
        return _warning("OTLP traces", "-", port, process, "not listening", exc, log_file)


def _check_prometheus_targets(*, host: str, port: int, http_get: HttpGet) -> tuple[list[str], list[str], str]:
    url = f"http://{format_host_port(host, port)}/api/v1/targets"
    try:
        response = http_get(url, timeout=REQUEST_TIMEOUT)
        if not 200 <= int(response.status_code) < 300:
            return [], [], f"Prometheus targets unavailable: HTTP {response.status_code}."
        payload = response.json()
        if payload.get("status") != "success":
            raise ValueError("Prometheus target response status is not success")
        active_targets = payload.get("data", {}).get("activeTargets", [])
        if not isinstance(active_targets, list):
            raise ValueError("Prometheus activeTargets must be a list")
    except requests.Timeout:
        return [], [], "Prometheus targets unavailable: request timed out."
    except requests.ConnectionError as exc:
        return [], [], f"Prometheus targets unavailable: {_connection_error_reason(exc)}."
    except (requests.RequestException, TypeError, ValueError) as exc:
        return [], [], f"Prometheus targets unavailable: {exc}."

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"up": 0, "down": 0})
    failures: list[str] = []
    for target in active_targets:
        labels = target.get("labels") or target.get("discoveredLabels") or {}
        job_name = str(labels.get("job") or "unknown")
        health = str(target.get("health") or "down")
        bucket = "up" if health == "up" else "down"
        counts[job_name][bucket] += 1
        if bucket == "down":
            address = str(target.get("scrapeUrl") or labels.get("instance") or target.get("globalUrl") or "unknown")
            error = str(target.get("lastError") or "unknown error")
            failures.append(f"{job_name}: {address} ({error})")

    if not active_targets:
        return ["0 active targets"], [], "No Prometheus targets are registered yet."

    summaries = [f"{job}: {values['up']} up, {values['down']} down" for job, values in sorted(counts.items())]
    return summaries, failures, ""


def _validate_server_health(response: Any) -> str | None:
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return "RL-Insight server health response must contain status=ok"
    return None


def _validate_grafana_health(response: Any) -> str | None:
    payload = response.json()
    if not isinstance(payload, dict):
        return "Grafana health response must be a JSON object"
    database = str(payload.get("database") or "").lower()
    if database and database != "ok":
        return f"Grafana database status is {database}"
    return None


def _process_fields(service: StartedService | None) -> tuple[str, str, str]:
    if service is None:
        return "-", "unknown", "-"
    pid = str(service.process.pid)
    process = "running" if service.process.poll() is None else "exited"
    return pid, process, str(service.log_file)


def _warning(
    service: str,
    pid: str,
    port: int,
    process: str,
    readiness: str,
    exc: BaseException,
    log_file: str,
) -> ServiceDiagnostic:
    return ServiceDiagnostic(
        service,
        pid,
        str(port),
        process,
        readiness,
        "WARNING",
        str(exc),
        log_file,
    )


def _connection_error_reason(exc: BaseException) -> str:
    message = str(exc).lower()
    if "name or service not known" in message or "temporary failure in name" in message:
        return "host resolution failed"
    if "refused" in message:
        return "connection refused"
    return "unreachable"
