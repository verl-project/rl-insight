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

"""Command-line helpers for RL-Insight observability stack management."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from omegaconf import DictConfig, OmegaConf

from .config import load_server_config_file


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``rl-insight``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    """Construct the root argument parser and ``server`` subcommands."""
    parser = argparse.ArgumentParser(prog="rl-insight")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser(
        "server", help="Manage Prometheus, Tempo, and Grafana."
    )
    server_subparsers = server.add_subparsers(dest="server_command", required=True)

    start = server_subparsers.add_parser(
        "start",
        help="Start Prometheus, Tempo, and Grafana.",
    )
    _add_common_config_args(start)
    mode_group = start.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--detach",
        action="store_true",
        help="Start in background and return immediately.",
    )
    mode_group.add_argument(
        "--attach-logs",
        action="store_true",
        help="Run in foreground and stream docker compose logs.",
    )
    start.set_defaults(func=_server_start)

    stop = server_subparsers.add_parser(
        "stop",
        help="Stop Prometheus, Tempo, and Grafana.",
    )
    _add_common_config_args(stop)
    stop.set_defaults(func=_server_stop)

    return parser


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--config`` shared by subcommands that read stack YAML."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Observability YAML; default is bundled experimental/config/services/config.yaml.",
    )


def _select_str(conf: DictConfig, key: str) -> str:
    """Return a stripped string for ``key``; empty becomes ``''``."""
    value = OmegaConf.select(conf, key)
    return str(value).strip() if value is not None else ""


def _require_stack_field(conf: DictConfig, key: str, desc: str) -> str:
    value = _select_str(conf, key)
    if not value:
        print(
            f"Error: missing required stack config field {desc} ({key}).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return value


def _require_stack_int(conf: DictConfig, key: str, desc: str) -> int:
    value = _require_stack_field(conf, key, desc)
    try:
        return int(value)
    except ValueError:
        print(
            f"Error: {desc} ({key}) must be an integer; got {value!r}.", file=sys.stderr
        )
        raise SystemExit(2)


def _otlp_http_publish_port(traces_endpoint: str) -> int:
    """Publish host port implied by ``otel.traces_endpoint``."""
    raw = traces_endpoint.strip()
    if not raw:
        return 4318
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme.lower() == "https":
        return 443
    return 4318


def _trainer_otlp_traces_url(host: str, traces_endpoint: str) -> str:
    """Resolve the trainer OTLP URL advertised to users."""
    if "127.0.0.1" in traces_endpoint or "localhost" in traces_endpoint.lower():
        port = _otlp_http_publish_port(traces_endpoint)
        return f"http://{host}:{port}/v1/traces".rstrip("/")
    return traces_endpoint.rstrip("/")


def _server_start(args: argparse.Namespace) -> int:
    """Start Docker Compose when ``server.backend`` is ``docker_compose``."""
    conf = load_server_config_file(config_path=args.config)
    if not _stack_management_enabled(conf, action="start"):
        return 0

    compose_file, project_name = _stack_compose_target(conf)
    traces_endpoint = _validate_start_config(conf)
    env = _stack_compose_env(conf, traces_endpoint)
    base_command = _compose_base_command(compose_file, project_name)

    _print_start_summary(conf, compose_file, traces_endpoint)

    if args.attach_logs:
        try:
            return subprocess.run(
                [*base_command, "up", "--quiet-pull"],
                check=False,
                env=env,
            ).returncode
        except KeyboardInterrupt:
            return 130

    result = _run_compose_command(
        [*base_command, "up", "--quiet-pull", "-d"],
        env=env,
        quiet=True,
    )
    if result.returncode != 0:
        _print_compose_error("start", result.returncode, result.stderr)
        return int(result.returncode)

    if args.detach:
        print("RL-Insight observability services are running in background mode.")
        return 0

    print("RL-Insight observability services are running. Press Ctrl+C to stop.")
    return _wait_for_stack_shutdown(base_command, env)


def _server_stop(args: argparse.Namespace) -> int:
    """Stop Docker Compose when ``server.backend`` is ``docker_compose``."""
    conf = load_server_config_file(config_path=args.config)
    if not _stack_management_enabled(conf, action="stop"):
        return 0

    compose_file, project_name = _stack_compose_target(conf)
    base_command = _compose_base_command(compose_file, project_name)

    print("Stopping RL-Insight observability services...")
    return _stop_compose_stack(
        base_command, env=_stack_compose_env(conf), announce_success=True
    )


def _stack_management_enabled(conf: DictConfig, action: str) -> bool:
    if not bool(conf.server.get("enable", True)):
        print("RL-Insight server management is disabled by config.")
        return False

    backend = str(conf.server.get("backend"))
    if backend != "docker_compose":
        print(f"Server backend {backend!r} is external; nothing to {action}.")
        return False

    return True


def _stack_compose_target(conf: DictConfig) -> tuple[Path, str]:
    _require_stack_field(conf, "server.project_name", "server project name")
    _require_stack_field(conf, "server.compose_file", "server compose file")
    compose_file = Path(str(conf.server.compose_file))
    project_name = _select_str(conf, "server.project_name")
    return compose_file, project_name


def _validate_start_config(conf: DictConfig) -> str:
    _require_stack_int(conf, "prometheus.prometheus_port", "Prometheus HTTP port")
    _require_stack_field(conf, "prometheus.config_file", "Prometheus config file")
    traces_endpoint = _require_stack_field(
        conf, "otel.traces_endpoint", "OTLP traces endpoint"
    )
    _require_stack_int(conf, "tempo.query_port", "Tempo query port")
    _require_stack_field(conf, "tempo.config_file", "Tempo config file")
    _require_stack_int(conf, "grafana.port", "Grafana port")
    _require_stack_field(conf, "grafana.config_file", "Grafana config file")
    _require_stack_field(
        conf, "grafana.provisioning_dir", "Grafana provisioning directory"
    )
    _require_stack_field(conf, "grafana.dashboards_dir", "Grafana dashboards directory")
    return traces_endpoint


def _print_start_summary(
    conf: DictConfig, compose_file: Path, traces_endpoint: str
) -> None:
    host = _advertised_host()
    prom_port = int(conf.prometheus.prometheus_port)
    tempo_query = int(conf.tempo.query_port)
    grafana_port = int(conf.grafana.port)
    otlp_trainer_url = _trainer_otlp_traces_url(host, traces_endpoint)

    print("Starting RL-Insight observability services...")
    print(f"Observability node IP (LAN): {host}")
    print(f"Prometheus config file: {conf.prometheus.config_file}")
    print(f"OpenTelemetry OTLP traces URL (trainers): {otlp_trainer_url}")
    print(f"Compose file: {compose_file}")
    print(
        f"Prometheus UI: {_http_url(host, prom_port)}  "
        f"(Tempo query: {_http_url(host, tempo_query)})"
    )
    print(f"Tempo config file: {conf.tempo.config_file}")
    print(
        "Ray monitor hub: training calls init(); set "
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or init otel.traces_endpoint."
    )
    print(f"Grafana URL: {_http_url(host, grafana_port)}")


def _compose_base_command(compose_file: Path, project_name: str) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), "-p", project_name]


def _run_compose_command(
    command: Sequence[str],
    *,
    env: dict[str, str],
    quiet: bool,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    if quiet:
        return subprocess.run(
            command,
            check=False,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    return subprocess.run(command, check=False, env=env)


def _wait_for_stack_shutdown(base_command: Sequence[str], env: dict[str, str]) -> int:
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping RL-Insight observability services...")
            return _stop_compose_stack(base_command, env=env, announce_success=True)


def _stop_compose_stack(
    base_command: Sequence[str],
    *,
    env: dict[str, str],
    announce_success: bool,
) -> int:
    result = _run_compose_command(
        [*base_command, "down"],
        env=env,
        quiet=True,
    )
    if result.returncode != 0:
        _print_compose_error("stop", result.returncode, result.stderr)
        return int(result.returncode)

    if announce_success:
        print("RL-Insight observability services stopped.")
    return 0


def _print_compose_error(
    action: str, return_code: int, stderr: str | bytes | None
) -> None:
    message = ""
    if isinstance(stderr, bytes):
        message = stderr.decode(errors="replace").strip()
    elif isinstance(stderr, str):
        message = stderr.strip()

    print(
        message or f"docker compose {action} failed with exit code {return_code}.",
        file=sys.stderr,
    )


def _http_url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}"


def _advertised_host() -> str:
    explicit = os.environ.get("RLINSIGHT_ADVERTISED_IP", "").strip()
    if explicit:
        return explicit

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("198.51.100.1", 1))
            addr = sock.getsockname()[0]
            if addr and not addr.startswith("127."):
                return addr
    except OSError:
        pass

    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass

    return "127.0.0.1"


def _stack_compose_env(
    conf: DictConfig, traces_endpoint: str | None = None
) -> dict[str, str]:
    """Map stack YAML fields to docker compose environment variables."""
    env = dict(os.environ)

    def _set_env_int(config_key: str, env_key: str) -> None:
        value = OmegaConf.select(conf, config_key)
        if value in (None, ""):
            return
        try:
            env[env_key] = str(int(value))
        except (TypeError, ValueError):
            return

    def _set_env_str(config_key: str, env_key: str) -> None:
        value = OmegaConf.select(conf, config_key)
        if value in (None, ""):
            return
        env[env_key] = str(value)

    _set_env_int("prometheus.prometheus_port", "RLINSIGHT_PROMETHEUS_PORT")
    _set_env_str("prometheus.config_file", "RLINSIGHT_PROMETHEUS_CONFIG")
    _set_env_int("tempo.query_port", "RLINSIGHT_TEMPO_QUERY_PORT")
    _set_env_str("tempo.config_file", "RLINSIGHT_TEMPO_CONFIG")
    _set_env_int("grafana.port", "RLINSIGHT_GRAFANA_PORT")
    _set_env_str("grafana.config_file", "RLINSIGHT_GRAFANA_CONFIG")
    _set_env_str("grafana.provisioning_dir", "RLINSIGHT_GRAFANA_PROVISIONING")
    _set_env_str("grafana.dashboards_dir", "RLINSIGHT_GRAFANA_DASHBOARDS")

    endpoint = (traces_endpoint or _select_str(conf, "otel.traces_endpoint")).rstrip(
        "/"
    )
    if endpoint:
        env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = endpoint
        env["OTEL_EXPORTER_OTLP_TRACES_PORT"] = str(_otlp_http_publish_port(endpoint))

    return env


if __name__ == "__main__":
    raise SystemExit(main())
