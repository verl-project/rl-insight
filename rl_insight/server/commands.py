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

"""Command handlers for ``rl-insight server``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import Sequence

import requests
from omegaconf import DictConfig, OmegaConf

from ..utils.monitor_config_loader import load_server_config_file
from ..utils.constants import MonitorEnv
from ..utils.prometheus_utils import PrometheusTarget, PrometheusTargetStore
from .dependencies import MissingDependencyError, ServiceStatus
from .display import (
    active_state_rows,
    dependency_rows,
    format_logo,
    format_panel,
    format_table,
)
from .network import format_host_port, local_addresses
from .runtime import StartedService
from .services import ServerServiceManager


class ServerCommands:
    """CLI command object for installing and running local server services."""

    def __init__(
        self,
        *,
        validator: ServerConfigValidator | None = None,
        console: ServerConsole | None = None,
    ):
        self.validator = validator or ServerConfigValidator()
        self.console = console or ServerConsole()

    def install(self, args: argparse.Namespace) -> int:
        """Install missing local service binaries."""
        conf = self._load_config(args)
        if not self._stack_management_enabled(conf, action="install"):
            return 0

        manager = ServerServiceManager(conf, install_dir=args.install_dir)
        local_archive_dir = getattr(args, "local_archive", None)
        before = manager.check_dependencies()
        missing = manager.missing_dependencies(before)
        if not missing and not args.force:
            print("RL-Insight server dependencies are already available.")
            self.console.print_dependencies(before)
            return 0

        if missing:
            print("Missing or incompatible server dependencies:")
            self.console.print_dependencies(missing)

        # Preview planned download URLs
        planned = manager.plan_install(
            targets=[s.name for s in (missing or before) if s.enabled]
        )
        if planned:
            print("\nPlanned downloads:")
            for plan in planned:
                print(f"  {plan['name']:12} {plan['version']:8}  {plan['url']}")
            if local_archive_dir:
                print(f"\nLooking for local archives in: {local_archive_dir}")
            print()

        try:
            statuses = manager.install_missing_dependencies(
                force=args.force,
                local_archive_dir=local_archive_dir,
                planned_releases=planned,
            )
        except RuntimeError as exc:
            print(f"Install failed: {exc}", file=sys.stderr)
            return 1

        print(f"Installed server dependencies under: {manager.install_root}")
        self.console.print_dependencies(statuses)
        return 0 if not manager.missing_dependencies(statuses) else 1

    def start(self, args: argparse.Namespace) -> int:
        """Start local RL-Insight server, Prometheus, Tempo, and Grafana processes."""
        conf = self._load_config(args)
        if not self._stack_management_enabled(conf, action="start"):
            return 0

        self.validator.validate_start(conf)
        manager = ServerServiceManager(conf)
        active_state = manager.active_state()
        if active_state:
            print("RL-Insight server services already appear to be running.")
            print(
                format_table(["Service", "PID", "Log"], active_state_rows(active_state))
            )
            return 0

        statuses = manager.check_dependencies()
        missing = manager.missing_dependencies(statuses)
        if missing:
            self.console.print_missing_start_dependencies(missing)
            return 2

        self.console.print_start_summary(manager, conf)

        try:
            stack = manager.start(detach=args.detach, attach_logs=args.attach_logs)
        except MissingDependencyError as exc:
            self.console.print_missing_start_dependencies(exc.missing)
            return 2
        except RuntimeError as exc:
            print(f"Failed to start RL-Insight server services: {exc}", file=sys.stderr)
            return 1

        if stack is None:
            print("RL-Insight server services already appear to be running.")
            return 0

        self.console.print_running_summary(conf, stack.services)

        if args.detach:
            print("RL-Insight server services are running in background mode.")
            return 0

        print("RL-Insight server services are running. Press Ctrl+C to stop.")
        return manager.wait(stack, attach_logs=args.attach_logs)

    def stop(self, args: argparse.Namespace) -> int:
        """Stop local RL-Insight server, Prometheus, Tempo, and Grafana processes."""
        conf = self._load_config(args)
        if not self._stack_management_enabled(conf, action="stop"):
            return 0

        manager = ServerServiceManager(conf)
        print("Stopping RL-Insight server services...")
        code, stopped = manager.stop()
        if stopped:
            print(
                format_table(
                    ["Service", "PID", "Status"],
                    [[row["name"], row["pid"], row["status"]] for row in stopped],
                )
            )
            print("RL-Insight server services stopped.")
        else:
            print("No running RL-Insight server services were found.")
        return code

    def add_targets(self, args: argparse.Namespace) -> int:
        """Register Prometheus scrape targets from a YAML file."""
        try:
            target_conf = OmegaConf.load(args.target_file.expanduser())
            raw_jobs = OmegaConf.select(target_conf, "jobs")
            if not OmegaConf.is_list(raw_jobs) or not raw_jobs:
                raise ValueError("jobs must be a non-empty list")

            jobs = []
            for index, raw_job in enumerate(raw_jobs):
                payload = OmegaConf.to_container(raw_job, resolve=True)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"jobs[{index}] must be an object")
                jobs.append(payload)
        except (OSError, ValueError) as exc:
            print(f"Invalid target file: {exc}", file=sys.stderr)
            return 2

        conf = self._load_config(args)
        store = PrometheusTargetStore.from_config(conf)

        try:
            for job in jobs:
                raw_targets = job.get("targets")
                if not isinstance(raw_targets, list) or not raw_targets:
                    raise ValueError("targets must be a non-empty list")

                default_labels = job.get("labels") or {}
                if not isinstance(default_labels, Mapping):
                    raise ValueError("labels must be an object")

                targets: list[PrometheusTarget] = []
                for target in raw_targets:
                    if isinstance(target, str):
                        address = target.strip()
                        target_labels = default_labels
                    elif isinstance(target, Mapping):
                        target_labels = target.get("labels") or {}
                        if not isinstance(target_labels, Mapping):
                            raise ValueError("target labels must be an object")
                        address = str(target.get("target") or "").strip()
                        target_labels = {**default_labels, **target_labels}
                    else:
                        raise ValueError(
                            "each target must be either a string or an object"
                        )
                    if not address:
                        raise ValueError("target must be a non-empty string")
                    targets.append(PrometheusTarget(address, target_labels))

                job_name = str(job.get("job_name") or "").strip()
                store.register(job_name, targets)
                print(f"Added {len(targets)} target(s) to job {job_name!r}.")
            store.reload()
        except (OSError, ValueError, requests.RequestException) as exc:
            print(f"Failed to add Prometheus targets: {exc}", file=sys.stderr)
            return 1
        return 0

    @staticmethod
    def _load_config(args: argparse.Namespace) -> DictConfig:
        return load_server_config_file(config_path=args.config)

    @staticmethod
    def _stack_management_enabled(conf: DictConfig, action: str) -> bool:
        if not bool(conf.server.get("enable", True)):
            print("RL-Insight server management is disabled by config.")
            return False

        backend = str(conf.server.get("backend", "local"))
        if backend != "local":
            print(
                f"Server backend {backend!r} is external; nothing to {action}. "
                "Manage the observability stack with your external deployment."
            )
            return False

        return True


class ServerConfigValidator:
    """Validate the service fields needed before starting the local stack."""

    def validate_start(self, conf: DictConfig) -> None:
        """Validate required fields before starting the local stack."""
        if bool(OmegaConf.select(conf, "server.enable", default=True)):
            self._require_int(conf, "server.port", "RL-Insight server port")

        if bool(OmegaConf.select(conf, "prometheus.enable", default=True)):
            self._require_int(
                conf, "prometheus.prometheus_port", "Prometheus HTTP port"
            )

        if bool(OmegaConf.select(conf, "tempo.enable", default=True)):
            self._require_int(conf, "otel.otel_port", "OTLP HTTP port")
            self._require_int(conf, "tempo.query_port", "Tempo query port")
            self._require_field(conf, "tempo.config_file", "Tempo config file")

        if bool(OmegaConf.select(conf, "grafana.enable", default=True)):
            self._require_int(conf, "grafana.port", "Grafana port")
            self._require_field(conf, "grafana.config_file", "Grafana config file")
            self._require_field(
                conf, "grafana.provisioning_dir", "Grafana provisioning directory"
            )
            self._require_field(
                conf, "grafana.dashboards_dir", "Grafana dashboards directory"
            )

    @staticmethod
    def _select_str(conf: DictConfig, key: str) -> str:
        value = OmegaConf.select(conf, key)
        return str(value).strip() if value is not None else ""

    def _require_field(self, conf: DictConfig, key: str, desc: str) -> str:
        value = self._select_str(conf, key)
        if value:
            return value
        print(
            f"Error: missing required server config field {desc} ({key}).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    def _require_int(self, conf: DictConfig, key: str, desc: str) -> int:
        value = self._require_field(conf, key, desc)
        try:
            return int(value)
        except ValueError:
            print(
                f"Error: {desc} ({key}) must be an integer; got {value!r}.",
                file=sys.stderr,
            )
            raise SystemExit(2) from None


class ServerConsole:
    """Render concise terminal output for server commands."""

    @staticmethod
    def print_dependencies(statuses: Sequence[ServiceStatus]) -> None:
        print(
            format_table(
                ["Service", "Status", "Source", "Version", "Location"],
                dependency_rows(statuses),
            )
        )

    @staticmethod
    def print_missing_start_dependencies(missing: Sequence[ServiceStatus]) -> None:
        print("Missing or incompatible server software:", file=sys.stderr)
        print(
            format_table(
                ["Service", "Status", "Source", "Version", "Location"],
                dependency_rows(missing),
            ),
            file=sys.stderr,
        )
        print(
            "\nInstall supported versions with:\n"
            "  rl-insight server install\n\n"
            "Or install Prometheus, Tempo, and Grafana with your Linux package manager.",
            file=sys.stderr,
        )

    def print_start_summary(
        self,
        manager: ServerServiceManager,
        conf: DictConfig,
    ) -> None:
        addresses = local_addresses()
        panel_rows = []
        if addresses["ipv4"]:
            panel_rows.append(("Node IPv4", addresses["ipv4"]))
        if addresses["ipv6"]:
            panel_rows.append(("Node IPv6", addresses["ipv6"]))
        if not panel_rows:
            panel_rows.append(("Node", "unknown"))
        panel_rows.extend(
            [
                (
                    "Listen",
                    "IPv6 (::)" if addresses["family"] == "ipv6" else "IPv4 (0.0.0.0)",
                ),
                ("Status", "starting"),
                ("Logs", "enabled"),
            ]
        )

        print(format_logo())
        print(format_panel("[RL-INSIGHT] Server Stack", panel_rows))
        print(
            format_table(
                ["Service", "Port", "Purpose"],
                [
                    [row["service"], row["port"], row["purpose"]]
                    for row in manager.service_rows()
                ],
            )
        )

    def print_running_summary(
        self,
        conf: DictConfig,
        services: Sequence[StartedService],
    ) -> None:
        addresses = local_addresses()
        host = addresses["host"]
        family_label = "IPv6" if addresses["family"] == "ipv6" else "IPv4"
        grafana_url = ""
        if bool(OmegaConf.select(conf, "grafana.enable", default=True)):
            grafana_url = _service_url(host, conf.grafana.port)
        server_url = _server_url(conf, host)
        rows = [
            [service.name, service.process.pid, service.log_file]
            for service in services
        ]
        print(format_table(["Service", "PID", "Log"], rows))
        print("RL-Insight server services are ready.")
        if server_url:
            print(
                "Training side: set "
                f"{MonitorEnv.SERVER_URL} to the RL-Insight server URL "
                "reachable from training workers."
            )
            print(f"Detected {family_label} URL candidate:")
            print(f"  export {MonitorEnv.SERVER_URL}={server_url}")
        if grafana_url:
            print(f"View monitoring dashboard ({family_label}):")
            print(f"  {grafana_url}")


def _service_url(host: str, port: object, path: str = "") -> str:
    if not host:
        return ""
    normalized_path = path if not path or path.startswith("/") else f"/{path}"
    return f"http://{format_host_port(host, port)}{normalized_path}"


def _server_url(conf: DictConfig, host: str) -> str:
    if not bool(OmegaConf.select(conf, "server.enable", default=True)):
        return ""
    return _service_url(host, conf.server.port)
