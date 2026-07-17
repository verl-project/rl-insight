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

"""High-level server service manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .dependencies import (
    DependencyManager,
    ServiceStatus,
    install_root_from_config,
)
from .runtime import LocalServiceRuntime, StartedStack


class ServerServiceManager:
    """Coordinate dependency checks, installation, and local service runtime."""

    def __init__(
        self,
        conf: DictConfig,
        install_root: Path | None = None,
        install_dir: str | Path | None = None,
    ):
        self.conf = conf
        install_location = install_root if install_root is not None else install_dir
        self.install_root = install_root_from_config(conf, install_dir=install_location)
        self.dependencies = DependencyManager(conf, self.install_root)
        self.runtime = LocalServiceRuntime(conf, self.install_root, self.dependencies)

    def check_dependencies(
        self, *, include_versions: bool = True
    ) -> list[ServiceStatus]:
        return self.dependencies.check(include_versions=include_versions)

    def missing_dependencies(
        self, statuses: list[ServiceStatus] | None = None
    ) -> list[ServiceStatus]:
        if statuses is None:
            statuses = self.check_dependencies()
        return self.dependencies.missing(statuses)

    def install_missing_dependencies(
        self,
        *,
        force: bool = False,
        local_archive_dir: str | Path | None = None,
        planned_releases: list[dict[str, Any]] | None = None,
    ) -> list[ServiceStatus]:
        return self.dependencies.install_missing(
            force=force,
            local_archive_dir=local_archive_dir,
            planned_releases=planned_releases,
        )

    def plan_install(self, *, targets: list[str]) -> list[dict[str, Any]]:
        return self.dependencies.plan_install(targets=targets)

    def active_state(self) -> dict[str, Any] | None:
        return self.runtime.active_state()

    def start(self, *, detach: bool, attach_logs: bool) -> StartedStack | None:
        return self.runtime.start(detach=detach, attach_logs=attach_logs)

    def wait(self, stack: StartedStack, *, attach_logs: bool) -> int:
        return self.runtime.wait(stack, attach_logs=attach_logs)

    def stop(self) -> tuple[int, list[dict[str, Any]]]:
        return self.runtime.stop()

    def service_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if bool(OmegaConf.select(self.conf, "server.enable", default=True)):
            rows.append(
                {
                    "service": "RL-Insight server",
                    "port": str(int(self.conf.server.port)),
                    "purpose": "service control and discovery",
                }
            )
        if bool(OmegaConf.select(self.conf, "prometheus.enable", default=True)):
            rows.append(
                {
                    "service": "Prometheus",
                    "port": str(int(self.conf.prometheus.prometheus_port)),
                    "purpose": "metrics UI",
                }
            )
        if bool(OmegaConf.select(self.conf, "tempo.enable", default=True)):
            rows.extend(
                [
                    {
                        "service": "Tempo",
                        "port": str(int(self.conf.tempo.query_port)),
                        "purpose": "trace query API",
                    },
                    {
                        "service": "OTLP traces",
                        "port": str(int(self.conf.otel.otel_port)),
                        "purpose": "trainer export URL",
                    },
                ]
            )
        if bool(OmegaConf.select(self.conf, "grafana.enable", default=True)):
            rows.append(
                {
                    "service": "Grafana",
                    "port": str(int(self.conf.grafana.port)),
                    "purpose": "dashboard UI",
                }
            )
        return rows


ObservabilityServiceManager = ServerServiceManager
