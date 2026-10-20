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

"""Service names, binary names, and standard local paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.constants import MonitorPaths

SERVICE_NAMES = ("prometheus", "tempo", "grafana")
DEFAULT_INSTALL_ROOT = Path.home() / ".rl-insight" / "services"
DEFAULT_STATE_ROOT = MonitorPaths.STATE_ROOT
MANIFEST_FILE = "manifest.json"
STATE_FILE = "rl-insight-services.json"
USER_AGENT = "rl-insight-service-installer"

SYSTEM_BINARY_PATHS = {
    "prometheus": (
        Path("/usr/bin/prometheus"),
        Path("/usr/local/bin/prometheus"),
        Path("/opt/prometheus/prometheus"),
    ),
    "tempo": (
        Path("/usr/bin/tempo"),
        Path("/usr/local/bin/tempo"),
        Path("/opt/tempo/tempo"),
    ),
    "grafana": (
        Path("/usr/sbin/grafana-server"),
        Path("/usr/bin/grafana-server"),
        Path("/usr/local/bin/grafana-server"),
        Path("/usr/share/grafana/bin/grafana-server"),
        Path("/usr/bin/grafana"),
        Path("/usr/local/bin/grafana"),
        Path("/usr/share/grafana/bin/grafana"),
    ),
}


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    executables: tuple[str, ...]
    asset_template: str
    download_url_template: str
    github_repo: str | None = None
    windows_archive_ext: str = ".zip"


SPECS = {
    "prometheus": ServiceSpec(
        name="prometheus",
        executables=("prometheus",),
        asset_template="prometheus-{version}.{os}-{arch}{ext}",
        download_url_template="https://github.com/prometheus/prometheus/releases/download/v{version}/{asset}",
        github_repo="prometheus/prometheus",
    ),
    "tempo": ServiceSpec(
        name="tempo",
        executables=("tempo",),
        asset_template="tempo_{version}_{os}_{arch}{ext}",
        download_url_template="https://github.com/grafana/tempo/releases/download/v{version}/{asset}",
        github_repo="grafana/tempo",
        windows_archive_ext=".tar.gz",
    ),
    "grafana": ServiceSpec(
        name="grafana",
        executables=("grafana-server", "grafana"),
        asset_template="grafana-{version}.{os}-{arch}{ext}",
        download_url_template="https://dl.grafana.com/oss/release/{asset}",
    ),
}
