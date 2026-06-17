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

"""Dependency discovery and version checks for server services."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from .catalog import (
    DEFAULT_INSTALL_ROOT,
    MANIFEST_FILE,
    SERVICE_NAMES,
    SPECS,
    SYSTEM_BINARY_PATHS,
)
from .installer import ServiceInstaller


@dataclass
class ServiceStatus:
    name: str
    enabled: bool
    binary: Path | None = None
    source: str = ""
    version: str = ""
    detail: str = ""
    min_version: str = ""
    current_version: str = ""

    @property
    def ok(self) -> bool:
        return (not self.enabled) or (
            self.binary is not None
            and not self.detail.startswith("version <")
            and self.detail != "version unknown"
        )


@dataclass(frozen=True)
class _BinaryCandidate:
    binary: Path | None
    source: str
    detail: str


class MissingDependencyError(RuntimeError):
    """Raised when required service binaries cannot be found."""

    def __init__(self, missing: Sequence[ServiceStatus]):
        super().__init__("missing service dependencies")
        self.missing = list(missing)


def install_root_from_config(
    conf: DictConfig, install_dir: str | Path | None = None
) -> Path:
    """Resolve the service install root from CLI override or config."""
    raw = install_dir or OmegaConf.select(conf, "server.install_dir")
    path = Path(str(raw)).expanduser() if raw else DEFAULT_INSTALL_ROOT
    return path.resolve()


class DependencyManager:
    """Find existing service binaries or install managed copies."""

    def __init__(self, conf: DictConfig, install_root: Path | None = None):
        self.conf = conf
        self.install_root = (install_root or install_root_from_config(conf)).resolve()

    def enabled_services(self) -> list[str]:
        """Return services enabled by stack YAML."""
        services: list[str] = []
        for name in SERVICE_NAMES:
            enabled = bool(OmegaConf.select(self.conf, f"{name}.enable", default=True))
            if enabled:
                services.append(name)
        return services

    def check(self, *, include_versions: bool = True) -> list[ServiceStatus]:
        """Find configured, installed, or PATH-provided service binaries."""
        manifest = self._read_manifest()
        statuses: list[ServiceStatus] = []
        for name in SERVICE_NAMES:
            enabled = bool(OmegaConf.select(self.conf, f"{name}.enable", default=True))
            if not enabled:
                statuses.append(
                    ServiceStatus(name=name, enabled=False, detail="disabled by config")
                )
                continue

            min_version = _select_str(self.conf, f"{name}.min_version")
            statuses.append(
                self._resolve_status(
                    name=name,
                    candidates=self._binary_candidates(name, manifest),
                    min_version=min_version,
                    include_versions=include_versions,
                )
            )
        return statuses

    @staticmethod
    def missing(statuses: Sequence[ServiceStatus]) -> list[ServiceStatus]:
        return [status for status in statuses if status.enabled and not status.ok]

    def install_missing(self, *, force: bool = False) -> list[ServiceStatus]:
        """Download and install enabled services that are missing locally."""
        self.install_root.mkdir(parents=True, exist_ok=True)
        before = self.check(include_versions=True)
        targets = [
            status.name
            for status in before
            if status.enabled and (force or not status.ok)
        ]
        if not targets:
            return self.check()

        manifest = self._read_manifest()
        installer = ServiceInstaller(
            self.conf,
            self.install_root,
            find_binary=self._find_binary_under,
            find_grafana_homepath=self.find_grafana_homepath,
        )
        for name in targets:
            info = installer.install(name)
            manifest.setdefault("services", {})[name] = info
            self._write_manifest(manifest)

        return self.check()

    def resolve_binary(
        self, name: str, manifest: dict[str, Any] | None = None
    ) -> tuple[Path | None, str, str]:
        candidate = self._binary_candidates(name, manifest)[0]
        return candidate.binary, candidate.source, candidate.detail

    def _binary_candidates(
        self, name: str, manifest: dict[str, Any] | None = None
    ) -> list[_BinaryCandidate]:
        spec = SPECS[name]
        manifest = manifest or self._read_manifest()
        candidates: list[_BinaryCandidate] = []
        seen: set[Path] = set()

        def add(path: Path, source: str) -> None:
            resolved = path.expanduser().resolve()
            if resolved in seen or not self._is_executable_file(resolved):
                return
            seen.add(resolved)
            candidates.append(_BinaryCandidate(resolved, source, str(resolved)))

        config_path = _select_str(self.conf, f"{name}.binary_path")
        if config_path:
            path = Path(config_path).expanduser()
            if self._is_executable_file(path):
                return [_BinaryCandidate(path.resolve(), "config", str(path.resolve()))]
            return [
                _BinaryCandidate(
                    None,
                    "config",
                    f"configured path does not exist: {path}",
                )
            ]

        service_info = manifest.get("services", {}).get(name, {})
        manifest_path = service_info.get("binary")
        if manifest_path:
            add(Path(str(manifest_path)), "managed")

        installed = self._find_installed_binary(name)
        if installed:
            add(installed, "managed")

        for exe in spec.executables:
            found = shutil.which(exe)
            if found:
                add(Path(found), "PATH")

        system_binary = self._find_system_binary(name)
        if system_binary:
            add(system_binary, "system")

        if candidates:
            return candidates

        return [_BinaryCandidate(None, "missing", "run `rl-insight server install`")]

    def _resolve_status(
        self,
        *,
        name: str,
        candidates: Sequence[_BinaryCandidate],
        min_version: str,
        include_versions: bool,
    ) -> ServiceStatus:
        statuses = [
            self._status_from_candidate(
                name=name,
                candidate=candidate,
                min_version=min_version,
                include_versions=include_versions,
            )
            for candidate in candidates
        ]
        for status in statuses:
            if status.ok:
                return status
        return statuses[0]

    def _status_from_candidate(
        self,
        *,
        name: str,
        candidate: _BinaryCandidate,
        min_version: str,
        include_versions: bool,
    ) -> ServiceStatus:
        version = ""
        current_version = ""
        detail = candidate.detail
        if candidate.binary and include_versions:
            version = self._binary_version(candidate.binary)
            current_version = _extract_semver(version)
            if min_version and not current_version:
                detail = "version unknown"
            elif min_version and _version_tuple(current_version) < _version_tuple(
                min_version
            ):
                detail = f"version < {min_version}"
        return ServiceStatus(
            name=name,
            enabled=True,
            binary=candidate.binary,
            source=candidate.source,
            version=version,
            detail=detail,
            min_version=min_version,
            current_version=current_version,
        )

    def resolve_grafana_homepath(self, binary: Path | None = None) -> Path | None:
        raw = _select_str(self.conf, "grafana.homepath")
        if raw:
            return Path(raw).expanduser().resolve()

        manifest = self._read_manifest()
        home = manifest.get("services", {}).get("grafana", {}).get("homepath")
        if home:
            return Path(str(home)).expanduser().resolve()

        if binary is None:
            binary, _, _ = self.resolve_binary("grafana", manifest)
        return self.find_grafana_homepath(binary) if binary else None

    @classmethod
    def find_grafana_homepath(cls, binary: Path | None) -> Path | None:
        if binary is None:
            return None
        for parent in [binary.parent, *binary.parents]:
            if (parent / "conf").exists() and (parent / "public").exists():
                return parent.resolve()
        for path in (
            Path("/usr/share/grafana"),
            Path("/usr/local/share/grafana"),
            Path("/opt/homebrew/share/grafana"),
            Path("/usr/local/opt/grafana/share/grafana"),
        ):
            if (path / "conf").exists() and (path / "public").exists():
                return path.resolve()
        return None

    @staticmethod
    def _is_executable_file(path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False
        return os.access(path, os.X_OK)

    @classmethod
    def _find_system_binary(cls, name: str) -> Path | None:
        for path in SYSTEM_BINARY_PATHS.get(name, ()):
            if cls._is_executable_file(path):
                return path
        return None

    def _find_installed_binary(self, name: str) -> Path | None:
        service_root = self.install_root / name
        if not service_root.exists():
            return None
        return self._find_binary_under(name, service_root)

    @classmethod
    def _find_binary_under(cls, name: str, root: Path) -> Path | None:
        spec = SPECS[name]
        candidates: list[Path] = []
        for exe in spec.executables:
            candidates.extend(root.rglob(exe))
        candidates = [path for path in candidates if cls._is_executable_file(path)]
        if not candidates:
            return None
        return sorted(candidates, key=lambda path: len(path.parts))[0]

    @staticmethod
    def _binary_version(binary: Path) -> str:
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        first_line = (result.stdout or "").strip().splitlines()
        return first_line[0].strip() if first_line else ""

    def _read_manifest(self) -> dict[str, Any]:
        path = self.install_root / MANIFEST_FILE
        if not path.exists():
            return {"services": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"services": {}}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.install_root.mkdir(parents=True, exist_ok=True)
        path = self.install_root / MANIFEST_FILE
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )


def _select_str(conf: DictConfig, key: str) -> str:
    value = OmegaConf.select(conf, key)
    return str(value).strip() if value is not None else ""


def _extract_semver(text: str) -> str:
    match = re.search(r"(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else ""


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(part) for part in version.split(".")[:3]]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])
