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

"""Managed download and extraction for server service binaries."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import shutil
import stat
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from .catalog import SPECS, USER_AGENT


class ServiceInstaller:
    """Download release archives into RL-Insight's managed install root."""

    def __init__(
        self,
        conf: DictConfig,
        install_root: Path,
        *,
        find_binary: Callable[[str, Path], Path | None],
        find_grafana_homepath: Callable[[Path | None], Path | None],
    ):
        self.conf = conf
        self.install_root = install_root.resolve()
        self._find_binary = find_binary
        self._find_grafana_homepath = find_grafana_homepath

    def resolve_release(self, name: str) -> dict[str, str]:
        """Resolve release info for a service without downloading."""
        return self._resolve_release(name)

    def install(
        self,
        name: str,
        local_archive_dir: str | Path | None = None,
        release: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Download, extract, and return manifest data for one service."""
        release = release or self._resolve_release(name)
        local_archive_dir = (
            Path(local_archive_dir).expanduser().resolve()
            if local_archive_dir
            else None
        )
        package_dir = self.install_root / name / release["version"]
        archive_dir = self.install_root / "_downloads"
        archive_dir.mkdir(parents=True, exist_ok=True)
        package_dir.mkdir(parents=True, exist_ok=True)

        archive_path = archive_dir / release["asset"]

        # Check local archive directory first
        use_local = False
        if local_archive_dir:
            if not local_archive_dir.is_dir():
                print(
                    f"--local-archive is not a directory ({local_archive_dir}), downloading..."
                )
            else:
                local_path = local_archive_dir / release["asset"]
                if local_path.is_file():
                    shutil.copy2(local_path, archive_path)
                    print(f"Using local archive: {local_path}")
                    use_local = True
                else:
                    print(f"Local archive not found ({local_path}), downloading...")
        if not use_local:
            print(f"Downloading {name} {release['version']} from {release['url']}")
            self._download_file(release["url"], archive_path)
        self._extract_archive(archive_path, package_dir)

        binary = self._find_binary(name, package_dir)
        if binary is None:
            raise RuntimeError(
                f"Could not find {name} executable after extracting {archive_path}"
            )

        homepath = ""
        if name == "grafana":
            home = self._find_grafana_homepath(binary)
            homepath = str(home) if home else ""

        return {
            "version": release["version"],
            "binary_path": str(binary.resolve()),
            "homepath": homepath,
            "installed_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "source_url": release["url"],
        }

    def _resolve_release(self, name: str) -> dict[str, str]:
        os_token, arch_token, archive_ext = _platform_archive_tokens()
        configured_version = _select_str(self.conf, f"{name}.install_version")
        if name == "grafana":
            version = configured_version or self._latest_grafana_version()
            asset = f"grafana-{version}.{os_token}-{arch_token}{archive_ext}"
            return {
                "version": version,
                "asset": asset,
                "url": self._build_download_url(name, version, os_token, arch_token),
            }

        spec = SPECS[name]
        if not spec.github_repo:
            raise RuntimeError(f"No release source configured for {name}")
        tag = (
            f"v{configured_version}"
            if configured_version
            else str(self._github_latest_release(spec.github_repo)["tag_name"])
        )
        version = tag[1:] if tag.startswith("v") else tag
        if name == "prometheus":
            asset = f"prometheus-{version}.{os_token}-{arch_token}{archive_ext}"
        elif name == "tempo":
            asset = f"tempo_{version}_{os_token}_{arch_token}{archive_ext}"
        else:
            raise RuntimeError(f"Unsupported service: {name}")
        return {
            "version": version,
            "asset": asset,
            "url": self._build_download_url(name, version, os_token, arch_token),
        }

    def _build_download_url(
        self, name: str, version: str, os_token: str, arch_token: str
    ) -> str:
        """Build the download URL from the configured template or built-in defaults."""
        template = _select_str(self.conf, f"{name}.download_url_template")
        if not template:
            if name == "grafana":
                template = "https://dl.grafana.com/oss/release/grafana-{version}.{os}-{arch}.tar.gz"
            elif name == "prometheus":
                template = "https://github.com/prometheus/prometheus/releases/download/v{version}/prometheus-{version}.{os}-{arch}.tar.gz"
            elif name == "tempo":
                template = "https://github.com/grafana/tempo/releases/download/v{version}/tempo_{version}_{os}_{arch}.tar.gz"
            else:
                raise RuntimeError(f"No download source configured for {name}")
        return template.format(version=version, os=os_token, arch=arch_token)

    @staticmethod
    def _github_latest_release(repo: str) -> dict[str, Any]:
        try:
            return ServiceInstaller._read_json(
                f"https://api.github.com/repos/{repo}/releases/latest"
            )
        except RuntimeError:
            return {"tag_name": ServiceInstaller._github_latest_tag_from_redirect(repo)}

    @staticmethod
    def _github_latest_tag_from_redirect(repo: str) -> str:
        request = urllib.request.Request(
            f"https://github.com/{repo}/releases/latest",
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                latest_url = response.geturl().rstrip("/")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to resolve latest release for {repo}: {exc}"
            ) from exc
        tag = latest_url.rsplit("/", 1)[-1]
        if not tag or tag == "latest":
            raise RuntimeError(f"Failed to resolve latest release tag for {repo}")
        return tag

    @staticmethod
    def _latest_grafana_version() -> str:
        data = ServiceInstaller._read_json("https://grafana.com/api/grafana/versions")
        for item in data.get("items", []):
            channels = item.get("channels") or {}
            if channels.get("stable"):
                return str(item["version"])
        raise RuntimeError("Could not resolve latest stable Grafana version")

    @staticmethod
    def _read_json(url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Failed to query {url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to query {url}: {exc.reason}") from exc

    @staticmethod
    def _download_file(url: str, target: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with target.open("wb") as output:
                    shutil.copyfileobj(response, output)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc

    @staticmethod
    def _extract_archive(archive_path: Path, target_dir: Path) -> None:
        with tarfile.open(archive_path) as archive:
            ServiceInstaller._safe_extract_tar(archive, target_dir)
        ServiceInstaller._mark_executables(target_dir)

    @staticmethod
    def _safe_extract_tar(archive: tarfile.TarFile, target_dir: Path) -> None:
        root = target_dir.resolve()
        for member in archive.getmembers():
            target = (target_dir / member.name).resolve()
            if root not in (target, *target.parents):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(target_dir)

    @staticmethod
    def _mark_executables(root: Path) -> None:
        names = {exe for spec in SPECS.values() for exe in spec.executables}
        for path in root.rglob("*"):
            if path.is_file() and path.name in names:
                path.chmod(
                    path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                )


def _platform_archive_tokens() -> tuple[str, str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux":
        raise RuntimeError(
            "RL-Insight automatic service install currently supports Linux only."
        )
    os_token = "linux"
    archive_ext = ".tar.gz"

    if machine in {"x86_64", "amd64"}:
        arch_token = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch_token = "arm64"
    else:
        raise RuntimeError(f"Unsupported CPU architecture for auto-install: {machine}")
    return os_token, arch_token, archive_ext


def _select_str(conf: DictConfig, key: str) -> str:
    value = OmegaConf.select(conf, key)
    return str(value).strip() if value is not None else ""
