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

"""Local process runtime for RL-Insight server, Prometheus, Tempo, and Grafana."""

from __future__ import annotations

import configparser
import datetime as _dt
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from omegaconf import DictConfig, OmegaConf

from .catalog import DEFAULT_STATE_ROOT, STATE_FILE
from .network import format_host_port, local_addresses
from ..utils.constants import (
    PrometheusScrape,
    prometheus_targets_file_from_config,
)
from .dependencies import (
    MissingDependencyError,
    DependencyManager,
)


@dataclass(frozen=True)
class RuntimeFiles:
    prometheus_config: Path
    tempo_config: Path
    grafana_config: Path
    server_config: Path
    grafana_homepath: Path | None


@dataclass
class StartedService:
    name: str
    process: subprocess.Popen[Any]
    command: list[str]
    log_file: Path


@dataclass
class StartedStack:
    services: list[StartedService]
    state_file: Path
    install_root: Path


class LocalServiceRuntime:
    """Prepare config files, launch services, and stop recorded processes."""

    def __init__(
        self,
        conf: DictConfig,
        install_root: Path,
        dependencies: DependencyManager | None = None,
    ):
        self.conf = conf
        self.install_root = install_root.resolve()
        self.dependencies = dependencies or DependencyManager(conf, self.install_root)
        self.state_file = _state_file_from_config(self.conf)

    def prepare_files(
        self,
        *,
        grafana_binary: Path | None = None,
        tempo_version: str = "",
    ) -> RuntimeFiles:
        """Render local runtime config files for managed services."""
        runtime_dir = _runtime_dir_from_config(self.conf)
        data_dir = _service_data_root(self.conf, self.install_root)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        runtime_config = _write_runtime_config(self.conf, runtime_dir)

        prometheus_config = Path(
            str(OmegaConf.select(self.conf, "prometheus.config_file"))
        )
        if bool(OmegaConf.select(self.conf, "prometheus.enable", default=True)):
            prometheus_config = _render_prometheus_config(self.conf, runtime_dir)
        tempo_config = Path(str(OmegaConf.select(self.conf, "tempo.config_file")))
        grafana_config = Path(str(OmegaConf.select(self.conf, "grafana.config_file")))
        if bool(OmegaConf.select(self.conf, "tempo.enable", default=True)):
            tempo_config = _render_tempo_config(
                self.conf, runtime_dir, data_dir, tempo_version
            )
        if bool(OmegaConf.select(self.conf, "grafana.enable", default=True)):
            grafana_config = _render_grafana_config(self.conf, runtime_dir, data_dir)
            dashboard_path = _stage_grafana_dashboards(self.conf, runtime_dir)
            _render_grafana_provisioning(self.conf, runtime_dir, dashboard_path)

        return RuntimeFiles(
            prometheus_config=prometheus_config,
            tempo_config=tempo_config,
            grafana_config=grafana_config,
            server_config=runtime_config,
            grafana_homepath=self.dependencies.resolve_grafana_homepath(grafana_binary),
        )

    def start(self, *, detach: bool, attach_logs: bool) -> StartedStack | None:
        """Start local service processes and write the PID state file."""
        active_state = load_active_state(self.state_file)
        if active_state:
            return None

        statuses = self.dependencies.check(include_versions=True)
        missing = self.dependencies.missing(statuses)
        if missing:
            raise MissingDependencyError(missing)

        status_by_name = {status.name: status for status in statuses}
        tempo_status = status_by_name.get("tempo")
        grafana_status = status_by_name.get("grafana")
        grafana_binary = grafana_status.binary if grafana_status else None
        runtime_files = self.prepare_files(
            grafana_binary=grafana_binary,
            tempo_version=tempo_status.current_version if tempo_status else "",
        )
        log_dir = (self.install_root / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        started: list[StartedService] = []
        try:
            for name in self.dependencies.enabled_services():
                binary = status_by_name[name].binary
                if binary is None:
                    raise MissingDependencyError([status_by_name[name]])
                command = _service_command(
                    name, binary, self.conf, runtime_files, self.install_root
                )
                log_file = log_dir / f"{name}.log"
                process = _spawn_service(name, command, log_file)
                started.append(
                    StartedService(
                        name=name,
                        process=process,
                        command=command,
                        log_file=log_file,
                    )
                )
                time.sleep(0.3)
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"{name} exited during startup with code {return_code}. "
                        f"See log: {log_file}"
                    )
            if _server_enabled(self.conf):
                name = "rl-insight-server"
                command = _server_command(self.conf, runtime_files)
                log_file = log_dir / "rl-insight-server.log"
                process = _spawn_service(name, command, log_file)
                started.append(
                    StartedService(
                        name=name,
                        process=process,
                        command=command,
                        log_file=log_file,
                    )
                )
                time.sleep(0.3)
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"{name} exited during startup with code {return_code}. "
                        f"See log: {log_file}"
                    )
        except BaseException:
            stop_started_services(started)
            raise

        stack = StartedStack(
            services=started,
            state_file=self.state_file,
            install_root=self.install_root,
        )
        _write_state(stack, self.conf)
        if detach:
            return stack
        if attach_logs:
            LogTailer([service.log_file for service in stack.services]).poll()
        return stack

    def active_state(self) -> dict[str, Any] | None:
        """Return active state for the configured stack, if any."""
        return load_active_state(self.state_file)

    @staticmethod
    def wait(stack: StartedStack, *, attach_logs: bool) -> int:
        """Wait for a foreground stack, stopping every service on Ctrl+C."""
        tailer = LogTailer([service.log_file for service in stack.services])
        try:
            while True:
                if attach_logs:
                    tailer.poll()
                for service in stack.services:
                    return_code = service.process.poll()
                    if return_code is not None:
                        print(
                            f"{service.name} exited with code {return_code}; stopping stack.",
                            file=sys.stderr,
                        )
                        stop_started_services(stack.services)
                        _remove_state(stack.state_file)
                        return int(return_code) if return_code else 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping RL-Insight server services...")
            stop_started_services(stack.services)
            _remove_state(stack.state_file)
            print("RL-Insight server services stopped.")
            return 130

    def stop(self) -> tuple[int, list[dict[str, Any]]]:
        """Stop processes recorded in the local state file."""
        state = _read_state(self.state_file)
        services = list(reversed(state.get("services", []))) if state else []
        if not services:
            return 0, []

        stopped: list[dict[str, Any]] = []
        for service in services:
            pid = int(service.get("pid", 0) or 0)
            name = str(service.get("name", "unknown"))
            if pid <= 0:
                stopped.append({"name": name, "pid": pid, "status": "invalid pid"})
                continue
            if not is_process_running(pid):
                stopped.append({"name": name, "pid": pid, "status": "already stopped"})
                continue
            _terminate_pid(pid)
            stopped.append({"name": name, "pid": pid, "status": "stopped"})

        _remove_state(self.state_file)
        return 0, list(reversed(stopped))


class LogTailer:
    """Tiny log tailer for foreground ``--attach-logs`` mode."""

    def __init__(self, paths: Sequence[Path]):
        self._offsets = {path: 0 for path in paths}

    def poll(self) -> None:
        for path in self._offsets:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self._offsets[path])
                text = stream.read()
                self._offsets[path] = stream.tell()
            if text:
                prefix = f"[{path.stem}] "
                for line in text.splitlines():
                    print(prefix + line)


def _state_file_from_config(conf: DictConfig) -> Path:
    raw = OmegaConf.select(conf, "server.state_file")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return (DEFAULT_STATE_ROOT / "run" / STATE_FILE).resolve()


def _runtime_dir_from_config(conf: DictConfig) -> Path:
    raw = OmegaConf.select(conf, "server.runtime_dir")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return (DEFAULT_STATE_ROOT / "runtime").resolve()


def _write_runtime_config(conf: DictConfig, runtime_dir: Path) -> Path:
    target = runtime_dir / "server.yaml"
    OmegaConf.save(config=conf, f=str(target))
    return target


def _server_enabled(conf: DictConfig) -> bool:
    return bool(OmegaConf.select(conf, "server.enable", default=True))


def _server_command(conf: DictConfig, runtime_files: RuntimeFiles) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rl_insight.server.http_api",
        "--config",
        str(runtime_files.server_config),
    ]


def load_active_state(state_file: Path) -> dict[str, Any] | None:
    """Return state only when at least one recorded process is still running."""
    state = _read_state(state_file)
    if not state:
        return None
    services = state.get("services", [])
    if any(is_process_running(int(service.get("pid", 0) or 0)) for service in services):
        return state
    _remove_state(state_file)
    return None


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_started_services(services: Sequence[StartedService]) -> None:
    for service in reversed(services):
        _terminate_process(service.process)


def _read_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(stack: StartedStack, conf: DictConfig) -> None:
    stack.state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "install_root": str(stack.install_root),
        "runtime_dir": str(_runtime_dir_from_config(conf)),
        "services": [
            {
                "name": service.name,
                "pid": service.process.pid,
                "command": service.command,
                "log_file": str(service.log_file),
            }
            for service in stack.services
        ],
    }
    stack.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _remove_state(state_file: Path) -> None:
    try:
        state_file.unlink()
    except FileNotFoundError:
        pass


def _service_data_root(conf: DictConfig, _install_root: Path) -> Path:
    raw = OmegaConf.select(conf, "server.data_dir")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return (DEFAULT_STATE_ROOT / "data").resolve()


def _render_prometheus_config(conf: DictConfig, runtime_dir: Path) -> Path:
    target = runtime_dir / "prometheus.yml"
    targets_file = prometheus_targets_file_from_config(conf)
    legacy_targets_file = (runtime_dir / PrometheusScrape.TARGETS_FILE_NAME).resolve()
    source = Path(str(OmegaConf.select(conf, "prometheus.config_file")))
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    scrape_configs = data.get("scrape_configs") or []
    if not isinstance(scrape_configs, list):
        raise ValueError("Prometheus scrape_configs must be a list")

    with _prometheus_targets_lock(targets_file):
        if not targets_file.exists():
            migrated_targets = (
                _read_file_sd_targets(legacy_targets_file)
                if legacy_targets_file.exists()
                else _migrate_prometheus_static_targets(
                    target, scrape_configs, targets_file
                )
            )
            _write_yaml_atomically(targets_file, migrated_targets)

    scrape_configs = [
        item
        for item in scrape_configs
        if not _is_managed_prometheus_job(item, targets_file)
    ]
    source_job_names: list[str] = []
    managed_jobs: list[dict[str, Any]] = []
    for source_job in scrape_configs:
        if not isinstance(source_job, dict):
            continue
        source_job_name = str(source_job.get("job_name") or "").strip()
        if not source_job_name:
            continue
        source_job_names.append(source_job_name)
        profile_name = _available_prometheus_job_name([*scrape_configs, *managed_jobs])
        managed_jobs.append(
            _profiled_prometheus_job(
                profile_name, source_job_name, source_job, targets_file
            )
        )

    dynamic_job_name = _available_prometheus_job_name([*scrape_configs, *managed_jobs])
    managed_jobs.append(
        _managed_prometheus_job(
            dynamic_job_name, targets_file, excluded_jobs=source_job_names
        )
    )
    scrape_configs.extend(managed_jobs)
    data["scrape_configs"] = scrape_configs
    _write_yaml_atomically(target, data)
    return target


def _managed_prometheus_job(
    job_name: str,
    targets_file: Path,
    *,
    excluded_jobs: Sequence[str] = (),
) -> dict[str, Any]:
    relabel_configs: list[dict[str, Any]] = []
    if excluded_jobs:
        relabel_configs.append(
            {
                "source_labels": [PrometheusScrape.DYNAMIC_JOB_LABEL],
                "regex": "|".join(re.escape(name) for name in excluded_jobs),
                "action": "drop",
            }
        )
    relabel_configs.extend(
        [
            {
                "source_labels": [PrometheusScrape.DYNAMIC_JOB_LABEL],
                "target_label": "job",
            },
            {
                "regex": PrometheusScrape.DYNAMIC_JOB_LABEL,
                "action": "labeldrop",
            },
        ]
    )
    return {
        "job_name": job_name,
        "file_sd_configs": [
            {
                "files": [str(targets_file)],
                "refresh_interval": PrometheusScrape.TARGETS_REFRESH_INTERVAL,
            }
        ],
        "relabel_configs": relabel_configs,
    }


def _profiled_prometheus_job(
    profile_name: str,
    source_job_name: str,
    source_job: dict[str, Any],
    targets_file: Path,
) -> dict[str, Any]:
    discovery_keys = {
        key
        for key in source_job
        if key == "static_configs" or key.endswith("_sd_configs")
    }
    profile = {
        "job_name": profile_name,
        **{
            key: value
            for key, value in source_job.items()
            if key not in discovery_keys and key not in {"job_name", "relabel_configs"}
        },
        "file_sd_configs": [
            {
                "files": [str(targets_file)],
                "refresh_interval": PrometheusScrape.TARGETS_REFRESH_INTERVAL,
            }
        ],
        "relabel_configs": [
            {
                "source_labels": [PrometheusScrape.DYNAMIC_JOB_LABEL],
                "regex": re.escape(source_job_name),
                "action": "keep",
            },
            {
                "source_labels": [PrometheusScrape.DYNAMIC_JOB_LABEL],
                "target_label": "job",
            },
            {
                "regex": PrometheusScrape.DYNAMIC_JOB_LABEL,
                "action": "labeldrop",
            },
            *(source_job.get("relabel_configs") or []),
        ],
    }
    return profile


def _is_managed_prometheus_job(item: Any, targets_file: Path) -> bool:
    if not isinstance(item, dict):
        return False
    name = str(item.get("job_name") or "")
    if not name.startswith(PrometheusScrape.DYNAMIC_CONFIG_JOB):
        return False
    file_sd_configs = item.get("file_sd_configs") or []
    if not any(
        isinstance(config, dict) and config.get("files") == [str(targets_file)]
        for config in file_sd_configs
    ):
        return False
    return any(
        isinstance(config, dict)
        and config.get("regex") == PrometheusScrape.DYNAMIC_JOB_LABEL
        and config.get("action") == "labeldrop"
        for config in item.get("relabel_configs") or []
    )


def _available_prometheus_job_name(scrape_configs: list[Any]) -> str:
    used_names = {
        str(item.get("job_name"))
        for item in scrape_configs
        if isinstance(item, dict) and item.get("job_name") is not None
    }
    base = PrometheusScrape.DYNAMIC_CONFIG_JOB
    if base not in used_names:
        return base
    suffix = 1
    while f"{base}-{suffix}" in used_names:
        suffix += 1
    return f"{base}-{suffix}"


def _write_yaml_atomically(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _prometheus_targets_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_name(f".{path.name}.lock")
    with lock_file.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_file_sd_targets(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError("Prometheus file_sd targets must be a list")
    return data


def _migrate_prometheus_static_targets(
    config_file: Path,
    source_scrape_configs: list[Any],
    targets_file: Path,
) -> list[dict[str, Any]]:
    if not config_file.exists():
        return []
    legacy = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    source_targets = {
        (str(job.get("job_name") or ""), str(target))
        for job in source_scrape_configs
        if isinstance(job, dict)
        for static_group in job.get("static_configs") or []
        if isinstance(static_group, dict)
        for target in static_group.get("targets") or []
    }
    groups: list[dict[str, Any]] = []
    for job in legacy.get("scrape_configs") or []:
        if not isinstance(job, dict):
            continue
        job_name = str(job.get("job_name") or "").strip()
        if not job_name or _is_managed_prometheus_job(job, targets_file):
            continue
        for static_group in job.get("static_configs") or []:
            if not isinstance(static_group, dict):
                continue
            targets = [
                str(item)
                for item in static_group.get("targets") or []
                if (job_name, str(item)) not in source_targets
            ]
            if not targets:
                continue
            labels = {
                str(key): str(value)
                for key, value in (static_group.get("labels") or {}).items()
            }
            labels[PrometheusScrape.DYNAMIC_JOB_LABEL] = job_name
            groups.append({"targets": targets, "labels": labels})
    return groups


def _render_tempo_config(
    conf: DictConfig, runtime_dir: Path, data_root: Path, tempo_version: str
) -> Path:
    source = Path(str(OmegaConf.select(conf, "tempo.config_file")))
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    query_port = int(OmegaConf.select(conf, "tempo.query_port"))
    otlp_port = int(OmegaConf.select(conf, "otel.otel_port"))
    tempo_data = _service_specific_data_dir(conf, "tempo", data_root)
    tempo_data.mkdir(parents=True, exist_ok=True)

    server_data = data.setdefault("server", {})
    server_data["http_listen_port"] = query_port
    server_data["http_listen_address"] = local_addresses()["bind"]
    receiver = (
        data.setdefault("distributor", {})
        .setdefault("receivers", {})
        .setdefault("otlp", {})
        .setdefault("protocols", {})
        .setdefault("http", {})
    )
    receiver["endpoint"] = format_host_port(local_addresses()["bind"], otlp_port)
    trace = data.setdefault("storage", {}).setdefault("trace", {})
    trace.setdefault("backend", "local")
    trace.setdefault("local", {})["path"] = str((tempo_data / "traces").resolve())
    trace.setdefault("wal", {})["path"] = str((tempo_data / "wal").resolve())
    retention_time = _select_str(conf, "tempo.retention_time")
    if retention_time:
        _set_tempo_retention(data, retention_time, tempo_version)

    target = runtime_dir / "tempo.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def _render_grafana_config(
    conf: DictConfig, runtime_dir: Path, data_root: Path
) -> Path:
    source = Path(str(OmegaConf.select(conf, "grafana.config_file")))
    parser = configparser.ConfigParser(interpolation=None)

    def _preserve_option_case(optionstr: str) -> str:
        return optionstr

    parser.optionxform = _preserve_option_case  # type: ignore[method-assign]
    parser.read(source, encoding="utf-8")
    for section in ("server", "security", "auth.anonymous", "paths"):
        if not parser.has_section(section):
            parser.add_section(section)

    grafana_data = _service_specific_data_dir(conf, "grafana", data_root)
    logs_dir = grafana_data / "logs"
    plugins_dir = grafana_data / "plugins"
    for path in (grafana_data, logs_dir, plugins_dir):
        path.mkdir(parents=True, exist_ok=True)

    parser.set("server", "http_addr", local_addresses()["bind"])
    parser.set("server", "http_port", str(int(OmegaConf.select(conf, "grafana.port"))))
    parser.set("paths", "provisioning", str((runtime_dir / "provisioning").resolve()))
    parser.set("paths", "data", str(grafana_data.resolve()))
    parser.set("paths", "logs", str(logs_dir.resolve()))
    parser.set("paths", "plugins", str(plugins_dir.resolve()))

    target = runtime_dir / "grafana.ini"
    with target.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    return target


def _render_grafana_provisioning(
    conf: DictConfig, runtime_dir: Path, dashboard_path: Path
) -> None:
    provisioning = runtime_dir / "provisioning"
    datasources_dir = provisioning / "datasources"
    dashboards_dir = provisioning / "dashboards"
    datasources_dir.mkdir(parents=True, exist_ok=True)
    dashboards_dir.mkdir(parents=True, exist_ok=True)

    datasources: list[dict[str, Any]] = []
    if bool(OmegaConf.select(conf, "prometheus.enable", default=True)):
        prometheus_port = int(OmegaConf.select(conf, "prometheus.prometheus_port"))
        datasources.append(
            {
                "name": "Prometheus",
                "uid": "prometheus",
                "type": "prometheus",
                "access": "proxy",
                "isDefault": True,
                "url": "http://"
                + format_host_port(local_addresses()["loopback"], prometheus_port),
                "editable": True,
            }
        )
    if bool(OmegaConf.select(conf, "tempo.enable", default=True)):
        tempo_query_port = int(OmegaConf.select(conf, "tempo.query_port"))
        datasources.append(
            {
                "name": "Tempo",
                "uid": "tempo",
                "type": "tempo",
                "access": "proxy",
                "isDefault": not datasources,
                "url": "http://"
                + format_host_port(local_addresses()["loopback"], tempo_query_port),
                "editable": True,
            }
        )
    datasource_data = {
        "apiVersion": 1,
        "datasources": datasources,
    }
    (datasources_dir / "default.yml").write_text(
        yaml.safe_dump(datasource_data, sort_keys=False), encoding="utf-8"
    )

    dashboard_data = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "RL-Insight",
                "orgId": 1,
                "folder": "RL-Insight",
                "type": "file",
                "disableDeletion": False,
                "updateIntervalSeconds": 10,
                "allowUiUpdates": True,
                "options": {"path": str(dashboard_path)},
            }
        ],
    }
    (dashboards_dir / "default.yml").write_text(
        yaml.safe_dump(dashboard_data, sort_keys=False), encoding="utf-8"
    )


def _stage_grafana_dashboards(conf: DictConfig, runtime_dir: Path) -> Path:
    source = Path(str(OmegaConf.select(conf, "grafana.dashboards_dir"))).resolve()
    target = (runtime_dir / "dashboards").resolve()
    if source == target:
        target.mkdir(parents=True, exist_ok=True)
        return target

    target.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        return target
    for item in source.iterdir():
        destination = target / item.name
        if item.is_file():
            if destination.is_dir():
                continue
            shutil.copy2(item, destination)
        elif item.is_dir():
            if destination.exists() and not destination.is_dir():
                continue
            shutil.copytree(item, destination, dirs_exist_ok=True)
    return target


def _service_specific_data_dir(conf: DictConfig, name: str, data_root: Path) -> Path:
    raw = OmegaConf.select(conf, f"{name}.data_dir")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return (data_root / name).resolve()


def _service_command(
    name: str,
    binary: Path,
    conf: DictConfig,
    runtime_files: RuntimeFiles,
    install_root: Path,
) -> list[str]:
    if name == "prometheus":
        data_dir = _service_specific_data_dir(
            conf, "prometheus", _service_data_root(conf, install_root)
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            f"--config.file={runtime_files.prometheus_config}",
            "--web.listen-address="
            + format_host_port(
                local_addresses()["bind"], conf.prometheus.prometheus_port
            ),
            "--web.enable-lifecycle",
            f"--storage.tsdb.path={data_dir.resolve()}",
        ]
        retention_time = _select_str(conf, "prometheus.retention_time")
        if retention_time:
            command.append(f"--storage.tsdb.retention.time={retention_time}")
        return command

    if name == "tempo":
        return [str(binary), f"-config.file={runtime_files.tempo_config}"]

    if name == "grafana":
        if binary.stem == "grafana":
            command = [
                str(binary),
                "server",
                "--config",
                str(runtime_files.grafana_config),
            ]
            if runtime_files.grafana_homepath:
                command.extend(["--homepath", str(runtime_files.grafana_homepath)])
            return command

        command = [str(binary), "--config", str(runtime_files.grafana_config)]
        if runtime_files.grafana_homepath:
            command.extend(["--homepath", str(runtime_files.grafana_homepath)])
        command.append("web")
        return command

    raise RuntimeError(f"Unsupported service: {name}")


def _spawn_service(
    name: str, command: Sequence[str], log_file: Path
) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    stdout = log_file.open("ab")
    try:
        process = subprocess.Popen(
            list(command),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        stdout.close()
        return process
    except OSError as exc:
        stdout.close()
        raise RuntimeError(f"Failed to start {name}: {exc}") from exc


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        process.terminate()
    _wait_or_kill(process)


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    deadline = time.time() + 8
    while time.time() < deadline:
        if not is_process_running(pid):
            return
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return


def _wait_or_kill(process: subprocess.Popen[Any]) -> None:
    try:
        process.wait(timeout=8)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except OSError:
        return


def _select_str(conf: DictConfig, key: str) -> str:
    value = OmegaConf.select(conf, key)
    return str(value).strip() if value is not None else ""


def _set_tempo_retention(
    data: dict[str, Any], retention_time: str, version: str
) -> None:
    retention = _tempo_duration(retention_time)
    if _major_version(version) >= 3:
        (
            data.setdefault("backend_scheduler", {})
            .setdefault("provider", {})
            .setdefault("compaction", {})
            .setdefault("compaction", {})
        )["block_retention"] = retention
        return
    data.setdefault("compactor", {}).setdefault("compaction", {})["block_retention"] = (
        retention
    )


def _tempo_duration(value: str) -> str:
    raw = value.strip()
    if raw.endswith("d") and raw[:-1].isdigit():
        return f"{int(raw[:-1]) * 24}h"
    return raw


def _major_version(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0
