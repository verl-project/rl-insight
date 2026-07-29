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

"""Incremental SSH/SFTP and Docker-log adapter for degradation detection.

The remote transport is deliberately separated from the detector. Remote
failures never become degradation events, and an offset is committed only
after a complete batch has been parsed and detected successfully.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import math
import os
import re
import shlex
import socket
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config_loader import get_default_config_dir, metric_to_safe_filename

try:  # Paramiko is optional for local-only degradation detection.
    import paramiko
except ModuleNotFoundError:  # pragma: no cover - exercised through lazy check
    paramiko = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_MAX_MONITOR_CONFIG_BYTES = 1024 * 1024
_DEFAULT_MAX_READ_BYTES = 10 * 1024 * 1024
_MAX_OFFSET_FILE_BYTES = 1024 * 1024
_FILE_HEAD_FINGERPRINT_BYTES = 256
_DOCKER_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DOCKER_TIMESTAMP_RE = re.compile(
    rb"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    rb"(?:\.\d+)?Z)\s+(?P<payload>.*)$"
)
_MONITOR_CONFIG_KEYS = {
    "host",
    "port",
    "username",
    "key_filename",
    "known_hosts",
    "password_env",
    "connect_timeout",
    "auth_timeout",
    "banner_timeout",
    "command_timeout",
    "source",
    "offset_path",
    "max_read_bytes",
    "metrics",
    "task_id",
    "start_time",
    "end_time",
    "config_dir",
}
_SOURCE_CONFIG_KEYS = {"type", "remote_path", "container", "tail_lines"}


class RemoteMonitorError(RuntimeError):
    """Base class for safe, user-facing remote-source failures."""

    code = "REMOTE_ERROR"


class RemoteDependencyError(RemoteMonitorError):
    code = "REMOTE_DEPENDENCY_MISSING"


class RemoteConnectionError(RemoteMonitorError):
    code = "REMOTE_CONNECTION_FAILED"


class RemoteAuthenticationError(RemoteMonitorError):
    code = "REMOTE_AUTHENTICATION_FAILED"


class RemoteHostKeyError(RemoteMonitorError):
    code = "REMOTE_HOST_KEY_FAILED"


class RemoteSourceNotFoundError(RemoteMonitorError):
    code = "REMOTE_SOURCE_NOT_FOUND"


class RemoteCommandError(RemoteMonitorError):
    code = "REMOTE_COMMAND_FAILED"


class RemoteDataError(RemoteMonitorError):
    code = "REMOTE_DATA_INVALID"


class OffsetStateError(RemoteMonitorError):
    code = "REMOTE_OFFSET_INVALID"


class DetectionExecutionError(RemoteMonitorError):
    code = "REMOTE_DETECTION_FAILED"


@dataclass(frozen=True)
class RemoteSourceConfig:
    """One supported remote source."""

    type: str
    remote_path: str | None = None
    container: str | None = None
    tail_lines: int = 1000


@dataclass(frozen=True)
class RemoteMonitorConfig:
    """Validated remote monitor settings."""

    host: str
    port: int
    username: str
    key_filename: str | None
    known_hosts: str | None
    password_env: str | None
    connect_timeout: float
    auth_timeout: float
    banner_timeout: float
    command_timeout: float
    source: RemoteSourceConfig
    offset_path: Path
    max_read_bytes: int
    metrics: tuple[str, ...]
    task_id: str
    start_time: float | None
    end_time: float | None
    config_dir: Path


@dataclass(frozen=True)
class RemoteFetchResult:
    """Complete application JSON lines and the cursor to commit later."""

    lines: tuple[bytes, ...]
    next_cursor: dict[str, Any]
    diagnostics: dict[str, Any]


class JsonOffsetStore:
    """Small single-writer cursor store with atomic replacement."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()

    def load(self, source_key: str) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            if not self.path.is_file():
                raise OffsetStateError(f"Offset path is not a file: {self.path}")
            if self.path.stat().st_size > _MAX_OFFSET_FILE_BYTES:
                raise OffsetStateError(f"Offset file is too large: {self.path}")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except OffsetStateError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OffsetStateError(f"Cannot load offset state: {exc}") from exc
        if (
            not isinstance(raw, Mapping)
            or isinstance(raw.get("version"), bool)
            or raw.get("version") != 1
        ):
            raise OffsetStateError("Offset state must be a version 1 object")
        sources = raw.get("sources")
        if not isinstance(sources, Mapping):
            raise OffsetStateError("Offset state sources must be an object")
        cursor = sources.get(source_key, {})
        if not isinstance(cursor, Mapping):
            raise OffsetStateError("Offset cursor must be an object")
        return dict(cursor)

    def save(self, source_key: str, cursor: Mapping[str, Any]) -> None:
        state: dict[str, Any] = {"version": 1, "sources": {}}
        if self.path.exists():
            try:
                if not self.path.is_file():
                    raise OffsetStateError(
                        f"Offset path is not a file: {self.path}"
                    )
                if self.path.stat().st_size > _MAX_OFFSET_FILE_BYTES:
                    raise OffsetStateError(
                        f"Offset file is too large: {self.path}"
                    )
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except OffsetStateError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise OffsetStateError(f"Cannot update offset state: {exc}") from exc
            if (
                not isinstance(loaded, Mapping)
                or isinstance(loaded.get("version"), bool)
                or loaded.get("version") != 1
                or not isinstance(loaded.get("sources"), Mapping)
            ):
                raise OffsetStateError("Offset state must be a version 1 object")
            state = {
                "version": 1,
                "sources": dict(loaded["sources"]),
            }
        state["sources"][source_key] = dict(cursor)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                json.dump(state, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
            temporary_name = None
        except (OSError, TypeError, ValueError) as exc:
            raise OffsetStateError(f"Cannot save offset state: {exc}") from exc
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass


class RemoteMonitor:
    """Fetch one remote batch and invoke ``DegradationPerception.detect``."""

    def __init__(
        self,
        config: RemoteMonitorConfig,
        *,
        detector_factory: Callable[..., Any] | None = None,
        ssh_factory: Callable[[], Any] | None = None,
        offset_store: JsonOffsetStore | None = None,
    ):
        self.config = config
        self.detector_factory = detector_factory or _default_detector_factory
        self.ssh_factory = ssh_factory
        self.offset_store = offset_store or JsonOffsetStore(config.offset_path)
        self._detector: Any | None = None

    def run(self) -> dict[str, Any]:
        """Return a detector response or an explicit operational error."""

        source_key = _source_key(self.config)
        try:
            try:
                cursor = self.offset_store.load(source_key)
            except RemoteMonitorError:
                raise
            except Exception as exc:
                raise OffsetStateError("Cannot load remote offset") from exc
            fetched = self._fetch(cursor)
            dataset = parse_remote_jsonl(fetched.lines, self.config.metrics)
            detector = self._detector
            history_snapshot: Any | None = None
            try:
                if detector is None:
                    detector = self.detector_factory(
                        path=None,
                        start_time=self.config.start_time,
                        end_time=self.config.end_time,
                        metrics=list(self.config.metrics),
                        task_id=self.config.task_id,
                        source_type="remote_monitor",
                        config_dir=self.config.config_dir,
                        dataset=dataset,
                    )
                    response = detector.detect()
                else:
                    history_snapshot = copy.deepcopy(
                        getattr(detector, "history", None)
                    )
                    detect_dataset = getattr(detector, "detect_dataset", None)
                    if not callable(detect_dataset):
                        raise TypeError(
                            "cached detector must implement detect_dataset"
                        )
                    response = detect_dataset(
                        dataset,
                        metrics=list(self.config.metrics),
                        start_time=self.config.start_time,
                        end_time=self.config.end_time,
                    )
            except Exception as exc:
                if self._detector is not None and history_snapshot is not None:
                    self._detector.history = history_snapshot
                raise DetectionExecutionError(
                    "Degradation detector could not process the remote batch"
                ) from exc
            if not isinstance(response, Mapping):
                if self._detector is not None and history_snapshot is not None:
                    self._detector.history = history_snapshot
                raise DetectionExecutionError("Detector response must be an object")
            try:
                result = dict(response)
            except Exception as exc:
                if self._detector is not None and history_snapshot is not None:
                    self._detector.history = history_snapshot
                raise DetectionExecutionError(
                    "Detector response could not be materialized"
                ) from exc
            # Commit only after detection has returned a complete response.
            try:
                self.offset_store.save(source_key, fetched.next_cursor)
            except RemoteMonitorError:
                if self._detector is not None and history_snapshot is not None:
                    self._detector.history = history_snapshot
                raise
            except Exception as exc:
                if self._detector is not None and history_snapshot is not None:
                    self._detector.history = history_snapshot
                raise OffsetStateError("Cannot save remote offset") from exc
            if self._detector is None:
                self._detector = detector
            result.setdefault("sourceStatus", "ok")
            result.setdefault("sourceDiagnostics", fetched.diagnostics)
            return result
        except RemoteMonitorError as exc:
            logger.warning("Remote monitor failed [%s]: %s", exc.code, exc)
            return _remote_error_response(self.config, exc)

    def _fetch(self, cursor: Mapping[str, Any]) -> RemoteFetchResult:
        module = _require_paramiko()
        ssh_client = None
        sftp = None
        remote_file = None
        stdin = None
        stdout = None
        stderr = None
        try:
            ssh_client = (
                self.ssh_factory()
                if self.ssh_factory is not None
                else module.SSHClient()
            )
            ssh_client.load_system_host_keys()
            if self.config.known_hosts is not None:
                ssh_client.load_host_keys(self.config.known_hosts)
            ssh_client.set_missing_host_key_policy(module.RejectPolicy())
            connect_kwargs: dict[str, Any] = {
                "hostname": self.config.host,
                "port": self.config.port,
                "username": self.config.username,
                "timeout": self.config.connect_timeout,
                "auth_timeout": self.config.auth_timeout,
                "banner_timeout": self.config.banner_timeout,
            }
            if self.config.key_filename is not None:
                connect_kwargs["key_filename"] = self.config.key_filename
            if self.config.password_env is not None:
                password = os.environ.get(self.config.password_env)
                if password:
                    connect_kwargs["password"] = password
            ssh_client.connect(**connect_kwargs)

            if self.config.source.type == "file":
                sftp = ssh_client.open_sftp()
                get_channel = getattr(sftp, "get_channel", None)
                if callable(get_channel):
                    sftp_channel = get_channel()
                    set_timeout = getattr(sftp_channel, "settimeout", None)
                    if callable(set_timeout):
                        set_timeout(self.config.command_timeout)
                try:
                    attributes = sftp.stat(self.config.source.remote_path)
                    remote_file = sftp.open(self.config.source.remote_path, "rb")
                except OSError as exc:
                    if getattr(exc, "errno", None) == errno.ENOENT:
                        raise RemoteSourceNotFoundError(
                            "Remote metric file does not exist"
                        ) from exc
                    raise
                return _read_sftp_batch(
                    remote_file,
                    int(attributes.st_size),
                    cursor,
                    self.config.max_read_bytes,
                )

            command = _docker_command(self.config.source, cursor)
            stdin, stdout, stderr = ssh_client.exec_command(
                command,
                timeout=self.config.command_timeout,
            )
            combine_stderr = getattr(stdout.channel, "set_combine_stderr", None)
            if callable(combine_stderr):
                combine_stderr(True)
            output = _require_bytes(
                stdout.read(self.config.max_read_bytes + 1),
                "Docker stdout",
            )
            if len(output) > self.config.max_read_bytes:
                raise RemoteDataError("Docker log batch exceeds max_read_bytes")
            error_output = _require_bytes(stderr.read(4097), "Docker stderr")
            status = stdout.channel.recv_exit_status()
            if status != 0:
                raise RemoteCommandError(
                    f"Docker logs command exited with status {status}"
                )
            if error_output:
                if output and not output.endswith(b"\n"):
                    output += b"\n"
                output += error_output
            return _read_docker_batch(output, cursor)
        except RemoteMonitorError:
            raise
        except Exception as exc:
            raise _map_transport_error(exc, module) from exc
        finally:
            for resource in (remote_file, stdin, stdout, stderr, sftp, ssh_client):
                _safe_close(resource)


def run_remote_monitor(
    config_path: str | os.PathLike[str],
    *,
    detector_factory: Callable[..., Any] | None = None,
    ssh_factory: Callable[[], Any] | None = None,
    offset_store: JsonOffsetStore | None = None,
) -> dict[str, Any]:
    """Load a monitor YAML and execute one programmatic detection cycle."""

    config = load_remote_monitor_config(config_path)
    return RemoteMonitor(
        config,
        detector_factory=detector_factory,
        ssh_factory=ssh_factory,
        offset_store=offset_store,
    ).run()


def load_remote_monitor_config(
    path: str | os.PathLike[str],
) -> RemoteMonitorConfig:
    """Load and strictly validate a credential-free monitor YAML."""

    config_path = Path(path).expanduser().resolve()
    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"Remote monitor config not found: {config_path}")
        if config_path.stat().st_size > _MAX_MONITOR_CONFIG_BYTES:
            raise ValueError(f"Remote monitor config is too large: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load remote monitor config: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Remote monitor config must be an object")
    config = dict(raw)
    _reject_unknown_config_keys(config, _MONITOR_CONFIG_KEYS, "remote monitor")

    host = _required_text(config.get("host"), "host")
    username = _required_text(config.get("username"), "username")
    port = _integer(config.get("port", 22), "port", minimum=1, maximum=65535)
    connect_timeout = _positive_number(
        config.get("connect_timeout", 10), "connect_timeout"
    )
    auth_timeout = _positive_number(
        config.get("auth_timeout", connect_timeout), "auth_timeout"
    )
    banner_timeout = _positive_number(
        config.get("banner_timeout", connect_timeout), "banner_timeout"
    )
    command_timeout = _positive_number(
        config.get("command_timeout", 30), "command_timeout"
    )

    key_filename = _optional_local_path(
        config.get("key_filename"), config_path.parent, "key_filename"
    )
    known_hosts = _optional_local_path(
        config.get("known_hosts"), config_path.parent, "known_hosts"
    )
    password_env_value = config.get("password_env")
    if password_env_value is None:
        password_env = None
    else:
        password_env = _required_text(password_env_value, "password_env")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", password_env):
            raise ValueError("password_env must be an environment variable name")

    source_raw = config.get("source")
    if not isinstance(source_raw, Mapping):
        raise ValueError("source must be an object")
    source_mapping = dict(source_raw)
    _reject_unknown_config_keys(source_mapping, _SOURCE_CONFIG_KEYS, "source")
    source_type = source_mapping.get("type")
    if source_type not in {"file", "docker"}:
        raise ValueError("source.type must be file or docker")
    tail_lines = _integer(
        source_mapping.get("tail_lines", 1000),
        "source.tail_lines",
        minimum=1,
        maximum=1_000_000,
    )
    if source_type == "file":
        remote_path = _required_text(
            source_mapping.get("remote_path"), "source.remote_path"
        )
        if "\x00" in remote_path:
            raise ValueError("source.remote_path must not contain NUL")
        container = None
    else:
        container = _required_text(
            source_mapping.get("container"), "source.container"
        )
        if not _DOCKER_CONTAINER_RE.fullmatch(container):
            raise ValueError("source.container contains unsafe characters")
        remote_path = None
    source = RemoteSourceConfig(
        type=source_type,
        remote_path=remote_path,
        container=container,
        tail_lines=tail_lines,
    )

    metrics_raw = config.get("metrics")
    if (
        not isinstance(metrics_raw, Sequence)
        or isinstance(metrics_raw, (str, bytes, bytearray))
        or not metrics_raw
    ):
        raise ValueError("metrics must be a non-empty list")
    metrics_list: list[str] = []
    for index, raw_metric in enumerate(metrics_raw):
        metric = _required_text(raw_metric, f"metrics[{index}]")
        metric_to_safe_filename(metric)
        if metric not in metrics_list:
            metrics_list.append(metric)
    metrics = tuple(metrics_list)

    task_id_value = config.get("task_id")
    task_id = (
        "default"
        if task_id_value is None
        else _required_text(task_id_value, "task_id")
    )
    start_time = _optional_finite_number(config.get("start_time"), "start_time")
    end_time = _optional_finite_number(config.get("end_time"), "end_time")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("start_time must not exceed end_time")

    offset_path = _resolve_local_path(
        config.get("offset_path", "remote-monitor.offset.json"),
        config_path.parent,
        "offset_path",
    )
    config_dir = _resolve_local_path(
        config.get("config_dir", str(get_default_config_dir())),
        config_path.parent,
        "config_dir",
    )
    max_read_bytes = _integer(
        config.get("max_read_bytes", _DEFAULT_MAX_READ_BYTES),
        "max_read_bytes",
        minimum=1,
        maximum=100 * 1024 * 1024,
    )
    return RemoteMonitorConfig(
        host=host,
        port=port,
        username=username,
        key_filename=key_filename,
        known_hosts=known_hosts,
        password_env=password_env,
        connect_timeout=connect_timeout,
        auth_timeout=auth_timeout,
        banner_timeout=banner_timeout,
        command_timeout=command_timeout,
        source=source,
        offset_path=offset_path,
        max_read_bytes=max_read_bytes,
        metrics=metrics,
        task_id=task_id,
        start_time=start_time,
        end_time=end_time,
        config_dir=config_dir,
    )


def parse_remote_jsonl(
    lines: Sequence[bytes], metrics: Sequence[str]
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Parse explicit-phase JSONL into the detector's canonical dataset."""

    dataset: dict[str, dict[str, dict[str, list[Any]]]] = {
        phase: {
            metric: {"timestamps": [], "values": []}
            for metric in metrics
        }
        for phase in ("standard", "inference")
    }
    for line_number, raw_line in enumerate(lines, start=1):
        if not isinstance(raw_line, bytes):
            raise RemoteDataError(f"Remote line {line_number} is not bytes")
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(
                raw_line.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value}")
                ),
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise RemoteDataError(
                f"Remote line {line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(record, Mapping):
            raise RemoteDataError(f"Remote line {line_number} must be an object")
        phase = record.get("phase")
        if phase not in {"standard", "inference"}:
            raise RemoteDataError(
                f"Remote line {line_number} phase must be standard or inference"
            )
        timestamp = record.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            raise RemoteDataError(
                f"Remote line {line_number} timestamp must be finite numeric"
            )
        nested_metrics = record.get("metrics")
        if nested_metrics is not None and not isinstance(nested_metrics, Mapping):
            raise RemoteDataError(
                f"Remote line {line_number} metrics must be an object"
            )
        values = nested_metrics if isinstance(nested_metrics, Mapping) else record
        for metric in metrics:
            if metric not in values:
                continue
            dataset[phase][metric]["timestamps"].append(timestamp)
            dataset[phase][metric]["values"].append(values[metric])
    return dataset


def _read_sftp_batch(
    remote_file: Any,
    remote_size: int,
    cursor: Mapping[str, Any],
    max_read_bytes: int,
) -> RemoteFetchResult:
    if isinstance(remote_size, bool) or not isinstance(remote_size, int):
        raise RemoteDataError("Remote file size must be an integer")
    source_type = cursor.get("sourceType")
    if source_type not in {None, "file"}:
        raise OffsetStateError("SFTP cursor sourceType must be file")
    if remote_size < 0:
        raise RemoteDataError("Remote file size must not be negative")
    raw_offset = cursor.get("offset", 0)
    if (
        isinstance(raw_offset, bool)
        or not isinstance(raw_offset, int)
        or raw_offset < 0
    ):
        raise OffsetStateError("SFTP offset must be a non-negative integer")
    offset = raw_offset
    reset_reason: str | None = None
    if offset > remote_size:
        offset = 0
        reset_reason = "truncate"

    previous_head_length = cursor.get("headLength", 0)
    previous_head_hash = cursor.get("headHash")
    if (
        isinstance(previous_head_length, bool)
        or not isinstance(previous_head_length, int)
        or previous_head_length < 0
        or previous_head_length > _FILE_HEAD_FINGERPRINT_BYTES
    ):
        raise OffsetStateError("SFTP headLength is invalid")
    if previous_head_hash is not None and not isinstance(previous_head_hash, str):
        raise OffsetStateError("SFTP headHash is invalid")
    if (
        previous_head_length
        and previous_head_hash
        and remote_size >= previous_head_length
    ):
        remote_file.seek(0)
        previous_head = _require_bytes(
            remote_file.read(previous_head_length), "SFTP fingerprint"
        )
        if hashlib.sha256(previous_head).hexdigest() != previous_head_hash:
            offset = 0
            reset_reason = "rotation"

    current_head_length = min(remote_size, _FILE_HEAD_FINGERPRINT_BYTES)
    remote_file.seek(0)
    current_head = _require_bytes(
        remote_file.read(current_head_length), "SFTP fingerprint"
    )
    if len(current_head) != current_head_length:
        raise RemoteDataError("Remote file changed while its fingerprint was read")

    remote_file.seek(offset)
    chunk = _require_bytes(
        remote_file.read(max_read_bytes + 1), "SFTP log batch"
    )
    if len(chunk) > max_read_bytes:
        raise RemoteDataError("SFTP log batch exceeds max_read_bytes")
    newline_index = chunk.rfind(b"\n")
    if newline_index < 0:
        complete = b""
        next_offset = offset
    else:
        complete = chunk[: newline_index + 1]
        next_offset = offset + newline_index + 1
    lines = tuple(complete.splitlines())
    next_cursor = {
        "sourceType": "file",
        "offset": next_offset,
        "headLength": current_head_length,
        "headHash": hashlib.sha256(current_head).hexdigest(),
    }
    diagnostics: dict[str, Any] = {
        "sourceType": "file",
        "previousOffset": raw_offset,
        "nextOffset": next_offset,
        "completeLineCount": len(lines),
        "partialBytes": len(chunk) - len(complete),
    }
    if reset_reason is not None:
        diagnostics["offsetResetReason"] = reset_reason
    return RemoteFetchResult(lines, next_cursor, diagnostics)


def _read_docker_batch(
    output: bytes, cursor: Mapping[str, Any]
) -> RemoteFetchResult:
    output = _require_bytes(output, "Docker log batch")
    source_type = cursor.get("sourceType")
    if source_type not in {None, "docker"}:
        raise OffsetStateError("Docker cursor sourceType must be docker")
    previous_timestamp = cursor.get("timestamp")
    previous_hashes_raw = cursor.get("hashes", [])
    if previous_timestamp is not None and (
        not isinstance(previous_timestamp, str)
        or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
            previous_timestamp,
        )
    ):
        raise OffsetStateError("Docker timestamp cursor is invalid")
    if not isinstance(previous_hashes_raw, list) or not all(
        isinstance(item, str) for item in previous_hashes_raw
    ):
        raise OffsetStateError("Docker hash cursor is invalid")
    previous_hashes = set(previous_hashes_raw)

    accepted: list[tuple[str, bytes, str]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        match = _DOCKER_TIMESTAMP_RE.match(line)
        if match is None:
            raise RemoteDataError(
                f"Docker log line {line_number} lacks an RFC3339 timestamp"
            )
        timestamp = match.group("timestamp").decode("ascii")
        payload = match.group("payload")
        digest = hashlib.sha256(payload).hexdigest()
        if previous_timestamp is not None and timestamp < previous_timestamp:
            continue
        if timestamp == previous_timestamp and digest in previous_hashes:
            continue
        accepted.append((timestamp, payload, digest))

    if not accepted:
        next_cursor = dict(cursor)
    else:
        latest_timestamp = max(item[0] for item in accepted)
        latest_hashes = sorted(
            item[2] for item in accepted if item[0] == latest_timestamp
        )
        if latest_timestamp == previous_timestamp:
            latest_hashes = sorted(set(latest_hashes) | previous_hashes)
        next_cursor = {
            "sourceType": "docker",
            "timestamp": latest_timestamp,
            "hashes": latest_hashes,
        }
    lines = tuple(item[1] for item in accepted)
    return RemoteFetchResult(
        lines,
        next_cursor,
        {
            "sourceType": "docker",
            "completeLineCount": len(lines),
            "deduplicatedLineCount": len(output.splitlines()) - len(lines),
        },
    )


def _docker_command(
    source: RemoteSourceConfig, cursor: Mapping[str, Any]
) -> str:
    if source.container is None or not _DOCKER_CONTAINER_RE.fullmatch(source.container):
        raise RemoteCommandError("Docker container name is invalid")
    arguments = ["docker", "logs", "--timestamps"]
    timestamp = cursor.get("timestamp")
    if timestamp is None:
        arguments.extend(["--tail", str(source.tail_lines)])
    else:
        if not isinstance(timestamp, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
            timestamp,
        ):
            raise OffsetStateError("Docker timestamp cursor is invalid")
        arguments.extend(["--since", timestamp])
    arguments.extend(["--", source.container])
    return " ".join(shlex.quote(argument) for argument in arguments)


def _source_key(config: RemoteMonitorConfig) -> str:
    identity = {
        "host": config.host,
        "port": config.port,
        "sourceType": config.source.type,
        "remotePath": config.source.remote_path,
        "container": config.source.container,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_paramiko() -> Any:
    if paramiko is None:
        raise RemoteDependencyError(
            'Paramiko is required for remote_monitor; install "rl-insight[degradation]"'
        )
    return paramiko


def _default_detector_factory(**kwargs: Any) -> Any:
    from .algorithm import DegradationPerception

    return DegradationPerception(**kwargs)


def _map_transport_error(exc: Exception, module: Any) -> RemoteMonitorError:
    authentication_type = getattr(module, "AuthenticationException", ())
    bad_host_key_type = getattr(module, "BadHostKeyException", ())
    no_connections_type = getattr(module, "NoValidConnectionsError", ())
    ssh_exception_type = getattr(module, "SSHException", ())
    if authentication_type and isinstance(exc, authentication_type):
        return RemoteAuthenticationError("SSH authentication failed")
    if bad_host_key_type and isinstance(exc, bad_host_key_type):
        return RemoteHostKeyError("SSH host key verification failed")
    if no_connections_type and isinstance(exc, no_connections_type):
        return RemoteConnectionError("No SSH connection could be established")
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return RemoteConnectionError("Remote connection or read timed out")
    if isinstance(exc, FileNotFoundError) or (
        isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOENT
    ):
        return RemoteSourceNotFoundError("Remote source does not exist")
    if ssh_exception_type and isinstance(exc, ssh_exception_type):
        if "known_hosts" in str(exc).lower():
            return RemoteHostKeyError("SSH host key verification failed")
        return RemoteConnectionError("SSH transport failed")
    if isinstance(exc, OSError):
        return RemoteConnectionError("Remote I/O failed")
    return RemoteConnectionError("Unexpected remote transport failure")


def _remote_error_response(
    config: RemoteMonitorConfig, exc: RemoteMonitorError
) -> dict[str, Any]:
    message = _safe_remote_text(str(exc).encode("utf-8"))
    results = {
        metric: {
            "message": message,
            "thresholds": [],
            "abnormalTimeRange": [],
        }
        for metric in config.metrics
    }
    return {
        "taskId": config.task_id,
        "states": {},
        "results": results,
        "abnormalTimeRange": {metric: [] for metric in config.metrics},
        "sourceStatus": "error",
        "sourceError": {"code": exc.code, "message": message},
    }


def _safe_close(resource: Any) -> None:
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        logger.warning("Failed to close a remote monitor resource", exc_info=True)


def _safe_remote_text(value: bytes) -> str:
    text = value[:4096].decode("utf-8", errors="replace")
    return " ".join(text.split()) or "remote operation failed"


def _require_bytes(value: Any, context: str) -> bytes:
    if not isinstance(value, bytes):
        raise RemoteDataError(f"{context} did not return bytes")
    return value


def _reject_unknown_config_keys(
    value: Mapping[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(repr(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {', '.join(unknown)}")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _positive_number(value: Any, name: str) -> float:
    result = _optional_finite_number(value, name)
    if result is None or result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


def _optional_finite_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number or null")
    return result


def _optional_local_path(value: Any, base: Path, name: str) -> str | None:
    if value is None:
        return None
    return str(_resolve_local_path(value, base, name))


def _resolve_local_path(value: Any, base: Path, name: str) -> Path:
    text = _required_text(value, name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
