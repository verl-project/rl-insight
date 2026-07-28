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

from __future__ import annotations

import io
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiment.degradation_perception import remote_monitor as module
from experiment.degradation_perception.remote_monitor import (
    JsonOffsetStore,
    OffsetStateError,
    RemoteConnectionError,
    RemoteFetchResult,
    RemoteMonitor,
    RemoteMonitorConfig,
    RemoteSourceConfig,
    load_remote_monitor_config,
    parse_remote_jsonl,
)


METRIC = "timing_s/step"


def config(tmp_path, *, source=None):
    return RemoteMonitorConfig(
        host="monitor.example.com",
        port=22,
        username="rl-insight",
        key_filename=None,
        known_hosts=None,
        password_env=None,
        connect_timeout=1.0,
        auth_timeout=1.0,
        banner_timeout=1.0,
        command_timeout=1.0,
        source=source or RemoteSourceConfig(type="file", remote_path="/metrics.jsonl"),
        offset_path=tmp_path / "offset.json",
        max_read_bytes=1024 * 1024,
        metrics=(METRIC,),
        task_id="default",
        start_time=None,
        end_time=None,
        config_dir=tmp_path / "config",
    )


def line(phase, timestamp, value):
    return json.dumps(
        {"phase": phase, "timestamp": timestamp, "metrics": {METRIC: value}}
    ).encode("utf-8")


def test_parse_remote_jsonl_requires_explicit_phase_and_finite_time():
    parsed = parse_remote_jsonl(
        [line("standard", 1, 1.0), line("inference", 2, 1.5)],
        [METRIC],
    )
    assert parsed["standard"][METRIC] == {"timestamps": [1], "values": [1.0]}
    assert parsed["inference"][METRIC] == {"timestamps": [2], "values": [1.5]}

    with pytest.raises(module.RemoteDataError, match="phase"):
        parse_remote_jsonl(
            [json.dumps({"timestamp": 1, METRIC: 1.0}).encode()], [METRIC]
        )
    with pytest.raises(module.RemoteDataError, match="finite numeric"):
        parse_remote_jsonl(
            [b'{"phase":"standard","timestamp":true,"metrics":{}}'],
            [METRIC],
        )


def test_json_offset_store_round_trip_and_corruption(tmp_path):
    store = JsonOffsetStore(tmp_path / "offset.json")
    assert store.load("source") == {}
    store.save("source", {"offset": 12})
    assert store.load("source") == {"offset": 12}
    store.path.write_text("{", encoding="utf-8")
    with pytest.raises(OffsetStateError):
        store.load("source")


def test_sftp_offset_beyond_size_resets_as_truncate():
    payload = line("standard", 1, 1.0) + b"\n"
    result = module._read_sftp_batch(
        io.BytesIO(payload),
        len(payload),
        {"offset": len(payload) + 10},
        1024,
    )
    assert result.diagnostics["offsetResetReason"] == "truncate"
    assert result.diagnostics["previousOffset"] == len(payload) + 10
    assert result.next_cursor["offset"] == len(payload)
    assert result.lines == (payload.rstrip(b"\n"),)


def test_sftp_head_fingerprint_mismatch_resets_as_rotation():
    payload = line("standard", 1, 1.0) + b"\n"
    result = module._read_sftp_batch(
        io.BytesIO(payload),
        len(payload),
        {
            "offset": 2,
            "headLength": 4,
            "headHash": hashlib.sha256(b"old!").hexdigest(),
        },
        1024,
    )
    assert result.diagnostics["offsetResetReason"] == "rotation"
    assert result.diagnostics["previousOffset"] == 2
    assert result.next_cursor["offset"] == len(payload)


class RecordingOffsetStore:
    def __init__(self):
        self.saved = []

    def load(self, _source_key):
        return {}

    def save(self, source_key, cursor):
        self.saved.append((source_key, dict(cursor)))


def test_remote_run_calls_programmatic_detector_and_commits_after_success(
    monkeypatch, tmp_path
):
    store = RecordingOffsetStore()
    captured = {}

    class Detector:
        def detect(self):
            return {
                "taskId": "default",
                "states": {METRIC: 0},
                "results": {},
                "abnormalTimeRange": {METRIC: []},
            }

    def detector_factory(**kwargs):
        captured.update(kwargs)
        return Detector()

    monitor = RemoteMonitor(
        config(tmp_path), detector_factory=detector_factory, offset_store=store
    )
    monkeypatch.setattr(
        monitor,
        "_fetch",
        lambda _cursor: RemoteFetchResult(
            (line("standard", 1, 1.0), line("inference", 2, 1.5)),
            {"offset": 200},
            {"completeLineCount": 2},
        ),
    )
    response = monitor.run()
    assert response["sourceStatus"] == "ok"
    assert captured["source_type"] == "remote_monitor"
    assert captured["dataset"]["standard"][METRIC]["values"] == [1.0]
    assert store.saved and store.saved[0][1] == {"offset": 200}


def test_remote_monitor_reuses_detector_and_history_across_polls(
    monkeypatch, tmp_path
):
    store = RecordingOffsetStore()
    created = []

    class Detector:
        def __init__(self):
            self.history = []
            self.detect_dataset_calls = 0

        def detect(self):
            self.history.append("first")
            return {
                "taskId": "default",
                "states": {METRIC: 0},
                "results": {},
                "abnormalTimeRange": {METRIC: []},
            }

        def detect_dataset(self, _dataset, **_kwargs):
            self.detect_dataset_calls += 1
            self.history.append("second")
            return {
                "taskId": "default",
                "states": {METRIC: 0},
                "results": {},
                "abnormalTimeRange": {METRIC: []},
            }

    def detector_factory(**_kwargs):
        instance = Detector()
        created.append(instance)
        return instance

    monitor = RemoteMonitor(
        config(tmp_path), detector_factory=detector_factory, offset_store=store
    )
    fetches = iter(
        [
            RemoteFetchResult((line("standard", 1, 1.0),), {"offset": 10}, {}),
            RemoteFetchResult((line("inference", 2, 1.5),), {"offset": 20}, {}),
        ]
    )
    monkeypatch.setattr(monitor, "_fetch", lambda _cursor: next(fetches))
    assert monitor.run()["sourceStatus"] == "ok"
    assert monitor.run()["sourceStatus"] == "ok"
    assert len(created) == 1
    assert created[0].detect_dataset_calls == 1
    assert created[0].history == ["first", "second"]


def test_remote_detection_or_fetch_failure_never_commits_or_reports_degradation(
    monkeypatch, tmp_path
):
    store = RecordingOffsetStore()

    def failing_factory(**_kwargs):
        raise RuntimeError("algorithm failed")

    monitor = RemoteMonitor(
        config(tmp_path), detector_factory=failing_factory, offset_store=store
    )
    monkeypatch.setattr(
        monitor,
        "_fetch",
        lambda _cursor: RemoteFetchResult(
            (line("standard", 1, 1.0),), {"offset": 10}, {}
        ),
    )
    response = monitor.run()
    assert response["states"] == {METRIC: 2}
    assert response["abnormalTimeRange"] == {METRIC: []}
    assert response["sourceStatus"] == "error"
    assert store.saved == []

    monkeypatch.setattr(
        monitor,
        "_fetch",
        lambda _cursor: (_ for _ in ()).throw(RemoteConnectionError("offline")),
    )
    response = monitor.run()
    assert response["states"] == {METRIC: 2}
    assert store.saved == []


class TrackingFile(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.close_called = False

    def close(self):
        self.close_called = True
        super().close()


class FakeChannel:
    def __init__(self):
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value


class FakeSFTP:
    def __init__(self, remote_file, size):
        self.remote_file = remote_file
        self.size = size
        self.close_called = False
        self.channel = FakeChannel()

    def get_channel(self):
        return self.channel

    def stat(self, _path):
        return SimpleNamespace(st_size=self.size)

    def open(self, _path, _mode):
        return self.remote_file

    def close(self):
        self.close_called = True


class MissingSFTP(FakeSFTP):
    def stat(self, _path):
        raise FileNotFoundError("missing")


class FakeSSH:
    def __init__(self, sftp, *, connect_error=None):
        self.sftp = sftp
        self.connect_error = connect_error
        self.close_called = False
        self.loaded_system_keys = False
        self.policy = None

    def load_system_host_keys(self):
        self.loaded_system_keys = True

    def load_host_keys(self, _path):
        pass

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **_kwargs):
        if self.connect_error is not None:
            raise self.connect_error

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.close_called = True


class FakeParamiko:
    class RejectPolicy:
        pass

    class AuthenticationException(Exception):
        pass

    class BadHostKeyException(Exception):
        pass

    class NoValidConnectionsError(Exception):
        pass

    class SSHException(Exception):
        pass


class TrackingStream(io.BytesIO):
    def __init__(self, value=b"", *, exit_status=0):
        super().__init__(value)
        self.close_called = False
        self.channel = SimpleNamespace(
            recv_exit_status=lambda: exit_status,
            set_combine_stderr=lambda _enabled: None,
        )

    def close(self):
        self.close_called = True
        super().close()


class DockerSSH(FakeSSH):
    def __init__(self, streams):
        super().__init__(sftp=None)
        self.streams = streams

    def exec_command(self, _command, timeout):
        assert timeout == 1.0
        return self.streams


def test_sftp_resources_close_and_partial_line_is_not_committed(
    monkeypatch, tmp_path
):
    complete = line("standard", 1, 1.0) + b"\n"
    partial = b'{"phase":"inference"'
    payload = complete + partial
    remote_file = TrackingFile(payload)
    sftp = FakeSFTP(remote_file, len(payload))
    ssh = FakeSSH(sftp)
    store = RecordingOffsetStore()
    captured = {}

    class Detector:
        def detect(self):
            return {
                "taskId": "default",
                "states": {METRIC: 2},
                "results": {},
                "abnormalTimeRange": {METRIC: []},
            }

    def factory(**kwargs):
        captured.update(kwargs)
        return Detector()

    monkeypatch.setattr(module, "paramiko", FakeParamiko)
    response = RemoteMonitor(
        config(tmp_path),
        detector_factory=factory,
        ssh_factory=lambda: ssh,
        offset_store=store,
    ).run()
    assert response["sourceStatus"] == "ok"
    assert captured["dataset"]["standard"][METRIC]["values"] == [1.0]
    assert captured["dataset"]["inference"][METRIC]["values"] == []
    assert store.saved[0][1]["offset"] == len(complete)
    assert remote_file.close_called is True
    assert sftp.close_called is True
    assert sftp.channel.timeout == 1.0
    assert ssh.close_called is True
    assert ssh.loaded_system_keys is True
    assert isinstance(ssh.policy, FakeParamiko.RejectPolicy)


def test_authentication_failure_closes_client_and_returns_empty_state_two(
    monkeypatch, tmp_path
):
    remote_file = TrackingFile(b"")
    sftp = FakeSFTP(remote_file, 0)
    ssh = FakeSSH(sftp, connect_error=FakeParamiko.AuthenticationException())
    store = RecordingOffsetStore()
    monkeypatch.setattr(module, "paramiko", FakeParamiko)
    response = RemoteMonitor(
        config(tmp_path), ssh_factory=lambda: ssh, offset_store=store
    ).run()
    assert response["sourceError"]["code"] == "REMOTE_AUTHENTICATION_FAILED"
    assert response["abnormalTimeRange"] == {METRIC: []}
    assert ssh.close_called is True
    assert store.saved == []


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (FakeParamiko.BadHostKeyException(), "REMOTE_HOST_KEY_FAILED"),
        (TimeoutError(), "REMOTE_CONNECTION_FAILED"),
    ],
)
def test_host_key_and_timeout_failures_close_client_and_return_state_two(
    monkeypatch, tmp_path, error, expected_code
):
    ssh = FakeSSH(FakeSFTP(TrackingFile(b""), 0), connect_error=error)
    store = RecordingOffsetStore()
    monkeypatch.setattr(module, "paramiko", FakeParamiko)
    response = RemoteMonitor(
        config(tmp_path), ssh_factory=lambda: ssh, offset_store=store
    ).run()
    assert response["sourceError"]["code"] == expected_code
    assert response["states"] == {METRIC: 2}
    assert ssh.close_called is True
    assert store.saved == []


def test_remote_file_not_found_closes_sftp_and_client_without_save(
    monkeypatch, tmp_path
):
    sftp = MissingSFTP(TrackingFile(b""), 0)
    ssh = FakeSSH(sftp)
    store = RecordingOffsetStore()
    monkeypatch.setattr(module, "paramiko", FakeParamiko)
    response = RemoteMonitor(
        config(tmp_path), ssh_factory=lambda: ssh, offset_store=store
    ).run()
    assert response["sourceError"]["code"] == "REMOTE_SOURCE_NOT_FOUND"
    assert response["states"] == {METRIC: 2}
    assert sftp.close_called is True
    assert ssh.close_called is True
    assert store.saved == []


def test_docker_nonzero_status_closes_all_streams_and_never_saves(
    monkeypatch, tmp_path
):
    stdin = TrackingStream()
    stdout = TrackingStream(b"", exit_status=125)
    stderr = TrackingStream(b"container does not exist")
    ssh = DockerSSH((stdin, stdout, stderr))
    store = RecordingOffsetStore()
    docker_config = config(
        tmp_path,
        source=RemoteSourceConfig(type="docker", container="missing-container"),
    )
    monkeypatch.setattr(module, "paramiko", FakeParamiko)
    response = RemoteMonitor(
        docker_config, ssh_factory=lambda: ssh, offset_store=store
    ).run()
    assert response["sourceError"]["code"] == "REMOTE_COMMAND_FAILED"
    assert response["states"] == {METRIC: 2}
    assert response["abnormalTimeRange"] == {METRIC: []}
    assert stdin.close_called is True
    assert stdout.close_called is True
    assert stderr.close_called is True
    assert ssh.close_called is True
    assert store.saved == []


def test_remote_config_defaults_none_task_and_rejects_unsafe_docker(tmp_path):
    path = tmp_path / "monitor.yaml"
    path.write_text(
        "\n".join(
            [
                "host: monitor.example.com",
                "username: rl-insight",
                "source:",
                "  type: file",
                "  remote_path: /metrics.jsonl",
                "metrics:",
                "  - timing_s/step",
                "task_id: null",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_remote_monitor_config(path)
    assert loaded.task_id == "default"
    assert loaded.metrics == (METRIC,)

    path.write_text(
        "\n".join(
            [
                "host: monitor.example.com",
                "username: rl-insight",
                "source:",
                "  type: docker",
                '  container: "bad; rm -rf"',
                "metrics:",
                "  - timing_s/step",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        load_remote_monitor_config(path)
