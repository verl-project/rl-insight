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

"""Unit tests for server startup diagnostics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import requests
from omegaconf import OmegaConf
from rl_insight.server import diagnostics
from rl_insight.server.runtime import StartedService


def _config(**overrides):
    conf = OmegaConf.create(
        {
            "server": {"enable": True, "port": 18080},
            "prometheus": {
                "enable": True,
                "prometheus_port": 9090,
            },
            "tempo": {"enable": True, "query_port": 3200},
            "otel": {"otel_port": 4318},
            "grafana": {"enable": True, "port": 3000},
        }
    )
    return OmegaConf.merge(conf, OmegaConf.create(overrides))


def _service(name: str, pid: int) -> StartedService:
    process = MagicMock()
    process.pid = pid
    process.poll.return_value = None
    return StartedService(
        name=name,
        process=process,
        command=[name],
        log_file=Path(f"/tmp/{name}.log"),
    )


def _response(status_code: int = 200, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    return response


def test_run_startup_diagnostics_should_report_healthy_services_and_targets(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "local_addresses",
        lambda: {
            "loopback": "127.0.0.1",
            "host": "10.0.0.8",
        },
    )
    responses = {
        "http://127.0.0.1:18080/healthz": _response(payload={"status": "ok"}),
        "http://127.0.0.1:9090/-/ready": _response(),
        "http://127.0.0.1:3200/ready": _response(),
        "http://127.0.0.1:3000/api/health": _response(payload={"database": "ok"}),
        "http://127.0.0.1:9090/api/v1/targets": _response(
            payload={
                "status": "success",
                "data": {
                    "activeTargets": [
                        {
                            "labels": {"job": "trainer_metrics"},
                            "scrapeUrl": "http://10.0.0.9:9092/metrics",
                            "health": "up",
                        }
                    ]
                },
            }
        ),
    }
    http_get = MagicMock(side_effect=lambda url, **_: responses[url])
    connection = MagicMock()
    socket_connect = MagicMock(return_value=connection)

    report = diagnostics.run_startup_diagnostics(
        _config(),
        [
            _service("prometheus", 101),
            _service("tempo", 102),
            _service("grafana", 103),
            _service("rl-insight-server", 104),
        ],
        http_get=http_get,
        socket_connect=socket_connect,
    )

    assert [result.status for result in report.services] == [
        "OK",
        "OK",
        "OK",
        "OK",
        "OK",
    ]
    assert report.target_summaries == ["trainer_metrics: 1 up, 0 down"]
    assert report.target_failures == []
    assert any("10.0.0.8:18080" in note for note in report.notes)
    assert all(call.kwargs["timeout"] == diagnostics.REQUEST_TIMEOUT for call in http_get.call_args_list)
    socket_connect.assert_called_once_with(("127.0.0.1", 4318), timeout=1.0)
    connection.close.assert_called_once_with()


def test_run_startup_diagnostics_should_continue_after_service_failures(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "local_addresses",
        lambda: {"loopback": "127.0.0.1", "host": ""},
    )

    def get(url, **_kwargs):
        if url.endswith("/healthz"):
            raise requests.ConnectionError("connection refused")
        if url.endswith("/-/ready"):
            raise requests.Timeout("timed out")
        if url.endswith("/ready"):
            return _response(status_code=503)
        return _response(payload={"database": "failed"})

    report = diagnostics.run_startup_diagnostics(
        _config(),
        [
            _service("prometheus", 201),
            _service("tempo", 202),
            _service("grafana", 203),
            _service("rl-insight-server", 204),
        ],
        http_get=get,
        socket_connect=MagicMock(side_effect=ConnectionRefusedError("refused")),
        sleep=MagicMock(),
    )

    assert [result.readiness for result in report.services] == [
        "connection refused",
        "timeout",
        "HTTP 503",
        "invalid response",
        "not listening",
    ]
    assert all(result.status == "WARNING" for result in report.services)
    assert report.target_summaries == []
    assert report.target_failures == []


def test_run_startup_diagnostics_should_skip_disabled_services(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostics,
        "local_addresses",
        lambda: {"loopback": "127.0.0.1", "host": ""},
    )
    http_get = MagicMock(side_effect=[_response(payload={"status": "ok"})])
    socket_connect = MagicMock()

    report = diagnostics.run_startup_diagnostics(
        _config(
            prometheus={"enable": False},
            tempo={"enable": False},
            grafana={"enable": False},
        ),
        [_service("rl-insight-server", 301)],
        http_get=http_get,
        socket_connect=socket_connect,
    )

    assert [result.readiness for result in report.services] == [
        "healthy",
        "disabled",
        "disabled",
        "disabled",
        "disabled",
    ]
    http_get.assert_called_once_with(
        "http://127.0.0.1:18080/healthz",
        timeout=diagnostics.REQUEST_TIMEOUT,
    )
    socket_connect.assert_not_called()


def test_prometheus_targets_should_report_down_target_details() -> None:
    http_get = MagicMock(
        return_value=_response(
            payload={
                "status": "success",
                "data": {
                    "activeTargets": [
                        {
                            "labels": {"job": "trainer_metrics"},
                            "scrapeUrl": "http://10.0.0.10:9092/metrics",
                            "health": "up",
                        },
                        {
                            "labels": {"job": "trainer_metrics"},
                            "scrapeUrl": "http://10.0.0.11:9092/metrics",
                            "health": "down",
                            "lastError": "connection refused",
                        },
                    ]
                },
            }
        )
    )

    summaries, failures, detail = diagnostics._check_prometheus_targets(host="127.0.0.1", port=9090, http_get=http_get)

    assert summaries == ["trainer_metrics: 1 up, 1 down"]
    assert failures == ["trainer_metrics: http://10.0.0.11:9092/metrics (connection refused)"]
    assert detail == ""


def test_run_startup_diagnostics_should_retry_until_services_are_ready(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "local_addresses",
        lambda: {"loopback": "127.0.0.1", "host": ""},
    )
    attempts = {
        "http://127.0.0.1:18080/healthz": [
            requests.ConnectionError("connection refused"),
            _response(payload={"status": "ok"}),
        ],
        "http://127.0.0.1:9090/-/ready": [_response()],
        "http://127.0.0.1:3200/ready": [
            _response(status_code=503),
            _response(),
        ],
        "http://127.0.0.1:3000/api/health": [
            requests.ConnectionError("connection refused"),
            _response(payload={"database": "ok"}),
        ],
        "http://127.0.0.1:9090/api/v1/targets": [
            _response(payload={"status": "success", "data": {"activeTargets": []}})
        ],
    }

    def get(url, **_kwargs):
        result = attempts[url].pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    sleep = MagicMock()
    report = diagnostics.run_startup_diagnostics(
        _config(),
        [
            _service("prometheus", 401),
            _service("tempo", 402),
            _service("grafana", 403),
            _service("rl-insight-server", 404),
        ],
        http_get=get,
        socket_connect=MagicMock(return_value=MagicMock()),
        sleep=sleep,
    )

    assert [result.status for result in report.services] == [
        "OK",
        "OK",
        "OK",
        "OK",
        "OK",
    ]
    assert sleep.call_count == 3
    sleep.assert_called_with(diagnostics.STARTUP_READINESS_RETRY_DELAY_SECONDS)
