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

"""Unit tests for local server runtime monitoring hooks."""

from pathlib import Path
from unittest.mock import MagicMock

from rl_insight.server import runtime


def _stack(process: MagicMock) -> runtime.StartedStack:
    return runtime.StartedStack(
        services=[
            runtime.StartedService(
                name="prometheus",
                process=process,
                command=["prometheus"],
                log_file=Path("/tmp/prometheus.log"),
            )
        ],
        state_file=Path("/tmp/rl-insight-state.json"),
        install_root=Path("/tmp/rl-insight"),
    )


def test_wait_should_run_diagnostic_tick_without_changing_process_exit(
    monkeypatch,
) -> None:
    process = MagicMock()
    process.poll.side_effect = [None, 3]
    on_tick = MagicMock()
    stop_services = MagicMock()
    remove_state = MagicMock()
    monkeypatch.setattr(runtime, "stop_started_services", stop_services)
    monkeypatch.setattr(runtime, "_remove_state", remove_state)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runtime.time, "sleep", MagicMock())

    result = runtime.LocalServiceRuntime.wait(
        _stack(process),
        attach_logs=False,
        on_tick=on_tick,
        tick_interval_seconds=5.0,
    )

    assert result == 3
    on_tick.assert_called_once_with()
    stop_services.assert_called_once()
    remove_state.assert_called_once()


def test_wait_should_continue_when_diagnostic_tick_raises(monkeypatch, capsys) -> None:
    process = MagicMock()
    process.poll.side_effect = [None, 2]
    stop_services = MagicMock()
    monkeypatch.setattr(runtime, "stop_started_services", stop_services)
    monkeypatch.setattr(runtime, "_remove_state", MagicMock())
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runtime.time, "sleep", MagicMock())

    result = runtime.LocalServiceRuntime.wait(
        _stack(process),
        attach_logs=False,
        on_tick=MagicMock(side_effect=RuntimeError("query failed")),
    )

    assert result == 2
    assert "Training diagnostics unavailable: query failed" in capsys.readouterr().err
    stop_services.assert_called_once()
