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

import json
from pathlib import Path

import numpy as np

from experiment.degradation_perception import main as main_module


def test_cli_only_parses_constructs_detects_and_prints_one_json_line(
    monkeypatch, capsys, tmp_path
):
    captured = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def detect(self):
            return {
                "taskId": "default",
                "states": {"timing_s/step": np.int64(0)},
                "results": {},
                "abnormalTimeRange": {},
            }

    monkeypatch.setattr(main_module, "DegradationPerception", FakeDetector)
    path = tmp_path / "input.json"
    code = main_module.main(
        [
            "--path",
            str(path),
            "--start-time",
            "10",
            "--end-time",
            "20",
            "--metrics",
            "timing_s/step",
            "latency",
            "--source-type",
            "training_log",
        ]
    )
    output = capsys.readouterr()
    assert code == 0
    assert output.err == ""
    assert len(output.out.splitlines()) == 1
    parsed = json.loads(output.out)
    assert parsed["states"]["timing_s/step"] == 0
    assert captured == {
        "path": path,
        "start_time": 10.0,
        "end_time": 20.0,
        "metrics": ["timing_s/step", "latency"],
        "task_id": None,
        "source_type": "training_log",
    }


def test_cli_passes_explicit_config_dir(monkeypatch, capsys, tmp_path):
    captured = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def detect(self):
            return {"ok": True}

    monkeypatch.setattr(main_module, "DegradationPerception", FakeDetector)
    config_dir = tmp_path / "config"
    assert main_module.main(
        [
            "--path",
            str(tmp_path / "input.json"),
            "--config-dir",
            str(config_dir),
        ]
    ) == 0
    capsys.readouterr()
    assert captured["config_dir"] == config_dir


def test_cli_runtime_error_is_strict_json_and_nonzero(monkeypatch, capsys, tmp_path):
    class FailingDetector:
        def __init__(self, **_kwargs):
            pass

        def detect(self):
            raise RuntimeError("detector failed")

    monkeypatch.setattr(main_module, "DegradationPerception", FailingDetector)
    code = main_module.main(["--path", str(tmp_path / "input.json")])
    parsed = json.loads(capsys.readouterr().out)
    assert code == 1
    assert parsed == {
        "ok": False,
        "error": {"type": "RuntimeError", "message": "detector failed"},
    }


def test_cli_rejects_nonstandard_nan_output_as_json_error(monkeypatch, capsys, tmp_path):
    class NanDetector:
        def __init__(self, **_kwargs):
            pass

        def detect(self):
            return {"value": float("nan")}

    monkeypatch.setattr(main_module, "DegradationPerception", NanDetector)
    code = main_module.main(["--path", str(tmp_path / "input.json")])
    raw = capsys.readouterr().out
    assert "NaN" not in raw
    parsed = json.loads(raw)
    assert code == 1
    assert parsed["error"]["type"] == "ValueError"


def test_sample_data_runs_through_the_real_public_cli(capsys, tmp_path):
    sample = Path(main_module.__file__).with_name("sample_data.json")
    code = main_module.main(
        [
            "--path",
            str(sample),
            "--metrics",
            "timing_s/step",
            "--config-dir",
            str(tmp_path / "config"),
        ]
    )
    parsed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert parsed["taskId"] == "default"
    assert parsed["states"]["timing_s/step"] == 0
    assert parsed["abnormalTimeRange"]["timing_s/step"]
