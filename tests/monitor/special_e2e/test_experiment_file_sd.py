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

"""Real Prometheus file_sd E2E for experiment partitions.

Requires a real Prometheus binary: set ``RL_INSIGHT_PROMETHEUS_BIN``, or have
``prometheus`` on PATH, or use the managed install directory. The test boots a
real Prometheus watching the legacy/global file plus the experiment
``active-targets.yml`` glob, then verifies discovery, archive, restore, and
restart recovery inside real file_sd refresh cycles.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from rl_insight.utils.experiment_targets import (
    ArchivedExperimentError,
    ExperimentTargetStore,
)
from rl_insight.utils.prometheus_utils import PrometheusTarget, write_file_sd_target_map

MANAGED_PROMETHEUS = (
    Path.home() / ".rl-insight" / "services" / "prometheus" / "prometheus"
)
DISCOVERY_TIMEOUT_SECONDS = 15.0


def _find_prometheus() -> Path | None:
    env = os.environ.get("RL_INSIGHT_PROMETHEUS_BIN")
    which = shutil.which("prometheus")
    candidates = [
        Path(env) if env else None,
        Path(which) if which else None,
        MANAGED_PROMETHEUS,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PrometheusProcess:
    """One real Prometheus subprocess bound to a rendered runtime config."""

    def __init__(self, binary: Path, config_file: Path, data_dir: Path, port: int):
        self.binary = binary
        self.config_file = config_file
        self.data_dir = data_dir
        self.port = port
        self.process: subprocess.Popen[Any] | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                str(self.binary),
                f"--config.file={self.config_file}",
                f"--storage.tsdb.path={self.data_dir / 'tsdb'}",
                f"--web.listen-address=127.0.0.1:{self.port}",
                "--storage.tsdb.retention.time=1d",
                "--web.enable-lifecycle",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                response = requests.get(self._url("/-/ready"), timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise RuntimeError("Prometheus did not become ready in time")

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def active_targets(self) -> list[dict[str, Any]]:
        response = requests.get(self._url("/api/v1/targets?state=active"), timeout=5)
        response.raise_for_status()
        return (response.json().get("data") or {}).get("activeTargets") or []

    def discovered_pairs(self) -> set[tuple[str, str]]:
        pairs = set()
        for target in self.active_targets():
            labels = target.get("labels") or {}
            pairs.add(
                (
                    str(labels.get("project") or ""),
                    str(labels.get("experiment_name") or ""),
                )
            )
        return pairs

    def wait_for_pairs(self, expected: set[tuple[str, str]]) -> None:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.discovered_pairs() >= expected:
                return
            time.sleep(0.25)
        raise AssertionError(
            f"Prometheus did not discover {expected}; saw {self.discovered_pairs()}"
        )


@pytest.fixture()
def prometheus(tmp_path: Any) -> Iterator[PrometheusProcess]:
    binary = _find_prometheus()
    if binary is None:
        pytest.skip("no real Prometheus binary available")
    assert binary is not None  # mypy can't see pytest.skip's NoReturn
    data_dir = tmp_path / "data"
    targets_file = data_dir / "targets" / "prometheus-targets.yml"
    targets_file.parent.mkdir(parents=True, exist_ok=True)
    write_file_sd_target_map(targets_file, {})

    experiment_glob = data_dir / "projects" / "*.active.yml"
    config_file = tmp_path / "prometheus.yml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "global": {"scrape_interval": "1s"},
                "scrape_configs": [
                    {
                        "job_name": "rl-insight-dynamic",
                        "file_sd_configs": [
                            {
                                "files": [str(targets_file), str(experiment_glob)],
                                "refresh_interval": "1s",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    process = PrometheusProcess(binary, config_file, tmp_path, _free_port())
    process.start()
    yield process
    process.stop()


def test_prometheus_file_sd_discovers_archives_and_restores_experiments(
    prometheus: PrometheusProcess, tmp_path: Any
) -> None:
    store = ExperimentTargetStore(tmp_path / "data", prometheus.port)
    store.register(
        "project-a", "exp-1", "trainer_metrics", [PrometheusTarget("127.0.0.1:19001")]
    )
    store.register(
        "project-b", "exp-1", "trainer_metrics", [PrometheusTarget("127.0.0.1:19002")]
    )

    prometheus.wait_for_pairs({("project-a", "exp-1"), ("project-b", "exp-1")})

    archived = store.archive("project-a", "exp-1")
    assert archived["prometheus_converged"] is True
    assert ("project-a", "exp-1") not in prometheus.discovered_pairs()
    assert ("project-b", "exp-1") in prometheus.discovered_pairs()

    with pytest.raises(ArchivedExperimentError):
        store.register(
            "project-a",
            "exp-1",
            "trainer_metrics",
            [PrometheusTarget("127.0.0.1:19003")],
        )

    restored = store.restore("project-a", "exp-1")
    assert restored["prometheus_converged"] is True
    prometheus.wait_for_pairs({("project-a", "exp-1"), ("project-b", "exp-1")})

    store.archive("project-a", "exp-1")
    assert ("project-a", "exp-1") not in prometheus.discovered_pairs()

    prometheus.stop()
    prometheus.start()
    prometheus.wait_for_pairs({("project-b", "exp-1")})
    assert ("project-a", "exp-1") not in prometheus.discovered_pairs()
