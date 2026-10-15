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

"""Run catalog with recoverable, readable per-Run file projections."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..protocol import RunContext, RunError, validate_id


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist observability fields; never persist arbitrary framework config."""
    if not isinstance(config, Mapping):
        raise RunError("Observability config must be an object")
    result: dict[str, Any] = {}
    for section, fields in {
        "server": ("namespace", "backend"),
        "prometheus": ("metrics_report_port",),
    }.items():
        raw = config.get(section, {})
        if not isinstance(raw, Mapping):
            raise RunError("Invalid observability config section")
        selected = {key: raw[key] for key in fields if key in raw}
        if any(not isinstance(value, (str, int, bool)) for value in selected.values()):
            raise RunError("Invalid observability config value")
        if selected:
            result[section] = selected
    return result


class RunRegistry:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.root / "runs.sqlite", timeout=10, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
            context TEXT NOT NULL, config TEXT NOT NULL, created_at REAL NOT NULL
        )""")
        self._db.commit()

    def create(
        self,
        request_id: str,
        project: str,
        experiment_name: str,
        server_url: str,
        config: Mapping[str, Any],
    ) -> RunContext:
        validate_id(request_id)
        context = RunContext(uuid4().hex, project, experiment_name, server_url)
        snapshot = _config_snapshot(config)
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT * FROM runs WHERE request_id=?", (request_id,)
            ).fetchone()
            if row:
                existing = RunContext.from_dict(json.loads(row["context"]))
                if (
                    existing.project,
                    existing.experiment_name,
                    json.loads(row["config"]),
                ) != (project, experiment_name, snapshot):
                    raise RunError(
                        "Run creation request was reused with different metadata", 409
                    )
                context = existing
            else:
                self._db.execute(
                    "INSERT INTO runs VALUES (?,?,?,?,?)",
                    (
                        context.run_id,
                        request_id,
                        _json(context.to_dict()),
                        _json(snapshot),
                        time.time(),
                    ),
                )
        # The DB owns identity. A failed projection can be rebuilt on get/retry.
        self.snapshot(context.run_id)
        return context

    def get(self, run_id: str) -> RunContext:
        return RunContext.from_dict(json.loads(self._read_run(run_id)["context"]))

    def list_runs(self) -> list[RunContext]:
        with self._lock:
            rows = self._db.execute(
                "SELECT context FROM runs ORDER BY created_at DESC, run_id DESC"
            ).fetchall()
        return [RunContext.from_dict(json.loads(row["context"])) for row in rows]

    def _read_run(self, run_id: str) -> sqlite3.Row:
        validate_id(run_id)
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunError("Run not found", 404)
            return row

    def snapshot(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._read_run(run_id)
            context = RunContext.from_dict(json.loads(row["context"]))
            directory = self.root / "runs" / run_id
            metadata = {
                "schema_version": 1,
                "context": context.to_dict(),
                "created_at": row["created_at"],
                "status": "registered",
                "ended_at": None,
            }
            self._write(directory / "run.json", metadata)
            self._write(directory / "config.json", json.loads(row["config"]))
            return metadata

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + uuid4().hex)
        try:
            temporary.write_text(_json(value) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            self._db.close()
