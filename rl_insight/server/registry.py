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

"""SQLite Run catalog for identity, idempotent creation and metadata queries.

Prometheus/Tempo retain shared storage. No per-Run file copies are maintained.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import __version__
from ..utils.protocol import RunContext, RunError, validate_id


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
    def __init__(self, root: Path, lease_ttl: float = 30):
        if not 1 <= lease_ttl <= 3600:
            raise RunError("Endpoint TTL must be between 1 and 3600 seconds")
        self._lease_ttl = lease_ttl
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
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(runs)")}
        for name, definition in {
            "status": "TEXT NOT NULL DEFAULT 'registered'",
            "ended_at": "REAL",
            "owner_hash": "TEXT NOT NULL DEFAULT ''",
            "final_confirmed": "INTEGER",
            "closing_targets": "TEXT NOT NULL DEFAULT '[]'",
        }.items():
            if name not in columns:
                self._db.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        self._db.commit()
        self._targets: dict[tuple[str, str], dict[str, Any]] = {}

    def create(
        self,
        run_id: str,
        project: str,
        experiment_name: str,
        server_url: str,
        config: Mapping[str, Any],
        owner_token: str = "",
    ) -> RunContext:
        if owner_token:
            validate_id(owner_token)
        context = RunContext(run_id, project, experiment_name, server_url)
        snapshot = _config_snapshot(config)
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row:
                expected_owner = (
                    hashlib.sha256(owner_token.encode()).hexdigest()
                    if owner_token
                    else ""
                )
                if row["owner_hash"] != expected_owner:
                    raise RunError(
                        "Run creation owner conflicts with its original request", 409
                    )
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
                # Caller-owned IDs are also idempotent request keys.
                self._db.execute(
                    "INSERT INTO runs (run_id,request_id,context,config,created_at,owner_hash) VALUES (?,?,?,?,?,?)",
                    (
                        context.run_id,
                        run_id,
                        _json(context.metadata()),
                        _json(snapshot),
                        time.time(),
                        hashlib.sha256(owner_token.encode()).hexdigest()
                        if owner_token
                        else "",
                    ),
                )
        return context

    def get(self, run_id: str) -> RunContext:
        return RunContext.from_dict(json.loads(self._read_run(run_id)["context"]))

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT context, created_at FROM runs ORDER BY created_at DESC, run_id DESC"
            ).fetchall()
        return [
            {"context": json.loads(row["context"]), "created_at": row["created_at"]}
            for row in rows
        ]

    def _read_run(self, run_id: str) -> sqlite3.Row:
        validate_id(run_id)
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunError("Run not found", 404)
            return row

    def describe(self, run_id: str) -> dict[str, Any]:
        """Read execution metadata and allowlisted config without writing files."""
        row = self._read_run(run_id)
        return {
            "version": __version__,
            "context": RunContext.from_dict(json.loads(row["context"])).to_dict(),
            "config": json.loads(row["config"]),
            "created_at": row["created_at"],
            "status": row["status"],
            "ended_at": row["ended_at"],
            "final_collection_confirmed": None
            if row["final_confirmed"] is None
            else bool(row["final_confirmed"]),
        }

    def assert_open(self, run_id: str, *, allow_closing: bool = True) -> None:
        status = self._read_run(run_id)["status"]
        if status == "ended" or (not allow_closing and status == "closing"):
            raise RunError("Run is closed", 409)

    def ended_runs(self) -> set[str]:
        with self._lock:
            return {
                row[0]
                for row in self._db.execute(
                    "SELECT run_id FROM runs WHERE status='ended'"
                )
            }

    def transition(
        self, run_id: str, owner_token: str, status: str, final_confirmed=None
    ) -> None:
        validate_id(owner_token)
        with self._lock, self._db:
            row = self._read_run(run_id)
            if (
                not row["owner_hash"]
                or row["owner_hash"] != hashlib.sha256(owner_token.encode()).hexdigest()
            ):
                raise RunError("Only the Run creator can close it", 409)
            if row["status"] == "ended":
                return
            if status == "closing" and row["status"] != "closing":
                # Preserve the admission fence across restart without restoring stale leases.
                admitted = [
                    [item["job_name"], item["target"], item["publisher_id"]]
                    for item in self.targets()
                    if item["run_id"] == run_id
                ]
                self._db.execute(
                    "UPDATE runs SET closing_targets=? WHERE run_id=?",
                    (_json(admitted), run_id),
                )
            self._db.execute(
                "UPDATE runs SET status=?,ended_at=? WHERE run_id=?",
                (status, time.time() if status == "ended" else None, run_id),
            )
            if status == "ended":
                self._db.execute(
                    "UPDATE runs SET final_confirmed=? WHERE run_id=?",
                    (
                        int(final_confirmed)
                        if isinstance(final_confirmed, bool)
                        else None,
                        run_id,
                    ),
                )
                self._targets = {
                    key: value
                    for key, value in self._targets.items()
                    if value["run_id"] != run_id
                }

    def register_targets(self, context, publisher_id, job_name, targets):
        """Renew target ownership; only existing targets may renew while closing."""
        validate_id(publisher_id)
        with self._lock:
            self.assert_open(context.run_id)
            active = self.targets()
            incoming = [(job_name, target.target) for target in targets]
            for target in targets:
                for item in active:
                    if item["target"] == target.target and (
                        item["run_id"] != context.run_id
                        or item["publisher_id"] != publisher_id
                    ):
                        raise RunError(
                            "Target already belongs to another Run or publisher", 409
                        )
            row = self._read_run(context.run_id)
            if row["status"] == "closing":
                admitted = json.loads(row["closing_targets"])
                if any(
                    [job_name, target.target, publisher_id] not in admitted
                    for target in targets
                ):
                    raise RunError("Run is closing; new targets are rejected", 409)
            for target, key in zip(targets, incoming):
                self._targets[key] = {
                    "run_id": context.run_id,
                    "publisher_id": publisher_id,
                    "job_name": job_name,
                    "target": target.target,
                    "labels": dict(target.labels),
                    "expires_at": time.monotonic() + self._lease_ttl,
                }
        return self._lease_ttl

    def targets(self):
        with self._lock:
            now = time.monotonic()
            self._targets = {
                key: val
                for key, val in self._targets.items()
                if val["expires_at"] > now
            }
            return list(self._targets.values())

    def close(self) -> None:
        with self._lock:
            self._db.close()
