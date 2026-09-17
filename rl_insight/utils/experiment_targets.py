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

"""Experiment-scoped Prometheus target partitions with archive/restore.

Experiments are identified by the case-sensitive ``(project, experiment_name)``
name pair. Each experiment owns an isolated manifest plus Prometheus discovery
files under ``<data_dir>/projects/<sha256(project)>__<sha256(experiment_name)>``
as ``.active.yml`` (watched), ``.archived.yml`` (archive snapshot) and
``.manifest.yaml`` siblings; the digest keys keep untrusted names out of the
filesystem paths. Archiving an experiment only moves its discovery file out of
the Prometheus watch set — history already written to the shared TSDB/Tempo
stays queryable until normal retention expires.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
import yaml
from omegaconf import DictConfig, OmegaConf

from ..server.network import format_host_port, local_addresses
from .constants import (
    ExperimentTargets,
    experiments_root_from_config,
)
from .prometheus_utils import (
    PrometheusTarget,
    file_sd_lock,
    read_file_sd_target_map,
    write_file_sd_target_map,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = [
    "ArchivedExperimentError",
    "ExperimentIdentityError",
    "ExperimentNotFoundError",
    "ExperimentTargetStore",
    "experiment_dir_key",
    "normalize_experiment_identity",
]


class ExperimentIdentityError(ValueError):
    """Raised when identity fields are missing, blank, or conflict with reserved labels."""


class ExperimentNotFoundError(LookupError):
    """Raised when an experiment partition does not exist."""


class ArchivedExperimentError(RuntimeError):
    """Raised when mutating an archived experiment; explicit ``restore`` is required first."""


def normalize_experiment_identity(
    project: str | None, experiment_name: str | None
) -> tuple[str, str] | None:
    """Normalize the ``(project, experiment_name)`` composite key.

    Returns ``None`` when both fields are absent or blank (legacy/global
    targets), raises :class:`ExperimentIdentityError` when exactly one is
    provided, and otherwise returns both values stripped of surrounding
    whitespace. Case and Unicode content are preserved exactly.
    """
    normalized_project = str(project or "").strip()
    normalized_experiment = str(experiment_name or "").strip()
    if not normalized_project and not normalized_experiment:
        return None
    if not normalized_project or not normalized_experiment:
        raise ExperimentIdentityError(
            "project and experiment_name must both be non-empty to scope an "
            "experiment; provide both for experiment-scoped targets, or neither "
            "to keep registering legacy/global targets"
        )
    return normalized_project, normalized_experiment


def experiment_dir_key(name: str) -> str:
    """Return the filesystem-safe directory key for one identity name.

    The full SHA-256 hex digest of the UTF-8 name is repeatable and cannot
    escape ``data_dir``; it is only a path key, not a business id — the
    manifest keeps the readable original name.
    """
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class ExperimentTargetStore:
    """Own the experiment partition tree under ``<data_dir>/projects``.

    All mutations of one experiment are serialized by the partition's lock
    file (``.<stem>.active.yml.lock``, shared with the file-format helpers);
    different experiments proceed in parallel and never see each other's
    partial writes.
    """

    def __init__(
        self,
        data_dir: str | Path,
        prometheus_port: int,
        convergence_timeout_seconds: float
        | None = ExperimentTargets.CONVERGENCE_TIMEOUT_SECONDS,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.projects_root = self.data_dir / ExperimentTargets.PROJECTS_DIR_NAME
        self.prometheus_port = int(prometheus_port)
        self.convergence_timeout_seconds = convergence_timeout_seconds
        self._scan_lock = threading.Lock()

    @classmethod
    def from_config(
        cls, conf: DictConfig, convergence_timeout_seconds: float | None = None
    ) -> ExperimentTargetStore:
        """Build a store rooted at the configured server data directory."""
        timeout: float | None = ExperimentTargets.CONVERGENCE_TIMEOUT_SECONDS
        if convergence_timeout_seconds is not None:
            timeout = convergence_timeout_seconds
        raw_timeout = OmegaConf.select(
            conf, "prometheus.experiment_convergence_timeout"
        )
        if raw_timeout is not None:
            timeout = float(raw_timeout)
        return cls(
            experiments_root_from_config(conf).parent,
            int(OmegaConf.select(conf, "prometheus.prometheus_port")),
            timeout,
        )

    # ------------------------------------------------------------------ layout

    def experiment_stem(self, project: str, experiment_name: str) -> str:
        """Return the flat file-name stem for the composite key.

        The stem is the two SHA-256 name digests joined by ``__``; all
        partition files (``*.active.yml``, ``*.archived.yml``,
        ``*.manifest.yaml``) share it. Digests keep untrusted names out of the
        filesystem; the readable names live in the manifest.
        """
        identity = normalize_experiment_identity(project, experiment_name)
        if identity is None:
            raise ExperimentIdentityError(
                "an experiment partition requires both project and experiment_name"
            )
        return ExperimentTargets.PROJECT_KEY_SEPARATOR.join(
            experiment_dir_key(name) for name in identity
        )

    def active_file(self, project: str, experiment_name: str) -> Path:
        """Return the watched discovery file of the experiment."""
        return self.projects_root / (
            self.experiment_stem(project, experiment_name)
            + ExperimentTargets.ACTIVE_TARGETS_SUFFIX
        )

    def archived_file(self, project: str, experiment_name: str) -> Path:
        """Return the archived discovery snapshot of the experiment."""
        return self.projects_root / (
            self.experiment_stem(project, experiment_name)
            + ExperimentTargets.ARCHIVED_TARGETS_SUFFIX
        )

    def manifest_file(self, project: str, experiment_name: str) -> Path:
        """Return the manifest file of the experiment."""
        return self.projects_root / (
            self.experiment_stem(project, experiment_name)
            + ExperimentTargets.MANIFEST_SUFFIX
        )

    def _active_file(self, stem: str) -> Path:
        return self.projects_root / f"{stem}{ExperimentTargets.ACTIVE_TARGETS_SUFFIX}"

    def _archived_file(self, stem: str) -> Path:
        return self.projects_root / f"{stem}{ExperimentTargets.ARCHIVED_TARGETS_SUFFIX}"

    def _manifest_file(self, stem: str) -> Path:
        return self.projects_root / f"{stem}{ExperimentTargets.MANIFEST_SUFFIX}"

    @contextmanager
    def _partition_lock(self, stem: str) -> Iterator[None]:
        with file_sd_lock(self._active_file(stem)):
            yield

    # -------------------------------------------------------------- operations

    def register(
        self,
        project: str,
        experiment_name: str,
        job_name: str,
        targets: Sequence[PrometheusTarget],
    ) -> dict[str, Any]:
        """Merge targets into the experiment's active discovery file.

        The first valid registration creates the partition in ``active`` state.
        Registering into an archived experiment raises
        :class:`ArchivedExperimentError` instead of silently re-enabling it.
        """
        identity = normalize_experiment_identity(project, experiment_name)
        if identity is None:
            raise ExperimentIdentityError(
                "experiment registration requires both project and experiment_name"
            )
        project, experiment_name = identity
        stem = self.experiment_stem(project, experiment_name)
        identity_labels = {"project": project, "experiment_name": experiment_name}
        for item in targets:
            for key, expected in identity_labels.items():
                if key in item.labels and str(item.labels[key]) != expected:
                    raise ExperimentIdentityError(
                        f"target label {key!r} is reserved for the experiment "
                        f"identity and cannot be overridden (got {item.labels[key]!r})"
                    )

        with self._partition_lock(stem):
            manifest = self._reconcile(stem, project, experiment_name)
            if manifest["state"] == ExperimentTargets.STATE_ARCHIVED:
                raise ArchivedExperimentError(
                    f"experiment ({project!r}, {experiment_name!r}) is archived; "
                    "run restore before registering new targets"
                )
            target_map = read_file_sd_target_map(self._active_file(stem))
            for item in targets:
                labels = {str(k): str(v) for k, v in item.labels.items()}
                labels.update(identity_labels)
                target_map[(str(job_name), str(item.target))] = labels
            write_file_sd_target_map(self._active_file(stem), target_map)
            manifest.update(
                state=ExperimentTargets.STATE_ACTIVE,
                target_count=len(target_map),
                updated_at=_utc_now(),
            )
            self._write_manifest(stem, manifest)

        return {
            "job_name": str(job_name),
            "target_count": len(target_map),
            "project": project,
            "experiment_name": experiment_name,
            "state": ExperimentTargets.STATE_ACTIVE,
            "targets_file": str(self._active_file(stem)),
        }

    def archive(self, project: str, experiment_name: str) -> dict[str, Any]:
        """Stop discovery for one experiment; idempotent, history is never deleted."""
        project, experiment_name, stem = self._require_experiment(
            project, experiment_name
        )
        with self._partition_lock(stem):
            manifest = self._reconcile(stem, project, experiment_name)
            if manifest["state"] == ExperimentTargets.STATE_ARCHIVED:
                return self._lifecycle_result(manifest, None)
            active_file = self._active_file(stem)
            if active_file.exists():
                # Atomic move out of the ``*.active.yml`` watch set; other
                # experiments and the legacy/global file are untouched.
                os.replace(active_file, self._archived_file(stem))
            manifest["state"] = ExperimentTargets.STATE_ARCHIVED
            manifest["target_count"] = len(
                read_file_sd_target_map(self._archived_file(stem))
            )
            manifest["updated_at"] = _utc_now()
            self._write_manifest(stem, manifest)

        converged, message = self._wait_for_discovery_state(
            {"project": project, "experiment_name": experiment_name}, present=False
        )
        return self._lifecycle_result(
            self._read_manifest(stem) or {}, converged, message
        )

    def restore(self, project: str, experiment_name: str) -> dict[str, Any]:
        """Re-enable discovery from the archived snapshot; idempotent."""
        project, experiment_name, stem = self._require_experiment(
            project, experiment_name
        )
        with self._partition_lock(stem):
            manifest = self._reconcile(stem, project, experiment_name)
            if manifest["state"] == ExperimentTargets.STATE_ACTIVE:
                return self._lifecycle_result(manifest, None)
            archived_file = self._archived_file(stem)
            if archived_file.exists():
                os.replace(archived_file, self._active_file(stem))
            manifest["state"] = ExperimentTargets.STATE_ACTIVE
            manifest["target_count"] = len(
                read_file_sd_target_map(self._active_file(stem))
            )
            manifest["updated_at"] = _utc_now()
            self._write_manifest(stem, manifest)

        expected = self._expected_labels(stem, project, experiment_name)
        converged, message = self._wait_for_discovery_state(expected, present=True)
        return self._lifecycle_result(
            self._read_manifest(stem) or {}, converged, message
        )

    def list_experiments(self, project: str | None = None) -> list[dict[str, Any]]:
        """List experiment partitions with state, composite key, and target count."""
        rows: list[dict[str, Any]] = []
        if not self.projects_root.exists():
            return rows
        with self._scan_lock:
            stems = (
                {
                    path.name[: -len(ExperimentTargets.MANIFEST_SUFFIX)]
                    for path in self.projects_root.glob(
                        f"*{ExperimentTargets.MANIFEST_SUFFIX}"
                    )
                }
                | {
                    path.name[: -len(ExperimentTargets.ACTIVE_TARGETS_SUFFIX)]
                    for path in self.projects_root.glob(
                        f"*{ExperimentTargets.ACTIVE_TARGETS_SUFFIX}"
                    )
                }
                | {
                    path.name[: -len(ExperimentTargets.ARCHIVED_TARGETS_SUFFIX)]
                    for path in self.projects_root.glob(
                        f"*{ExperimentTargets.ARCHIVED_TARGETS_SUFFIX}"
                    )
                }
            )
            for stem in sorted(stems):
                stored = self._read_manifest(stem)
                if stored is None:
                    # Crash debris before the manifest existed: heal the
                    # readable names from the injected discovery labels.
                    stored = self._heal_manifest_from_labels(stem)
                if stored is None:
                    logger.warning(
                        "[rl-insight] Skipping partition %s: manifest missing and "
                        "discovery labels carry no experiment identity.",
                        stem,
                    )
                    continue
                stored_project = str(stored.get("project") or "")
                if project is not None and stored_project != str(project).strip():
                    continue
                with self._partition_lock(stem):
                    manifest = self._reconcile(
                        stem,
                        stored_project,
                        str(stored.get("experiment_name") or ""),
                    )
                rows.append(
                    {
                        "project": manifest["project"],
                        "experiment_name": manifest["experiment_name"],
                        "state": manifest["state"],
                        "target_count": manifest["target_count"],
                        "created_at": manifest.get("created_at"),
                        "updated_at": manifest.get("updated_at"),
                    }
                )
        return rows

    def _heal_manifest_from_labels(self, stem: str) -> dict[str, Any] | None:
        """Rebuild a lost manifest from the injected identity labels, if any."""
        for suffix in (
            ExperimentTargets.ACTIVE_TARGETS_SUFFIX,
            ExperimentTargets.ARCHIVED_TARGETS_SUFFIX,
        ):
            path = self.projects_root / f"{stem}{suffix}"
            for labels in read_file_sd_target_map(path).values():
                identity = normalize_experiment_identity(
                    labels.get("project"), labels.get("experiment_name")
                )
                if identity is not None:
                    return {
                        "schema_version": ExperimentTargets.SCHEMA_VERSION,
                        "project": identity[0],
                        "experiment_name": identity[1],
                    }
        return None

    def show_targets(self, project: str, experiment_name: str) -> dict[str, Any]:
        """Return the manifest state and discovery records of one experiment."""
        project, experiment_name, stem = self._require_experiment(
            project, experiment_name
        )
        with self._partition_lock(stem):
            manifest = self._reconcile(stem, project, experiment_name)
            source = (
                self._active_file(stem)
                if manifest["state"] == ExperimentTargets.STATE_ACTIVE
                else self._archived_file(stem)
            )
            target_map = read_file_sd_target_map(source)
        targets = [
            {"job_name": job, "target": target, "labels": dict(labels)}
            for (job, target), labels in sorted(target_map.items())
        ]
        return {
            "project": manifest["project"],
            "experiment_name": manifest["experiment_name"],
            "state": manifest["state"],
            "target_count": len(targets),
            "updated_at": manifest.get("updated_at"),
            "targets": targets,
        }

    # --------------------------------------------------------------- migration

    def migrate_legacy_targets(
        self, global_targets_file: str | Path, backup_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """Move fully-identified records from the legacy/global file into partitions.

        Records carrying both reserved labels are merged into the matching
        experiment partition; records missing one label, conflicting, or
        targeting an archived experiment stay in the legacy/global file. The
        global file is only replaced after every partition write succeeded, so
        a failure leaves the old discovery state untouched and re-runs are
        idempotent. A byte-level backup of the original file is kept for
        rollback when anything actually changed.
        """
        global_file = Path(global_targets_file)
        if not global_file.exists():
            return {
                "scanned": 0,
                "migrated": 0,
                "kept_legacy": 0,
                "backup_file": None,
                "changed": False,
            }

        backup_file: Path | None = None
        with file_sd_lock(global_file):
            original = read_file_sd_target_map(global_file)
            plan: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = {}
            legacy: dict[tuple[str, str], dict[str, str]] = {}
            for key, labels in original.items():
                # Missing/blank/conflicting identity labels stay in the legacy
                # partition; only fully-identified records move.
                record_project = str(labels.get("project") or "").strip()
                record_experiment = str(labels.get("experiment_name") or "").strip()
                if not record_project or not record_experiment:
                    legacy[key] = labels
                    continue
                identity = (record_project, record_experiment)
                stem = self.experiment_stem(*identity)
                with self._partition_lock(stem):
                    manifest = self._reconcile(stem, *identity)
                if manifest["state"] == ExperimentTargets.STATE_ARCHIVED:
                    logger.warning(
                        "[rl-insight] Migration keeps record %s in the global file: "
                        "experiment (%s, %s) is archived.",
                        key,
                        *identity,
                    )
                    legacy[key] = labels
                    continue
                plan.setdefault(identity, {})[key] = {
                    **labels,
                    "project": identity[0],
                    "experiment_name": identity[1],
                }

            if not plan:
                return {
                    "scanned": len(original),
                    "migrated": 0,
                    "kept_legacy": len(legacy),
                    "backup_file": None,
                    "changed": False,
                }

            backup_root = (
                Path(backup_dir)
                if backup_dir
                else self.data_dir / ExperimentTargets.BACKUPS_DIR_NAME
            )
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_file = backup_root / f"{global_file.name}.{time.time_ns()}.bak"
            shutil.copy2(global_file, backup_file)

            migrated = 0
            try:
                for identity, records in sorted(plan.items()):
                    migrated += self._merge_partition_records(identity, records)
            except BaseException:
                # Keep the old global file; partial partition writes are merged
                # away by the next idempotent run.
                logger.error(
                    "[rl-insight] Experiment target migration failed; the legacy "
                    "global discovery file was left unchanged."
                )
                raise

            write_file_sd_target_map(global_file, legacy)

        return {
            "scanned": len(original),
            "migrated": migrated,
            "kept_legacy": len(legacy),
            "backup_file": str(backup_file) if backup_file else None,
            "changed": True,
        }

    def _merge_partition_records(
        self,
        identity: tuple[str, str],
        records: dict[tuple[str, str], dict[str, str]],
    ) -> int:
        """Merge not-yet-present records into one experiment partition."""
        project, experiment_name = identity
        stem = self.experiment_stem(project, experiment_name)
        with self._partition_lock(stem):
            manifest = self._reconcile(stem, project, experiment_name)
            if manifest["state"] == ExperimentTargets.STATE_ARCHIVED:
                return 0
            target_map = read_file_sd_target_map(self._active_file(stem))
            missing = {k: v for k, v in records.items() if k not in target_map}
            if not missing:
                return 0
            target_map.update(missing)
            write_file_sd_target_map(self._active_file(stem), target_map)
            manifest["target_count"] = len(target_map)
            manifest["updated_at"] = _utc_now()
            self._write_manifest(stem, manifest)
            return len(missing)

    # ---------------------------------------------------------------- internal

    def _require_experiment(
        self, project: str, experiment_name: str
    ) -> tuple[str, str, str]:
        identity = normalize_experiment_identity(project, experiment_name)
        if identity is None:
            raise ExperimentIdentityError(
                "an experiment partition requires both project and experiment_name"
            )
        stem = self.experiment_stem(*identity)
        has_manifest = self._manifest_file(stem).exists()
        has_discovery = (
            self._active_file(stem).exists() or self._archived_file(stem).exists()
        )
        if not has_manifest and not has_discovery:
            raise ExperimentNotFoundError(
                f"experiment ({identity[0]!r}, {identity[1]!r}) does not exist"
            )
        return identity[0], identity[1], stem

    def _read_manifest(self, stem: str) -> dict[str, Any] | None:
        manifest_file = self._manifest_file(stem)
        if not manifest_file.exists():
            return None
        data = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def _write_manifest(self, stem: str, manifest: dict[str, Any]) -> None:
        manifest_file = self._manifest_file(stem)
        payload = yaml.safe_dump(manifest, sort_keys=False)
        if (
            manifest_file.is_file()
            and manifest_file.read_text(encoding="utf-8") == payload
        ):
            return
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_file.with_name(f".{manifest_file.name}.{os.getpid()}.tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, manifest_file)
        except BaseException:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _reconcile(
        self, stem: str, project: str, experiment_name: str
    ) -> dict[str, Any]:
        """Return the manifest, repairing state/count from discovery file locations.

        The discovery file position is the source of truth: a crash between the
        archive/restore rename and the manifest update is fixed on the next
        read, so a stale manifest never misstates an experiment. Exception: when
        both discovery files are gone, the stored manifest state wins so an
        archived experiment is not silently resurrected.
        """
        active_exists = self._active_file(stem).exists()
        archived_exists = self._archived_file(stem).exists()
        manifest = self._read_manifest(stem) or {
            "schema_version": ExperimentTargets.SCHEMA_VERSION,
            "project": project,
            "experiment_name": experiment_name,
            "created_at": _utc_now(),
        }
        if archived_exists and not active_exists:
            state = ExperimentTargets.STATE_ARCHIVED
        elif active_exists or archived_exists:
            state = ExperimentTargets.STATE_ACTIVE
        elif manifest.get("state") == ExperimentTargets.STATE_ARCHIVED:
            state = ExperimentTargets.STATE_ARCHIVED
        else:
            state = ExperimentTargets.STATE_ACTIVE
        manifest.setdefault("schema_version", ExperimentTargets.SCHEMA_VERSION)
        manifest["project"] = project
        manifest["experiment_name"] = experiment_name
        manifest["state"] = state
        manifest["target_count"] = len(
            read_file_sd_target_map(
                self._active_file(stem)
                if state == ExperimentTargets.STATE_ACTIVE
                else self._archived_file(stem)
            )
        )
        self._write_manifest(stem, manifest)
        return manifest

    def _expected_labels(
        self, stem: str, project: str, experiment_name: str
    ) -> dict[str, Any]:
        target_map = read_file_sd_target_map(self._active_file(stem))
        return {
            "project": project,
            "experiment_name": experiment_name,
            "addresses": {address for _job, address in target_map},
        }

    def _lifecycle_result(
        self,
        manifest: dict[str, Any],
        converged: bool | None,
        message: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "project": manifest.get("project"),
            "experiment_name": manifest.get("experiment_name"),
            "state": manifest.get("state"),
            "target_count": manifest.get("target_count"),
            "prometheus_converged": converged,
        }
        if message:
            result["prometheus_message"] = message
        return result

    def _prometheus_url(self, path: str) -> str:
        return (
            "http://"
            + format_host_port(local_addresses()["loopback"], self.prometheus_port)
            + path
        )

    def _wait_for_discovery_state(
        self, expected: dict[str, Any], *, present: bool
    ) -> tuple[bool | None, str]:
        """Reload Prometheus and wait until the experiment's targets appear/vanish.

        Best-effort confirmation of one file_sd refresh cycle: returns
        ``False`` when Prometheus is unreachable or the deadline passes, while
        the persisted state already reflects the requested lifecycle change.
        """
        if (
            self.convergence_timeout_seconds is not None
            and self.convergence_timeout_seconds <= 0
        ):
            return None, "convergence polling disabled"
        deadline = time.monotonic() + (self.convergence_timeout_seconds or 0) + 0.001
        if not self._reload_prometheus():
            return (
                False,
                "Prometheus reload failed; file_sd will converge on its refresh interval",
            )
        while time.monotonic() < deadline:
            labels = self._active_target_labels()
            if labels is not None:
                matching = {
                    str(item.get("discoveredLabels", {}).get("__address__") or "")
                    for item in labels
                    if item.get("labels", {}).get("project") == expected["project"]
                    and item.get("labels", {}).get("experiment_name")
                    == expected["experiment_name"]
                }
                if not present and not matching:
                    return True, ""
                if (
                    present
                    and expected.get("addresses")
                    and expected["addresses"] <= matching
                ):
                    return True, ""
                if present and not expected.get("addresses"):
                    return True, ""
            time.sleep(0.2)
        return False, "Prometheus file_sd did not converge before the deadline"

    def _reload_prometheus(self) -> bool:
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(self._prometheus_url("/-/reload"), timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.warning("[rl-insight] Failed to reload Prometheus: %s", exc)
            return False

    def _active_target_labels(self) -> list[dict[str, Any]] | None:
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(
                    self._prometheus_url("/api/v1/targets?state=active"), timeout=5
                )
            response.raise_for_status()
            data = response.json().get("data") or {}
            targets = data.get("activeTargets") or []
            return targets if isinstance(targets, list) else None
        except (requests.RequestException, ValueError) as exc:
            logger.warning("[rl-insight] Failed to query Prometheus targets: %s", exc)
            return None
