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

"""Unit tests for migrating the legacy/global target file into experiment partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rl_insight.utils.experiment_targets import ExperimentTargetStore
from rl_insight.utils.prometheus_utils import write_file_sd_target_map


@pytest.fixture()
def store(tmp_path: Any) -> ExperimentTargetStore:
    return ExperimentTargetStore(
        data_dir=tmp_path / "data", prometheus_port=9090, convergence_timeout_seconds=0
    )


@pytest.fixture()
def global_file(tmp_path: Any):
    file = tmp_path / "data" / "targets" / "prometheus-targets.yml"
    file.parent.mkdir(parents=True, exist_ok=True)
    return file


def _group(target: str, **labels: Any) -> dict[str, Any]:
    labels.setdefault("rl_insight_job", "trainer_metrics")
    return {"targets": [target], "labels": {str(k): str(v) for k, v in labels.items()}}


def test_migrate_should_move_fully_identified_records_into_partitions_and_rewrite_global(
    store: ExperimentTargetStore, global_file: Any
) -> None:
    write_file_sd_target_map(
        global_file,
        {
            ("trainer_metrics", "a:9000"): {
                "project": "project-a",
                "experiment_name": "exp-1",
            },
            ("trainer_metrics", "b:9000"): {
                "project": "project-b",
                "experiment_name": "exp-1",
            },
            ("node_exporter", "c:9100"): {},
        },
    )

    result = store.migrate_legacy_targets(global_file)

    assert result["migrated"] == 2
    assert result["kept_legacy"] == 1
    assert result["changed"] is True
    backup_path = result["backup_file"]
    assert backup_path and yaml.safe_load(Path(backup_path).read_text(encoding="utf-8"))

    project_a = store.show_targets("project-a", "exp-1")
    assert project_a["target_count"] == 1
    assert project_a["targets"][0]["labels"]["project"] == "project-a"
    project_b = store.show_targets("project-b", "exp-1")
    assert project_b["target_count"] == 1

    remaining = yaml.safe_load(global_file.read_text(encoding="utf-8"))
    assert remaining == [
        {"targets": ["c:9100"], "labels": {"rl_insight_job": "node_exporter"}}
    ]


def test_migrate_should_keep_partial_and_blank_labels_in_legacy_partition(
    store: ExperimentTargetStore, global_file: Any
) -> None:
    write_file_sd_target_map(
        global_file,
        {
            ("job", "a:9000"): {"project": "project-a"},
            ("job", "b:9000"): {"experiment_name": "exp-1"},
            ("job", "c:9000"): {"project": "  ", "experiment_name": "exp-1"},
            ("job", "d:9000"): {"project": "p", "experiment_name": " "},
        },
    )

    result = store.migrate_legacy_targets(global_file)

    assert result["migrated"] == 0
    assert result["kept_legacy"] == 4
    assert result["changed"] is False
    assert result["backup_file"] is None
    assert len(yaml.safe_load(global_file.read_text(encoding="utf-8"))) == 4


def test_migrate_should_deduplicate_repeated_records(
    store: ExperimentTargetStore, global_file: Any
) -> None:
    groups = [
        _group("a:9000", project="project-a", experiment_name="exp-1"),
        _group("a:9000", project="project-a", experiment_name="exp-1", extra="x"),
    ]
    global_file.write_text(yaml.safe_dump(groups), encoding="utf-8")

    result = store.migrate_legacy_targets(global_file)

    assert result["scanned"] == 1  # duplicate (job, target) collapses at read time
    assert result["migrated"] == 1
    assert store.show_targets("project-a", "exp-1")["target_count"] == 1


def test_migrate_should_be_idempotent_across_repeated_starts(
    store: ExperimentTargetStore, global_file: Any
) -> None:
    write_file_sd_target_map(
        global_file,
        {
            ("job", "a:9000"): {"project": "project-a", "experiment_name": "exp-1"},
            ("job", "b:9000"): {},
        },
    )
    first = store.migrate_legacy_targets(global_file)
    snapshot = global_file.read_bytes()

    second = store.migrate_legacy_targets(global_file)

    assert first["changed"] is True and second["changed"] is False
    assert second["migrated"] == 0
    assert second["backup_file"] is None
    assert global_file.read_bytes() == snapshot


def test_migrate_should_not_change_archived_experiments(
    store: ExperimentTargetStore, global_file: Any
) -> None:
    store.register("project-a", "exp-1", "job", [])
    store.archive("project-a", "exp-1")
    write_file_sd_target_map(
        global_file,
        {("job", "a:9000"): {"project": "project-a", "experiment_name": "exp-1"}},
    )

    result = store.migrate_legacy_targets(global_file)

    assert result["migrated"] == 0
    assert result["kept_legacy"] == 1
    assert store.show_targets("project-a", "exp-1")["state"] == "archived"
    assert store.show_targets("project-a", "exp-1")["target_count"] == 0


def test_migrate_should_leave_global_untouched_when_a_partition_write_fails(
    store: ExperimentTargetStore, global_file: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_file_sd_target_map(
        global_file,
        {("job", "a:9000"): {"project": "project-a", "experiment_name": "exp-1"}},
    )
    original = global_file.read_bytes()

    def explode(path: Any, target_map: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "rl_insight.utils.experiment_targets.write_file_sd_target_map", explode
    )

    with pytest.raises(OSError):
        store.migrate_legacy_targets(global_file)

    assert global_file.read_bytes() == original


def test_migrate_should_noop_when_global_file_is_missing(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    result = store.migrate_legacy_targets(tmp_path / "missing.yml")
    assert result == {
        "scanned": 0,
        "migrated": 0,
        "kept_legacy": 0,
        "backup_file": None,
        "changed": False,
    }
