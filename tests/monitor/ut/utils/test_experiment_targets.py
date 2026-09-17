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

"""Unit tests for experiment-scoped Prometheus target partitions."""

from __future__ import annotations

import threading
from typing import Any

import pytest
import yaml

from rl_insight.utils.experiment_targets import (
    ExperimentNotFoundError,
    ExperimentTargetStore,
    experiment_dir_key,
    normalize_experiment_identity,
)
from rl_insight.utils.prometheus_utils import PrometheusTarget


@pytest.fixture()
def store(tmp_path: Any) -> ExperimentTargetStore:
    return ExperimentTargetStore(
        data_dir=tmp_path / "data", prometheus_port=9090, convergence_timeout_seconds=0
    )


def _target(address: str, **labels: Any) -> PrometheusTarget:
    return PrometheusTarget(address, dict(labels))


def test_normalize_identity_should_return_stripped_pair_when_both_names_are_valid() -> (
    None
):
    assert normalize_experiment_identity(" project-a ", " exp-1 ") == (
        "project-a",
        "exp-1",
    )


def test_normalize_identity_should_preserve_case_and_unicode_exactly() -> None:
    assert normalize_experiment_identity("Project-A", "Exp—1") == ("Project-A", "Exp—1")


def test_normalize_identity_should_return_none_when_both_names_are_absent() -> None:
    assert normalize_experiment_identity(None, None) is None
    assert normalize_experiment_identity("", "  ") is None


def test_normalize_identity_should_reject_when_exactly_one_name_is_given() -> None:
    with pytest.raises(Exception, match="both"):
        normalize_experiment_identity("project-a", None)
    with pytest.raises(Exception, match="both"):
        normalize_experiment_identity(None, "exp-1")


def test_dir_key_should_be_reversible_free_digest_that_keeps_names_out_of_paths() -> (
    None
):
    key = experiment_dir_key("../../etc/passwd")
    assert key == experiment_dir_key("../../etc/passwd")
    assert "/" not in key and ".." not in key
    assert key != experiment_dir_key("..//etc/passwd")


def test_register_should_create_partition_with_manifest_and_active_file_when_first_used(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    result = store.register(
        "project-a",
        "exp-1",
        "trainer_metrics",
        [_target("10.0.0.1:9092", role="trainer")],
    )

    assert result["state"] == "active"
    manifest = yaml.safe_load(
        store.manifest_file("project-a", "exp-1").read_text(encoding="utf-8")
    )
    assert store.manifest_file("project-a", "exp-1").parent.name == "projects"
    assert manifest["project"] == "project-a"
    assert manifest["experiment_name"] == "exp-1"
    assert manifest["state"] == "active"
    assert manifest["schema_version"] == 1
    assert manifest["target_count"] == 1

    saved = yaml.safe_load(
        store.active_file("project-a", "exp-1").read_text(encoding="utf-8")
    )
    assert saved == [
        {
            "targets": ["10.0.0.1:9092"],
            "labels": {
                "role": "trainer",
                "project": "project-a",
                "experiment_name": "exp-1",
                "rl_insight_job": "trainer_metrics",
            },
        }
    ]


def test_register_should_not_let_client_labels_override_reserved_identity_labels(
    store: ExperimentTargetStore,
) -> None:
    with pytest.raises(Exception, match="reserved"):
        store.register(
            "project-a",
            "exp-1",
            "trainer_metrics",
            [_target("10.0.0.1:9092", project="other")],
        )


def test_register_should_isolate_same_experiment_name_across_projects(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000")])
    store.register("project-b", "exp-1", "job", [_target("b:9000")])
    store.archive("project-a", "exp-1")

    assert store.show_targets("project-a", "exp-1")["state"] == "archived"
    project_b = store.show_targets("project-b", "exp-1")
    assert project_b["state"] == "active"
    assert [t["target"] for t in project_b["targets"]] == ["b:9000"]


def test_register_should_allow_same_address_in_different_experiments(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("10.0.0.1:9092")])
    store.register("project-a", "exp-2", "job", [_target("10.0.0.1:9092")])

    assert store.show_targets("project-a", "exp-1")["target_count"] == 1
    assert store.show_targets("project-a", "exp-2")["target_count"] == 1
    assert (
        store.list_experiments("project-a")
        and len(store.list_experiments("project-a")) == 2
    )


def test_register_should_reject_untrusted_names_that_escape_data_dir(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    from rl_insight.utils.experiment_targets import ExperimentIdentityError

    with pytest.raises(ExperimentIdentityError):
        store.register("project-a", "  \n\t", "job", [_target("t:9000")])
    names = ("../../evil", "a/b", "c\\d", "..", ".", "x" * 600)
    projects_root = (tmp_path / "data" / "projects").resolve()
    for name in names:
        store.register("project-a", name, "job", [_target("t:9000")])
        stem = store.experiment_stem("project-a", name)
        assert stem == experiment_dir_key("project-a") + "__" + experiment_dir_key(name)
        assert store.active_file("project-a", name).is_file()
        assert projects_root == store.active_file("project-a", name).resolve().parent
    assert not (tmp_path / "data" / "a").exists()
    assert list((tmp_path / "data").glob("**/evil")) == []
    assert len(store.list_experiments("project-a")) == 6


def test_archive_should_move_active_file_out_of_discovery_and_keep_snapshot(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000", rank="0")])

    result = store.archive("project-a", "exp-1")

    assert result["state"] == "archived"
    assert result["prometheus_converged"] is None
    assert not store.active_file("project-a", "exp-1").exists()
    archived = yaml.safe_load(
        store.archived_file("project-a", "exp-1").read_text(encoding="utf-8")
    )
    assert archived[0]["targets"] == ["a:9000"]
    manifest = yaml.safe_load(
        store.manifest_file("project-a", "exp-1").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "archived"
    assert manifest["target_count"] == 1


def test_archive_should_be_idempotent_and_keep_files_unchanged(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000")])
    store.archive("project-a", "exp-1")
    archived_file = store.archived_file("project-a", "exp-1")
    snapshot = archived_file.read_bytes()

    second = store.archive("project-a", "exp-1")

    assert second["state"] == "archived"
    assert archived_file.read_bytes() == snapshot


def test_register_should_fail_after_archive_without_recreating_discovery(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000")])
    store.archive("project-a", "exp-1")

    with pytest.raises(Exception, match="archived"):
        store.register("project-a", "exp-1", "job", [_target("a:9001")])

    assert not store.active_file("project-a", "exp-1").exists()


def test_archive_should_not_affect_other_experiments(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000")])
    store.register("project-a", "exp-2", "job", [_target("a:9001")])

    store.archive("project-a", "exp-1")

    assert store.show_targets("project-a", "exp-2")["state"] == "active"
    assert store.show_targets("project-a", "exp-2")["target_count"] == 1


def test_archive_should_report_unknown_experiment(store: ExperimentTargetStore) -> None:
    with pytest.raises(ExperimentNotFoundError):
        store.archive("project-a", "missing")


def test_restore_should_reactivate_saved_snapshot_and_be_idempotent(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000", rank="0")])
    store.archive("project-a", "exp-1")

    result = store.restore("project-a", "exp-1")
    assert result["state"] == "active"

    shown = store.show_targets("project-a", "exp-1")
    assert shown["target_count"] == 1
    assert shown["targets"][0]["labels"]["rank"] == "0"

    again = store.restore("project-a", "exp-1")
    assert again["state"] == "active"
    assert (
        shown["target_count"]
        == store.show_targets("project-a", "exp-1")["target_count"]
    )


def test_restore_should_report_unknown_experiment(store: ExperimentTargetStore) -> None:
    with pytest.raises(ExperimentNotFoundError):
        store.restore("project-a", "missing")


def test_list_should_return_composite_keys_states_and_counts(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000")])
    store.register("project-a", "exp-2", "job", [_target("a:9001"), _target("a:9002")])
    store.register("project-b", "exp-1", "job", [_target("b:9000")])
    store.archive("project-b", "exp-1")

    rows = store.list_experiments()
    by_key = {(row["project"], row["experiment_name"]): row for row in rows}
    assert set(by_key) == {
        ("project-a", "exp-1"),
        ("project-a", "exp-2"),
        ("project-b", "exp-1"),
    }
    assert by_key[("project-b", "exp-1")]["state"] == "archived"
    assert by_key[("project-a", "exp-2")]["target_count"] == 2
    assert by_key[("project-a", "exp-1")]["updated_at"]

    assert [row["experiment_name"] for row in store.list_experiments("project-a")] == [
        "exp-1",
        "exp-2",
    ]


def test_show_should_report_unknown_experiment(store: ExperimentTargetStore) -> None:
    with pytest.raises(ExperimentNotFoundError):
        store.show_targets("project-a", "missing")


def test_recovery_should_trust_discovery_file_location_over_stale_manifest(
    store: ExperimentTargetStore, tmp_path: Any
) -> None:
    store.register("project-a", "exp-1", "job", [_target("a:9000")])
    manifest_file = store.manifest_file("project-a", "exp-1")
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    manifest["state"] = "active"  # crash between archive rename and manifest write
    manifest_file.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    store.active_file("project-a", "exp-1").rename(
        store.archived_file("project-a", "exp-1")
    )

    assert store.show_targets("project-a", "exp-1")["state"] == "archived"
    with pytest.raises(Exception, match="archived"):
        store.register("project-a", "exp-1", "job", [_target("a:9001")])


def test_concurrent_register_should_serialize_same_experiment_and_keep_other_experiments(
    store: ExperimentTargetStore,
) -> None:
    errors: list[Exception] = []

    def register_same(index: int) -> None:
        try:
            for round_index in range(5):
                store.register(
                    "project-a",
                    "exp-1",
                    "job",
                    [_target(f"same-{index}:9000", attempt=str(round_index))],
                )
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    def register_other() -> None:
        try:
            for index in range(5):
                store.register(
                    "project-b", "exp-1", "job", [_target(f"other-{index}:9000")]
                )
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=register_same, args=(index,)) for index in range(4)
    ]
    threads.append(threading.Thread(target=register_other))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    shown = store.show_targets("project-a", "exp-1")
    assert shown["target_count"] == 4
    assert {t["target"] for t in shown["targets"]} == {
        f"same-{i}:9000" for i in range(4)
    }
    assert store.show_targets("project-b", "exp-1")["target_count"] == 5
    active = yaml.safe_load(
        store.active_file("project-a", "exp-1").read_text(encoding="utf-8")
    )
    assert isinstance(active, list) and len(active) == 4


def test_reads_should_not_rewrite_an_unchanged_manifest(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("10.0.0.1:9092")])
    manifest = store.manifest_file("project-a", "exp-1")
    inode_before = manifest.stat().st_ino

    store.list_experiments("project-a")
    store.show_targets("project-a", "exp-1")

    assert manifest.stat().st_ino == inode_before


def test_reads_should_keep_archived_state_when_all_discovery_files_vanish(
    store: ExperimentTargetStore,
) -> None:
    store.register("project-a", "exp-1", "job", [_target("10.0.0.1:9092")])
    store.archive("project-a", "exp-1")
    store.archived_file("project-a", "exp-1").unlink()

    assert store.list_experiments("project-a")[0]["state"] == "archived"
    assert store.show_targets("project-a", "exp-1")["state"] == "archived"
