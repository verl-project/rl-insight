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

"""Unit tests for Grafana dashboard preparation at server startup.

The runtime evaluates the Jsonnet sources shipped inside the installed package
in process, so no test here runs a generate/Jsonnet CLI. The composition
registry ships empty in this change, which is exactly what the empty-registry
and static-compatibility tests below exercise; the production compositions and
their business tests live in the change that depends on this one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module
from rl_insight.utils.constants import MonitorPaths

BUNDLED_DASHBOARDS_DIR = MonitorPaths.GRAFANA_DASHBOARDS_DIR
JSONNET_DIR = MonitorPaths.GRAFANA_JSONNET_DIR
JSONNET_OUTPUT_DIR = MonitorPaths.GRAFANA_JSONNET_OUTPUT_DIR

#: Source files the runtime must never write to.
MONITORED_SOURCE_FILES = (
    MonitorPaths.GRAFANA_JSONNET_ENTRYPOINT,
    JSONNET_DIR / "dashboard_compositions.libsonnet",
    JSONNET_OUTPUT_DIR / "verl_tainer_v1_with_vllm_engine.json",
    JSONNET_OUTPUT_DIR / "verl_tainer_v1_with_sglang_engine.json",
    MonitorPaths.CONFIG_FILE,
)


def _conf(builtin: Path | None = None, extra: Path | None = None, dashboard_config=""):
    grafana = {
        "dashboards_dir": str(
            builtin if builtin is not None else BUNDLED_DASHBOARDS_DIR
        )
    }
    if dashboard_config:
        grafana["dashboard_config"] = dashboard_config
    if extra is not None:
        grafana["extra_dashboard_dir"] = str(extra)
    return OmegaConf.create({"grafana": grafana})


def _renderer():
    """Import the packaged renderer, skipping when the framework is absent."""
    return pytest.importorskip("rl_insight.grafana.renderer")


def _write_custom_config(directory: Path, name: str = "custom_board") -> Path:
    """Write a user composition config next to a copy of the framework assets.

    The assets are copied instead of addressed by a relative path because the
    checkout and the temporary directory can be on different Windows drives,
    where ``os.path.relpath`` raises.
    """
    for asset in ("composer.libsonnet", "viz.libsonnet"):
        shutil.copy(JSONNET_DIR / "framework" / asset, directory / asset)
    config = directory / "custom.jsonnet"
    config.write_text(
        "local composer = import 'composer.libsonnet';\n"
        f"{{ {name}: composer.compose([{{\n"
        "  panels: [{ key: 'toy.panel', outputKey: 'toy-panel', id: 1,\n"
        "             title: 'Toy panel', queries: [{ expr: 'toy_metric' }] }],\n"
        "}], {\n"
        f"  metadata: {{ name: '{name}', labels: {{}}, annotations: {{}} }},\n"
        f"  title: '{name}', tags: ['RL-Insight'], spec: {{}},\n"
        "  variableOrder: [], rowOrder: [],\n"
        "}) }\n",
        encoding="utf-8",
    )
    return config


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_stages_static_dashboards_with_an_empty_composition_set(
    tmp_path,
) -> None:
    _renderer()
    # An explicitly empty composition set is valid: nothing is rendered and the
    # bundled static dashboards are still staged, byte for byte.
    config = tmp_path / "empty.jsonnet"
    config.write_text("{}\n", encoding="utf-8")

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(dashboard_config=str(config)), tmp_path / "runtime"
    )

    assert list(staged.glob("*/*_jsonnet.json")) == []
    for name in (
        "verl_tainer_v1_with_vllm_engine",
        "verl_tainer_v1_with_sglang_engine",
    ):
        staged_file = staged / "verl" / f"{name}.json"
        staged_source = JSONNET_OUTPUT_DIR / f"{name}.json"
        assert staged_file.read_bytes() == staged_source.read_bytes()
    assert (staged / "quick_start_demo" / "quick_start_demo.json").is_file()
    assert (staged / "agent_loop_trajectory" / "agent_loop_trajectory.json").is_file()
    assert (
        staged / "verl-omni" / "verl_omni_trainer_v1_with_vllm_omni_engine.json"
    ).is_file()


def test_prepare_writes_only_into_the_runtime_directory(tmp_path) -> None:
    _renderer()
    before = {path: path.read_bytes() for path in MONITORED_SOURCE_FILES}

    runtime_module._prepare_grafana_dashboards(_conf(), tmp_path / "runtime")

    assert {path: path.read_bytes() for path in MONITORED_SOURCE_FILES} == before


def test_prepare_renders_explicit_dashboard_config(tmp_path) -> None:
    _renderer()
    config = _write_custom_config(tmp_path)

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(dashboard_config=str(config)), tmp_path / "runtime"
    )

    written = _read_json(
        staged / MonitorPaths.GRAFANA_JSONNET_OUTPUT_SUBDIR / "custom_board.json"
    )
    assert written["spec"]["title"] == "custom_board"
    assert written["metadata"]["name"] == "custom_board"
    # The bundled static dashboards are staged as well: the custom composition
    # replaces the bundled entrypoint, not the other bundled dashboards.
    assert (staged / "verl" / "verl_tainer_v1_with_vllm_engine.json").is_file()
    assert (staged / "quick_start_demo" / "quick_start_demo.json").is_file()


@pytest.mark.parametrize(
    ("config_name", "body", "expected_error"),
    [
        ("nope.jsonnet", None, "does not exist"),
        ("broken.jsonnet", "{ this is not jsonnet", "generation failed"),
    ],
)
def test_prepare_rejects_bad_dashboard_config(
    tmp_path, config_name: str, body: str | None, expected_error: str
) -> None:
    _renderer()
    config = tmp_path / config_name
    if body is not None:
        config.write_text(body, encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected_error) as error:
        runtime_module._prepare_grafana_dashboards(
            _conf(dashboard_config=str(config)), tmp_path / "runtime"
        )

    assert str(config) in str(error.value)


def test_prepare_rejects_a_custom_config_that_hits_a_static_dashboard(tmp_path) -> None:
    _renderer()
    static = JSONNET_OUTPUT_DIR / "verl_tainer_v1_with_vllm_engine.json"
    before = static.read_bytes()
    config = _write_custom_config(tmp_path, name="verl_tainer_v1_with_vllm_engine")
    runtime_dir = tmp_path / "runtime"

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runtime_module._prepare_grafana_dashboards(
            _conf(dashboard_config=str(config)), runtime_dir
        )

    # Nothing was overwritten: neither the packaged source nor the staged copy.
    staged = runtime_dir / "dashboards" / "verl" / static.name
    assert static.read_bytes() == before
    assert staged.read_bytes() == before


def test_stage_copies_builtin_and_extra_without_parsing_json(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "verl").mkdir(parents=True)
    (extra / "custom").mkdir(parents=True)
    (builtin / "verl" / "board.json").write_text("{}", encoding="utf-8")
    (extra / "custom" / "router.json").write_text("not valid json", encoding="utf-8")

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(builtin, extra), tmp_path / "runtime"
    )

    assert (staged / "verl" / "board.json").is_file()
    assert (staged / "custom" / "router.json").read_text(encoding="utf-8") == (
        "not valid json"
    )


def test_stage_rejects_json_collision_before_copying_extra(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "shared").mkdir(parents=True)
    (extra / "shared").mkdir(parents=True)
    (builtin / "shared" / "board.json").write_text("builtin", encoding="utf-8")
    (extra / "unique.json").write_text("extra", encoding="utf-8")
    (extra / "shared" / "board.json").write_text("conflict", encoding="utf-8")

    runtime_dir = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="already exists in runtime dashboards"):
        runtime_module._prepare_grafana_dashboards(_conf(builtin, extra), runtime_dir)

    staged = runtime_dir / "dashboards"
    assert (staged / "shared" / "board.json").read_text(encoding="utf-8") == "builtin"
    assert not (staged / "unique.json").exists()


def test_stage_refreshes_extra_in_shared_builtin_directory(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "shared").mkdir(parents=True)
    (extra / "shared").mkdir(parents=True)
    (builtin / "shared" / "builtin.json").write_text("builtin", encoding="utf-8")
    extra_dashboard = extra / "shared" / "extra.json"
    extra_dashboard.write_text("first", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"

    runtime_module._prepare_grafana_dashboards(_conf(builtin, extra), runtime_dir)
    extra_dashboard.write_text("second", encoding="utf-8")
    staged = runtime_module._prepare_grafana_dashboards(
        _conf(builtin, extra), runtime_dir
    )

    assert (staged / "shared" / "extra.json").read_text(encoding="utf-8") == "second"


def test_stage_drops_stale_files_between_starts(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    (builtin / "verl").mkdir(parents=True)
    board = builtin / "verl" / "board.json"
    board.write_text("{}", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"

    runtime_module._prepare_grafana_dashboards(_conf(builtin), runtime_dir)
    board.unlink()
    staged = runtime_module._prepare_grafana_dashboards(_conf(builtin), runtime_dir)

    assert not (staged / "verl" / "board.json").exists()


def test_prepare_rejects_missing_legacy_dashboards_dir(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        runtime_module._prepare_grafana_dashboards(
            _conf(tmp_path / "absent"), tmp_path / "runtime"
        )


@pytest.mark.parametrize(
    ("entry_type", "expected_error"),
    [("missing", "does not exist"), ("file", "is not a directory")],
)
def test_stage_rejects_invalid_extra_directory(
    tmp_path, entry_type: str, expected_error: str
) -> None:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    extra = tmp_path / entry_type
    if entry_type == "file":
        extra.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected_error):
        runtime_module._prepare_grafana_dashboards(
            _conf(builtin, extra), tmp_path / "runtime"
        )


def test_prepare_generated_and_extra_dashboards_coexist(tmp_path) -> None:
    _renderer()
    extra = tmp_path / "extra"
    (extra / "custom").mkdir(parents=True)
    (extra / "custom" / "router.json").write_text("{}", encoding="utf-8")
    config = _write_custom_config(tmp_path)

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(extra=extra, dashboard_config=str(config)), tmp_path / "runtime"
    )

    assert (staged / "custom" / "router.json").is_file()
    assert (staged / "verl" / "custom_board.json").is_file()


@pytest.mark.parametrize("name", ["verl_tainer_v1_with_vllm_engine", "custom_board"])
def test_prepare_keeps_collision_detection_against_staged_dashboards(
    tmp_path, name: str
) -> None:
    _renderer()
    extra = tmp_path / "extra"
    (extra / "verl").mkdir(parents=True)
    (extra / "verl" / f"{name}.json").write_text("{}", encoding="utf-8")
    # The first name collides with a bundled static dashboard, the second with
    # a dashboard rendered by the custom config; both are detected the same way.
    conf = _conf(extra=extra)
    if name == "custom_board":
        conf = _conf(extra=extra, dashboard_config=str(_write_custom_config(tmp_path)))

    with pytest.raises(RuntimeError, match="already exists in runtime dashboards"):
        runtime_module._prepare_grafana_dashboards(conf, tmp_path / "runtime")


# --------------------------------------------------------------------------
# Production VERL compositions
# --------------------------------------------------------------------------

STATIC_NAMES = (
    "verl_tainer_v1_with_vllm_engine",
    "verl_tainer_v1_with_sglang_engine",
)
COMPOSITION_NAMES = (
    "verl_tainer_v1_with_vllm_engine_jsonnet",
    "verl_tainer_v1_with_sglang_engine_jsonnet",
)
PRODUCTION_SOURCE_FILES = (
    MonitorPaths.GRAFANA_JSONNET_ENTRYPOINT,
    JSONNET_DIR / "dashboard_compositions.libsonnet",
    JSONNET_DIR / "compositions" / "verl_vllm.libsonnet",
    JSONNET_DIR / "compositions" / "verl_sglang.libsonnet",
    JSONNET_DIR / "dashboards" / "trainer.libsonnet",
    JSONNET_DIR / "dashboards" / "vllm.libsonnet",
    MonitorPaths.CONFIG_FILE,
)


def _normalized(dashboard: dict) -> dict:
    """Strip the identity the two dashboards intentionally do not share."""
    normalized = json.loads(json.dumps(dashboard))
    normalized["metadata"]["name"] = "identity"
    normalized["spec"]["title"] = "identity"
    return normalized


def _staged(tmp_path) -> Path:
    return runtime_module._prepare_grafana_dashboards(_conf(), tmp_path / "runtime")


def test_registry_defines_the_two_verl_compositions() -> None:
    renderer = _renderer()

    rendered = renderer.render_dashboards(MonitorPaths.GRAFANA_JSONNET_ENTRYPOINT)

    assert tuple(sorted(rendered)) == tuple(sorted(COMPOSITION_NAMES))


def test_prepare_materializes_four_verl_dashboards_without_any_cli(tmp_path) -> None:
    _renderer()

    folder = _staged(tmp_path) / MonitorPaths.GRAFANA_JSONNET_OUTPUT_SUBDIR

    assert sorted(path.name for path in folder.glob("*.json")) == sorted(
        f"{name}.json" for name in STATIC_NAMES + COMPOSITION_NAMES
    )


def test_jsonnet_dashboards_keep_their_own_identity(tmp_path) -> None:
    _renderer()
    staged = _staged(tmp_path)

    for composition, static in zip(COMPOSITION_NAMES, STATIC_NAMES):
        static_dashboard = _read_json(JSONNET_OUTPUT_DIR / f"{static}.json")
        jsonnet_dashboard = _read_json(staged / "verl" / f"{composition}.json")
        assert (
            jsonnet_dashboard["metadata"]["name"]
            != static_dashboard["metadata"]["name"]
        )
        assert (
            jsonnet_dashboard["spec"]["title"]
            == f"{static_dashboard['spec']['title']}_jsonnet"
        )


def test_jsonnet_and_static_dashboards_are_semantically_equivalent(tmp_path) -> None:
    _renderer()
    staged = _staged(tmp_path)

    for composition, static in zip(COMPOSITION_NAMES, STATIC_NAMES):
        jsonnet_dashboard = _read_json(staged / "verl" / f"{composition}.json")
        static_dashboard = _read_json(JSONNET_OUTPUT_DIR / f"{static}.json")
        assert _normalized(jsonnet_dashboard) == _normalized(static_dashboard)


def test_sglang_composition_never_includes_the_npu_content(tmp_path) -> None:
    _renderer()
    staged = _staged(tmp_path)

    vllm = _read_json(staged / "verl" / f"{COMPOSITION_NAMES[0]}.json")
    sglang = _read_json(staged / "verl" / f"{COMPOSITION_NAMES[1]}.json")
    variables = lambda dashboard: [  # noqa: E731
        variable["spec"]["name"] for variable in dashboard["spec"]["variables"]
    ]
    rows = lambda dashboard: [  # noqa: E731
        row["spec"]["title"] for row in dashboard["spec"]["layout"]["spec"]["rows"]
    ]

    assert "npu_instance" in variables(vllm)
    assert "npu_instance" not in variables(sglang)
    assert "hardware metric" in rows(vllm)
    assert "hardware metric" not in rows(sglang)


def test_prepare_never_writes_to_production_sources(tmp_path) -> None:
    _renderer()
    before = {path: path.read_bytes() for path in PRODUCTION_SOURCE_FILES}

    _staged(tmp_path)

    assert {path: path.read_bytes() for path in PRODUCTION_SOURCE_FILES} == before


def test_legacy_dashboards_dir_skips_the_production_compositions(tmp_path) -> None:
    _renderer()
    legacy = tmp_path / "legacy"
    (legacy / "board").mkdir(parents=True)
    (legacy / "board" / "board.json").write_text("{}", encoding="utf-8")

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(legacy), tmp_path / "runtime"
    )

    assert (staged / "board" / "board.json").is_file()
    assert not (staged / "verl").exists()


def test_custom_config_can_reuse_a_production_module(tmp_path) -> None:
    _renderer()
    # Copied for the same cross-drive reason as _write_custom_config.
    for asset in ("composer.libsonnet", "viz.libsonnet"):
        shutil.copy(JSONNET_DIR / "framework" / asset, tmp_path / asset)
    shutil.copy(JSONNET_DIR / "dashboards" / "npu.libsonnet", tmp_path)
    config = tmp_path / "custom.jsonnet"
    config.write_text(
        "local composer = import 'composer.libsonnet';\n"
        "local npu = import 'npu.libsonnet';\n"
        "{ composed: composer.compose([npu], {\n"
        "  metadata: { name: 'npu-only', labels: {}, annotations: {} },\n"
        "  title: 'npu_only', tags: ['RL-Insight'], spec: {},\n"
        "  variableOrder: ['npu_instance'], rowOrder: [],\n"
        "}) }\n",
        encoding="utf-8",
    )

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(dashboard_config=str(config)), tmp_path / "runtime"
    )

    assert _read_json(staged / "verl" / "composed.json")["spec"]["title"] == "npu_only"
    assert (staged / "verl" / f"{STATIC_NAMES[0]}.json").is_file()


def test_extra_collision_against_a_generated_jsonnet_dashboard(tmp_path) -> None:
    _renderer()
    extra = tmp_path / "extra"
    (extra / "verl").mkdir(parents=True)
    (extra / "verl" / f"{COMPOSITION_NAMES[0]}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists in runtime dashboards"):
        runtime_module._prepare_grafana_dashboards(
            _conf(extra=extra), tmp_path / "runtime"
        )


def test_generated_and_extra_dashboards_coexist(tmp_path) -> None:
    _renderer()
    extra = tmp_path / "extra"
    (extra / "custom").mkdir(parents=True)
    (extra / "custom" / "router.json").write_text("{}", encoding="utf-8")

    staged = runtime_module._prepare_grafana_dashboards(
        _conf(extra=extra), tmp_path / "runtime"
    )

    assert (staged / "custom" / "router.json").is_file()
    assert (staged / "verl" / f"{COMPOSITION_NAMES[0]}.json").is_file()
