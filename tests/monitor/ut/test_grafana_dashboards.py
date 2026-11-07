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

"""Unit tests for Grafana dashboard directory staging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module
from rl_insight.utils.constants import MonitorPaths

_BUNDLED_DASHBOARDS = Path(MonitorPaths.GRAFANA_DASHBOARDS_DIR)


def _conf(builtin: Path, extra: Path | None = None):
    grafana = {"dashboards_dir": str(builtin)}
    if extra is not None:
        grafana["extra_dashboard_dir"] = str(extra)
    return OmegaConf.create({"grafana": grafana})


def test_stage_copies_builtin_and_extra_without_parsing_json(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "verl").mkdir(parents=True)
    (extra / "custom").mkdir(parents=True)
    (builtin / "verl" / "board.json").write_text("{}", encoding="utf-8")
    (extra / "custom" / "router.json").write_text("not valid json", encoding="utf-8")

    staged = runtime_module._stage_grafana_dashboards(
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
        runtime_module._stage_grafana_dashboards(
            _conf(builtin, extra),
            runtime_dir,
        )

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

    runtime_module._stage_grafana_dashboards(_conf(builtin, extra), runtime_dir)
    extra_dashboard.write_text("second", encoding="utf-8")
    staged = runtime_module._stage_grafana_dashboards(
        _conf(builtin, extra), runtime_dir
    )

    assert (staged / "shared" / "extra.json").read_text(encoding="utf-8") == "second"


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
        runtime_module._stage_grafana_dashboards(
            _conf(builtin, extra), tmp_path / "runtime"
        )


def _bundled_dashboard_files() -> list[Path]:
    return sorted(_BUNDLED_DASHBOARDS.rglob("*.json"))


def _iter_element_references(node: object):
    """Yield every ``ElementReference`` name in a layout subtree."""
    if isinstance(node, dict):
        if node.get("kind") == "ElementReference":
            yield node.get("name")
        for value in node.values():
            yield from _iter_element_references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_element_references(value)


@pytest.mark.parametrize(
    "dashboard_path", _bundled_dashboard_files(), ids=lambda p: p.name
)
def test_bundled_dashboard_should_place_every_element_in_its_layout(
    dashboard_path: Path,
) -> None:
    """Every layout reference resolves, no element is orphaned, panel ids are unique."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    elements = dashboard["spec"]["elements"]

    references = list(_iter_element_references(dashboard["spec"]["layout"]))
    undeclared = [name for name in references if name not in elements]
    assert not undeclared, f"{dashboard_path.name} references undeclared elements"

    unplaced = set(elements) - set(references)
    assert not unplaced, f"{dashboard_path.name} declares unplaced elements"

    panel_ids = [
        element["spec"]["id"]
        for element in elements.values()
        if element.get("kind") == "Panel"
    ]
    assert len(panel_ids) == len(set(panel_ids)), (
        f"{dashboard_path.name} reuses a panel id"
    )


def test_sglang_dashboard_should_plot_prompt_and_generation_length() -> None:
    """The verl SGLang dashboard plots both SGLang token-length histograms."""
    dashboard_path = (
        _BUNDLED_DASHBOARDS / "verl" / "verl_tainer_v1_with_sglang_engine.json"
    )
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    elements = dashboard["spec"]["elements"]

    titles = {
        element["spec"]["title"]
        for element in elements.values()
        if element.get("kind") == "Panel"
    }
    assert "SGLang: Request Prompt Length" in titles
    assert "SGLang: Request Generation Length" in titles

    expressions = [
        query["spec"]["query"]["spec"].get("expr", "")
        for element in elements.values()
        if element.get("kind") == "Panel"
        for query in element["spec"]["data"]["spec"]["queries"]
    ]
    joined = "\n".join(expressions)
    assert "sglang[:_]prompt_tokens_histogram_bucket" in joined
    assert "sglang[:_]generation_tokens_histogram_bucket" in joined
