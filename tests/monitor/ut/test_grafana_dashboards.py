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
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module
from rl_insight.utils.constants import MonitorDefaults, MonitorPaths

_BUNDLED_DASHBOARDS = Path(MonitorPaths.GRAFANA_DASHBOARDS_DIR)
# Integration-owned dashboards that ship in the source tree, one subdirectory per
# Grafana folder, meant to be passed to ``--extra-dashboard-dir``.
_RECIPE_DASHBOARDS = Path(__file__).resolve().parents[3] / "recipe" / "dashboards"

# ``$name`` and ``${name:format}`` template references inside PromQL expressions.
_TEMPLATE_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


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


def test_stage_allows_same_filename_in_different_subdirectories(tmp_path) -> None:
    """Collisions are decided on the relative path, not on the file name alone."""
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "verl").mkdir(parents=True)
    (extra / "slime").mkdir(parents=True)
    (builtin / "verl" / "board.json").write_text("builtin", encoding="utf-8")
    (extra / "slime" / "board.json").write_text("extra", encoding="utf-8")

    staged = runtime_module._stage_grafana_dashboards(
        _conf(builtin, extra), tmp_path / "runtime"
    )

    assert (staged / "verl" / "board.json").read_text(encoding="utf-8") == "builtin"
    assert (staged / "slime" / "board.json").read_text(encoding="utf-8") == "extra"


def test_stage_rejects_collision_nested_below_the_top_level(tmp_path) -> None:
    """The collision scan walks the whole tree, not just the first level."""
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "shared" / "nested").mkdir(parents=True)
    (extra / "shared" / "nested").mkdir(parents=True)
    (builtin / "shared" / "nested" / "board.json").write_text(
        "builtin", encoding="utf-8"
    )
    (extra / "shared" / "nested" / "board.json").write_text(
        "conflict", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match=r"'shared/nested/board\.json'"):
        runtime_module._stage_grafana_dashboards(
            _conf(builtin, extra), tmp_path / "runtime"
        )

    staged = tmp_path / "runtime" / "dashboards"
    assert (staged / "shared" / "nested" / "board.json").read_text(
        encoding="utf-8"
    ) == "builtin"


def test_stage_treats_extra_equal_to_target_as_a_no_op(tmp_path) -> None:
    """Pointing extra at the staged directory returns early instead of self-copying."""
    runtime_dir = tmp_path / "runtime"
    staged_dir = runtime_dir / "dashboards"
    staged_dir.mkdir(parents=True)
    (staged_dir / "board.json").write_text("staged", encoding="utf-8")

    staged = runtime_module._stage_grafana_dashboards(
        _conf(staged_dir, staged_dir), runtime_dir
    )

    assert staged == staged_dir.resolve()
    assert sorted(path.name for path in staged.iterdir()) == ["board.json"]
    assert (staged / "board.json").read_text(encoding="utf-8") == "staged"


def test_stage_does_not_wipe_target_when_it_is_the_builtin_source(tmp_path) -> None:
    """When the source already is the target the staged files survive untouched."""
    runtime_dir = tmp_path / "runtime"
    staged_dir = runtime_dir / "dashboards"
    staged_dir.mkdir(parents=True)
    (staged_dir / "keep.json").write_text("keep", encoding="utf-8")

    staged = runtime_module._stage_grafana_dashboards(_conf(staged_dir), runtime_dir)

    assert (staged / "keep.json").read_text(encoding="utf-8") == "keep"


def _bundled_dashboard_files() -> list[Path]:
    return sorted(_BUNDLED_DASHBOARDS.rglob("*.json"))


def _shipped_dashboard_files() -> list[Path]:
    """Every dashboard in the source tree, whether provisioned by default or not."""
    return sorted(_BUNDLED_DASHBOARDS.rglob("*.json")) + sorted(
        _RECIPE_DASHBOARDS.rglob("*.json")
    )


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
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_place_every_element_in_its_layout(
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


_RECIPE_SGLANG_DASHBOARD = _RECIPE_DASHBOARDS / "sglang" / "sglang_request_length.json"


def test_recipe_dashboard_should_plot_prompt_and_generation_length() -> None:
    """The recipe SGLang dashboard plots both SGLang token-length histograms."""
    dashboard_path = _RECIPE_SGLANG_DASHBOARD
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


def _dashboard_panels(dashboard: dict) -> list[dict]:
    return [
        element["spec"]
        for element in dashboard["spec"]["elements"].values()
        if element.get("kind") == "Panel"
    ]


def _layout_variables(layout: object) -> list[dict]:
    """Collect variables declared on rows or tabs in a layout subtree."""
    if not isinstance(layout, dict):
        return []
    spec = layout.get("spec")
    if not isinstance(spec, dict):
        return []
    variables = list(spec.get("variables") or [])
    kind = layout.get("kind")
    if kind == "RowsLayout":
        for row in spec.get("rows") or []:
            row_spec = row.get("spec") or {}
            variables.extend(row_spec.get("variables") or [])
            variables.extend(_layout_variables(row_spec.get("layout")))
    elif kind == "TabsLayout":
        for tab in spec.get("tabs") or []:
            variables.extend(_layout_variables((tab.get("spec") or {}).get("layout")))
    return variables


def _panel_expressions(panel: dict) -> list[str]:
    return [
        query["spec"]["query"]["spec"].get("expr")
        for query in panel["data"]["spec"]["queries"]
        if query.get("kind") == "PanelQuery"
    ]


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_declare_a_variable_for_every_template(
    dashboard_path: Path,
) -> None:
    """Every ``$name`` used in a PromQL expression resolves to a dashboard variable.

    ``$__rate_interval`` and friends are Grafana built-ins, so they are exempt.
    """
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    declared = {
        variable["spec"]["name"]
        for variable in dashboard["spec"].get("variables", [])
        + _layout_variables(dashboard["spec"].get("layout"))
    }
    used: set[str] = set()
    for panel in _dashboard_panels(dashboard):
        for expression in _panel_expressions(panel):
            used.update(_TEMPLATE_VAR.findall(expression or ""))

    undeclared = {
        name for name in used if name not in declared and not name.startswith("__")
    }
    assert not undeclared, (
        f"{dashboard_path.name} uses undeclared dashboard variables: "
        f"{sorted(undeclared)}"
    )


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_use_bucket_metrics_in_heatmaps(
    dashboard_path: Path,
) -> None:
    """A heatmap needs a le-labelled bucket series fed through increase()."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    heatmaps = [
        panel
        for panel in _dashboard_panels(dashboard)
        if panel.get("vizConfig", {}).get("group") == "heatmap"
    ]
    if not heatmaps:
        pytest.skip(f"{dashboard_path.name} has no heatmap panels")

    for panel in heatmaps:
        for expression in _panel_expressions(panel):
            assert expression, f"{panel['title']} has an empty query"
            assert "_bucket" in expression, (
                f"{panel['title']} does not select a _bucket series: {expression}"
            )
            assert "increase(" in expression, (
                f"{panel['title']} does not accumulate with increase(): {expression}"
            )
            assert "le" in expression, (
                f"{panel['title']} does not group or select on le: {expression}"
            )


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_label_heatmap_y_axis_in_metric_units(
    dashboard_path: Path,
) -> None:
    """A heatmap's y axis is the bucketed value, so its unit must match it.

    The length heatmaps were copied from the latency heatmap and kept
    ``"unit": "s"``, which renders token counts as seconds and then scales them
    into minutes, hours and days. Nothing in Grafana rejects that: the panel
    renders and is simply mislabelled.
    """
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    heatmaps = [
        panel
        for panel in _dashboard_panels(dashboard)
        if panel.get("vizConfig", {}).get("group") == "heatmap"
    ]
    if not heatmaps:
        pytest.skip(f"{dashboard_path.name} has no heatmap panels")

    for panel in heatmaps:
        expressions = " ".join(_panel_expressions(panel) or [])
        # A duration heatmap buckets a *_seconds metric; anything else buckets a
        # count (tokens, requests) and must not claim to be a time.
        unit = panel["vizConfig"]["spec"]["options"]["yAxis"]["unit"]
        if "_seconds" in expressions:
            assert unit == "s", (
                f"{panel['title']} buckets a seconds metric but labels the y axis "
                f"as {unit!r}"
            )
        else:
            assert unit == "none", (
                f"{panel['title']} buckets a non-time metric but labels the y axis "
                f"as {unit!r}"
            )


def test_recipe_dashboard_should_stage_beside_the_bundled_dashboards(
    tmp_path,
) -> None:
    """The recipe dashboard loads through ``--extra-dashboard-dir`` on its own.

    ``foldersFromFilesStructure`` turns the subdirectory into a Grafana folder, so
    the recipe dashboard sits in its own folder instead of joining a bundled one,
    and no relative path collides with a bundled dashboard.
    """
    staged = runtime_module._stage_grafana_dashboards(
        _conf(_BUNDLED_DASHBOARDS, _RECIPE_DASHBOARDS), tmp_path / "runtime"
    )

    recipe_dashboard = staged / "sglang" / "sglang_request_length.json"
    assert recipe_dashboard.is_file()
    assert json.loads(recipe_dashboard.read_text(encoding="utf-8")) == json.loads(
        _RECIPE_SGLANG_DASHBOARD.read_text(encoding="utf-8")
    )
    for bundled in _bundled_dashboard_files():
        relative = bundled.relative_to(_BUNDLED_DASHBOARDS)
        assert (staged / relative).is_file(), f"{relative} did not survive staging"


def _provisioning_conf(**sections) -> OmegaConf:
    base = {
        "prometheus": {"prometheus_port": 9090, "enable": True},
        "tempo": {"query_port": 3200, "retention_time": "3d", "enable": True},
    }
    base.update(sections)
    return OmegaConf.create(base)


def _render_provisioning(tmp_path: Path, **sections) -> tuple[Path, dict, dict]:
    runtime_dir = tmp_path / "runtime"
    dashboard_dir = runtime_dir / "dashboards"
    dashboard_dir.mkdir(parents=True)
    runtime_module._render_grafana_provisioning(
        _provisioning_conf(**sections), runtime_dir, dashboard_dir
    )
    provisioning = runtime_dir / "provisioning"
    datasources = yaml.safe_load(
        (provisioning / "datasources" / "default.yml").read_text(encoding="utf-8")
    )
    dashboards = yaml.safe_load(
        (provisioning / "dashboards" / "default.yml").read_text(encoding="utf-8")
    )
    return dashboard_dir, datasources, dashboards


def test_provisioning_should_point_each_datasource_at_its_own_port(tmp_path) -> None:
    """Datasource URLs carry the configured ports, so a moved port stays reachable."""
    _, datasources, _ = _render_provisioning(
        tmp_path,
        prometheus={"prometheus_port": 19090, "enable": True},
        tempo={"query_port": 13200, "retention_time": "3d", "enable": True},
    )

    entries = {entry["name"]: entry for entry in datasources["datasources"]}
    assert entries["Prometheus"]["url"].endswith(":19090")
    assert entries["Tempo"]["url"].endswith(":13200")
    assert entries["Prometheus"]["uid"] == "prometheus"
    assert entries["Tempo"]["uid"] == "tempo"


def test_provisioning_should_make_the_first_enabled_datasource_the_default(
    tmp_path,
) -> None:
    """Exactly one datasource is default, and disabling Prometheus promotes Tempo."""
    _, both, _ = _render_provisioning(tmp_path / "both")
    assert [entry["isDefault"] for entry in both["datasources"]] == [True, False]

    _, tempo_only, _ = _render_provisioning(
        tmp_path / "tempo_only", prometheus={"prometheus_port": 9090, "enable": False}
    )
    assert [entry["name"] for entry in tempo_only["datasources"]] == ["Tempo"]
    assert tempo_only["datasources"][0]["isDefault"] is True


def test_provisioning_should_emit_no_datasource_when_both_are_disabled(
    tmp_path,
) -> None:
    """Disabling both services writes an empty datasource list, not a broken entry."""
    _, datasources, _ = _render_provisioning(
        tmp_path,
        prometheus={"prometheus_port": 9090, "enable": False},
        tempo={"query_port": 3200, "retention_time": "3d", "enable": False},
    )

    assert datasources == {"apiVersion": 1, "datasources": []}


@pytest.mark.parametrize(
    ("retention_time", "expected_seconds"),
    [("3d", 259200), ("1h", 3600), ("45m", 2700)],
)
def test_provisioning_should_pass_tempo_retention_to_the_tag_time_range(
    tmp_path, retention_time: str, expected_seconds: int
) -> None:
    """Grafana clamps tag queries to this window, so it has to match retention."""
    _, datasources, _ = _render_provisioning(
        tmp_path,
        tempo={"query_port": 3200, "retention_time": retention_time, "enable": True},
    )

    tempo = next(e for e in datasources["datasources"] if e["name"] == "Tempo")
    assert tempo["jsonData"]["timeRangeForTags"] == expected_seconds


def test_provisioning_should_default_tempo_tag_range_to_thirty_days(tmp_path) -> None:
    """Without an explicit retention the tag window falls back to 30d."""
    _, datasources, _ = _render_provisioning(
        tmp_path, tempo={"query_port": 3200, "enable": True}
    )

    tempo = next(e for e in datasources["datasources"] if e["name"] == "Tempo")
    assert tempo["jsonData"]["timeRangeForTags"] == 30 * 24 * 3600


def test_provisioning_should_serve_dashboards_from_the_staged_directory(
    tmp_path, monkeypatch
) -> None:
    """The provider must point at the staged directory so folders map to subfolders.

    Grafana starts in its own working directory, so the path has to be absolute.
    """
    monkeypatch.chdir(tmp_path)
    runtime_dir = Path("runtime")
    dashboard_dir = runtime_dir / "dashboards"
    dashboard_dir.mkdir(parents=True)
    runtime_module._render_grafana_provisioning(
        _provisioning_conf(), runtime_dir, dashboard_dir
    )
    dashboards = yaml.safe_load(
        (runtime_dir / "provisioning" / "dashboards" / "default.yml").read_text(
            encoding="utf-8"
        )
    )

    provider = dashboards["providers"][0]
    assert provider["options"]["path"] == str(dashboard_dir.resolve())
    assert Path(provider["options"]["path"]).is_absolute()
    assert provider["options"]["foldersFromFilesStructure"] is True
    assert provider["type"] == "file"


def _iter_grid_layouts(node: object):
    """Yield every ``GridLayout`` spec in a layout subtree."""
    if isinstance(node, dict):
        if node.get("kind") == "GridLayout":
            yield node["spec"]
        for value in node.values():
            yield from _iter_grid_layouts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_grid_layouts(value)


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_fit_items_inside_the_twenty_four_column_grid(
    dashboard_path: Path,
) -> None:
    """Grafana's grid is 24 columns wide; an item past the edge is clipped."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    for grid in _iter_grid_layouts(dashboard["spec"]["layout"]):
        for item in grid["items"]:
            spec = item["spec"]
            right = spec["x"] + spec["width"]
            assert spec["x"] >= 0 and right <= 24, (
                f"{dashboard_path.name}: {item.get('name')} spans "
                f"x={spec['x']} w={spec['width']}"
            )
            assert spec["width"] > 0 and spec["height"] > 0, (
                f"{dashboard_path.name}: {item.get('name')} has empty geometry"
            )


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_not_overlap_grid_items(
    dashboard_path: Path,
) -> None:
    """Two items sharing a rectangle means one of them is invisible in Grafana."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    for grid in _iter_grid_layouts(dashboard["spec"]["layout"]):
        boxes = [
            (
                item.get("name"),
                item["spec"]["x"],
                item["spec"]["y"],
                item["spec"]["width"],
                item["spec"]["height"],
            )
            for item in grid["items"]
        ]
        for index, (name, x, y, width, height) in enumerate(boxes):
            for other, ox, oy, ow, oh in boxes[index + 1 :]:
                overlaps = (
                    x < ox + ow and ox < x + width and y < oy + oh and oy < y + height
                )
                assert not overlaps, f"{dashboard_path.name}: {name} overlaps {other}"


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_use_a_unique_ref_id_per_query(
    dashboard_path: Path,
) -> None:
    """Duplicate refIds inside one panel make Grafana drop or merge a query."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    for panel in _dashboard_panels(dashboard):
        ref_ids = [
            query["spec"].get("refId") for query in panel["data"]["spec"]["queries"]
        ]
        assert all(ref_ids), f"{panel['title']} has a query without a refId"
        assert len(ref_ids) == len(set(ref_ids)), (
            f"{panel['title']} repeats a refId: {ref_ids}"
        )


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_switch_heatmap_queries_to_heatmap_format(
    dashboard_path: Path,
) -> None:
    """A heatmap panel whose query is not in heatmap format renders as a line chart."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    heatmaps = [
        panel
        for panel in _dashboard_panels(dashboard)
        if panel.get("vizConfig", {}).get("group") == "heatmap"
    ]
    if not heatmaps:
        pytest.skip(f"{dashboard_path.name} has no heatmap panels")

    for panel in heatmaps:
        for query in panel["data"]["spec"]["queries"]:
            data_query = query["spec"]["query"]["spec"]
            assert data_query.get("format") == "heatmap", (
                f"{panel['title']} queries with format={data_query.get('format')!r}"
            )


@pytest.mark.parametrize(
    "dashboard_path", _shipped_dashboard_files(), ids=lambda p: p.name
)
def test_shipped_dashboard_should_keep_prometheus_queries_on_prometheus(
    dashboard_path: Path,
) -> None:
    """Prometheus queries need an expr; Tempo queries carry a trace query instead."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    for panel in _dashboard_panels(dashboard):
        for query in panel["data"]["spec"]["queries"]:
            data_query = query["spec"]["query"]
            payload = data_query.get("spec", {})
            group = data_query.get("group")
            assert group in {"prometheus", "tempo"}, (
                f"{panel['title']} queries unknown group {group!r}"
            )
            if group == "prometheus":
                assert payload.get("expr"), f"{panel['title']} has an empty expr"
            else:
                assert payload.get("queryType") == "traceql", (
                    f"{panel['title']} issues a Tempo query without a queryType"
                )


# ---------------------------------------------------------------------------
# Trainer dashboard hardware panels.
# ---------------------------------------------------------------------------

_DASHBOARDS_DIR = (
    Path(__file__).resolve().parents[3]
    / "rl_insight"
    / "config"
    / "services"
    / "grafana"
    / "dashboards"
)
_TRAINER_DASHBOARDS = [
    _DASHBOARDS_DIR / "verl" / "verl_trainer_v1_with_vllm_engine.json",
    _DASHBOARDS_DIR / "verl-omni" / "verl_omni_trainer_v1_with_vllm_omni_engine.json",
]
_METRIC_PREFIX = f"{MonitorDefaults.NAMESPACE}_rlinsight_"
_DEVICE_FILTER = 'device_index=~"$device_index"'

# Panel id -> built-in metric family backing the visible query.
_BUILTIN_PANELS = {
    "325": "cpu_utilization_percent",
    "326": "memory_used_bytes",
    "328": "network_receive_bytes_per_second",
    "329": "npu_chip_info_utilization",
    "330": "npu_chip_info_hbm_utilization",
    "331": "npu_chip_info_hbm_used_memory",
    "332": "npu_chip_info_power",
    "333": "npu_chip_info_temperature",
    "334": "npu_chip_info_health_status",
    "335": "npu_chip_info_hbm_bandwidth_utilization",
    "337": "gpu_utilization_percent",
    "338": "gpu_memory_utilization_percent",
    "339": "gpu_memory_used_bytes",
    "340": "gpu_power_watts",
    "341": "gpu_temperature_celsius",
    # The vllm dashboard numbers its GPU panels separately.
    "350": "gpu_utilization_percent",
    "356": "gpu_memory_utilization_percent",
    "352": "gpu_memory_used_bytes",
    "353": "gpu_power_watts",
    "354": "gpu_temperature_celsius",
    "344": "disk_read_bytes_per_second",
}
_RAY_PROCESS_PANELS = {"345": "ray_process_cpu_percent", "346": "ray_process_rss_bytes"}
_SELF_PROCESS_PANELS = {"347": "process_cpu_percent", "348": "process_rss_bytes"}


@pytest.fixture(
    scope="module", params=_TRAINER_DASHBOARDS, ids=lambda path: path.parent.name
)
def trainer_dashboard(request: pytest.FixtureRequest) -> dict[str, Any]:
    with open(request.param, encoding="utf-8") as f:
        return json.load(f)["spec"]


def _panel_queries(spec: dict[str, Any], panel_id: str) -> list[tuple[str, str, bool]]:
    """(expr, legendFormat, hidden) triples of one hardware panel."""
    element = spec["elements"].get(f"panel-{panel_id}")
    if element is None:
        return []
    queries = element["spec"]["data"]["spec"]["queries"]
    return [
        (
            query["spec"]["query"]["spec"]["expr"],
            query["spec"]["query"]["spec"].get("legendFormat", ""),
            bool(query["spec"].get("hidden")),
        )
        for query in queries
    ]


def _query_values(spec: dict[str, Any], panel_id: str, *, hidden: bool) -> list[str]:
    return [
        expr
        for expr, _, is_hidden in _panel_queries(spec, panel_id)
        if is_hidden == hidden
    ]


def test_hardware_panels_should_query_builtin_series_with_device_filters(
    trainer_dashboard: dict[str, Any],
) -> None:
    for panel_id, family in _BUILTIN_PANELS.items():
        visible = _query_values(trainer_dashboard, panel_id, hidden=False)
        if not visible:
            continue  # the omni dashboard has no disk panel
        assert all(_METRIC_PREFIX in expr for expr in visible), panel_id
        assert any(f"{_METRIC_PREFIX}{family}" in expr for expr in visible), panel_id
        assert not _query_values(trainer_dashboard, panel_id, hidden=True), panel_id

    # Both directions are present on the network and disk panels.
    for panel_id in ("328", "344"):
        visible_exprs = " ".join(
            _query_values(trainer_dashboard, panel_id, hidden=False)
        )
        if visible_exprs:
            assert "network_receive" in visible_exprs or "disk_read" in visible_exprs
            assert "network_transmit" in visible_exprs or "disk_write" in visible_exprs

    # NPU memory stays in megabytes, like the exporter it replaces.
    npu_memory = _query_values(trainer_dashboard, "331", hidden=False)
    if npu_memory:
        assert all("/ 1048576" not in expr for expr in npu_memory)

    # Device labels filter device panels; process panels carry pid/role labels.
    for panel_ids, legend, shows_card in (
        (
            ("329", "330", "331", "332", "333", "337", "338", "339", "340", "341"),
            None,
            True,
        ),
        (
            tuple(_RAY_PROCESS_PANELS),
            "{{node}} {{device_index}} {{worker_role}} {{pid}}",
            True,
        ),
        (tuple(_SELF_PROCESS_PANELS), "{{node}} {{worker_role}} {{pid}}", False),
    ):
        for panel_id in panel_ids:
            queries = _panel_queries(trainer_dashboard, panel_id)
            if not queries:
                continue
            for expr, legend_format, hidden in queries:
                if hidden:
                    continue  # only visible queries drive the panels
                assert (_DEVICE_FILTER in expr) is shows_card, expr
                if legend is not None:
                    assert legend_format == legend, legend_format


def test_hardware_variables_and_rows_should_use_builtin_sources(
    trainer_dashboard: dict[str, Any],
) -> None:
    hardware_row = _find_row_spec(trainer_dashboard["layout"], "hardware metric")
    assert hardware_row is not None
    hardware_variables = {
        variable["spec"]["name"]: variable["spec"]
        for variable in hardware_row["variables"]
    }
    assert {spec["label"] for spec in hardware_variables.values()} == {
        "Node",
        "Device",
    }
    node_variable = hardware_variables["node"]
    assert "device_index" in hardware_variables

    query = node_variable["definition"]
    assert "query_result(" in query
    assert f"{_METRIC_PREFIX}cpu_utilization_percent" in query
    assert f"{_METRIC_PREFIX}npu_chip_info_name" in query
    assert f"{_METRIC_PREFIX}gpu_info" in query
    assert "npu-exporter" not in query
    assert "DCGM_FI_DEV_GPU_UTIL" not in query
    assert "node-exporter" not in query
    assert node_variable["regex"] == '/node="(?<value>[^"]+)"/'

    def _condition(title: str) -> dict[str, Any]:
        row = _find_row_spec(trainer_dashboard["layout"], title)
        assert row is not None, title
        assert "repeat" not in json.dumps(row.get("layout") or {}), title
        return row["conditionalRendering"]["spec"]

    cpu_row = _find_row_spec(trainer_dashboard["layout"], "CPU / Host Metrics")
    assert cpu_row is not None
    assert "conditionalRendering" not in cpu_row
    assert "repeat" not in json.dumps(cpu_row.get("layout") or {})

    for title, variable, value in (
        ("Ascend NPU Metrics", "npu_vendor", "ascend"),
        ("NVIDIA GPU Metrics", "gpu_vendor", "nvidia"),
    ):
        condition = _condition(title)
        assert condition["visibility"] == "show"
        assert condition["condition"] == "and"
        assert condition["items"] == [
            {
                "kind": "ConditionalRenderingVariable",
                "spec": {"variable": variable, "operator": "equals", "value": value},
            }
        ]

    variables = {
        variable["spec"]["name"]: variable["spec"]
        for variable in trainer_dashboard["variables"]
    }
    for name in ("npu_vendor", "gpu_vendor"):
        assert name in variables
        assert variables[name]["hide"] == "hideVariable"
        assert variables[name]["refresh"] == "onTimeRangeChanged"
    assert (
        f"{_METRIC_PREFIX}npu_chip_info_name" in variables["npu_vendor"]["definition"]
    )
    assert f"{_METRIC_PREFIX}gpu_info" in variables["gpu_vendor"]["definition"]
    assert "DCGM_FI_DEV_GPU_UTIL" not in variables["gpu_vendor"]["definition"]


def _find_row_spec(layout: Any, title: str) -> dict[str, Any] | None:
    if not isinstance(layout, dict):
        return None
    spec = layout.get("spec")
    if not isinstance(spec, dict):
        return None
    kind = layout.get("kind")
    if kind == "RowsLayout":
        for row in spec.get("rows") or []:
            row_spec = row.get("spec") or {}
            if row_spec.get("title") == title:
                return row_spec
            found = _find_row_spec(row_spec.get("layout"), title)
            if found is not None:
                return found
    elif kind == "TabsLayout":
        for tab in spec.get("tabs") or []:
            found = _find_row_spec((tab.get("spec") or {}).get("layout"), title)
            if found is not None:
                return found
    return None
