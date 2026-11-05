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

"""Unit tests for Grafana dashboard staging and hardware panels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module
from rl_insight.utils.constants import MonitorDefaults

# ---------------------------------------------------------------------------
# Dashboard runtime staging.
# ---------------------------------------------------------------------------


def _conf(builtin: Path, data_dir: Path, extra: Path | None = None):
    grafana = {"dashboards_dir": str(builtin)}
    if extra is not None:
        grafana["extra_dashboard_dir"] = str(extra)
    return OmegaConf.create({"grafana": grafana, "server": {"data_dir": str(data_dir)}})


def test_stage_copies_builtin_and_extra_without_parsing_json(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "verl").mkdir(parents=True)
    (extra / "custom").mkdir(parents=True)
    (builtin / "verl" / "board.json").write_text("{}", encoding="utf-8")
    (extra / "custom" / "router.json").write_text("not valid json", encoding="utf-8")

    staged = runtime_module._stage_grafana_dashboards(
        _conf(builtin, tmp_path / "state", extra), tmp_path / "runtime"
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
            _conf(builtin, tmp_path / "state", extra), runtime_dir
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

    runtime_module._stage_grafana_dashboards(
        _conf(builtin, tmp_path / "state", extra), runtime_dir
    )
    extra_dashboard.write_text("second", encoding="utf-8")
    staged = runtime_module._stage_grafana_dashboards(
        _conf(builtin, tmp_path / "state", extra), runtime_dir
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
            _conf(builtin, tmp_path / "state", extra), tmp_path / "runtime"
        )


# ---------------------------------------------------------------------------
# Hardware source selection.
# ---------------------------------------------------------------------------


def _dual_source_panel(builtin: str, exporter: str) -> dict[str, Any]:
    return {
        "spec": {
            "data": {
                "spec": {
                    "queries": [
                        {
                            "spec": {
                                "refId": "A",
                                "hidden": False,
                                "query": {"spec": {"expr": builtin}},
                            }
                        },
                        {
                            "spec": {
                                "refId": "B",
                                "hidden": True,
                                "query": {"spec": {"expr": exporter}},
                            }
                        },
                    ]
                }
            }
        }
    }


def _dashboard(**panels: dict[str, Any]) -> dict[str, Any]:
    return {"spec": {"elements": panels}}


def _hidden_by_ref(panel: dict[str, Any]) -> dict[str, bool]:
    queries = runtime_module._panel_queries(panel["spec"])
    return {
        query["spec"]["refId"]: bool(query["spec"].get("hidden")) for query in queries
    }


def test_set_hardware_source_should_switch_each_family_independently() -> None:
    dashboard = _dashboard(
        **{
            "panel-node": _dual_source_panel(
                "rl_insight_monitor_rlinsight_cpu_utilization_percent",
                'node_cpu_seconds_total{job="node-exporter"}',
            ),
            "panel-npu": _dual_source_panel(
                "rl_insight_monitor_rlinsight_npu_chip_info_utilization",
                'npu_chip_info_utilization{job="npu-exporter"}',
            ),
            "panel-gpu": _dual_source_panel(
                "rl_insight_monitor_rlinsight_gpu_utilization_percent",
                "DCGM_FI_DEV_GPU_UTIL",
            ),
        }
    )
    elements = dashboard["spec"]["elements"]

    assert runtime_module._set_hardware_source(dashboard, {"npu-exporter"}) is True
    assert _hidden_by_ref(elements["panel-node"]) == {"A": False, "B": True}
    assert _hidden_by_ref(elements["panel-npu"]) == {"A": True, "B": False}
    assert _hidden_by_ref(elements["panel-gpu"]) == {"A": False, "B": True}

    assert (
        runtime_module._set_hardware_source(dashboard, {"node-exporter", "dcgm"})
        is True
    )
    assert _hidden_by_ref(elements["panel-node"]) == {"A": True, "B": False}
    assert _hidden_by_ref(elements["panel-npu"]) == {"A": False, "B": True}
    assert _hidden_by_ref(elements["panel-gpu"]) == {"A": True, "B": False}


def test_refresh_staged_dashboards_should_follow_registered_exporters(
    tmp_path: Path,
) -> None:
    targets = tmp_path / "targets" / "prometheus-targets.yml"
    targets.parent.mkdir(parents=True)
    targets.write_text(
        yaml.safe_dump(
            [
                {
                    "targets": ["10.0.0.1:9100"],
                    "labels": {"rl_insight_job": "node-exporter"},
                }
            ]
        ),
        encoding="utf-8",
    )
    dashboard_dir = tmp_path / "runtime" / "dashboards"
    dashboard_dir.mkdir(parents=True)
    panel = _dual_source_panel(
        "rl_insight_monitor_rlinsight_cpu_utilization_percent",
        'node_cpu_seconds_total{job="node-exporter"}',
    )
    (dashboard_dir / "board.json").write_text(
        json.dumps(_dashboard(**{"panel-325": panel})), encoding="utf-8"
    )
    conf = OmegaConf.create(
        {
            "server": {
                "data_dir": str(tmp_path),
                "runtime_dir": str(tmp_path / "runtime"),
            }
        }
    )

    assert runtime_module._registered_hardware_exporters(conf) == {"node-exporter"}
    assert runtime_module.refresh_staged_dashboards(conf) == 1
    stored = json.loads((dashboard_dir / "board.json").read_text(encoding="utf-8"))
    assert _hidden_by_ref(stored["spec"]["elements"]["panel-325"]) == {
        "A": True,
        "B": False,
    }
    # A second pass has nothing left to change.
    assert runtime_module.refresh_staged_dashboards(conf) == 0


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


def test_builtin_panels_should_hide_the_exporter_query(
    trainer_dashboard: dict[str, Any],
) -> None:
    for panel_id, family in _BUILTIN_PANELS.items():
        visible = _query_values(trainer_dashboard, panel_id, hidden=False)
        hidden = _query_values(trainer_dashboard, panel_id, hidden=True)
        if not visible:
            continue  # the omni dashboard has no disk panel
        assert hidden, panel_id
        assert any(f"{_METRIC_PREFIX}{family}" in expr for expr in visible), panel_id
        assert all(_METRIC_PREFIX not in expr for expr in hidden), panel_id

    # Both directions are present on the network and disk panels.
    for panel_id in ("328", "344"):
        visible = " ".join(_query_values(trainer_dashboard, panel_id, hidden=False))
        if visible:
            assert "network_receive" in visible or "disk_read" in visible
            assert "network_transmit" in visible or "disk_write" in visible


def test_npu_memory_panel_should_use_megabytes_like_the_exporter(
    trainer_dashboard: dict[str, Any],
) -> None:
    visible = _query_values(trainer_dashboard, "331", hidden=False)
    if not visible:
        pytest.skip("dashboard has no NPU memory panel")

    assert all("/ 1048576" not in expr for expr in visible)


@pytest.mark.parametrize(
    ("panel_ids", "legend", "shows_card"),
    [
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
    ],
    ids=["devices", "ray-process", "self-process"],
)
def test_process_and_device_panels_should_filter_by_device(
    trainer_dashboard: dict[str, Any],
    panel_ids: tuple[str, ...],
    legend: str | None,
    shows_card: bool,
) -> None:
    for panel_id in panel_ids:
        queries = _panel_queries(trainer_dashboard, panel_id)
        if not queries:
            continue
        for expr, legend_format, hidden in queries:
            if hidden:
                continue  # exporter queries are covered by the source tests
            assert (_DEVICE_FILTER in expr) is shows_card, expr
            if legend is not None:
                assert legend_format == legend, legend_format


def test_node_variable_should_cover_builtin_and_exporter_sources(
    trainer_dashboard: dict[str, Any],
) -> None:
    variable = next(
        variable
        for variable in trainer_dashboard["variables"]
        if variable["spec"]["name"] == "node"
    )
    query = variable["spec"]["definition"]

    assert "query_result(" in query
    assert f"{_METRIC_PREFIX}cpu_utilization_percent" in query
    assert f"{_METRIC_PREFIX}npu_chip_info_name" in query
    assert "npu_chip_info_name" in query
    assert "DCGM_FI_DEV_GPU_UTIL" in query
    assert variable["spec"]["regex"] == '/node="(?<value>[^"]+)"/'


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


def test_hardware_vendor_rows_should_render_only_for_present_vendors(
    trainer_dashboard: dict[str, Any],
) -> None:
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
    assert "npu_chip_info_name" in variables["npu_vendor"]["definition"]
    assert "DCGM_FI_DEV_GPU_UTIL" in variables["gpu_vendor"]["definition"]


def test_hardware_descriptions_should_not_refer_to_internal_sources(
    trainer_dashboard: dict[str, Any],
) -> None:
    for panel_id in (*_BUILTIN_PANELS, *_RAY_PROCESS_PANELS, *_SELF_PROCESS_PANELS):
        element = trainer_dashboard["elements"].get(f"panel-{panel_id}")
        if element is None:
            continue
        description = element["spec"].get("description", "")
        assert description
        assert "in-process" not in description
        assert len(description) <= 120, (panel_id, description)
