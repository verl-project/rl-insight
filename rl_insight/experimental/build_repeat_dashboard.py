#!/usr/bin/env python3
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

"""Build the Grafana 13.1 section-scoped Agent Loop dashboard.

Run is repeated from a dashboard variable. Each repeated section owns the
query variable for its direct children, so Sample, Session, and Trajectory
queries resolve against that specific parent rather than a dashboard-global
``All`` value.

Session rows show an overview of their trajectories. Each repeated trajectory
row contains its own Tempo turn timeline and turn-details table.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from rl_insight.experimental.agent_loop_constants import (
    GRAFANA_DASHBOARD_FILE,
    GRAFANA_DASHBOARD_TITLE,
    GRAFANA_DASHBOARD_UID,
)

_HERE = Path(__file__).resolve().parent
_DASHBOARD_DIR = _HERE.parent / "config" / "services" / "grafana" / "dashboards"
_RUNTIME = Path.home() / ".rl-insight" / "runtime" / "dashboards" / GRAFANA_DASHBOARD_FILE
_PANEL_TEMPLATES = _HERE / "agent_loop_panel_templates.json"

_OVERVIEW_ELEMENT = "agent-loop-repeat-overview"
_SEQUENCE_ELEMENT = "agent-loop-repeat-sequence"
_DETAILS_ELEMENT = "agent-loop-repeat-details"
_EMPTY_STATE_ELEMENT = "agent-loop-empty-state"

_RE_RUN = r'/run_id="(?<value>[^"]+)".*title="(?<text>[^"]+)"/'
_RE_SAMPLE = r'/sample="(?<value>[^"]+)".*title="(?<text>[^"]+)"/'
_RE_SESSION = r'/session="(?<value>[^"]+)".*title="(?<text>[^"]+)"/'
# Prometheus serializes labels alphabetically, so title precedes traj.
_RE_TRAJ = r'/title="(?<text>[^"]+)".*traj="(?<value>[^"]+)"/'


def _grid_item(name: str, *, y: int, height: int) -> dict[str, Any]:
    return {
        "kind": "GridLayoutItem",
        "spec": {
            "x": 0,
            "y": y,
            "width": 24,
            "height": height,
            "element": {"kind": "ElementReference", "name": name},
        },
    }


def _grid_layout(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "GridLayout", "spec": {"items": items}}


def _panel_query(panel: dict[str, Any], query: str) -> None:
    panel["spec"]["data"]["spec"]["queries"][0]["spec"]["query"]["spec"][
        "query"
    ] = query


def _tempo_elements() -> dict[str, Any]:
    templates = json.loads(_PANEL_TEMPLATES.read_text(encoding="utf-8"))
    overview = copy.deepcopy(templates["overview"])
    sequence = copy.deepcopy(templates["sequence"])
    details = copy.deepcopy(templates["details"])

    session_lane = (
        "run=${run_id}/sample=${sample}/session=${session}"
    )
    traj_lane = f"{session_lane}/traj=${{traj}}"
    service_filter = 'resource.service.name="agent-loop-poc"'
    run_filter = 'span.run_id = "${run_id}"'

    _panel_query(
        overview,
        (
            f'{{span.state_lane_id =~ "{session_lane}/.*" && '
            f"{run_filter} && {service_filter}}} "
            "| select(span.turn, span.finish_reason)"
        ),
    )
    _panel_query(
        sequence,
        (
            f'{{span.state_lane_id = "{traj_lane}" && '
            f"{run_filter} && {service_filter}}} "
            "| select(span.turn, span.finish_reason)"
        ),
    )
    _panel_query(
        details,
        (
            f'{{span.state_lane_id = "{traj_lane}" && '
            f"{run_filter} && {service_filter}}} "
            "| select(span.turn, span.type, span.tools, span.finish_reason, "
            "span.content)"
        ),
    )
    overview["spec"]["description"] = (
        "All trajectories in the current repeated Session."
    )

    return {
        _OVERVIEW_ELEMENT: overview,
        _SEQUENCE_ELEMENT: sequence,
        _DETAILS_ELEMENT: details,
    }


def _empty_state_element() -> dict[str, Any]:
    """Plain markdown panel shown only while ``has_agent_loop_data == 0``.

    The push-batch export model (``export_to_tempo.py`` flushes a finished
    sample tree atomically) has no observable "in progress" state to report,
    so this only distinguishes "no data yet" from "has data" -- not a
    third "generating" state.
    """
    return {
        "kind": "Panel",
        "spec": {
            "id": 329,
            "title": "",
            "description": "",
            "links": [],
            "data": {
                "kind": "QueryGroup",
                "spec": {"queries": [], "transformations": [], "queryOptions": {}},
            },
            "vizConfig": {
                "kind": "VizConfig",
                "group": "text",
                "version": "13.0.2",
                "spec": {
                    "options": {
                        "mode": "markdown",
                        "content": (
                            "### No Agent Loop data yet\n\n"
                            "No trajectories have been exported for the "
                            "current time range / run filter. Generate some:\n\n"
                            "```\npython rl_insight/experimental/export_to_tempo.py "
                            "--samples 2 --seed 42\n```\n\n"
                            "This panel is replaced by the real Run / Sample / "
                            "Session / Trajectory hierarchy on the next refresh "
                            "once data lands."
                        ),
                    }
                },
            },
        },
    }


def _row(
    title: str,
    layout: dict[str, Any],
    *,
    collapse: bool = True,
    hide_header: bool = False,
    repeat: str | None = None,
    variables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "title": title,
        "collapse": collapse,
        "hideHeader": hide_header,
        "layout": layout,
    }
    if repeat:
        spec["repeat"] = {"mode": "variable", "value": repeat}
    if variables:
        spec["variables"] = variables
    return {"kind": "RowsLayoutRow", "spec": spec}


def _rows_layout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "RowsLayout", "spec": {"rows": rows}}


def _show_when_variable_equals(name: str, value: str) -> dict[str, Any]:
    return {
        "kind": "ConditionalRenderingGroup",
        "spec": {
            "visibility": "show",
            "condition": "and",
            "items": [
                {
                    "kind": "ConditionalRenderingVariable",
                    "spec": {
                        "variable": name,
                        "operator": "equals",
                        "value": value,
                    },
                }
            ],
        },
    }


def _query_variable(
    name: str,
    query: str,
    *,
    label: str,
    regex: str,
    include_all: bool = True,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": name,
        "label": label,
        "hide": "hideVariable",
        "refresh": "onTimeRangeChanged",
        "skipUrlSync": False,
        "description": "Grafana 13.1 section-scoped Repeat variable.",
        "query": {
            "kind": "DataQuery",
            "group": "prometheus",
            "version": "v0",
            "datasource": {"name": "${datasource}"},
            "spec": {
                "query": query,
                "refId": "StandardVariableQuery",
            },
        },
        "regex": regex,
        "regexApplyTo": "value",
        "sort": "alphabeticalAsc",
        "definition": query,
        "options": [],
        "allowCustomValue": False,
    }
    if include_all:
        spec["current"] = {"text": "All", "value": "$__all"}
        spec["multi"] = True
        spec["includeAll"] = True
    else:
        spec["current"] = {"text": "", "value": ""}
        spec["multi"] = False
        spec["includeAll"] = False
    return {"kind": "QueryVariable", "spec": spec}


def build_dashboard() -> dict[str, Any]:
    traj_variable = _query_variable(
        "traj",
        (
            "query_result(agent_loop_traj_info"
            '{run_id=~"$run_id",sample=~"$sample",session=~"$session"})'
        ),
        label="traj",
        regex=_RE_TRAJ,
    )
    traj_row = _row(
        "${traj:text}",
        _grid_layout(
            [
                _grid_item(_SEQUENCE_ELEMENT, y=0, height=6),
                _grid_item(_DETAILS_ELEMENT, y=6, height=8),
            ]
        ),
        collapse=True,
        repeat="traj",
    )
    overview_row = _row(
        "",
        _grid_layout([_grid_item(_OVERVIEW_ELEMENT, y=0, height=8)]),
        collapse=False,
        hide_header=True,
    )
    session_variable = _query_variable(
        "session",
        (
            "query_result(agent_loop_session_info"
            '{run_id=~"$run_id",sample=~"$sample"})'
        ),
        label="session",
        regex=_RE_SESSION,
    )
    session_row = _row(
        "${session:text}",
        _rows_layout([overview_row, traj_row]),
        collapse=True,
        repeat="session",
        variables=[traj_variable],
    )
    sample_variable = _query_variable(
        "sample",
        'query_result(agent_loop_sample_info{run_id=~"$run_id"})',
        label="sample",
        regex=_RE_SAMPLE,
    )
    sample_row = _row(
        "${sample:text}",
        _rows_layout([session_row]),
        collapse=True,
        repeat="sample",
        variables=[session_variable],
    )
    run_row = _row(
        "${run_id:text}",
        _rows_layout([sample_row]),
        collapse=False,
        repeat="run_id",
        variables=[sample_variable],
    )
    # includeAll keeps every real option selected for Repeat, but Grafana also
    # exposes a synthetic "All" option when the query is empty. Hide that
    # placeholder hierarchy unless Prometheus confirms at least one run.
    run_row["spec"]["conditionalRendering"] = _show_when_variable_equals(
        "has_agent_loop_data", "1"
    )
    empty_state_row = _row(
        "",
        _grid_layout([_grid_item(_EMPTY_STATE_ELEMENT, y=0, height=4)]),
        collapse=False,
        hide_header=True,
    )
    empty_state_row["spec"]["conditionalRendering"] = _show_when_variable_equals(
        "has_agent_loop_data", "0"
    )
    outer = _row(
        "Agent Loop Trajectory",
        _rows_layout([run_row, empty_state_row]),
        collapse=False,
    )

    has_data_variable = _query_variable(
        "has_agent_loop_data",
        (
            "query_result("
            "clamp_max(count(agent_loop_run_info), 1) or vector(0)"
            ")"
        ),
        label="has Agent Loop data",
        regex=r"/\s(?<value>[01])(?:\s|$)/",
        include_all=False,
    )
    has_data_variable["spec"]["current"] = {"text": "0", "value": "0"}

    run_id_variable = _query_variable(
        "run_id",
        "query_result(agent_loop_run_info)",
        label="run_id",
        regex=_RE_RUN,
    )
    # run_id embeds a unix timestamp (tempo_export.new_run_id: "export-<ts>-<hex>"),
    # so descending string sort is descending recency. Leave `current` unset
    # (rather than the synthetic "$__all") so Grafana auto-selects the first
    # (newest) option instead of expanding every run in the time window by
    # default; includeAll/multi stay on so users can still opt into comparing
    # multiple runs.
    run_id_variable["spec"]["sort"] = "alphabeticalDesc"
    run_id_variable["spec"]["current"] = {"text": "", "value": ""}

    variables = [
        {
            "kind": "DatasourceVariable",
            "spec": {
                "name": "datasource",
                "pluginId": "prometheus",
                "refresh": "onDashboardLoad",
                "regex": "",
                "current": {"text": "Prometheus", "value": "Prometheus"},
                "options": [],
                "multi": False,
                "includeAll": False,
                "hide": "hideVariable",
                "skipUrlSync": False,
                "allowCustomValue": True,
            },
        },
        run_id_variable,
        has_data_variable,
    ]

    elements = _tempo_elements()
    elements[_EMPTY_STATE_ELEMENT] = _empty_state_element()

    return {
        "apiVersion": "dashboard.grafana.app/v2",
        "kind": "Dashboard",
        "metadata": {
            "name": GRAFANA_DASHBOARD_UID,
            "generation": 1,
            "creationTimestamp": "2026-07-21T00:00:00Z",
            "labels": {},
            "annotations": {},
        },
        "spec": {
            "annotations": [],
            "cursorSync": "Crosshair",
            "editable": True,
            "elements": elements,
            "layout": _rows_layout([outer]),
            "links": [],
            "liveNow": False,
            "preload": False,
            "tags": [
                "rl-insight",
                "agent-loop",
                "experimental",
                "repeat-poc",
                "titles-from-info",
            ],
            "timeSettings": {
                "timezone": "browser",
                "from": "now-15m",
                "to": "now",
                "autoRefresh": "5s",
                "autoRefreshIntervals": [
                    "5s",
                    "10s",
                    "30s",
                    "1m",
                    "5m",
                    "15m",
                    "30m",
                    "1h",
                    "2h",
                    "1d",
                ],
                "hideTimepicker": False,
                "fiscalYearStartMonth": 0,
            },
            "title": GRAFANA_DASHBOARD_TITLE,
            "variables": variables,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DASHBOARD_DIR / GRAFANA_DASHBOARD_FILE,
    )
    args = parser.parse_args()

    dashboard = build_dashboard()
    text = json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    _RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    _RUNTIME.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.output} (+ runtime) Grafana 13.1 "
        "section-scoped hierarchy with Tempo panels"
    )


if __name__ == "__main__":
    main()
