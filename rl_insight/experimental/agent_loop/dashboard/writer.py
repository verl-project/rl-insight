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

"""Rebuild Agent Loop Trajectory Grafana panels from Tempo-derived hierarchies.

Visualization only sees Tempo attributes. Every run in the Rebuild time window
gets a full expandable tree: Run → Sample → Session → Trajectory.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

from rl_insight.experimental.agent_loop.constants import (
    DEFAULT_REBUILD_API_BASE,
    GRAFANA_DASHBOARD_FILE,
    SERVICE_NAME_VALUE,
)

logger = logging.getLogger(__name__)

AGENT_LOOP_TITLE = "Agent Loop Trajectory"
_TEMPLATES_PATH = Path(__file__).resolve().parent / "panel_templates.json"


def _rebuild_api_base() -> str:
    try:
        from rl_insight.server.network import local_addresses

        ipv4 = (local_addresses() or {}).get("ipv4")
        if ipv4:
            return f"http://{ipv4}:18080"
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_REBUILD_API_BASE


def rebuild_dashboard_link() -> dict[str, Any]:
    """Top-bar link near time range / Refresh; returns to Grafana via Referer."""
    api = _rebuild_api_base()
    return {
        "title": "Rebuild Agent Loop",
        "type": "link",
        "icon": "sync",
        "tooltip": (
            "Rebuild nested Agent Loop trees for the current dashboard time range, "
            "then return here"
        ),
        "url": f"{api}/api/v1/agent-loop/rebuild/go?from=${{__from}}&to=${{__to}}",
        "targetBlank": False,
        "includeVars": False,
        "keepTime": False,
        "asDropdown": False,
        "tags": [],
    }


_DEFAULT_BUNDLED = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "services"
    / "grafana"
    / "dashboards"
    / GRAFANA_DASHBOARD_FILE
)
_DEFAULT_RUNTIME = (
    Path.home() / ".rl-insight" / "runtime" / "dashboards" / GRAFANA_DASHBOARD_FILE
)


def _load_templates() -> dict[str, Any]:
    return json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


def _slug(run_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("-")
    return (s or "run")[:80]


def _set_query(panel: dict[str, Any], query: str) -> None:
    panel["spec"]["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]["query"] = query


def _panel_from_template(
    templates: dict[str, Any],
    kind: str,
    *,
    name: str,
    query: str,
) -> dict[str, Any]:
    panel = copy.deepcopy(templates[kind])
    _set_query(panel, query)
    return panel


def _overview_query(
    run_id: str,
    sample_i: int,
    session_i: int,
    service_name: str,
) -> str:
    """Per-run session overview (not mixed across runs)."""
    if run_id.startswith("auto-") or run_id.startswith("derived-"):
        return (
            f'{{span.state_lane_id =~ "sample={sample_i}/session={session_i}/.*"'
            f' && resource.service.name="{service_name}"}}'
        )
    return (
        f'{{span.state_lane_id =~ "run={run_id}/sample={sample_i}/session={session_i}/.*"'
        f' && span.run_id = "{run_id}"'
        f' && resource.service.name="{service_name}"}}'
    )


def _traj_query(
    run_id: str,
    sample_i: int,
    session_i: int,
    traj_i: int,
    service_name: str,
) -> str:
    if run_id.startswith("auto-") or run_id.startswith("derived-"):
        lane = f"sample={sample_i}/session={session_i}/traj={traj_i}"
        return (
            f'{{span.state_lane_id = "{lane}"'
            f' && resource.service.name="{service_name}"}}'
        )
    lane = f"run={run_id}/sample={sample_i}/session={session_i}/traj={traj_i}"
    return (
        f'{{span.state_lane_id = "{lane}"'
        f' && span.run_id = "{run_id}"'
        f' && resource.service.name="{service_name}"}}'
    )


def _details_query(
    run_id: str,
    sample_i: int,
    session_i: int,
    traj_i: int,
    service_name: str,
) -> str:
    base = _traj_query(run_id, sample_i, session_i, traj_i, service_name)
    return (
        f"{base} | select(span.turn, span.state_name, span.type, span.tools, "
        f"span.finish_reason, span.content)"
    )


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


def _rows_layout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "RowsLayout", "spec": {"rows": rows}}


def _row(
    title: str,
    layout: dict[str, Any],
    *,
    collapse: bool = True,
    hide_header: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "RowsLayoutRow",
        "spec": {
            "title": title,
            "collapse": collapse,
            "hideHeader": hide_header,
            "layout": layout,
        },
    }


def ensure_rebuild_dashboard_link(dashboard: dict[str, Any]) -> None:
    """Put Rebuild next to the time picker / Refresh (dashboard ``spec.links``)."""
    links = dashboard.setdefault("spec", {}).setdefault("links", [])
    link = rebuild_dashboard_link()
    kept = [
        item
        for item in links
        if not (isinstance(item, dict) and item.get("title") == link["title"])
    ]
    kept.insert(0, link)
    dashboard["spec"]["links"] = kept


def strip_agent_loop(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Remove all ``agent-*`` elements and clear the Agent Loop row body."""
    elements = dashboard.get("spec", {}).get("elements", {})
    for key in list(elements):
        if key.startswith("agent-"):
            del elements[key]

    rows = dashboard.get("spec", {}).get("layout", {}).get("spec", {}).get("rows", [])
    for row in rows:
        if row.get("spec", {}).get("title") != AGENT_LOOP_TITLE:
            continue
        row["spec"]["collapse"] = True
        row["spec"]["hideHeader"] = False
        row["spec"]["layout"] = _rows_layout([])
        break
    else:
        rows.append(_row(AGENT_LOOP_TITLE, _rows_layout([]), collapse=True))
    ensure_rebuild_dashboard_link(dashboard)
    return dashboard


def _sample_stats(sample: dict[str, Any]) -> tuple[int, int, int, int]:
    total = success = turns = 0
    for sess in sample.get("sessions") or []:
        for traj in sess.get("trajectories") or []:
            total += 1
            turns += int(traj.get("num_turns") or 0)
            if traj.get("success"):
                success += 1
    return success, total, turns, len(sample.get("sessions") or [])


def _session_stats(session: dict[str, Any]) -> tuple[int, int, int]:
    total = success = turns = 0
    for traj in session.get("trajectories") or []:
        total += 1
        turns += int(traj.get("num_turns") or 0)
        if traj.get("success"):
            success += 1
    return success, total, turns


def _build_sample_rows(
    *,
    templates: dict[str, Any],
    elements: dict[str, Any],
    run_id: str,
    samples: list[dict[str, Any]],
    service_name: str,
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    slug = _slug(run_id)
    sample_rows: list[dict[str, Any]] = []
    samples = sorted(samples, key=lambda s: int(s.get("sample_index", 0)))

    for sample in samples:
        si = int(sample.get("sample_index", 0))
        succ, total, turns, n_sess = _sample_stats(sample)
        sample_title = (
            f"Sample {si} · success {succ}/{total} · {turns} turns · {n_sess} sessions"
        )
        session_rows: list[dict[str, Any]] = []
        sessions = sorted(
            sample.get("sessions") or [], key=lambda s: int(s.get("session_index", 0))
        )
        for session in sessions:
            counters["sessions"] += 1
            sess_i = int(session.get("session_index", 0))
            s_succ, s_total, s_turns = _session_stats(session)
            session_title = (
                f"Session {sess_i} · success {s_succ}/{s_total} · "
                f"{s_turns} turns · {s_total} trajectories"
            )

            ov_name = f"agent-session-panel-{slug}-{si}-{sess_i}"
            elements[ov_name] = _panel_from_template(
                templates,
                "overview",
                name=ov_name,
                query=_overview_query(run_id, si, sess_i, service_name),
            )
            counters["panels"] += 1

            child_rows: list[dict[str, Any]] = [
                _row(
                    "",
                    {
                        "kind": "GridLayout",
                        "spec": {"items": [_grid_item(ov_name, y=0, height=8)]},
                    },
                    collapse=False,
                    hide_header=True,
                )
            ]

            trajs = sorted(
                session.get("trajectories") or [],
                key=lambda t: int(t.get("trajectory_index", 0)),
            )
            for traj in trajs:
                counters["trajectories"] += 1
                ti = int(traj.get("trajectory_index", 0))
                traj_title = (
                    f"Trajectory #{ti} · "
                    f"{int(traj.get('num_turns') or 0)} turns"
                )
                seq_name = f"agent-panel-seq-{slug}-{si}-{sess_i}-{ti}"
                det_name = f"agent-panel-det-{slug}-{si}-{sess_i}-{ti}"
                elements[seq_name] = _panel_from_template(
                    templates,
                    "sequence",
                    name=seq_name,
                    query=_traj_query(run_id, si, sess_i, ti, service_name),
                )
                elements[det_name] = _panel_from_template(
                    templates,
                    "details",
                    name=det_name,
                    query=_details_query(run_id, si, sess_i, ti, service_name),
                )
                counters["panels"] += 2
                child_rows.append(
                    _row(
                        traj_title,
                        {
                            "kind": "GridLayout",
                            "spec": {
                                "items": [
                                    _grid_item(seq_name, y=0, height=6),
                                    _grid_item(det_name, y=6, height=8),
                                ]
                            },
                        },
                        collapse=False,
                    )
                )

            session_rows.append(
                _row(session_title, _rows_layout(child_rows), collapse=True)
            )

        sample_rows.append(_row(sample_title, _rows_layout(session_rows), collapse=True))
        counters["samples"] += 1

    return sample_rows


def rebuild_agent_loop_from_runs(
    dashboard: dict[str, Any],
    run_hierarchies: list[dict[str, Any]],
    *,
    service_name: str = SERVICE_NAME_VALUE,
    window_from: int | None = None,
    window_to: int | None = None,
) -> dict[str, Any]:
    """Replace Agent Loop with one expandable tree per run in the window."""
    templates = _load_templates()
    dashboard = strip_agent_loop(dashboard)
    elements = dashboard["spec"]["elements"]
    counters = {"samples": 0, "sessions": 0, "trajectories": 0, "panels": 0}

    run_rows: list[dict[str, Any]] = []
    if not run_hierarchies:
        run_rows.append(
            _row(
                "No agent-loop runs in this time range",
                _rows_layout([]),
                collapse=False,
            )
        )
    else:
        for hier in run_hierarchies:
            run_id = str(hier.get("run_id") or "")
            samples = list(hier.get("samples") or [])
            succ = sum(
                1
                for s in samples
                for sess in s.get("sessions") or []
                for t in sess.get("trajectories") or []
                if t.get("success")
            )
            total = sum(
                len(sess.get("trajectories") or [])
                for s in samples
                for sess in s.get("sessions") or []
            )
            run_title = (
                f"Run · {run_id} · samples {len(samples)} · "
                f"success {succ}/{total}"
            )
            sample_rows = _build_sample_rows(
                templates=templates,
                elements=elements,
                run_id=run_id,
                samples=samples,
                service_name=service_name,
                counters=counters,
            )
            run_rows.append(_row(run_title, _rows_layout(sample_rows), collapse=True))

    rows = dashboard["spec"]["layout"]["spec"]["rows"]
    for row in rows:
        if row.get("spec", {}).get("title") == AGENT_LOOP_TITLE:
            row["spec"]["collapse"] = True
            row["spec"]["layout"] = _rows_layout(run_rows)
            break

    logger.info(
        "Agent Loop rebuilt: runs=%s samples=%s sessions=%s trajs=%s panels=%s",
        len(run_hierarchies),
        counters["samples"],
        counters["sessions"],
        counters["trajectories"],
        counters["panels"],
    )
    return {
        "dashboard": dashboard,
        "stats": {
            "runs": len(run_hierarchies),
            "samples": counters["samples"],
            "sessions": counters["sessions"],
            "trajectories": counters["trajectories"],
            "panels": counters["panels"],
        },
    }


def write_agent_loop_from_runs(
    run_hierarchies: list[dict[str, Any]],
    *,
    bundled_path: Path | None = None,
    runtime_path: Path | None = None,
    write_bundled: bool = False,
    service_name: str = SERVICE_NAME_VALUE,
    window_from: int | None = None,
    window_to: int | None = None,
) -> dict[str, Any]:
    """Load base dashboard, rebuild all runs, write runtime JSON."""
    bundled = Path(bundled_path) if bundled_path else _DEFAULT_BUNDLED
    runtime = Path(runtime_path) if runtime_path else _DEFAULT_RUNTIME

    base = json.loads(bundled.read_text(encoding="utf-8"))
    result = rebuild_agent_loop_from_runs(
        base,
        run_hierarchies,
        service_name=service_name,
        window_from=window_from,
        window_to=window_to,
    )
    dashboard = result["dashboard"]
    ensure_rebuild_dashboard_link(dashboard)
    text = json.dumps(dashboard, indent=2) + "\n"

    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(text, encoding="utf-8")
    logger.info("Wrote Agent Loop dashboard → %s", runtime)

    if write_bundled:
        bundled.write_text(text, encoding="utf-8")
        logger.info("Wrote Agent Loop dashboard → %s", bundled)

    return result["stats"]


def slim_bundled_dashboard(bundled_path: Path | None = None) -> Path:
    """Strip baked agent panels from the committed dashboard (empty Agent Loop shell)."""
    path = Path(bundled_path) if bundled_path else _DEFAULT_BUNDLED
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    strip_agent_loop(dashboard)
    path.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    return path


class AgentLoopDashboardWriter:
    """Write nested Agent Loop Grafana panels from run hierarchies."""

    def __init__(
        self,
        service_name: str = SERVICE_NAME_VALUE,
        bundled_path: Path | None = None,
        runtime_path: Path | None = None,
    ) -> None:
        self.service_name = service_name
        self.bundled_path = bundled_path
        self.runtime_path = runtime_path

    def write(
        self,
        run_hierarchies: list[dict[str, Any]],
        *,
        write_bundled: bool = False,
        window_from: int | None = None,
        window_to: int | None = None,
    ) -> dict[str, Any]:
        return write_agent_loop_from_runs(
            run_hierarchies,
            bundled_path=self.bundled_path,
            runtime_path=self.runtime_path,
            write_bundled=write_bundled,
            service_name=self.service_name,
            window_from=window_from,
            window_to=window_to,
        )

    def rebuild_in_memory(
        self,
        dashboard: dict[str, Any],
        run_hierarchies: list[dict[str, Any]],
        *,
        window_from: int | None = None,
        window_to: int | None = None,
    ) -> dict[str, Any]:
        return rebuild_agent_loop_from_runs(
            dashboard,
            run_hierarchies,
            service_name=self.service_name,
            window_from=window_from,
            window_to=window_to,
        )


# Back-compat wrappers
def write_agent_loop_from_hierarchy(
    hierarchy: dict[str, Any],
    *,
    overview_run_ids: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del overview_run_ids
    return write_agent_loop_from_runs([hierarchy], **kwargs)


def rebuild_agent_loop_from_hierarchy(
    dashboard: dict[str, Any],
    hierarchy: dict[str, Any],
    *,
    overview_run_ids: list[str] | None = None,
    service_name: str = SERVICE_NAME_VALUE,
) -> dict[str, Any]:
    del overview_run_ids
    return rebuild_agent_loop_from_runs(
        dashboard, [hierarchy], service_name=service_name
    )
