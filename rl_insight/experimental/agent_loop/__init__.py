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

"""Agent Loop visualization package: SampleRecord → Tempo → Grafana Rebuild.

Public names are loaded lazily so the py3.9 ``rl-insight-server`` can import
Rebuild / dashboard helpers without evaluating SampleRecord (Pydantic) annotations.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # constants
    "SERVICE_NAME_VALUE",
    "GRAFANA_DASHBOARD_FILE",
    "GRAFANA_DASHBOARD_UID",
    "GRAFANA_DASHBOARD_SLUG",
    "GRAFANA_DASHBOARD_TITLE",
    "DEFAULT_GRAFANA_BASE",
    "DEFAULT_REBUILD_API_BASE",
    "DEFAULT_TEMPO_URL",
    "DEFAULT_GAP_S",
    "AGENT_LOOP_TITLE",
    # classes
    "TempoSpanMapper",
    "TempoClient",
    "TempoSpan",
    "RunHierarchyBuilder",
    "AgentLoopDashboardWriter",
    "AgentLoopRebuild",
    # primary entry points
    "export_samples_to_tempo",
    "rebuild_from_tempo",
    "write_agent_loop_from_runs",
    # mapper helpers
    "new_run_id",
    "lane_id",
    "wait_for_otlp",
    "samples_to_span_dicts",
    "compress_span_times",
    "flush_span_dicts",
    # tempo / hierarchy helpers
    "fetch_spans",
    "assign_run_keys",
    "complete_stamped_runs",
    "hierarchy_from_spans",
    "pick_latest_run_id",
    # dashboard helpers
    "rebuild_agent_loop_from_runs",
    "rebuild_agent_loop_from_hierarchy",
    "write_agent_loop_from_hierarchy",
    "rebuild_dashboard_link",
    "strip_agent_loop",
    "slim_bundled_dashboard",
]

_LAZY: dict[str, tuple[str, str]] = {
    # constants
    "SERVICE_NAME_VALUE": (
        "rl_insight.experimental.agent_loop.constants",
        "SERVICE_NAME_VALUE",
    ),
    "GRAFANA_DASHBOARD_FILE": (
        "rl_insight.experimental.agent_loop.constants",
        "GRAFANA_DASHBOARD_FILE",
    ),
    "GRAFANA_DASHBOARD_UID": (
        "rl_insight.experimental.agent_loop.constants",
        "GRAFANA_DASHBOARD_UID",
    ),
    "GRAFANA_DASHBOARD_SLUG": (
        "rl_insight.experimental.agent_loop.constants",
        "GRAFANA_DASHBOARD_SLUG",
    ),
    "GRAFANA_DASHBOARD_TITLE": (
        "rl_insight.experimental.agent_loop.constants",
        "GRAFANA_DASHBOARD_TITLE",
    ),
    "DEFAULT_GRAFANA_BASE": (
        "rl_insight.experimental.agent_loop.constants",
        "DEFAULT_GRAFANA_BASE",
    ),
    "DEFAULT_REBUILD_API_BASE": (
        "rl_insight.experimental.agent_loop.constants",
        "DEFAULT_REBUILD_API_BASE",
    ),
    # write (SampleRecord — only load when requested)
    "TempoSpanMapper": (
        "rl_insight.experimental.agent_loop.write.mapper",
        "TempoSpanMapper",
    ),
    "export_samples_to_tempo": (
        "rl_insight.experimental.agent_loop.write.mapper",
        "export_samples_to_tempo",
    ),
    "new_run_id": ("rl_insight.experimental.agent_loop.write.mapper", "new_run_id"),
    "lane_id": ("rl_insight.experimental.agent_loop.write.mapper", "lane_id"),
    "wait_for_otlp": (
        "rl_insight.experimental.agent_loop.write.mapper",
        "wait_for_otlp",
    ),
    "samples_to_span_dicts": (
        "rl_insight.experimental.agent_loop.write.mapper",
        "samples_to_span_dicts",
    ),
    "compress_span_times": (
        "rl_insight.experimental.agent_loop.write.mapper",
        "compress_span_times",
    ),
    "flush_span_dicts": (
        "rl_insight.experimental.agent_loop.write.mapper",
        "flush_span_dicts",
    ),
    # read
    "DEFAULT_TEMPO_URL": (
        "rl_insight.experimental.agent_loop.read.client",
        "DEFAULT_TEMPO_URL",
    ),
    "TempoClient": (
        "rl_insight.experimental.agent_loop.read.client",
        "TempoClient",
    ),
    "TempoSpan": ("rl_insight.experimental.agent_loop.read.client", "TempoSpan"),
    "fetch_spans": (
        "rl_insight.experimental.agent_loop.read.client",
        "fetch_spans",
    ),
    "DEFAULT_GAP_S": (
        "rl_insight.experimental.agent_loop.read.hierarchy",
        "DEFAULT_GAP_S",
    ),
    "RunHierarchyBuilder": (
        "rl_insight.experimental.agent_loop.read.hierarchy",
        "RunHierarchyBuilder",
    ),
    "assign_run_keys": (
        "rl_insight.experimental.agent_loop.read.hierarchy",
        "assign_run_keys",
    ),
    "complete_stamped_runs": (
        "rl_insight.experimental.agent_loop.read.hierarchy",
        "complete_stamped_runs",
    ),
    "hierarchy_from_spans": (
        "rl_insight.experimental.agent_loop.read.hierarchy",
        "hierarchy_from_spans",
    ),
    "pick_latest_run_id": (
        "rl_insight.experimental.agent_loop.read.hierarchy",
        "pick_latest_run_id",
    ),
    # dashboard
    "AGENT_LOOP_TITLE": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "AGENT_LOOP_TITLE",
    ),
    "AgentLoopDashboardWriter": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "AgentLoopDashboardWriter",
    ),
    "write_agent_loop_from_runs": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "write_agent_loop_from_runs",
    ),
    "rebuild_agent_loop_from_runs": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "rebuild_agent_loop_from_runs",
    ),
    "rebuild_agent_loop_from_hierarchy": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "rebuild_agent_loop_from_hierarchy",
    ),
    "write_agent_loop_from_hierarchy": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "write_agent_loop_from_hierarchy",
    ),
    "rebuild_dashboard_link": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "rebuild_dashboard_link",
    ),
    "strip_agent_loop": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "strip_agent_loop",
    ),
    "slim_bundled_dashboard": (
        "rl_insight.experimental.agent_loop.dashboard.writer",
        "slim_bundled_dashboard",
    ),
    # rebuild
    "AgentLoopRebuild": (
        "rl_insight.experimental.agent_loop.rebuild.service",
        "AgentLoopRebuild",
    ),
    "rebuild_from_tempo": (
        "rl_insight.experimental.agent_loop.rebuild.service",
        "rebuild_from_tempo",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    mod = importlib.import_module(module_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
