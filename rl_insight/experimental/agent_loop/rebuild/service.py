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

"""Orchestrate Tempo fetch → hierarchy → Grafana dashboard rebuild."""

from __future__ import annotations

import logging
import time
from typing import Any

from rl_insight.experimental.agent_loop.constants import SERVICE_NAME_VALUE
from rl_insight.experimental.agent_loop.dashboard.writer import (
    AgentLoopDashboardWriter,
    write_agent_loop_from_runs,
)
from rl_insight.experimental.agent_loop.read.hierarchy import (
    DEFAULT_GAP_S,
    RunHierarchyBuilder,
    _run_end_ns,
    _run_start_ns,
)
from rl_insight.experimental.agent_loop.read.client import (
    DEFAULT_TEMPO_URL,
    TempoClient,
    TempoSpan,
)

logger = logging.getLogger(__name__)


class AgentLoopRebuild:
    """Compose TempoClient + RunHierarchyBuilder + AgentLoopDashboardWriter."""

    def __init__(
        self,
        client: TempoClient | None = None,
        hierarchy: RunHierarchyBuilder | None = None,
        writer: AgentLoopDashboardWriter | None = None,
        *,
        tempo_url: str = DEFAULT_TEMPO_URL,
        service_name: str = SERVICE_NAME_VALUE,
        gap_s: float = DEFAULT_GAP_S,
    ) -> None:
        self.client = client or TempoClient(
            tempo_url=tempo_url, service_name=service_name
        )
        self.hierarchy = hierarchy or RunHierarchyBuilder(
            gap_s=gap_s, client=self.client
        )
        self.writer = writer or AgentLoopDashboardWriter(service_name=service_name)
        self.service_name = service_name
        self.gap_s = gap_s

    def run(
        self,
        *,
        start_unix: int | None = None,
        end_unix: int | None = None,
        run_id: str | None = None,
        write_bundled: bool = False,
        ingest_wait_s: float = 0.0,
        retries: int = 8,
        retry_pause_s: float = 1.5,
    ) -> dict[str, Any]:
        """Query Tempo for ``[start_unix, end_unix]`` and expand every run in window."""
        if ingest_wait_s > 0:
            time.sleep(ingest_wait_s)

        now = int(time.time())
        end = end_unix if end_unix is not None else now
        start = start_unix if start_unix is not None else end - 3 * 3600

        runs: dict[str, list[TempoSpan]] = {}
        last_err = "no spans"
        for attempt in range(max(1, retries)):
            spans = self.client.fetch_spans(start_unix=start, end_unix=end)
            runs = self.hierarchy.group(spans)
            if run_id:
                if run_id in runs:
                    break
            elif runs:
                break
            last_err = (
                f"attempt {attempt + 1}/{retries}: spans={len(spans)} "
                f"runs={list(runs)[:5]!r} requested={run_id!r}"
            )
            logger.info("rebuild waiting for Tempo: %s", last_err)
            time.sleep(retry_pause_s)
        else:
            # Empty window: clear nested panels and return (do not error-page the user).
            logger.info(
                "no agent-loop runs in window service=%s from=%s to=%s (%s)",
                self.service_name,
                start,
                end,
                last_err,
            )
            stats = self.writer.write(
                [],
                write_bundled=write_bundled,
                window_from=start,
                window_to=end,
            )
            return {
                "status": "ok",
                "empty": True,
                "from": start,
                "to": end,
                "service_name": self.service_name,
                "selected_run_id": run_id,
                "runs": [],
                "run_count": 0,
                "samples": 0,
                "sessions": 0,
                "trajectories": 0,
                "panels": stats.get("panels", 0),
                "message": last_err,
            }

        # Short Grafana windows can clip early samples of a compressed export;
        # refill each stamped run_id, then keep only spans that overlap [start, end]
        # so every panel in the tree has data inside the dashboard time range.
        runs = self.hierarchy.complete_stamped_runs(runs, end_unix=end)
        runs = self.hierarchy.filter_to_window(runs, start, end)
        if run_id:
            runs = {run_id: runs[run_id]} if run_id in runs else {}
        if not runs:
            logger.info(
                "no in-window agent-loop spans after filter service=%s from=%s to=%s",
                self.service_name,
                start,
                end,
            )
            stats = self.writer.write(
                [],
                write_bundled=write_bundled,
                window_from=start,
                window_to=end,
            )
            return {
                "status": "ok",
                "empty": True,
                "from": start,
                "to": end,
                "service_name": self.service_name,
                "selected_run_id": run_id,
                "runs": [],
                "run_count": 0,
                "samples": 0,
                "sessions": 0,
                "trajectories": 0,
                "panels": stats.get("panels", 0),
                "message": "no spans overlapping rebuild window after filter",
            }

        # Newest runs first so the dashboard opens on data that matches "now".
        ordered_ids = sorted(
            runs.keys(), key=lambda rid: _run_end_ns(runs[rid]), reverse=True
        )

        run_summaries = [
            {
                "run_id": rid,
                "spans": len(runs[rid]),
                "end_ns": _run_end_ns(runs[rid]),
                "start_ns": _run_start_ns(runs[rid]),
            }
            for rid in ordered_ids
        ]

        hierarchies = [
            self.hierarchy.build(runs[rid], rid) for rid in ordered_ids
        ]
        stats = self.writer.write(
            hierarchies,
            write_bundled=write_bundled,
            window_from=start,
            window_to=end,
        )
        return {
            "status": "ok",
            "from": start,
            "to": end,
            "service_name": self.service_name,
            "selected_run_id": run_id,
            # Panel stats also has a numeric "runs" key — keep summaries as a list.
            "runs": run_summaries,
            "run_count": int(stats.get("runs") or len(run_summaries)),
            "samples": stats.get("samples", 0),
            "sessions": stats.get("sessions", 0),
            "trajectories": stats.get("trajectories", 0),
            "panels": stats.get("panels", 0),
        }


def rebuild_from_tempo(
    *,
    tempo_url: str = DEFAULT_TEMPO_URL,
    service_name: str = SERVICE_NAME_VALUE,
    start_unix: int | None = None,
    end_unix: int | None = None,
    run_id: str | None = None,
    gap_s: float = DEFAULT_GAP_S,
    write_bundled: bool = False,
    ingest_wait_s: float = 0.0,
    retries: int = 8,
    retry_pause_s: float = 1.5,
) -> dict[str, Any]:
    """Query Tempo for ``[start_unix, end_unix]`` and expand **every** run in window.

    ``run_id`` is optional: if set, only that run's tree is written; otherwise all
    runs found in the window get a full Run → Sample → Session → Traj tree.
    """
    rebuild = AgentLoopRebuild(
        tempo_url=tempo_url,
        service_name=service_name,
        gap_s=gap_s,
    )
    return rebuild.run(
        start_unix=start_unix,
        end_unix=end_unix,
        run_id=run_id,
        write_bundled=write_bundled,
        ingest_wait_s=ingest_wait_s,
        retries=retries,
        retry_pause_s=retry_pause_s,
    )


__all__ = [
    "AgentLoopRebuild",
    "DEFAULT_GAP_S",
    "DEFAULT_TEMPO_URL",
    "TempoSpan",
    "rebuild_from_tempo",
    "write_agent_loop_from_runs",
]
