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

"""Group Tempo spans into runs and build Sample→Session→Traj hierarchies."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from rl_insight.experimental.agent_loop.constants import SERVICE_NAME_VALUE
from rl_insight.experimental.agent_loop.read.client import (
    DEFAULT_TEMPO_URL,
    TempoClient,
    TempoSpan,
)

logger = logging.getLogger(__name__)

DEFAULT_GAP_S = 300.0


def _is_stamped_run_id(run_id: str) -> bool:
    return bool(run_id) and not (
        run_id.startswith("auto-") or run_id.startswith("derived-")
    )


def assign_run_keys(
    spans: list[TempoSpan],
    *,
    gap_s: float = DEFAULT_GAP_S,
) -> dict[str, list[TempoSpan]]:
    """Group spans by ``run_id`` attribute, else time-gap clusters.

    Gap is silence *between* runs (not run duration): sort by start, start a
    new cluster when ``start - prev_end > gap_s``.
    """
    by_attr: dict[str, list[TempoSpan]] = defaultdict(list)
    unlabeled: list[TempoSpan] = []
    for sp in spans:
        rid = sp.run_id.strip()
        if rid:
            by_attr[rid].append(sp)
        else:
            unlabeled.append(sp)

    if unlabeled:
        unlabeled.sort(key=lambda s: s.start_ns)
        gap_ns = int(gap_s * 1_000_000_000)
        cluster_idx = 0
        cluster: list[TempoSpan] = []
        prev_end = 0
        for sp in unlabeled:
            if cluster and sp.start_ns - prev_end > gap_ns:
                key = f"auto-{cluster_idx}-{cluster[0].start_ns // 1_000_000_000}"
                by_attr[key].extend(cluster)
                cluster_idx += 1
                cluster = []
            cluster.append(sp)
            prev_end = max(prev_end, sp.end_ns)
        if cluster:
            key = f"auto-{cluster_idx}-{cluster[0].start_ns // 1_000_000_000}"
            by_attr[key].extend(cluster)

    return dict(by_attr)


def complete_stamped_runs(
    runs: dict[str, list[TempoSpan]],
    *,
    tempo_url: str = DEFAULT_TEMPO_URL,
    service_name: str = SERVICE_NAME_VALUE,
    end_unix: int,
    lookback_s: int = 24 * 3600,
    client: TempoClient | None = None,
) -> dict[str, list[TempoSpan]]:
    """For mapper ``run_id``s seen in-window, pull the full run (all samples).

    Span timestamps are compressed across ~30m, so a short Grafana window can
    clip early samples; hierarchy should still show the whole export.
    """
    if client is not None:

        def _fetch(rid: str, start: int) -> list[TempoSpan]:
            return client.fetch_spans(
                start_unix=start,
                end_unix=end_unix,
                extra_traceql=f'span.run_id = "{rid}"',
            )

    else:
        from rl_insight.experimental.agent_loop.read.client import fetch_spans as _fetch_spans

        def _fetch(rid: str, start: int) -> list[TempoSpan]:
            return _fetch_spans(
                tempo_url=tempo_url,
                service_name=service_name,
                start_unix=start,
                end_unix=end_unix,
                extra_traceql=f'span.run_id = "{rid}"',
            )

    out = dict(runs)
    start = max(0, int(end_unix) - int(lookback_s))
    for rid in list(runs):
        if not _is_stamped_run_id(rid):
            continue
        full = _fetch(rid, start)
        if full:
            out[rid] = full
            logger.info(
                "completed run %s: window_partial=%s full=%s samples=%s",
                rid,
                len(runs[rid]),
                len(full),
                sorted({s.sample for s in full}),
            )
    return out


def _run_end_ns(spans: list[TempoSpan]) -> int:
    return max((s.end_ns for s in spans), default=0)


def _run_start_ns(spans: list[TempoSpan]) -> int:
    return min((s.start_ns for s in spans), default=0)


def span_overlaps_window(span: TempoSpan, start_unix: int, end_unix: int) -> bool:
    """True if span time range intersects ``[start_unix, end_unix]`` (unix seconds)."""
    start_ns = int(start_unix) * 1_000_000_000
    end_ns = int(end_unix) * 1_000_000_000
    return span.start_ns < end_ns and span.end_ns > start_ns


def filter_spans_to_window(
    spans: list[TempoSpan],
    start_unix: int,
    end_unix: int,
) -> list[TempoSpan]:
    return [s for s in spans if span_overlaps_window(s, start_unix, end_unix)]


def filter_runs_to_window(
    runs: dict[str, list[TempoSpan]],
    start_unix: int,
    end_unix: int,
) -> dict[str, list[TempoSpan]]:
    """Keep only spans overlapping the Grafana/Rebuild window; drop empty runs.

    Tempo search can return traces whose span timestamps fall outside ``from``/``to``.
    Building trees from those runs leaves empty Timeline panels. Strict window
    filtering keeps the dashboard tree aligned with what panels can query.
    """
    out: dict[str, list[TempoSpan]] = {}
    for rid, spans in runs.items():
        kept = filter_spans_to_window(spans, start_unix, end_unix)
        if kept:
            out[rid] = kept
        else:
            logger.info(
                "drop run %s: 0/%s spans overlap window [%s, %s]",
                rid,
                len(spans),
                start_unix,
                end_unix,
            )
    return out


def pick_latest_run_id(runs: dict[str, list[TempoSpan]]) -> str | None:
    if not runs:
        return None
    return max(runs.keys(), key=lambda rid: _run_end_ns(runs[rid]))


def hierarchy_from_spans(
    spans: list[TempoSpan],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Build Sample→Session→Traj dict used by the dashboard generator."""
    # sample -> session -> traj -> spans
    tree: dict[int, dict[int, dict[int, list[TempoSpan]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    uids: dict[int, str] = {}
    for sp in spans:
        tree[sp.sample][sp.session][sp.traj].append(sp)
        if sp.uid:
            uids[sp.sample] = sp.uid

    samples_out: list[dict[str, Any]] = []
    for si in sorted(tree.keys()):
        sessions_out: list[dict[str, Any]] = []
        for sess_i in sorted(tree[si].keys()):
            trajs_out: list[dict[str, Any]] = []
            for ti in sorted(tree[si][sess_i].keys()):
                turns = tree[si][sess_i][ti]
                turns.sort(key=lambda s: (s.start_ns, s.attrs.get("turn", "")))
                last = turns[-1] if turns else None
                finish = last.finish_reason if last else ""
                success = finish == "stop"
                trajs_out.append(
                    {
                        "trajectory_index": ti,
                        "num_turns": len(turns),
                        "success": success,
                        "finish_reason": finish,
                    }
                )
            sessions_out.append(
                {
                    "session_index": sess_i,
                    "trajectories": trajs_out,
                }
            )
        samples_out.append(
            {
                "sample_index": si,
                "uid": uids.get(si, ""),
                "sessions": sessions_out,
            }
        )
    return {"run_id": run_id, "samples": samples_out}


class RunHierarchyBuilder:
    """Group spans by run and build nested Sample→Session→Traj dicts."""

    def __init__(
        self,
        gap_s: float = DEFAULT_GAP_S,
        client: TempoClient | None = None,
    ) -> None:
        self.gap_s = gap_s
        self._client = client

    def group(self, spans: list[TempoSpan]) -> dict[str, list[TempoSpan]]:
        return assign_run_keys(spans, gap_s=self.gap_s)

    def build(self, spans: list[TempoSpan], run_id: str) -> dict[str, Any]:
        return hierarchy_from_spans(spans, run_id=run_id)

    def complete_stamped_runs(
        self,
        runs: dict[str, list[TempoSpan]],
        end_unix: int,
        lookback_s: int = 24 * 3600,
    ) -> dict[str, list[TempoSpan]]:
        if self._client is not None:
            return complete_stamped_runs(
                runs,
                end_unix=end_unix,
                lookback_s=lookback_s,
                client=self._client,
            )
        return complete_stamped_runs(
            runs,
            tempo_url=DEFAULT_TEMPO_URL,
            service_name=SERVICE_NAME_VALUE,
            end_unix=end_unix,
            lookback_s=lookback_s,
        )

    def filter_to_window(
        self,
        runs: dict[str, list[TempoSpan]],
        start_unix: int,
        end_unix: int,
    ) -> dict[str, list[TempoSpan]]:
        return filter_runs_to_window(runs, start_unix, end_unix)
