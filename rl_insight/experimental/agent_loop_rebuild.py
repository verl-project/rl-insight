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

"""Rebuild Agent Loop dashboard from Tempo only (mapper is a black box).

Reads spans via Tempo HTTP search, groups by ``run_id`` (or time-gap clusters
when the attribute is missing), then writes nested Grafana panels. Details /
row-title stats use the selected (default: latest) run only; overview
timelines may include all runs in the window.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rl_insight.experimental.agent_loop_constants import SERVICE_NAME_VALUE
from rl_insight.experimental.generate_agent_loop_dashboard import (
    write_agent_loop_from_runs,
)

logger = logging.getLogger(__name__)

DEFAULT_TEMPO_URL = "http://127.0.0.1:3200"
DEFAULT_GAP_S = 300.0


@dataclass
class TempoSpan:
    """Minimal span view for visualization (attributes as seen in Tempo)."""

    name: str
    start_ns: int
    end_ns: int
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.attrs.get("run_id", "")

    @property
    def sample(self) -> int:
        return int(self.attrs.get("sample", "0") or 0)

    @property
    def session(self) -> int:
        return int(self.attrs.get("session", "0") or 0)

    @property
    def traj(self) -> int:
        return int(self.attrs.get("traj", "0") or 0)

    @property
    def uid(self) -> str:
        return self.attrs.get("uid", "")

    @property
    def finish_reason(self) -> str:
        return self.attrs.get("finish_reason") or self.attrs.get("state_name") or self.name

    @property
    def reward(self) -> str:
        return self.attrs.get("reward", "")


def _http_json(url: str, *, timeout_s: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _attr_map(span_json: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in span_json.get("attributes") or []:
        key = item.get("key")
        if not key:
            continue
        val = item.get("value") or {}
        if "stringValue" in val:
            out[key] = str(val["stringValue"])
        elif "intValue" in val:
            out[key] = str(val["intValue"])
        elif "doubleValue" in val:
            out[key] = str(val["doubleValue"])
        elif "boolValue" in val:
            out[key] = str(val["boolValue"]).lower()
    return out


def _parse_trace_spans(trace_json: dict[str, Any]) -> list[TempoSpan]:
    spans: list[TempoSpan] = []
    batches = trace_json.get("batches") or []
    for batch in batches:
        for scope_spans in batch.get("scopeSpans") or []:
            for sp in scope_spans.get("spans") or []:
                start = int(sp.get("startTimeUnixNano") or 0)
                end = int(sp.get("endTimeUnixNano") or start)
                spans.append(
                    TempoSpan(
                        name=str(sp.get("name") or ""),
                        start_ns=start,
                        end_ns=end,
                        attrs=_attr_map(sp),
                    )
                )
    # Jaeger-style fallback (some Tempo builds)
    if not spans and "resourceSpans" in trace_json:
        for rs in trace_json["resourceSpans"]:
            for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
                for sp in ss.get("spans") or []:
                    start = int(sp.get("startTimeUnixNano") or 0)
                    end = int(sp.get("endTimeUnixNano") or start)
                    spans.append(
                        TempoSpan(
                            name=str(sp.get("name") or ""),
                            start_ns=start,
                            end_ns=end,
                            attrs=_attr_map(sp),
                        )
                    )
    return spans


def fetch_spans(
    *,
    tempo_url: str = DEFAULT_TEMPO_URL,
    service_name: str = SERVICE_NAME_VALUE,
    start_unix: int | None = None,
    end_unix: int | None = None,
    limit: int = 5000,
    extra_traceql: str = "",
) -> list[TempoSpan]:
    """Search Tempo for agent-loop spans in ``[start_unix, end_unix]``."""
    now = int(time.time())
    end = end_unix if end_unix is not None else now
    start = start_unix if start_unix is not None else end - 3 * 3600
    q = f'{{resource.service.name="{service_name}"'
    if extra_traceql:
        q += f" && {extra_traceql}"
    q += "}"
    params = urllib.parse.urlencode(
        {"q": q, "start": str(start), "end": str(end), "limit": str(limit)}
    )
    base = tempo_url.rstrip("/")
    search_url = f"{base}/api/search?{params}"
    try:
        payload = _http_json(search_url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Tempo search failed: {search_url} ({exc})") from exc

    traces = payload.get("traces") or []
    spans: list[TempoSpan] = []
    seen: set[str] = set()
    for tr in traces:
        tid = tr.get("traceID") or tr.get("traceId")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        try:
            trace = _http_json(f"{base}/api/traces/{tid}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("skip trace %s: %s", tid, exc)
            continue
        for sp in _parse_trace_spans(trace):
            # Keep only state_interval / agent-loop shaped spans when possible.
            seg = sp.attrs.get("monitor.trace_segment", "")
            if seg and seg != "state_interval":
                continue
            if "sample" not in sp.attrs and "state_lane_id" not in sp.attrs:
                continue
            spans.append(sp)
    logger.info(
        "fetched %s spans from %s traces (service=%s window=%s..%s extra=%r)",
        len(spans),
        len(seen),
        service_name,
        start,
        end,
        extra_traceql,
    )
    return spans


def _is_stamped_run_id(run_id: str) -> bool:
    return bool(run_id) and not (
        run_id.startswith("auto-") or run_id.startswith("derived-")
    )


def complete_stamped_runs(
    runs: dict[str, list[TempoSpan]],
    *,
    tempo_url: str = DEFAULT_TEMPO_URL,
    service_name: str = SERVICE_NAME_VALUE,
    end_unix: int,
    lookback_s: int = 24 * 3600,
) -> dict[str, list[TempoSpan]]:
    """For mapper ``run_id``s seen in-window, pull the full run (all samples).

    Span timestamps are compressed across ~30m, so a short Grafana window can
    clip early samples; hierarchy should still show the whole export.
    """
    out = dict(runs)
    start = max(0, int(end_unix) - int(lookback_s))
    for rid in list(runs):
        if not _is_stamped_run_id(rid):
            continue
        full = fetch_spans(
            tempo_url=tempo_url,
            service_name=service_name,
            start_unix=start,
            end_unix=end_unix,
            extra_traceql=f'span.run_id = "{rid}"',
        )
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


def _run_end_ns(spans: list[TempoSpan]) -> int:
    return max((s.end_ns for s in spans), default=0)


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
                reward = ""
                for t in reversed(turns):
                    if t.reward:
                        reward = t.reward
                        break
                success = finish == "stop"
                trajs_out.append(
                    {
                        "trajectory_index": ti,
                        "num_turns": len(turns),
                        "reward": reward,
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
    if ingest_wait_s > 0:
        time.sleep(ingest_wait_s)

    now = int(time.time())
    end = end_unix if end_unix is not None else now
    start = start_unix if start_unix is not None else end - 3 * 3600

    runs: dict[str, list[TempoSpan]] = {}
    last_err = "no spans"
    for attempt in range(max(1, retries)):
        spans = fetch_spans(
            tempo_url=tempo_url,
            service_name=service_name,
            start_unix=start,
            end_unix=end,
        )
        runs = assign_run_keys(spans, gap_s=gap_s)
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
            service_name,
            start,
            end,
            last_err,
        )
        stats = write_agent_loop_from_runs(
            [],
            service_name=service_name,
            write_bundled=write_bundled,
            window_from=start,
            window_to=end,
        )
        return {
            "status": "ok",
            "empty": True,
            "from": start,
            "to": end,
            "service_name": service_name,
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
    # refill each stamped run_id to the full sample set.
    runs = complete_stamped_runs(
        runs,
        tempo_url=tempo_url,
        service_name=service_name,
        end_unix=end,
    )

    if run_id:
        ordered_ids = [run_id]
    else:
        ordered_ids = sorted(runs.keys(), key=lambda rid: _run_end_ns(runs[rid]))

    run_summaries = [
        {
            "run_id": rid,
            "spans": len(runs[rid]),
            "end_ns": _run_end_ns(runs[rid]),
            "start_ns": min((s.start_ns for s in runs[rid]), default=0),
        }
        for rid in ordered_ids
    ]

    hierarchies = [
        hierarchy_from_spans(runs[rid], run_id=rid) for rid in ordered_ids
    ]
    stats = write_agent_loop_from_runs(
        hierarchies,
        service_name=service_name,
        write_bundled=write_bundled,
        window_from=start,
        window_to=end,
    )
    return {
        "status": "ok",
        "from": start,
        "to": end,
        "service_name": service_name,
        "selected_run_id": run_id,
        # Panel stats also has a numeric "runs" key — keep summaries as a list.
        "runs": run_summaries,
        "run_count": int(stats.get("runs") or len(run_summaries)),
        "samples": stats.get("samples", 0),
        "sessions": stats.get("sessions", 0),
        "trajectories": stats.get("trajectories", 0),
        "panels": stats.get("panels", 0),
    }
