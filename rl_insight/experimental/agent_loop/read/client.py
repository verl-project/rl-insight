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

"""Tempo HTTP client: search + fetch agent-loop state spans."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from rl_insight.experimental.agent_loop.constants import SERVICE_NAME_VALUE

logger = logging.getLogger(__name__)

DEFAULT_TEMPO_URL = "http://127.0.0.1:3200"


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


class TempoClient:
    """HTTP client for Tempo search + per-trace span fetch."""

    def __init__(
        self,
        tempo_url: str = DEFAULT_TEMPO_URL,
        service_name: str = SERVICE_NAME_VALUE,
    ) -> None:
        self.tempo_url = tempo_url
        self.service_name = service_name

    def fetch_spans(
        self,
        *,
        start_unix: int | None = None,
        end_unix: int | None = None,
        limit: int = 5000,
        extra_traceql: str = "",
    ) -> list[TempoSpan]:
        return fetch_spans(
            tempo_url=self.tempo_url,
            service_name=self.service_name,
            start_unix=start_unix,
            end_unix=end_unix,
            limit=limit,
            extra_traceql=extra_traceql,
        )
