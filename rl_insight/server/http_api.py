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

"""HTTP API and remote helpers for the RL-Insight server."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Event, Thread
from urllib.parse import parse_qsl
from typing import Any, Sequence

import requests
import uvicorn
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response
from omegaconf import DictConfig, OmegaConf
from starlette.concurrency import run_in_threadpool

from ..protocol import RunContext, RunError, endpoint_description
from ..utils.constants import MonitorEnv, MonitorPaths, MonitorServer, PrometheusScrape
from ..utils.monitor_config_loader import load_server_config_file
from ..utils.prometheus_utils import PrometheusTarget, PrometheusTargetStore
from .network import format_host_port, local_addresses
from .registry import RunRegistry
from .collection import Collection

logger = logging.getLogger(__name__)


def server_url() -> str:
    """Return the configured RL-Insight server URL without a trailing slash."""
    return str(os.environ.get(MonitorEnv.SERVER_URL, "")).strip().rstrip("/")


def get_server_services() -> dict[str, Any]:
    """Fetch service endpoints from the RL-Insight server."""
    base_url = server_url()
    if not base_url:
        logger.error(
            "[rl-insight] RL-Insight server URL is required; set %s",
            MonitorEnv.SERVER_URL,
        )
        return {}

    url = f"{base_url}{MonitorServer.API_PREFIX}/services"
    last_error: requests.RequestException | ValueError | None = None
    for attempt in range(MonitorServer.SERVICE_DISCOVERY_RETRIES):
        try:
            response = requests.get(
                url,
                timeout=MonitorServer.SERVICE_DISCOVERY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(
                    f"services response must be an object, got {type(data).__name__}"
                )
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc

        if attempt + 1 < MonitorServer.SERVICE_DISCOVERY_RETRIES:
            time.sleep(MonitorServer.SERVICE_DISCOVERY_RETRY_DELAY_SECONDS)

    logger.error(
        "[rl-insight] Failed to fetch RL-Insight server services at %s: %s",
        url,
        last_error,
    )
    return {}


def create_app(conf: DictConfig) -> FastAPI:
    """Create the RL-Insight server application."""
    registry = RunRegistry(
        Path(
            str(
                OmegaConf.select(
                    conf, "server.data_dir", default=MonitorPaths.STATE_ROOT / "data"
                )
            )
        ),
        lease_ttl=float(OmegaConf.select(conf, "server.endpoint_ttl", default=30)),
    )

    collection = Collection(registry)
    stop_projection = Event()
    store = PrometheusTargetStore.from_config(conf)

    def project():
        address = format_host_port(
            local_addresses()["loopback"],
            int(OmegaConf.select(conf, "server.port", default=18080)),
        )
        store.project_runs(registry.endpoints(), address, registry.ended_runs())

    def projection_loop():
        while not stop_projection.wait(5):
            try:
                project()
            except (OSError, ValueError):
                logger.exception("Run discovery projection failed")

    @asynccontextmanager
    async def lifespan(_app):
        try:
            await run_in_threadpool(refresh_views)
            await run_in_threadpool(project)
            thread = Thread(
                target=projection_loop, name="insight-run-discovery", daemon=True
            )
            thread.start()
            yield
        finally:
            stop_projection.set()
            if "thread" in locals():
                await run_in_threadpool(thread.join, 6)
            await run_in_threadpool(collection.close)
            registry.close()

    app = FastAPI(title="RL-Insight server", version="0.1.0", lifespan=lifespan)
    install_queries(app, registry, conf)
    refresh_views = install_views(app, registry, conf)

    @app.exception_handler(RunError)
    async def run_error(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.post("/api/v1/runs")
    def create_run(
        request: Request,
        background: BackgroundTasks,
        payload: dict[str, Any] = Body(...),
    ):
        context = registry.create(
            payload.get("request_id", ""),
            payload.get("project", "uncategorized"),
            payload.get("experiment_name", "unnamed"),
            str(OmegaConf.select(conf, "server.public_url") or request.base_url),
            payload.get("config", {}),
            payload.get("owner_token", ""),
        )
        background.add_task(refresh_views, str(request.base_url))
        return context.to_dict()

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str):
        return registry.snapshot(run_id)

    @app.post("/api/v1/runs/{run_id}/endpoints")
    def register_endpoint(run_id: str, payload: dict[str, Any] = Body(...)):
        description = endpoint_description(registry.get(run_id), payload)
        registry.assert_open(run_id)
        collection.verify(description)
        lease = registry.register_endpoint(description)
        project()
        return lease

    @app.post("/api/v1/runs/{run_id}/endpoints/{lease_id}")
    def renew_endpoint(run_id: str, lease_id: str):
        return registry.renew_endpoint(run_id, lease_id)

    @app.delete("/api/v1/runs/{run_id}/endpoints/{lease_id}")
    def revoke_endpoint(run_id: str, lease_id: str):
        registry.revoke_endpoint(run_id, lease_id)
        project()
        return {"status": "ok"}

    @app.get("/api/v1/runs/{run_id}/metrics")
    def collect_metrics(run_id: str, request: Request):
        timeout = 9.0
        supplied = request.headers.get("X-Prometheus-Scrape-Timeout-Seconds")
        if supplied is not None:
            try:
                seconds = float(supplied)
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError
                timeout = min(timeout, seconds - min(1.0, seconds * 0.2))
            except ValueError as exc:
                raise RunError("Invalid scrape timeout") from exc
        code, body = collection.collect(run_id, timeout)
        return Response(body, status_code=code, media_type="text/plain; version=0.0.4")

    @app.post("/api/v1/runs/{run_id}/close")
    def close_run(run_id: str, payload: dict[str, Any] = Body(...)):
        state = "closing" if payload.get("prepare") is True else "ended"
        registry.transition(
            run_id,
            payload.get("owner_token", ""),
            state,
            payload.get("final_collection_confirmed"),
        )
        project()
        return {"status": state}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Return a lightweight liveness response for process checks."""
        return {"status": "ok"}

    @app.get(f"{MonitorServer.API_PREFIX}/services")
    def services() -> dict[str, Any]:
        """Expose enabled component ports for trainer-side endpoint discovery."""
        return {
            "status": "ok",
            "run_context_version": 1,
            "endpoint_registration_version": 1,
            "otlp_port": (
                int(OmegaConf.select(conf, "otel.otel_port"))
                if bool(OmegaConf.select(conf, "tempo.enable", default=True))
                else None
            ),
            "prometheus_port": (
                int(OmegaConf.select(conf, "prometheus.prometheus_port"))
                if bool(OmegaConf.select(conf, "prometheus.enable", default=True))
                else None
            ),
            "grafana_port": (
                int(OmegaConf.select(conf, "grafana.port"))
                if bool(OmegaConf.select(conf, "grafana.enable", default=True))
                else None
            ),
        }

    @app.post(f"{MonitorServer.API_PREFIX}/prometheus/targets")
    def register_prometheus_targets(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Register metric scrape targets into the runtime Prometheus config."""
        raw_targets = payload.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="targets must be a non-empty list",
            )

        default_labels = payload.get("labels") or {}
        if not isinstance(default_labels, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="labels must be an object",
            )

        targets: list[PrometheusTarget] = []
        # Apply request-level labels first, then let each target override them.
        for item in raw_targets:
            if isinstance(item, str):
                targets.append(
                    PrometheusTarget(target=item, labels=dict(default_labels))
                )
                continue
            if isinstance(item, dict):
                item_labels = item.get("labels") or {}
                if not isinstance(item_labels, dict):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="target labels must be an object",
                    )
                targets.append(
                    PrometheusTarget(
                        target=str(item.get("target")),
                        labels={**default_labels, **item_labels},
                    )
                )
                continue
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="each target must be either a string or an object",
            )

        job_name = str(payload.get("job_name") or PrometheusScrape.TRAINER_METRICS_JOB)
        if "run_context" in payload:
            supplied = RunContext.from_dict(payload["run_context"])
            context = registry.get(supplied.run_id)
            registry.assert_open(context.run_id, allow_closing=False)
            if context != supplied:
                raise RunError("RunContext conflicts with registered metadata", 409)
            targets = [
                PrometheusTarget(target.target, context.labels(target.labels))
                for target in targets
            ]
            if payload.get("identity_in_metrics") is True:
                # Hub samples already have run_id; avoid duplicate exported_run_id labels.
                targets = [
                    PrometheusTarget(
                        target.target,
                        {k: v for k, v in target.labels.items() if k != "run_id"},
                    )
                    for target in targets
                ]
        active = {
            item["metrics_url"].split("://", 1)[1].split("/", 1)[0]: item
            for item in registry.endpoints()
        }
        for target in targets:
            if (
                target.target in active
                and payload.get("run_context", {}).get("run_id")
                != active[target.target]["run_id"]
            ):
                raise RunError("Target already belongs to a proved publisher", 409)
            if target.target in active:
                registry.annotate_endpoint(
                    active[target.target]["lease_id"],
                    {"job": job_name, **target.labels},
                )
        targets = [target for target in targets if target.target not in active]
        try:
            result = store.register(job_name, targets)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        reloaded = False
        try:
            reloaded = store.reload()
        except requests.RequestException as exc:
            logger.warning(
                "[rl-insight] Failed to reload Prometheus after target update: %s",
                exc,
            )
        return {"status": "ok", "prometheus_reloaded": reloaded, **result}

    return app


MAX_BYTES = 8 * 1024 * 1024
MAX_SPANS = 10000

_TOKEN = re.compile(
    r'\s+|"(?:[^"\\]|\\.)*"|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?(?:ms|[smhdwy])?|[A-Za-z_:][A-Za-z0-9_:]*|=~|!~|!=|==|>=|<=|[-+*/%^(),{}\[\]@=<>]'
)
_AGG = {
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "count_values",
    "stddev",
    "stdvar",
    "topk",
    "bottomk",
    "quantile",
    "group",
}
_GROUP = {"by", "without", "on", "ignoring", "group_left", "group_right"}


def _tokens(query):
    result = []
    pos = 0
    while pos < len(query):
        match = _TOKEN.match(query, pos)
        if match is None:
            raise RunError("Unsupported PromQL token", 422)
        if not match.group().isspace():
            result.append((match.group(), match.start(), match.end()))
        pos = match.end()
    return result


def prometheus(query: str, identity: RunContext) -> str:
    """Constrain every metric/brace selector, including nested function args."""
    if len(query) > 65536:
        raise RunError("Query too large", 413)
    tokens = _tokens(query)
    if not tokens:
        return "{run_id=" + json.dumps(identity.run_id) + "}"
    edits = []
    index = 0
    matcher = "run_id=" + json.dumps(identity.run_id, ensure_ascii=True)
    operand_expected = True
    while index < len(tokens):
        value, _start, end = tokens[index]
        if value in _GROUP and index + 1 < len(tokens) and tokens[index + 1][0] == "(":
            index += 2
            while index < len(tokens) and tokens[index][0] != ")":
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|,", tokens[index][0]):
                    raise RunError("Unsupported query", 422)
                index += 1
        elif value == "{":
            close = index + 1
            while close < len(tokens) and tokens[close][0] != "}":
                close += 1
            if close == len(tokens):
                raise RunError("Unsupported query", 422)
            content = tokens[index + 1 : close]
            cursor = 0
            while cursor < len(content):
                if cursor + 2 >= len(content):
                    raise RunError("Unsupported query", 422)
                label, operator, literal = (
                    item[0] for item in content[cursor : cursor + 3]
                )
                if (
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label)
                    or operator not in {"=", "!=", "=~", "!~"}
                    or not literal.startswith('"')
                ):
                    raise RunError("Unsupported query", 422)
                if label == "run_id" and (
                    operator != "=" or json.loads(literal) != identity.run_id
                ):
                    raise RunError("Query run_id conflicts with selected Run", 403)
                cursor += 3
                if cursor < len(content):
                    if content[cursor][0] != ",":
                        raise RunError("Unsupported query", 422)
                    cursor += 1
            insertion = ("," if content else "") + matcher
            edits.append((tokens[close][1], tokens[close][1], insertion))
            index = close
            operand_expected = False
        elif re.fullmatch(r"[A-Za-z_:][A-Za-z0-9_:]*", value):
            following = tokens[index + 1][0] if index + 1 < len(tokens) else ""
            previous = tokens[index - 1][0] if index else ""
            # Keyword spelling does not make an identifier syntax: `sum`, `min`
            # and `offset` can also be metric names. Recognize their actual role.
            aggregate = value in _AGG and (
                following == "(" or following in {"by", "without"}
            )
            binary = value in {"and", "or", "unless"} and not operand_expected
            modifier = value == "bool" and previous in {
                "==",
                "!=",
                ">",
                "<",
                ">=",
                "<=",
            }
            duration = (
                tokens[index + 2][0]
                if following == "-" and index + 2 < len(tokens)
                else following
            )
            offset = (
                value == "offset"
                and index > 0
                and re.fullmatch(r"[0-9]+(?:ms|[smhdwy])", duration)
            )
            if following not in {"(", "{"} and not (
                aggregate or binary or modifier or offset
            ):
                edits.append((end, end, "{" + matcher + "}"))
                operand_expected = False
            elif binary or aggregate or following == "(":
                operand_expected = True
        elif value in {
            "(",
            ",",
            "+",
            "-",
            "*",
            "/",
            "%",
            "^",
            "==",
            "!=",
            ">",
            "<",
            ">=",
            "<=",
        }:
            operand_expected = True
        elif value in {")", "]"} or value.startswith('"') or value[0].isdigit():
            operand_expected = False
        index += 1
    for left, right, replacement in reversed(edits):
        query = query[:left] + replacement + query[right:]
    return query


def trace_search(query: str, identity: RunContext) -> str:
    return traces(query, identity) + (
        " | select(resource.rl_insight.run_id, name, span.state_lane_id)"
    )


def traces(query: str, identity: RunContext) -> str:
    """Boolean combinations of spansets; structural/pipeline syntax rejected."""
    if len(query) > 65536:
        raise RunError("Query too large", 413)
    result, cursor, count = [], 0, 0
    condition = "resource.rl_insight.run_id = " + json.dumps(identity.run_id)
    while cursor < len(query):
        char = query[cursor]
        if char == "{":
            end = cursor + 1
            quoted = escaped = False
            while end < len(query):
                token = query[end]
                if escaped:
                    escaped = False
                elif token == "\\" and quoted:
                    escaped = True
                elif token == '"':
                    quoted = not quoted
                elif token == "}" and not quoted:
                    break
                elif token in "{}|" and not quoted:
                    raise RunError("Unsupported query", 422)
                end += 1
            if end == len(query):
                raise RunError("Unsupported query", 422)
            inner = query[cursor + 1 : end].strip()
            # Existing identity conditions intersect the mandatory equality.
            result.append("{ " + (f"({inner}) && " if inner else "") + condition + " }")
            cursor = end + 1
            count += 1
        elif char.isspace() or char in "()":
            result.append(char)
            cursor += 1
        elif query[cursor : cursor + 2] in {"&&", "||"}:
            result.append(query[cursor : cursor + 2])
            cursor += 2
        else:
            raise RunError("Unsupported TraceQL operator", 422)
    if not count:
        return "{ " + condition + " }" if not query.strip() else _unsupported()
    return "".join(result)


def _unsupported():
    raise RunError("Unsupported query", 422)


def read_backend(url: str, fields: list[tuple[str, str]], *, auth=None, headers=None):
    """Bounded JSON reads from deployment-controlled services only."""
    deadline = time.monotonic() + 10
    with requests.Session() as session:
        session.trust_env = False
        try:
            with session.get(
                url,
                params=fields,
                auth=auth,
                headers=headers,
                timeout=(3, 3),
                allow_redirects=False,
                stream=True,
            ) as response:
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > MAX_BYTES or time.monotonic() >= deadline:
                        raise RunError("Backend response exceeds query budget", 502)
                if 300 <= response.status_code < 400:
                    raise RunError("Backend redirect rejected", 502)
                return response.status_code, bytes(body)
        except requests.RequestException as exc:
            raise RunError("Backend query failed", 502) from exc


def project_search(payload, context: RunContext) -> int:
    """Keep only proven Run-local spans, never trace-wide root metadata."""
    required, output = 0, []
    for trace in payload.get("traces", []):
        groups_out, members = [], []
        for group in trace.get("spanSets") or [trace.get("spanSet", {})]:
            spans = group["spans"]
            matched = int(group["matched"])
            if matched < len(spans) or matched > MAX_SPANS:
                raise RunError("Trace search exceeds span budget", 502)
            if matched > len(spans):
                required = max(required, matched)
            for span in spans:
                pairs = [
                    a
                    for a in span.get("attributes", [])
                    if a.get("key") == "rl_insight.run_id"
                ]
                if (
                    len(pairs) != 1
                    or pairs[0].get("value", {}).get("stringValue") != context.run_id
                ):
                    raise RunError(
                        "Backend returned a foreign or unidentified span", 502
                    )
                # Ownership is checked above; keep the original timeline fields.
                span["attributes"] = [
                    a
                    for a in span.get("attributes", [])
                    if a.get("key") != "rl_insight.run_id"
                ]
            groups_out.append({"spans": spans, "matched": matched})
            members.extend(spans)
        if not members:
            raise RunError("Backend trace has no identifiable spans", 502)
        if not re.fullmatch(r"[0-9a-fA-F]{1,32}", trace["traceID"]):
            raise RunError("Invalid backend trace ID", 502)
        start = min(int(s["startTimeUnixNano"]) for s in members)
        end = max(
            int(s["startTimeUnixNano"]) + int(s["durationNanos"]) for s in members
        )
        output.append(
            {
                "traceID": trace["traceID"].zfill(32),
                "rootTraceName": members[0]["name"],
                "rootServiceName": "",
                "startTimeUnixNano": str(start),
                "durationMs": max(0, end - start) // 1000000,
                "spanSets": groups_out,
                "spanSet": groups_out[0],
            }
        )
    payload["traces"] = output
    return required


def trace_owned(payload, context: RunContext) -> bool:
    resources = payload.get("batches", payload.get("resourceSpans", []))
    if not resources:
        return False
    for resource in resources:
        pairs = [
            a
            for a in resource.get("resource", {}).get("attributes", [])
            if a.get("key") == "rl_insight.run_id"
        ]
        if (
            len(pairs) != 1
            or pairs[0].get("value", {}).get("stringValue") != context.run_id
        ):
            return False
    return True


def install_queries(app, registry, conf):
    host = local_addresses()["loopback"]
    endpoints = {
        "prometheus": "http://"
        + format_host_port(
            host,
            int(OmegaConf.select(conf, "prometheus.prometheus_port", default=9090)),
        ),
        "tempo": "http://"
        + format_host_port(
            host, int(OmegaConf.select(conf, "tempo.query_port", default=3200))
        ),
    }

    def query(run_id, backend, path, fields):
        context = registry.get(run_id)
        if not bool(OmegaConf.select(conf, backend + ".enable", default=True)):
            raise RunError("Backend disabled", 503)
        if backend == "prometheus":
            if path not in {
                "query",
                "query_range",
                "series",
                "labels",
            } and not re.fullmatch(r"label/[A-Za-z_][A-Za-z0-9_]*/values", path):
                raise RunError("Unsupported query path", 404)
            field = "query" if path in {"query", "query_range"} else "match[]"
            values = [v for k, v in fields if k == field]
            if field == "query" and len(values) != 1:
                raise RunError("Expected one query expression")
            fields = [(k, v) for k, v in fields if k != field] + [
                (field, prometheus(v, context)) for v in (values or [""])
            ]
            url = endpoints[backend] + "/api/v1/" + path
        else:
            if path == "search":
                values = [v for k, v in fields if k == "q"]
                if len(values) > 1 or any(k == "tags" for k, _ in fields):
                    raise RunError("Unsupported search options", 422)
                budgets = [v for k, v in fields if k == "spss"]
                if len(budgets) > 1 or (
                    budgets
                    and (
                        not re.fullmatch(r"[0-9]{1,5}", budgets[0])
                        or not 1 <= int(budgets[0]) <= 10000
                    )
                ):
                    raise RunError("Invalid span search budget", 422)
                budget = max(1000, int(budgets[0]) if budgets else 1000)
                fields = [(k, v) for k, v in fields if k not in {"q", "spss"}] + [
                    ("q", trace_search(values[0] if values else "", context)),
                    ("spss", str(budget)),
                ]
            elif not re.fullmatch(r"traces/[0-9a-fA-F]{32}", path):
                raise RunError("Unsupported trace query path", 404)
            url = endpoints[backend] + "/api/" + path
        status, body = read_backend(url, fields, headers={"Accept": "application/json"})
        if backend == "tempo" and 200 <= status < 300:
            try:
                payload = json.loads(body)
                if path == "search":
                    needed = project_search(payload, context)
                    if needed:
                        fields = [(k, v) for k, v in fields if k != "spss"] + [
                            ("spss", str(needed))
                        ]
                        status, body = read_backend(
                            url, fields, headers={"Accept": "application/json"}
                        )
                        if not 200 <= status < 300:
                            raise RunError("Complete trace search failed", 502)
                        payload = json.loads(body)
                        if project_search(payload, context):
                            raise RunError("Trace search is incomplete", 502)
                elif not trace_owned(payload, context):
                    raise RunError("Trace not found in this Run", 404)
                body = json.dumps(payload).encode()
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                if isinstance(exc, RunError):
                    raise
                raise RunError("Invalid backend trace response", 502) from exc
        return Response(body, status_code=status, media_type="application/json")

    @app.get("/api/v1/runs/{run_id}/prometheus/api/v1/{path:path}")
    def metrics(run_id: str, path: str, request: Request):
        return query(
            run_id, "prometheus", path, list(request.query_params.multi_items())
        )

    @app.post("/api/v1/runs/{run_id}/prometheus/api/v1/{path:path}")
    async def metrics_post(run_id: str, path: str, request: Request):
        if (
            request.headers.get("content-type", "").split(";")[0]
            != "application/x-www-form-urlencoded"
        ):
            raise RunError("Expected form-encoded query", 415)
        import asyncio

        async def read_form():
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 65536:
                    raise RunError("Query body too large", 413)
            return body.decode("utf-8")

        try:
            body = await asyncio.wait_for(read_form(), timeout=5)
        except asyncio.TimeoutError as exc:
            raise RunError("Query body timed out", 408) from exc
        except UnicodeError as exc:
            raise RunError("Invalid query encoding") from exc
        fields = list(request.query_params.multi_items()) + parse_qsl(
            body, keep_blank_values=True
        )
        return await run_in_threadpool(query, run_id, "prometheus", path, fields)

    @app.get("/api/v1/runs/{run_id}/tempo/api/{path:path}")
    def tempo(run_id: str, path: str, request: Request):
        return query(run_id, "tempo", path, list(request.query_params.multi_items()))


def install_views(app, registry, conf):
    from .runtime import DEFAULT_TEMPLATE, prepare_run_view

    lock = Lock()

    def prepare(run_id, base_url, template):
        with lock:
            return prepare_run_view(registry, conf, run_id, base_url, template)

    def refresh(base_url=None):
        runs = registry.list_runs()
        if runs and bool(OmegaConf.select(conf, "grafana.enable", default=True)):
            try:
                prepare(
                    runs[0].run_id, base_url or runs[0].server_url, DEFAULT_TEMPLATE
                )
            except (RunError, OSError) as exc:
                logger.warning(
                    "[rl-insight] Could not refresh the original dashboard: %s", exc
                )

    @app.get("/api/v1/runs/{run_id}/view")
    def view(run_id: str, request: Request, template: str = DEFAULT_TEMPLATE):
        return prepare(run_id, str(request.base_url), template)

    @app.get("/ui/runs/{run_id}")
    def page(run_id: str, request: Request, template: str = DEFAULT_TEMPLATE):
        return RedirectResponse(prepare(run_id, str(request.base_url), template)["url"])

    return refresh


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl_insight.server.http_api")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved server config YAML used by the RL-Insight server.",
    )
    args = parser.parse_args(argv)

    conf = load_server_config_file(args.config)
    port = int(OmegaConf.select(conf, "server.port", default=18080))
    uvicorn.run(create_app(conf), host=local_addresses()["bind"], port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
