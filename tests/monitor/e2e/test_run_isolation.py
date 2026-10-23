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

"""Representative Run flows through real HTTP and Ray, with protocol backends."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from rl_insight import __version__
from rl_insight.server.http_api import create_app
from rl_insight.utils.protocol import RunContext


@contextmanager
def serve(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 10
        try:
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.01)
            yield url
        finally:
            server.should_exit = True
            thread.join(10)


@pytest.fixture
def stack(tmp_path):
    backend = FastAPI()
    state = SimpleNamespace(
        ids=[],
        span_attributes=[],
        lose_response=False,
    )

    @backend.get("/{path:path}")
    def query(path: str, request: Request):
        fields = dict(request.query_params)
        if path == "api/v1/label/run_id/values":
            return {"status": "success", "data": state.ids}
        expression = fields.get("query") or fields.get("q", "")
        selected = [run_id for run_id in state.ids if run_id in expression]
        if path.startswith("api/v1/"):
            return {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {"metric": {"run_id": run_id}, "value": [1, "1"]}
                        for run_id in selected
                    ],
                },
            }
        if path == "api/search":
            return {
                "traces": [
                    {
                        "traceID": str(i + 1).zfill(32),
                        "spanSet": {
                            "matched": 1,
                            "spans": [
                                {
                                    "name": "actor_update",
                                    "startTimeUnixNano": "1000000000",
                                    "durationNanos": "1000000",
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": v}}
                                        for k, v in {
                                            "rl_insight.run_id": run_id,
                                            "state_lane_id": "actor_0",
                                        }.items()
                                    ],
                                }
                            ],
                        },
                    }
                    for i, run_id in enumerate(state.ids)
                    if run_id in selected
                ]
            }
        if path.startswith("api/traces/"):
            return {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [
                                {
                                    "key": "rl_insight.run_id",
                                    "value": {"stringValue": state.ids[0]},
                                }
                            ]
                        }
                    }
                ]
            }
        return JSONResponse({}, status_code=404)

    @backend.post("/{path:path}")
    async def ingest(path: str, request: Request):
        if path == "v1/traces":
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )

            batch = ExportTraceServiceRequest.FromString(await request.body())
            for resource in batch.resource_spans:
                attrs = {
                    a.key: a.value.string_value for a in resource.resource.attributes
                }
                state.span_attributes.extend(
                    {
                        "run_id": attrs.get("rl_insight.run_id"),
                        "name": span.name,
                        "attributes": {
                            attribute.key: attribute.value.string_value
                            for attribute in span.attributes
                        },
                    }
                    for scope in resource.scope_spans
                    for span in scope.spans
                )
        return {}

    with serve(backend) as backend_url:
        port = urlsplit(backend_url).port
        conf = OmegaConf.create(
            {
                "server": {
                    "data_dir": str(tmp_path / "data"),
                    "runtime_dir": str(tmp_path / "runtime"),
                },
                "prometheus": {"prometheus_port": port},
                "tempo": {"query_port": port},
                "otel": {"otel_port": port},
                "grafana": {"enable": False},
            }
        )
        app = create_app(conf)

        @app.middleware("http")
        async def lost_response(request, call_next):
            response = await call_next(request)
            if request.method == "POST" and request.url.path == "/api/v1/runs":
                if state.lose_response:
                    state.lose_response = False
                    return JSONResponse(
                        {"detail": "retry after committed registration"},
                        status_code=503,
                    )
            return response

        with serve(app) as url:
            yield SimpleNamespace(url=url, conf=conf, backend=state, path=tmp_path)


def test_same_named_runs_support_selection_isolation_and_catalog_restart(stack):
    ids = [RunContext.create("p", "same", stack.url).run_id for _ in range(3)]
    stack.backend.ids = ids
    for i, run_id in enumerate(ids):
        payload = {
            "version": __version__,
            "run_id": run_id,
            "project": "p" if i < 2 else "other",
            "experiment_name": "same",
        }
        response = requests.post(stack.url + "/api/v1/runs", json=payload, timeout=5)
        assert response.status_code == 200
        assert response.json()["run_id"] == run_id
        assert response.json()["version"] == __version__
        assert (
            requests.post(
                stack.url + "/api/v1/runs",
                json={**payload, "project": "conflict"},
                timeout=5,
            ).status_code
            == 409
        )
        assert (
            requests.post(stack.url + "/api/v1/runs", json=payload, timeout=5).json()
            == response.json()
        )
    base = stack.url + "/api/v1/query/"
    for chosen in (ids[:1], ids[:2], ids):
        literal = "(" + "|".join(chosen) + ")"
        metrics = requests.get(
            base + "prometheus/api/v1/query",
            params={"query": 'reward{run_id=~"' + literal + '"}'},
            timeout=5,
        )
        assert {
            sample["metric"]["run_id"] for sample in metrics.json()["data"]["result"]
        } == set(chosen)
        traces = requests.get(
            base + "tempo/api/search",
            params={"q": '{resource.rl_insight.run_id=~"' + literal + '"}'},
            timeout=5,
        )
        assert len(traces.json()["traces"]) == len(chosen)
    for query in ("reward", 'reward{run_id=~".*"}', 'reward{run_id!=""}'):
        assert requests.get(
            base + "prometheus/api/v1/query", params={"query": query}, timeout=5
        ).status_code in {400, 422}
    conflicting = 'reward{run_id="' + ids[0] + '"} + reward{run_id="' + ids[1] + '"}'
    assert (
        requests.get(
            base + "prometheus/api/v1/query", params={"query": conflicting}, timeout=5
        ).status_code
        == 403
    )
    assert (
        requests.get(
            stack.url + "/api/v1/runs/" + ids[1] + "/tempo/api/traces/" + "1" * 32,
            timeout=5,
        ).status_code
        == 404
    )
    assert (
        requests.get(base + "tempo/api/traces/" + "1" * 32, timeout=5).status_code
        == 200
    )
    with serve(create_app(stack.conf)) as restarted:
        for run_id in ids:
            assert (
                requests.get(
                    restarted + "/api/v1/runs/" + run_id, timeout=5
                ).status_code
                == 200
            )
        catalog = restarted + "/api/v1/catalog/prometheus/api/v1/label/execution/values"
        params = {"match[]": '{__name__=~"vllm:.*",project=~"p"}'}
        options = requests.get(catalog, params=params, timeout=5).json()["data"]
        assert len(options) == 2 and all(ids[2] not in item for item in options)
        params["start"], params["end"] = "0", "1"
        assert requests.get(catalog, params=params, timeout=5).json()["data"] == options


def test_ray_workers_share_local_identity_and_registration_recovers(stack):
    pytest.importorskip("ray")
    stack.backend.lose_response = True
    script = r"""
import json, socket, sys, time
import ray
import rl_insight as insight
from rl_insight.client.ray_monitor_client import current_job_context

url = sys.argv[1]
ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
try:
    insight.init()
    with insight.trace_state("driver_prepare", state_lane_id="early_driver"):
        pass
    assert insight.get_context() is None

    @ray.remote(num_cpus=0)
    class EarlyWorker:
        def trace_before_training(self):
            insight.init()
            with insight.trace_state("sleep", state_lane_id="early_replica"):
                pass
            context = insight.get_context()
            return context.to_dict() if context else None

        def wait_for_context(self):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                context = insight.get_context()
                if context:
                    return context.to_dict()
                time.sleep(0.1)
            raise TimeoutError("Early worker did not connect to the training Run")

    early_workers = [EarlyWorker.remote(), EarlyWorker.remote()]
    early_contexts = ray.get(
        [worker.trace_before_training.remote() for worker in early_workers], timeout=20
    )
    assert early_contexts == [None, None], (
        "Argument-free worker init must defer Run creation until training init", early_contexts
    )
    delayed_context = early_workers[1].wait_for_context.remote()

    @ray.remote
    def native():
        insight.update_prometheus_config(["127.0.0.1:19999"])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            context = current_job_context()
            if context:
                return context.run_id
            time.sleep(0.1)
        raise TimeoutError("Hub did not become ready")
    waiting = native.remote()
    time.sleep(1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    insight.init(project="ray", experiment_name="same", config={"server": {"url": url}, "prometheus": {"metrics_report_port": port}})
    first = insight.get_context()
    assert ray.get(delayed_context)["run_id"] == first.run_id
    assert ray.get(early_workers[0].wait_for_context.remote())["run_id"] == first.run_id
    insight.metric_gauge("reward", 2)
    @ray.remote
    def worker():
        insight.init(project="ray", experiment_name="same", config={"server": {"url": url}})
        insight.metric_gauge("reward", 3)
        now = time.time_ns()
        insight.trace_span(name="actor_update", start_time_ns=now, end_time_ns=now + 1000, attributes={})
        context = insight.get_context()
        insight.finish()
        return context.run_id
    run_id = ray.get(worker.remote())
    assert run_id == ray.get(waiting) == first.run_id
    assert insight.get_context().events_applied > first.events_applied
    print("RESULT=" + json.dumps(insight.get_context().to_dict()))
    # Observe normal batch export; this test does not introduce a flush guarantee.
    time.sleep(6)
    insight.finish()
finally:
    ray.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, stack.url],
        text=True,
        capture_output=True,
        timeout=100,
        env={**os.environ, "RL_INSIGHT_SERVER_URL": stack.url},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    context = json.loads(
        next(
            line[7:]
            for line in result.stdout.splitlines()
            if line.startswith("RESULT=")
        )
    )
    assert re.fullmatch(r"[0-9]{8}T[0-9]{6}Z_[0-9a-f]{32}", context["run_id"])
    assert (
        requests.get(
            stack.url + "/api/v1/runs/" + context["run_id"], timeout=5
        ).status_code
        == 200
    )
    assert any(
        s["run_id"] == context["run_id"] and s["name"] == "actor_update"
        for s in stack.backend.span_attributes
    )
    for name, lane, count in (
        ("sleep", "early_replica", 2),
        ("driver_prepare", "early_driver", 1),
    ):
        spans = [s for s in stack.backend.span_attributes if s["name"] == name]
        assert len(spans) == count
        for span in spans:
            assert span["run_id"] == context["run_id"]
            assert {
                "project": "ray",
                "experiment_name": "same",
                "state_lane_id": lane,
            }.items() <= span["attributes"].items()
    import yaml

    targets = yaml.safe_load(
        (stack.path / "data/targets/prometheus-targets.yml").read_text()
    )
    assert any(
        "127.0.0.1:19999" in group["targets"]
        and group["labels"]["run_id"] == context["run_id"]
        for group in targets
    )
