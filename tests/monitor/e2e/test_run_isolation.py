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
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from rl_insight.server.http_api import create_app
from rl_insight.utils.protocol import RunContext


@contextmanager
def serve(app, port=0):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
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
        flush_queries=[],
        flush_blocked=False,
        lose_response=False,
    )

    @backend.get("/{path:path}")
    def query(path: str, request: Request):
        fields = dict(request.query_params)
        if path == "api/v1/label/run_id/values":
            return {"status": "success", "data": state.ids}
        expression = fields.get("query") or fields.get("q", "")
        selected = [run_id for run_id in state.ids if run_id in expression]
        if "rl_insight_sdk_flush_sequence" in expression:
            # Protocol backend acknowledges collection; real backends verify actual scrapes.
            selected = re.findall(r'run_id="([^"]+)"', expression)
            state.flush_queries.extend(selected)
            if state.flush_blocked:
                selected = []
        if path.startswith("api/v1/"):
            return {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {"run_id": run_id},
                            "value": [
                                1,
                                "1000000"
                                if "rl_insight_sdk_flush_sequence" in expression
                                else "1",
                            ],
                        }
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
            try:
                yield SimpleNamespace(url=url, conf=conf, backend=state, path=tmp_path)
            finally:
                from rl_insight.utils import prometheus_utils

                with prometheus_utils._publication_lock:
                    prometheus_utils._target_publications.clear()


def test_same_named_runs_support_selection_isolation_and_catalog_restart(stack):
    ids = [RunContext.create("p", "same", stack.url).run_id for _ in range(3)]
    stack.backend.ids = ids
    for i, run_id in enumerate(ids):
        payload = {
            "run_id": run_id,
            "project": "p" if i < 2 else "other",
            "experiment_name": "same",
        }
        response = requests.post(stack.url + "/api/v1/runs", json=payload, timeout=5)
        assert response.status_code == 200
        from rl_insight import __version__

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
    # Allow early workers' background submissions to reach the Hub.
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
    assert all(
        context["run_id"] != group["labels"].get("__rl_insight_run_id")
        for group in targets or []
    )
    metadata = requests.get(
        stack.url + "/api/v1/runs/" + context["run_id"], timeout=5
    ).json()
    assert metadata["status"] == "ended"
    assert metadata["final_collection_confirmed"] is True


def test_direct_targets_have_run_ownership_and_independent_lifecycle(stack):
    import yaml

    owners = [uuid4().hex, uuid4().hex]
    contexts = [
        requests.post(
            stack.url + "/api/v1/runs",
            json={
                "run_id": RunContext.create("same", "same", stack.url).run_id,
                "project": "same",
                "experiment_name": "same",
                "owner_token": owner,
            },
            timeout=3,
        ).json()
        for owner in owners
    ]
    root = stack.url + "/api/v1/prometheus/targets"
    payloads = [
        {
            "run_context": context,
            "publisher_id": uuid4().hex,
            "job_name": "native",
            "targets": [{"target": f"worker-{i}:9000"}],
        }
        for i, context in enumerate(contexts)
    ]
    # No exporter needs to be reachable by the server; only Prometheus scrapes it.
    for payload in payloads:
        response = requests.post(root, json=payload, timeout=3)
        assert response.status_code == 200, response.text
        assert response.json()["ttl"] > 0
    groups = yaml.safe_load(
        (stack.path / "data" / "targets/prometheus-targets.yml").read_text()
    )
    assert {g["targets"][0] for g in groups} == {"worker-0:9000", "worker-1:9000"}
    assert {g["labels"]["run_id"] for g in groups} == {c["run_id"] for c in contexts}
    assert not any("__metrics_path__" in g["labels"] for g in groups)
    assert (
        requests.get(
            stack.url + "/api/v1/runs/" + contexts[0]["run_id"] + "/metrics", timeout=3
        ).status_code
        == 404
    )
    stolen = dict(payloads[1], targets=payloads[0]["targets"])
    assert requests.post(root, json=stolen, timeout=3).status_code == 409
    assert (
        requests.post(
            root, json=dict(payloads[0], publisher_id=uuid4().hex), timeout=3
        ).status_code
        == 409
    )
    closing = stack.url + "/api/v1/runs/" + contexts[0]["run_id"] + "/close"
    assert (
        requests.post(closing, json={"owner_token": owners[1]}, timeout=3).status_code
        == 409
    )
    requests.post(
        closing, json={"owner_token": owners[0], "prepare": True}, timeout=3
    ).raise_for_status()
    assert requests.post(root, json=payloads[0], timeout=3).status_code == 200
    assert (
        requests.post(
            root, json=dict(payloads[0], targets=[{"target": "late:9000"}]), timeout=3
        ).status_code
        == 409
    )
    requests.post(
        closing, json={"owner_token": owners[0]}, timeout=3
    ).raise_for_status()
    assert requests.post(root, json=payloads[0], timeout=3).status_code == 409
    assert requests.post(root, json=payloads[1], timeout=3).status_code == 200
    groups = yaml.safe_load(
        (stack.path / "data" / "targets/prometheus-targets.yml").read_text()
    )
    assert [g["targets"] for g in groups] == [["worker-1:9000"]]


def _ray_scenario(url, port, managed=False, address=None, release=None):
    import ray

    import rl_insight as insight

    ray.init(
        address=address or "local",
        runtime_env={"env_vars": {"RL_INSIGHT_SERVER_URL": url}},
        **({} if address else {"num_cpus": 2, "include_dashboard": False}),
        logging_level="ERROR",
    )
    try:
        insight.init(
            project="concurrent",
            experiment_name="same",
            config={
                "server": {"url": url},
                "prometheus": {"metrics_report_port": port},
            },
        )
        insight.metric_gauge("loss", 0.5)
        context = insight.get_context()
        print("CONTEXT " + json.dumps(context.to_dict()), flush=True)

        @ray.remote(num_cpus=0)
        class Worker:
            def ready(self):
                insight.init()
                return insight.get_context().run_id

            def finish(self):
                insight.metric_count("worker_finished", 1)
                insight.metric_log("reward_mean", 1.5, step=0)
                insight.finish()

        workers = [Worker.remote() for _ in range(5)]
        assert (
            ray.get([w.ready.remote() for w in workers], timeout=40)
            == [context.run_id] * 5
        )
        ray.get([w.finish.remote() for w in workers], timeout=10)
        assert insight.get_context().events_applied >= 11
        print("WORKERS_DONE", flush=True)
        while release and not Path(release).exists():
            insight.metric_gauge("loss", 0.25)
            time.sleep(0.1)
        insight.metric_gauge("final_step", 2)
        insight.metric_log("final_step", 2, step=1)
    finally:
        insight.finish()
        ray.shutdown()


def test_two_ray_jobs_keep_monitoring_when_one_finishes(stack, tmp_path):
    import ray

    cluster = ray.init(num_cpus=4, include_dashboard=False, logging_level="ERROR")
    stack.backend.flush_blocked = True
    children = []
    files = []
    releases = [tmp_path / "release-a", tmp_path / "release-b"]
    try:
        for index in range(2):
            output = tmp_path / f"job-{index}.log"
            stream = output.open("w")
            files.append(stream)
            script = "import runpy,sys;runpy.run_path(sys.argv[1])['_ray_scenario'](sys.argv[2],0,False,sys.argv[3],sys.argv[4])"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(Path(__file__).resolve()),
                    stack.url,
                    cluster.address_info["gcs_address"],
                    str(releases[index]),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            children.append((child, output))
        deadline = time.monotonic() + 50
        contexts = []
        workers_done = False
        while len(contexts) != 2 or not workers_done:
            contexts = []
            workers_done = True
            for child, output in children:
                text = output.read_text(encoding="utf-8")
                workers_done = workers_done and "WORKERS_DONE" in text
                lines = [
                    line[8:]
                    for line in text.splitlines()
                    if line.startswith("CONTEXT ")
                ]
                if lines:
                    contexts.append(json.loads(lines[0]))
                assert child.poll() is None, text
            assert time.monotonic() < deadline, [
                output.read_text(encoding="utf-8") for _, output in children
            ]
            time.sleep(0.05)
        assert contexts[0]["run_id"] != contexts[1]["run_id"]
        # Worker shutdown drains its own events without waiting for any scrape.
        assert stack.backend.flush_queries == []
        stack.backend.flush_blocked = False
        for context in contexts:
            root = stack.url + "/api/v1/runs/" + context["run_id"]
            import yaml

            groups = yaml.safe_load(
                (stack.path / "data" / "targets/prometheus-targets.yml").read_text()
            )
            address = next(
                g["targets"][0]
                for g in groups
                if g["labels"].get("__rl_insight_run_id") == context["run_id"]
            )
            response = requests.get("http://" + address + "/metrics", timeout=5)
            assert response.status_code == 200, response.text
            assert context["metrics_endpoint"] == "http://" + address + "/metrics"
            assert 'run_id="' + context["run_id"] + '"' in response.text
            assert "rl_insight_monitor_loss" in response.text
            assert re.search(
                r"rl_insight_monitor_worker_finished_total\{[^\n]+\} 5\.0",
                response.text,
            )
            assert requests.get(root, timeout=3).json()["status"] == "registered"
        releases[0].touch()
        assert children[0][0].wait(30) == 0, children[0][1].read_text()
        metadata = requests.get(
            stack.url + "/api/v1/runs/" + contexts[0]["run_id"], timeout=3
        ).json()
        assert (
            metadata["status"] == "ended"
            and metadata["final_collection_confirmed"] is True
        )
        root_b = stack.url + "/api/v1/runs/" + contexts[1]["run_id"]
        assert (
            "rl_insight_monitor_loss"
            in requests.get(
                "http://"
                + next(
                    g["targets"][0]
                    for g in yaml.safe_load(
                        (
                            stack.path / "data" / "targets/prometheus-targets.yml"
                        ).read_text()
                    )
                    if g["labels"].get("__rl_insight_run_id") == contexts[1]["run_id"]
                )
                + "/metrics",
                timeout=5,
            ).text
        )
        assert requests.get(root_b, timeout=3).json()["status"] == "registered"
        releases[1].touch()
        assert children[1][0].wait(30) == 0, children[1][1].read_text()
        discovery = (stack.path / "data" / "targets/prometheus-targets.yml").read_text()
        assert (
            contexts[0]["run_id"] not in discovery
            and contexts[1]["run_id"] not in discovery
        )
        assert {
            span["run_id"]
            for span in stack.backend.span_attributes
            if span["name"] == "metric_log"
        } == {context["run_id"] for context in contexts}
        for context in contexts:
            spans = [
                span
                for span in stack.backend.span_attributes
                if span["run_id"] == context["run_id"] and span["name"] == "metric_log"
            ]
            assert len(spans) == 6
            assert {span["attributes"]["metric.name"] for span in spans} == {
                "reward_mean",
                "final_step",
            }
            assert stack.backend.flush_queries.count(context["run_id"]) == 1
    finally:
        for release in releases:
            release.touch()
        for child, _ in children:
            if child.poll() is None:
                child.kill()
            child.wait(10)
        for stream in files:
            stream.close()
        ray.shutdown()


def test_direct_sdk_republishes_after_restart_and_expires(stack, tmp_path, monkeypatch):
    import yaml

    from rl_insight.utils.prometheus_utils import (
        _forget_targets,
        _register_targets,
    )
    from rl_insight.utils.protocol import RunContext

    conf = OmegaConf.create(OmegaConf.to_container(stack.conf, resolve=True))
    conf.server.data_dir = str(tmp_path / "recovery")
    conf.server.endpoint_ttl = 1
    conf.grafana.enable = False
    target_file = Path(conf.server.data_dir) / "targets/prometheus-targets.yml"
    with serve(create_app(conf)) as url:
        context = RunContext.from_dict(
            requests.post(
                url + "/api/v1/runs",
                json={"run_id": RunContext.create("same", "same", stack.url).run_id},
                timeout=3,
            ).json()
        )
        assert _register_targets(["native.example:9000"], "native", None, context)
        assert "native.example:9000" in target_file.read_text()
    with serve(create_app(conf), port=urlsplit(url).port):
        deadline = time.monotonic() + 8
        while "native.example:9000" not in target_file.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert (
            requests.get(url + "/api/v1/runs/" + context.run_id, timeout=3).json()[
                "context"
            ]
            == context.to_dict()
        )
        # A restarted publisher first conflicts, then acquires the address after TTL.
        import rl_insight.utils.prometheus_utils as tools

        _forget_targets(context)
        replacement = uuid4().hex
        monkeypatch.setattr(tools, "_publisher_id", replacement)
        assert not _register_targets(["native.example:9000"], "native", None, context)
        deadline = time.monotonic() + 8
        while not any(
            g["labels"].get("__rl_insight_publisher") == replacement
            for g in yaml.safe_load(target_file.read_text()) or []
        ):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        _forget_targets(context)
        deadline = time.monotonic() + 8
        while yaml.safe_load(target_file.read_text()):
            assert time.monotonic() < deadline
            time.sleep(0.05)


def test_closing_run_restores_only_existing_targets_after_server_restart(
    stack, tmp_path
):
    conf = OmegaConf.create(OmegaConf.to_container(stack.conf, resolve=True))
    conf.server.data_dir = str(tmp_path / "closing")
    conf.grafana.enable = False
    owner = uuid4().hex
    with serve(create_app(conf)) as url:
        context = requests.post(
            url + "/api/v1/runs",
            json={
                "run_id": RunContext.create("same", "same", stack.url).run_id,
                "owner_token": owner,
            },
            timeout=3,
        ).json()
        payload = {
            "run_context": context,
            "publisher_id": uuid4().hex,
            "targets": ["closing.example:9000"],
        }
        requests.post(
            url + "/api/v1/prometheus/targets", json=payload, timeout=3
        ).raise_for_status()
        close = url + "/api/v1/runs/" + context["run_id"] + "/close"
        requests.post(
            close, json={"owner_token": owner, "prepare": True}, timeout=3
        ).raise_for_status()
    with serve(create_app(conf), port=urlsplit(url).port):
        assert (
            requests.post(
                url + "/api/v1/prometheus/targets", json=payload, timeout=3
            ).status_code
            == 200
        )
        assert (
            requests.post(
                url + "/api/v1/prometheus/targets",
                json=dict(payload, targets=["new.example:9000"]),
                timeout=3,
            ).status_code
            == 409
        )
        requests.post(close, json={"owner_token": owner}, timeout=3).raise_for_status()
