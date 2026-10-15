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

"""Run isolation through the real HTTP API and existing collector path."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests
import uvicorn
from omegaconf import OmegaConf
from fastapi.responses import JSONResponse

from rl_insight.server.http_api import create_app


@contextmanager
def serve(app, port=0):
    sock = socket.socket()
    sock.bind(("127.0.0.1", port))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    try:
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.02)
        yield base
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()


@pytest.fixture
def stack(tmp_path):
    state = SimpleNamespace(
        calls=[],
        traces=[],
        results={},
        fail_writes=False,
        reject_native_once=False,
        native_rejections=0,
    )
    dashboard_uid = "a4cabbce-92af-4a84-9399-4deba24ae6d1"
    dashboard_path = (
        "/apis/dashboard.grafana.app/v2/namespaces/default/dashboards/" + dashboard_uid
    )
    state.results[dashboard_path] = {
        "metadata": {
            "name": dashboard_uid,
            "resourceVersion": "1",
            "annotations": {"grafana.app/folder": "existing-verl"},
        }
    }

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            state.calls.append((self.path, b""))
            path = self.path.split("?")[0]
            body = json.dumps(state.results.get(path, {})).encode()
            if path == "/api/v1/query" and "rl_insight_sdk_flush_sequence" in self.path:
                body = json.dumps(
                    {
                        "status": "success",
                        "data": {"result": [{"value": [time.time(), "1"]}]},
                    }
                ).encode()
            self.send_response(
                404
                if (path.startswith("/api/datasources/") or path.startswith("/apis/"))
                and path not in state.results
                else 200
            )
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.calls.append((self.path, body))
            if self.path == "/v1/traces":
                state.traces.append(body)
            elif self.path.startswith("/api/datasources"):
                data = json.loads(body)
                state.results["/api/datasources/uid/" + data["uid"]] = data
            elif self.path.startswith("/apis/"):
                data = json.loads(body)
                data["metadata"]["resourceVersion"] = "1"
                path = (
                    "/apis/dashboard.grafana.app/v2/namespaces/default/dashboards/"
                    + data["metadata"]["name"]
                )
                state.results[path] = data
            self.send_response(404 if state.fail_writes else 200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_PUT = do_POST

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever)
    thread.start()
    conf = OmegaConf.create(
        {
            "server": {
                "data_dir": str(tmp_path / "data"),
                "runtime_dir": str(tmp_path / "runtime"),
            },
            "prometheus": {"prometheus_port": backend.server_port},
            "tempo": {"query_port": backend.server_port},
            "otel": {"otel_port": backend.server_port},
            "grafana": {"port": backend.server_port},
        }
    )
    app = create_app(conf)

    @app.middleware("http")
    async def reject_native_registration_once(request, call_next):
        if (
            state.reject_native_once
            and request.url.path == "/api/v1/prometheus/targets"
        ):
            payload = await request.json()
            if any(item["target"] == "127.0.0.1:9998" for item in payload["targets"]):
                state.reject_native_once = False
                state.native_rejections += 1
                return JSONResponse(
                    {"detail": "Temporary registration failure"}, status_code=503
                )
        return await call_next(request)

    try:
        with serve(app) as url:
            yield SimpleNamespace(
                url=url, conf=conf, data=tmp_path / "data", backend=state
            )
    finally:
        backend.shutdown()
        thread.join(10)
        backend.server_close()


def test_run_directories_and_scoped_queries(stack, monkeypatch):
    payload = {
        "request_id": uuid4().hex,
        "project": "../项目",
        "experiment_name": "same-name <script>alert(1)</script> 100%_",
        "config": {
            "server": {"namespace": "demo", "password": "secret"},
            "token": "secret",
        },
    }
    response = requests.post(stack.url + "/api/v1/runs", json=payload, timeout=3)
    assert response.status_code == 200, response.text
    first = response.json()
    assert (
        requests.post(
            stack.url + "/api/v1/runs",
            json=dict(payload, request_id=uuid4().hex, config=[]),
            timeout=3,
        ).status_code
        == 400
    )
    assert (
        requests.post(stack.url + "/api/v1/runs", json=payload, timeout=3).json()
        == first
    )
    changed = dict(payload, experiment_name="other")
    assert (
        requests.post(stack.url + "/api/v1/runs", json=changed, timeout=3).status_code
        == 409
    )
    second = requests.post(
        stack.url + "/api/v1/runs",
        json=dict(payload, request_id=uuid4().hex),
        timeout=3,
    ).json()
    assert first["run_id"] != second["run_id"]
    directories = list((stack.data / "runs").iterdir())
    assert {p.name for p in directories} == {first["run_id"], second["run_id"]}
    for directory in directories:
        assert directory.resolve().is_relative_to(stack.data)
        context = json.loads((directory / "run.json").read_text(encoding="utf-8"))[
            "context"
        ]
        assert context["run_id"] == directory.name
        assert context["project"] == payload["project"]
        assert context["experiment_name"] == payload["experiment_name"]
        assert "secret" not in (directory / "config.json").read_text(encoding="utf-8")
    first_directory = stack.data / "runs" / first["run_id"]
    (first_directory / "run.json").unlink()
    (first_directory / "config.json").unlink()
    with serve(create_app(stack.conf)) as reopened:
        restored = requests.get(reopened + "/api/v1/runs/" + first["run_id"], timeout=3)
        assert restored.status_code == 200
        assert restored.json()["context"] == first
        assert (
            json.loads((first_directory / "run.json").read_text(encoding="utf-8"))[
                "context"
            ]
            == first
        )
        assert json.loads(
            (first_directory / "config.json").read_text(encoding="utf-8")
        ) == {"server": {"namespace": "demo"}}
    run_id = first["run_id"]
    root = stack.url + "/api/v1/runs/" + run_id
    stack.backend.results["/api/v1/query"] = {
        "status": "success",
        "data": {"result": []},
    }
    response = requests.get(
        root + "/prometheus/api/v1/query",
        params={"query": "sum(rate(loss[5m])) or reward"},
        timeout=3,
    )
    assert response.status_code == 200, response.text
    from urllib.parse import parse_qs, urlsplit

    sent = parse_qs(urlsplit(stack.backend.calls[-1][0]).query)["query"][0]
    assert sent.count('run_id="' + run_id + '"') == 2
    for expression, selectors in (
        ("sum(or)", 1),
        ("sum(and)", 1),
        ("sum(unless)", 1),
        ("foo + or + bar", 3),
        ("foo and on(rank) or", 2),
    ):
        assert (
            requests.get(
                root + "/prometheus/api/v1/query",
                params={"query": expression},
                timeout=3,
            ).status_code
            == 200
        )
        sent = parse_qs(urlsplit(stack.backend.calls[-1][0]).query)["query"][0]
        assert sent.count('run_id="' + run_id + '"') == selectors
    wrong = 'loss{run_id="' + second["run_id"] + '"}'
    assert (
        requests.post(
            root + "/prometheus/api/v1/query", data={"query": wrong}, timeout=3
        ).status_code
        == 403
    )
    assert (
        requests.get(root + "/prometheus/api/v1/labels", timeout=3).status_code == 200
    )
    assert "match[]" in parse_qs(urlsplit(stack.backend.calls[-1][0]).query)
    assert (
        requests.get(
            root + "/prometheus/api/v1/admin/tsdb/delete_series", timeout=3
        ).status_code
        == 404
    )

    def attrs(owned):
        return [{"key": "rl_insight.run_id", "value": {"stringValue": owned}}]

    trace_id = "0" + "a" * 31
    stack.backend.results["/api/traces/" + trace_id] = {
        "batches": [{"resource": {"attributes": attrs(second["run_id"])}}]
    }
    assert (
        requests.get(root + "/tempo/api/traces/" + trace_id, timeout=3).status_code
        == 404
    )
    stack.backend.results["/api/traces/" + trace_id]["batches"][0]["resource"][
        "attributes"
    ] = attrs(run_id)
    assert (
        requests.get(root + "/tempo/api/traces/" + trace_id, timeout=3).status_code
        == 200
    )
    span = {
        "spanID": "1",
        "name": "train",
        "startTimeUnixNano": "1000000",
        "durationNanos": "2000000",
        "attributes": attrs(run_id)
        + [{"key": "state_lane_id", "value": {"stringValue": "rank_0"}}],
    }
    stack.backend.results["/api/search"] = {
        "traces": [
            {
                "traceID": trace_id.lstrip("0"),
                "rootTraceName": "untrusted-root",
                "spanSets": [{"matched": 1, "spans": [span]}],
            }
        ]
    }
    searched = requests.get(
        root + "/tempo/api/search",
        params={"q": '{name="train"}', "spss": "100"},
        timeout=3,
    )
    assert searched.status_code == 200, searched.text
    assert searched.json()["traces"][0]["traceID"] == trace_id
    assert searched.json()["traces"][0]["rootTraceName"] == "train"
    assert searched.json()["traces"][0]["spanSets"][0]["spans"][0]["attributes"] == [
        {"key": "state_lane_id", "value": {"stringValue": "rank_0"}}
    ]
    assert (
        "resource.rl_insight.run_id"
        in parse_qs(urlsplit(stack.backend.calls[-1][0]).query)["q"][0]
    )
    assert parse_qs(urlsplit(stack.backend.calls[-1][0]).query)["spss"] == ["1000"]
    selected = parse_qs(urlsplit(stack.backend.calls[-1][0]).query)["q"][0]
    assert selected.endswith(
        " | select(resource.rl_insight.run_id, name, span.state_lane_id)"
    )
    assert (
        requests.get(
            root + "/tempo/api/search", params={"spss": "10001"}, timeout=3
        ).status_code
        == 422
    )
    span["attributes"] = attrs(second["run_id"])
    assert requests.get(root + "/tempo/api/search", timeout=3).status_code == 502
    from rl_insight.server.runtime import run_dashboard
    from rl_insight.protocol import RunContext

    dashboard = run_dashboard(
        RunContext.from_dict(first), {"prometheus": "run-p", "tempo": "run-t"}
    )
    template = json.loads(
        (
            Path(__file__).parents[3]
            / "rl_insight/config/services/grafana/dashboards/verl/verl_tainer_v1_with_vllm_engine.json"
        ).read_text(encoding="utf-8")
    )
    assert dashboard["spec"]["layout"] == template["spec"]["layout"]
    assert dashboard["metadata"]["name"] == template["metadata"]["name"]
    assert dashboard["spec"]["title"] == template["spec"]["title"]
    timeline = dashboard["spec"]["elements"]["panel-40"]["spec"]["vizConfig"]
    assert timeline["spec"]["fieldConfig"]["overrides"] == [
        {
            "matcher": {"id": "byName", "options": "traceName"},
            "properties": [
                {"id": "displayName", "value": "${__field.labels.state_lane_id}"}
            ],
        }
    ]
    assert len(dashboard["spec"]["elements"]) == len(template["spec"]["elements"])
    for name, element in dashboard["spec"]["elements"].items():
        assert (
            element["spec"]["vizConfig"]
            == template["spec"]["elements"][name]["spec"]["vizConfig"]
        )
    assert "rl_insight_monitor_" in json.dumps(dashboard)
    prepared = requests.get(root + "/view", timeout=3)
    assert prepared.status_code == 200, prepared.text
    assert all(
        "rli-" + run_id in value for value in prepared.json()["datasources"].values()
    )
    posted = json.loads(
        next(
            body
            for path, body in reversed(stack.backend.calls)
            if path.endswith(template["metadata"]["name"]) and body
        )
    )
    assert posted["metadata"]["annotations"]["grafana.app/folder"] == "existing-verl"
    assert not any(
        path == "/apis/dashboard.grafana.app/v2/namespaces/default/dashboards" and body
        for path, body in stack.backend.calls
    )
    variables = {v["spec"]["name"]: v for v in posted["spec"]["variables"]}
    assert len(variables) == len(posted["spec"]["variables"])
    selector = variables["run_id"]["spec"]
    assert selector["label"] == "Run ID"
    assert selector["current"]["value"] == run_id
    assert selector["multi"] is False and selector["includeAll"] is False
    choices = json.loads(selector["query"])
    assert {item["value"] for item in choices} == {first["run_id"], second["run_id"]}
    for item in choices:
        assert item["text"] == item["value"]
        assert item["prometheus"] == "rli-" + item["value"] + "-p"
        assert item["tempo"] == "rli-" + item["value"] + "-t"
    assert variables["experiment_name"]["kind"] == "QueryVariable"
    assert variables["project"]["kind"] == "QueryVariable"
    assert "${run_id.prometheus}" in json.dumps(posted)
    assert "${run_id.tempo}" in json.dumps(posted)
    registered = [
        json.loads(body)
        for path, body in stack.backend.calls
        if path == "/api/datasources" and body
    ]
    assert {body["uid"] for body in registered} >= {
        "rli-" + item["value"] + suffix for item in choices for suffix in ("-p", "-t")
    }
    opened = requests.get(
        stack.url + "/ui/runs/" + run_id, allow_redirects=False, timeout=3
    )
    assert opened.status_code == 307
    assert "/d/" + template["metadata"]["name"] in opened.headers["Location"]
    assert "var-run_id=" + run_id in opened.headers["Location"]
    other_view = requests.get(
        stack.url + "/api/v1/runs/" + second["run_id"] + "/view", timeout=3
    ).json()
    assert other_view["url"].split("?")[0] == prepared.json()["url"].split("?")[0]
    assert (
        requests.get(
            root + "/view", params={"template": "../../config.yaml"}, timeout=3
        ).status_code
        == 404
    )
    stack.backend.fail_writes = True
    assert requests.get(root + "/view", timeout=3).status_code == 502
    stack.backend.fail_writes = False
    import rl_insight as monitor
    from rl_insight.client.base import MonitorClient, MONITOR_CLIENT_REGISTRY

    class LegacyClient(MonitorClient):
        def apply_event(self, event):
            pass

    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", stack.url)
    monkeypatch.setitem(MONITOR_CLIENT_REGISTRY, "legacy", lambda conf: LegacyClient())
    try:
        monitor.init(config={"server": {"url": stack.url, "backend": "legacy"}})
        assert monitor.get_context() is None
        monitor.update_prometheus_config(["127.0.0.1:19999"])
        assert "19999" in (stack.data / "targets/prometheus-targets.yml").read_text(
            encoding="utf-8"
        )
    finally:
        monitor.finish()
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL")
    monitor.update_prometheus_config(
        ["127.0.0.1:19998"], context=RunContext.from_dict(first)
    )
    assert "19998" in (stack.data / "targets/prometheus-targets.yml").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("context_delay", [0, 35], ids=["immediate", "late-context"])
def test_existing_ray_api_keeps_one_run_identity(stack, context_delay):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    from rl_insight.utils.constants import MonitorPaths
    from rl_insight.server.runtime import DEFAULT_TEMPLATE

    template = json.loads(
        (MonitorPaths.GRAFANA_DASHBOARDS_DIR / DEFAULT_TEMPLATE).read_text(
            encoding="utf-8"
        )
    )
    selector = next(
        variable["spec"]
        for variable in template["spec"]["variables"]
        if variable["spec"]["name"] == "run_id"
    )
    assert selector["hide"] == "dontHide" and selector["options"] == []

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    stack.backend.reject_native_once = bool(context_delay)
    script = "import runpy,sys; runpy.run_path(sys.argv[1])['_ray_scenario'](sys.argv[2],int(sys.argv[3]),context_delay=float(sys.argv[4]))"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            stack.url,
            str(port),
            str(context_delay),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert stack.backend.native_rejections == int(bool(context_delay))
    context = json.loads(
        next(
            line[8:]
            for line in child.stdout.splitlines()
            if line.startswith("CONTEXT ")
        )
    )
    assert stack.backend.traces
    request = ExportTraceServiceRequest.FromString(stack.backend.traces[0])
    attrs = {
        a.key: a.value.string_value
        for a in request.resource_spans[0].resource.attributes
    }
    assert attrs["rl_insight.run_id"] == context["run_id"]
    assert attrs["service.name"] == "rl_insight_monitor"
    assert request.resource_spans[0].scope_spans[0].spans[0].name == "update_actor"
    targets = (stack.data / "targets/prometheus-targets.yml").read_text(
        encoding="utf-8"
    )
    assert (
        targets.count(context["run_id"]) == 2
        and "replica: '0'" in targets
        and "replica: '1'" in targets
    )
    assert len(list((stack.data / "runs").iterdir())) == 1
    path = (
        "/apis/dashboard.grafana.app/v2/namespaces/default/dashboards/"
        + template["metadata"]["name"]
    )
    selector = next(
        variable["spec"]
        for variable in stack.backend.results[path]["spec"]["variables"]
        if variable["spec"]["name"] == "run_id"
    )
    assert [option["value"] for option in selector["options"]] == [context["run_id"]]
    # Restore the template, change ports, then inspect sources before any view request.
    for public_url in ("", "https://monitor.example.invalid/insight"):
        stack.backend.results[path] = template
        stack.conf.server.public_url = public_url
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            stack.conf.server.port = reserve.getsockname()[1]
        with serve(create_app(stack.conf), port=stack.conf.server.port) as restarted:
            for kind in ("prometheus", "tempo"):
                source = stack.backend.results[
                    "/api/datasources/uid/rli-" + context["run_id"] + "-" + kind[0]
                ]
                assert source["url"] == (
                    (public_url or restarted)
                    + "/api/v1/runs/"
                    + context["run_id"]
                    + "/"
                    + kind
                )
            selector = next(
                variable["spec"]
                for variable in stack.backend.results[path]["spec"]["variables"]
                if variable["spec"]["name"] == "run_id"
            )
            assert [option["value"] for option in selector["options"]] == [
                context["run_id"]
            ]
            assert (
                requests.get(
                    restarted + "/api/v1/runs/" + context["run_id"], timeout=3
                ).status_code
                == 200
            )


def _ray_scenario(url, port, real_backends=False, context_delay=0):
    import ray
    import rl_insight as insight
    from rl_insight.client.ray_monitor_client import _current_job_actor_name
    from rl_insight.utils.constants import MonitorRayActor

    os.environ["RL_INSIGHT_SERVER_URL"] = url
    os.environ["OTEL_BSP_SCHEDULE_DELAY"] = "100"
    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR")
    conf = {"prometheus": {"metrics_report_port": port}}
    try:

        @ray.remote
        class NativeExporter:
            def __init__(self):
                import rl_insight as monitor

                monitor.update_prometheus_config(
                    ["127.0.0.1:9998"], job_name="native", labels=[{"replica": "1"}]
                )

            def ready(self):
                return True

            def registered(self):
                from rl_insight.utils.prometheus_utils import _pending_targets

                return not _pending_targets

        if not real_backends:
            exporter = NativeExporter.remote()
            assert ray.get(exporter.ready.remote())
            insight.update_prometheus_config(
                ["127.0.0.1:9999"], job_name="native", labels=[{"replica": "0"}]
            )
            time.sleep(context_delay)
        insight.init(project="demo", experiment_name="same", config=conf)
        context = insight.get_context()
        assert context is not None
        if not real_backends:
            deadline = time.monotonic() + 10
            while not ray.get(exporter.registered.remote()):
                assert time.monotonic() < deadline
                time.sleep(0.1)

        @ray.remote(max_calls=1)
        def worker(config):
            import rl_insight as monitor

            monitor.init(project="demo", experiment_name="same", config=config)
            monitor.metric_gauge("loss", 0.25, rank="0")
            with monitor.trace_state("update_actor", state_lane_id="actor_0"):
                pass
            return monitor.get_context().to_dict()

        assert ray.get(worker.remote(conf)) == context.to_dict()
        hub = ray.get_actor(
            _current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE
        )
        endpoint = ray.get(hub.get_status.remote())["metrics_endpoint"]
        deadline = time.monotonic() + 10
        while True:
            metrics = requests.get(endpoint, timeout=3).text
            if "rl_insight_monitor_loss" in metrics:
                break
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert 'run_id="' + context.run_id + '"' in metrics
        assert "0.25" in metrics
        with pytest.raises(ValueError, match="run_id"):
            insight.metric_gauge("loss", 0.5, run_id=uuid4().hex)

        if real_backends:
            root = url + "/api/v1/runs/" + context.run_id
            for path, field in (
                ("prometheus/api/v1/query", "data"),
                ("tempo/api/search", "traces"),
            ):
                deadline = time.monotonic() + 45
                while True:
                    response = requests.get(
                        root + "/" + path,
                        params={"query": "rl_insight_monitor_loss"}
                        if field == "data"
                        else {},
                        timeout=12,
                    )
                    assert response.status_code == 200, response.text
                    payload = response.json()
                    ready = (
                        payload.get("data", {}).get("result")
                        if field == "data"
                        else payload.get("traces")
                    )
                    if ready:
                        break
                    assert time.monotonic() < deadline, response.text
                    time.sleep(0.2)
            view = requests.get(root + "/view", timeout=20)
            assert view.status_code == 200, view.text

        time.sleep(0.5)  # The final metric below intentionally has no scrape delay.
        if real_backends:
            insight.metric_gauge("final_step", 2)
        print("CONTEXT " + json.dumps(context.to_dict()))
    finally:
        insight.finish()
        ray.shutdown()


@pytest.mark.skipif(
    not os.environ.get("RL_INSIGHT_RUN_E2E_URL"),
    reason="requires a managed Prometheus/Tempo/Grafana stack",
)
def test_sequential_runs_on_managed_backends():
    url = os.environ["RL_INSIGHT_RUN_E2E_URL"].rstrip("/")
    contexts = []
    start = time.time()
    for port in (19102, 19103):
        script = "import runpy,sys; runpy.run_path(sys.argv[1])['_ray_scenario'](sys.argv[2],int(sys.argv[3]),True)"
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(Path(__file__).resolve()),
                url,
                str(port),
            ],
            capture_output=True,
            text=True,
            timeout=150,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        contexts.append(
            json.loads(
                next(
                    line[8:]
                    for line in child.stdout.splitlines()
                    if line.startswith("CONTEXT ")
                )
            )
        )
    assert contexts[0]["run_id"] != contexts[1]["run_id"]
    trace_ids = []
    for context in contexts:
        root = url + "/api/v1/runs/" + context["run_id"]
        metrics = requests.get(
            root + "/prometheus/api/v1/query_range",
            params={
                "query": "rl_insight_monitor_loss",
                "start": start,
                "end": time.time(),
                "step": 1,
            },
            timeout=12,
        ).json()
        assert metrics["data"]["result"]
        assert all(
            item["metric"]["run_id"] == context["run_id"]
            for item in metrics["data"]["result"]
        )
        final = requests.get(
            root + "/prometheus/api/v1/query",
            params={"query": "max_over_time(rl_insight_monitor_final_step[10m])"},
            timeout=12,
        ).json()
        assert (
            final["data"]["result"]
            and float(final["data"]["result"][0]["value"][1]) == 2
        )
        traces = requests.get(root + "/tempo/api/search", timeout=12).json()["traces"]
        assert traces
        trace_ids.append(traces[0]["traceID"])
        assert (
            requests.get(
                root + "/tempo/api/traces/" + trace_ids[-1], timeout=12
            ).status_code
            == 200
        )
    for current, foreign in ((0, 1), (1, 0)):
        root = url + "/api/v1/runs/" + contexts[current]["run_id"]
        assert (
            requests.get(
                root + "/tempo/api/traces/" + trace_ids[foreign], timeout=12
            ).status_code
            == 404
        )
