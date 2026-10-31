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

"""Representative execution-name flows through real HTTP and Ray."""

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
        span_attributes=[],
        lose_response=False,
    )

    @backend.post("/{path:path}")
    async def ingest(path: str, request: Request):
        if path == "v1/traces":
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )

            batch = ExportTraceServiceRequest.FromString(await request.body())
            for resource in batch.resource_spans:
                state.span_attributes.extend(
                    {
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
            if (
                request.method == "POST"
                and request.url.path == "/api/v1/prometheus/targets"
            ):
                if state.lose_response:
                    state.lose_response = False
                    return JSONResponse(
                        {"detail": "retry after target registration"},
                        status_code=503,
                    )
            return response

        with serve(app) as url:
            yield SimpleNamespace(url=url, conf=conf, backend=state, path=tmp_path)


def test_same_named_executions_register_distinct_native_labels(stack):
    import yaml

    from rl_insight.utils.prometheus_utils import _register_targets

    contexts = [
        RunContext.create(project, "same", stack.url) for project in ("p", "p", "other")
    ]
    assert len({context.experiment_name for context in contexts}) == 3
    for i, context in enumerate(contexts):
        assert _register_targets(
            [f"host{i}:9000"], "workers", [{"experiment_name": "same"}], context
        )
    targets_file = stack.path / "data/targets/prometheus-targets.yml"
    targets = yaml.safe_load(targets_file.read_text())
    assert {group["labels"]["experiment_name"] for group in targets} == {
        context.experiment_name for context in contexts
    }
    assert sum(group["labels"]["project"] == "p" for group in targets) == 2
    assert all("run_id" not in group["labels"] for group in targets)
    with serve(create_app(stack.conf)) as restarted:
        assert requests.get(restarted + "/healthz", timeout=5).status_code == 200
        assert yaml.safe_load(targets_file.read_text()) == targets


def test_ray_workers_share_execution_name_and_registration_recovers(stack):
    pytest.importorskip("ray")
    stack.backend.lose_response = True
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config_file = stack.path / "training.yaml"
    config_file.write_text(json.dumps({"prometheus": {"metrics_report_port": port}}))
    script = r"""
import json, os, sys, time, warnings
import requests
import ray
import rl_insight as insight
import yaml
from pathlib import Path
from rl_insight.client.ray_monitor_client import _current_job_actor_name
from rl_insight.utils.constants import MonitorRayActor
from rl_insight.utils.protocol import RunError

url = sys.argv[1]
ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
try:
    os.environ.pop("RL_INSIGHT_SERVER_URL", None)
    insight.init()
    assert insight.get_context() is None
    insight.metric_gauge("before_init", 1)
    assert insight.get_context() is None
    os.environ["RL_INSIGHT_SERVER_URL"] = url
    try:
        insight.init(config={})
    except RunError as exc:
        assert exc.status == 503
        insight.init(config={})
    default = insight.get_context()
    assert default.project == "default"
    assert default.original_experiment_name == "default"
    assert default.events_applied == 0
    insight.metric_gauge("default_names", 1)
    assert insight.get_context().events_applied == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        insight.init(project="ignored")
    assert caught and insight.get_context().experiment_name == default.experiment_name
    insight.init(refresh=True)
    assert insight.get_context().experiment_name == default.experiment_name
finally:
    insight.finish()
    ray.shutdown()

ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
try:
    @ray.remote(num_cpus=0)
    class Worker:
        def trace_with_defaults(self):
            insight.init()
            with insight.trace_state("sleep", state_lane_id="early_replica"):
                pass
            context = insight.get_context()
            return context.to_dict() if context else None

        def refresh(self):
            insight.init(refresh=True)
            return insight.get_context().to_dict()

    @ray.remote
    def native(address):
        os.environ.pop("RL_INSIGHT_SERVER_URL", None)
        insight.update_prometheus_config([address])
        hub = ray.get_actor(_current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE)
        return hub._actor_id.hex()
    early_actor = ray.get(native.remote("127.0.0.1:19999"))
    targets = yaml.safe_load(Path(sys.argv[2]).read_text())
    assert not any("127.0.0.1:19999" in group["targets"] for group in targets)
    workers = [Worker.remote(), Worker.remote()]
    assert ray.get([worker.trace_with_defaults.remote() for worker in workers], timeout=20) == [None, None]
    insight.init(project="ray", experiment_name="same", config={"server": {"url": url}})
    first = insight.get_context()
    actor = ray.get(native.remote("127.0.0.1:19998"))
    assert actor == early_actor
    with insight.trace_state("driver_prepare", state_lane_id="early_driver"):
        pass
    contexts = ray.get([worker.trace_with_defaults.remote() for worker in workers], timeout=20)
    assert {context["experiment_name"] for context in contexts} == {first.experiment_name}
    assert ray.get(workers[0].refresh.remote())["experiment_name"] == first.experiment_name
    insight.metric_gauge("reward", 2)
    @ray.remote
    def worker():
        insight.init(project="ray", experiment_name="same", config={"server": {"url": url}})
        insight.metric_gauge("reward", 3)
        now = time.time_ns()
        insight.trace_span(name="actor_update", start_time_ns=now, end_time_ns=now + 1000, attributes={})
        context = insight.get_context()
        insight.finish()
        return context.experiment_name
    experiment_name = ray.get(worker.remote())
    assert experiment_name == first.experiment_name
    assert insight.get_context().events_applied > first.events_applied
    snapshot = insight.get_context().to_dict()
    snapshot["metrics"] = requests.get(snapshot["metrics_endpoint"], timeout=5).text
    print("RESULT=" + json.dumps(snapshot))
    # Observe normal batch export; this test does not introduce a flush guarantee.
    time.sleep(6)
    insight.finish()
finally:
    ray.shutdown()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            stack.url,
            str(stack.path / "data/targets/prometheus-targets.yml"),
        ],
        text=True,
        capture_output=True,
        timeout=100,
        env={
            **os.environ,
            "RL_INSIGHT_SERVER_URL": stack.url,
            "RL_INSIGHT_CONFIG": str(config_file),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    context = json.loads(
        next(
            line[7:]
            for line in result.stdout.splitlines()
            if line.startswith("RESULT=")
        )
    )
    assert "run_id" not in context
    assert context["original_experiment_name"] == "same"
    assert re.fullmatch(
        r"same_[0-9]{8}-[0-9]{6}_[0-9a-f]{8}", context["experiment_name"]
    )
    metrics = context["metrics"]
    assert 'experiment_name="' + context["experiment_name"] + '"' in metrics
    assert "rl_insight_monitor_experiment_info" in metrics
    assert not re.search(r"[{,]run_id=", metrics)
    assert any(
        s["name"] == "actor_update"
        and s["attributes"]["experiment_name"] == context["experiment_name"]
        for s in stack.backend.span_attributes
    )
    for name, lane, count in (
        ("sleep", "early_replica", 2),
        ("driver_prepare", "early_driver", 1),
    ):
        spans = [s for s in stack.backend.span_attributes if s["name"] == name]
        assert len(spans) == count
        for span in spans:
            assert {
                "project": "ray",
                "experiment_name": context["experiment_name"],
                "state_lane_id": lane,
            }.items() <= span["attributes"].items()
    import yaml

    targets = yaml.safe_load(
        (stack.path / "data/targets/prometheus-targets.yml").read_text()
    )
    for address in ("127.0.0.1:19999", "127.0.0.1:19998"):
        assert any(
            address in group["targets"]
            and group["labels"]["experiment_name"] == context["experiment_name"]
            for group in targets
        )
