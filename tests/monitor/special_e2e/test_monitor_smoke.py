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

"""Data-path tests for a monitor stack managed by the CI workflow."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Generator
from typing import Any, cast

import pytest
import ray
import requests

import rl_insight as insight
from rl_insight.client.ray_monitor_client import _current_job_actor_name
from rl_insight.utils.constants import MonitorRayActor


pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the managed server stack only runs on Linux"
)

SERVER_URL = os.environ.get("RL_INSIGHT_SERVER_URL", "http://127.0.0.1:18080")
TEMPO_QUERY_URL = os.environ.get("RL_INSIGHT_TEMPO_QUERY_URL", "http://127.0.0.1:3200")
READY_TIMEOUT_SECONDS = 60
TEST_RUN_ID = uuid.uuid4().hex


def _wait_for_json(
    url: str,
    *,
    params: dict[str, str] | None = None,
    ready: Callable[[dict[str, Any]], bool] = bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, params=params, timeout=3)
            response.raise_for_status()
            payload = response.json()
            if ready(payload):
                return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
        time.sleep(1)
    raise AssertionError(
        f"{url} was not ready within {READY_TIMEOUT_SECONDS}s: {last_error}"
    )


@pytest.fixture(scope="module")
def monitor_stack() -> Generator[dict[str, str], None, None]:
    """Connect the reporting client to the stack started by the workflow."""
    os.environ["RL_INSIGHT_SERVER_URL"] = SERVER_URL
    services = _wait_for_json(
        f"{SERVER_URL}/api/v1/services",
        ready=lambda data: data.get("status") == "ok",
    )
    endpoints = {
        "prometheus": f"http://127.0.0.1:{services['prometheus_port']}",
        "otlp": f"http://127.0.0.1:{services['otlp_port']}",
        "tempo": TEMPO_QUERY_URL,
    }

    ray.init(namespace=MonitorRayActor.NAMESPACE, ignore_reinit_error=True)
    insight.init(project="rl-insight-e2e", experiment_name="monitor-smoke")
    try:
        yield endpoints
    finally:
        insight.finish()
        try:
            hub = ray.get_actor(
                _current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE
            )
            ray.kill(hub, no_restart=True)
        except ValueError:
            pass
        ray.shutdown()


def test_monitor_services_should_be_reachable_when_stack_is_running(
    monitor_stack: dict[str, str],
) -> None:
    """Verify the control, metrics, and trace data paths are reachable."""
    health = requests.get(f"{SERVER_URL}/healthz", timeout=3)
    prometheus = requests.get(f"{monitor_stack['prometheus']}/-/ready", timeout=3)
    tempo = requests.get(f"{monitor_stack['tempo']}/ready", timeout=3)
    otlp = requests.get(f"{monitor_stack['otlp']}/v1/traces", timeout=3)
    hub = ray.get_actor(_current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE)
    hub_status = cast(dict[str, Any], ray.get(hub.get_status.remote()))

    assert health.json() == {"status": "ok"}
    assert prometheus.ok
    assert tempo.ok
    assert otlp.status_code < 500
    assert requests.get(hub_status["metrics_endpoint"], timeout=3).ok
    assert hub_status["otel_traces_enabled"] is True


def test_monitor_metrics_should_match_reported_values_when_events_are_emitted(
    monitor_stack: dict[str, str],
) -> None:
    """Report counter, gauge, and histogram events and verify stored values."""
    for step in range(3):
        insight.metric_count(
            "train_step", amount=1, worker="trainer_0", test_run=TEST_RUN_ID
        )
        insight.metric_gauge(
            "reward_mean",
            value=1.0 + step * 0.01,
            worker="trainer_0",
            test_run=TEST_RUN_ID,
        )
        insight.metric_histogram(
            "step_latency_ms",
            value=200 + step * 20,
            worker="trainer_0",
            test_run=TEST_RUN_ID,
        )

    hub = ray.get_actor(_current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE)
    ray.get(hub.get_status.remote())

    expected_values = {
        "rl_insight_monitor_train_step_total": 3.0,
        "rl_insight_monitor_reward_mean": 1.02,
        "rl_insight_monitor_step_latency_ms_count": 3.0,
        "rl_insight_monitor_step_latency_ms_sum": 660.0,
    }
    for metric_name, expected in expected_values.items():
        payload = _wait_for_json(
            f"{monitor_stack['prometheus']}/api/v1/query",
            params={
                "query": (f'{metric_name}{{test_run="{TEST_RUN_ID}"}} == {expected}')
            },
            ready=lambda data: bool(data.get("data", {}).get("result")),
        )
        result = payload["data"]["result"]
        assert result[0]["metric"]["worker"] == "trainer_0"
        assert float(result[0]["value"][1]) == pytest.approx(expected)


def test_monitor_trace_should_be_queryable_when_trace_is_reported(
    monitor_stack: dict[str, str],
) -> None:
    """Report a state trace and verify Tempo stores its name and attributes."""
    trace_name = "monitor_e2e_rollout_generate"
    with insight.trace_state(
        trace_name,
        state_lane_id="replica_0",
        step=7,
        test_run=TEST_RUN_ID,
    ):
        time.sleep(0.1)

    hub = ray.get_actor(_current_job_actor_name(), namespace=MonitorRayActor.NAMESPACE)
    status = cast(dict[str, Any], ray.get(hub.get_status.remote()))
    assert status["otel_traces_enabled"] is True

    search = _wait_for_json(
        f"{monitor_stack['tempo']}/api/search",
        params={"q": f'{{ name = "{trace_name}" && span.test_run = "{TEST_RUN_ID}" }}'},
        ready=lambda data: bool(data.get("traces")),
    )
    trace_id = search["traces"][0]["traceID"]
    trace = _wait_for_json(f"{monitor_stack['tempo']}/api/traces/{trace_id}")
    serialized_trace = json.dumps(trace)

    assert trace_name in serialized_trace
    assert TEST_RUN_ID in serialized_trace
