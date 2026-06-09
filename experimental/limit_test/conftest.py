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

"""
Shared fixtures and pytest configuration.

This file is auto-loaded by test files under tests/experimental/.

Fixture overview:
  - reset_monitor_state  : resets the api._STATE singleton before and after each test
  - mock_client          : injects a MagicMock client so emit calls can be asserted
  - make_hub             : factory fixture that returns a MonitorHubActor without Ray dependencies
  - make_trace_event     : helper for building state_interval trace event dictionaries

Marker overview:
  - integration : tests that need a local Ray process (ray.init()), skipped by default
  - multinode   : tests that need a multi-node Ray cluster, skipped by default
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from experimental import api
from experimental.utils import MonitorEventKind


# ---------------------------------------------------------------------------
# pytest marker registration to avoid PytestUnknownMarkWarning.
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires local ray.init(); enable with --run-integration",
    )
    config.addinivalue_line(
        "markers",
        "multinode: requires a multi-node Ray cluster; enable with --run-multinode",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require a local Ray process",
    )
    parser.addoption(
        "--run-multinode",
        action="store_true",
        default=False,
        help="run end-to-end tests that require a multi-node Ray cluster",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration / multinode tests based on command-line options."""
    skip_integration = pytest.mark.skip(reason="requires --run-integration")
    skip_multinode = pytest.mark.skip(reason="requires --run-multinode")

    for item in items:
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "multinode" in item.keywords and not config.getoption("--run-multinode"):
            item.add_marker(skip_multinode)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def reset_monitor_state():
    """
    Reset the global api._STATE singleton.

    Use this fixture for any test that needs a clean _STATE.
    Declare `usefixtures("reset_monitor_state")` on a test class or function,
    or list it directly as a function argument.
    """
    api.close()
    yield
    api.close()


@pytest.fixture
def mock_client(reset_monitor_state):
    """
    Inject a MagicMock client and put _STATE into the enabled state.

    Returns a MagicMock object for assert_called / call_args assertions on apply_event.

    Example:
        def test_foo(mock_client):
            api.metric_value("reward", 1.0)
            mock_client.apply_event.assert_called_once()
    """
    client = MagicMock()
    api._STATE.enabled = True
    api._STATE.client = client
    api._STATE.process_id = "test-pid-0"
    return client


@pytest.fixture
def make_hub():
    """
    Factory fixture that returns a MonitorHubActor instantiated without Ray.

    Usage:
        def test_foo(make_hub):
            hub = make_hub()          # default configuration, OTLP disabled
            hub = make_hub(otlp=True) # inject a mock trace_collector

    Returns (hub, mock_trace_collector):
        - hub                 : MonitorHubActor instance; apply_event can be called directly
        - mock_trace_collector: None when otlp=False, otherwise a MagicMock
    """
    # Delay the import so environments without ray installed can still import this file.
    from experimental.collector.ray_monitor_hub import MonitorHubActor

    _hub_counter = [0]

    def _factory(port: int = 19092, otlp: bool = False):
        # Generate a unique namespace for each call to avoid prometheus_client registry conflicts.
        _hub_counter[0] += 1
        ns = f"test_ns_{port}_{_hub_counter[0]}"
        conf = {
            "namespace": ns,
            "prometheus": {
                "metrics_report_port": port,
                "reload": {"mode": "none"},
            },
            "otel": {
                "traces_endpoint": "http://fake:4318/v1/traces" if otlp else "",
            },
        }
        mock_collector = MagicMock() if otlp else None

        patch_targets = [
            patch("experimental.collector.ray_monitor_hub.start_metrics_http_server"),
            patch("experimental.collector.ray_monitor_hub.update_prometheus_config"),
        ]
        if otlp:
            patch_targets.append(
                patch(
                    "experimental.collector.ray_monitor_hub.OpenTelemetryTraceCollector",
                    return_value=mock_collector,
                )
            )

        for p in patch_targets:
            p.start()

        # MonitorHubActor is wrapped into an ActorClass by @ray.remote, so __new__ cannot
        # be called directly. Use __ray_actor_class__ to instantiate the underlying class.
        underlying_cls = MonitorHubActor.__ray_actor_class__

        hub = underlying_cls.__new__(underlying_cls)
        underlying_cls.__init__(hub, conf)

        for p in patch_targets:
            p.stop()

        if otlp:
            # Ensure hub references the mock instead of the original object.
            hub._trace_collector = mock_collector

        return hub, mock_collector

    return _factory


@pytest.fixture
def make_trace_event():
    """
    Helper fixture that builds a state_interval trace event dictionary.

    Usage:
        def test_foo(make_trace_event):
            event = make_trace_event("rollout", "worker_0", 0, 100_000_000)

    Arguments:
        name     : span name, such as "rollout" or "actor_update"
        lane     : state_lane_id, usually a worker identifier
        start_ns : start time in nanoseconds
        end_ns   : end time in nanoseconds
        **attrs  : extra attributes
    """
    def _make(name: str, lane: str, start_ns: int, end_ns: int, **attrs) -> dict:
        return {
            "kind": MonitorEventKind.TRACE,
            "name": name,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "attributes": {
                "monitor.trace_segment": "state_interval",
                "state_lane_id": lane,
                "state_name": name,
                **attrs,
            },
        }
    return _make
