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

"""End-to-end tests for the push backend through the public rl_insight API."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import types
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from omegaconf import DictConfig, OmegaConf

from rl_insight import api
from rl_insight.client.push import poller as poller_module
from rl_insight.client.push.sinks.base import PushSink
from rl_insight.utils.prometheus_utils import update_prometheus_config

LOGGING_DRIVER = "rl_insight.client.push.sinks.logging:create_sink"


@pytest.fixture(autouse=True)
def reset_monitor_state() -> Generator[None, None, None]:
    api.finish()
    yield
    api.finish()
    poller_module.set_active_registry(None)


def test_init_push_without_server_url_is_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api.init(
        project="p",
        experiment_name="e",
        config={
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "driver": LOGGING_DRIVER,
                        "prefix": "root",
                        "streams": ["metric"],
                    }
                ]
            },
        },
    )

    assert api._STATE.enabled is True
    with caplog.at_level(logging.INFO, logger="rl_insight.push.logging"):
        api.metric_gauge("reward_mean", value=0.9)

    assert "store root.trainer.reward_mean=0.9" in caplog.text


def test_register_targets_does_not_contact_ray_for_trace_only_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    from rl_insight.client import ray_monitor_client

    monkeypatch.delenv("RL_INSIGHT_CONFIG", raising=False)
    monkeypatch.delenv("RL_INSIGHT_SERVER_BACKEND", raising=False)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    create_hub = Mock(side_effect=AssertionError("unexpected Ray Hub connection"))
    monkeypatch.setattr(ray_monitor_client, "get_or_create_monitor_hub", create_hub)
    api.init(
        config={
            "server": {"backend": "push"},
            "push": {"sinks": [{"driver": LOGGING_DRIVER, "streams": ["trace"]}]},
        }
    )
    assert api._STATE.enabled
    assert poller_module.get_active_registry() is None

    update_prometheus_config(["127.0.0.1:9000"])

    create_hub.assert_not_called()


def test_init_push_registers_rollout_targets_and_finish_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks: list[Any] = []
    module = types.ModuleType("fake_e2e_sink_mod")

    class RecordingSink(PushSink):
        def __init__(self, prefix: str = "") -> None:
            super().__init__(prefix)
            self.closed = False
            sinks.append(self)

        def emit_counter(self, name, value, tags):
            pass

        def emit_store(self, name, value, tags):
            pass

        def emit_timer(self, name, value_us, tags):
            pass

        def close(self):
            self.closed = True

    def factory(conf: Any) -> RecordingSink:
        return RecordingSink(str(OmegaConf.select(conf, "prefix") or ""))

    setattr(module, "create_sink", factory)
    sys.modules["fake_e2e_sink_mod"] = module

    api.init(
        config={
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "driver": "fake_e2e_sink_mod:create_sink",
                        "streams": ["metric", "rollout"],
                    }
                ],
                "rollout": {
                    "interval_seconds": 0,
                    "metrics": [
                        {
                            "source": "eng:ttft",
                            "type": "summary_mean_us",
                            "name": "TTFT",
                        }
                    ],
                },
            },
        }
    )

    assert api._STATE.enabled is True
    assert poller_module.get_active_registry() is not None

    api.finish()

    assert poller_module.get_active_registry() is None
    assert sinks and sinks[0].closed is True
    assert api._STATE.enabled is False


def test_init_ray_without_server_url_stays_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    api.init(config={"server": {"backend": "ray", "url": ""}})
    assert api._STATE.enabled is False


@pytest.fixture
def early_push_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DictConfig:
    conf = OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "driver": LOGGING_DRIVER,
                        "prefix": "root",
                        "streams": ["trace", "rollout"],
                    }
                ],
                "rollout": {
                    "interval_seconds": 60,
                    "metrics": [
                        {
                            "source": "eng:ttft",
                            "type": "summary_mean_us",
                            "name": "TTFT",
                        }
                    ],
                },
            },
        }
    )
    path = tmp_path / "push.yaml"
    OmegaConf.save(conf, path)
    monkeypatch.setenv("RL_INSIGHT_CONFIG", str(path))
    monkeypatch.delenv("RL_INSIGHT_SERVER_BACKEND", raising=False)
    return conf


def test_register_targets_reuses_push_client_prepared_by_state_trace(
    early_push_config: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    with api.trace_state("early"):
        pass
    monkeypatch.delenv("RL_INSIGHT_CONFIG")

    update_prometheus_config(["host-a:9000", "host-b:9000"], labels=[{"rank": 0}, None])

    registry = poller_module.get_active_registry()
    assert registry is not None
    assert [(t.address, t.labels) for t in registry.snapshot()] == [
        ("host-a:9000", {"rank": "0"}),
        ("host-b:9000", {}),
    ]


@pytest.mark.parametrize("environment_override", [False, True])
def test_push_state_traces_work_before_init_and_finish_releases_resources(
    early_push_config: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    environment_override: bool,
) -> None:
    if environment_override:
        early_push_config.server.backend = "ray"
        path = tmp_path / "ray.yaml"
        OmegaConf.save(early_push_config, path)
        monkeypatch.setenv("RL_INSIGHT_CONFIG", str(path))
        monkeypatch.setenv("RL_INSIGHT_SERVER_BACKEND", "push")

    async def one(index: int) -> None:
        with api.trace_state(f"early_{index}", state_lane_id=f"lane_{index}"):
            await asyncio.sleep(0)

    async def run() -> None:
        await asyncio.gather(*(one(index) for index in range(16)))

    with caplog.at_level(logging.INFO, logger="rl_insight.push.logging"):
        asyncio.run(run())
        assert caplog.text.count("[push] span root.trace.early_") == 16
        assert "'project'" not in caplog.text
        with (
            pytest.raises(ValueError, match="business failure"),
            api.trace_state("business"),
        ):
            raise ValueError("business failure")

    client = api._STATE.client
    assert client is not None
    assert api._STATE.enabled is False
    assert api._STATE.labels == {}
    thread = client._poller._thread
    assert thread.is_alive()
    api.finish()
    assert client._closed
    assert not thread.is_alive()
    assert poller_module.get_active_registry() is None


@pytest.mark.parametrize("bootstrap_first", [False, True])
def test_formal_init_reuses_early_push_client_and_keeps_configured_names(
    early_push_config: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    bootstrap_first: bool,
) -> None:
    monkeypatch.setenv("PROJECT_NAME", "configured-project")
    monkeypatch.setenv("EXPERIMENT_NAME", "configured-experiment")
    monkeypatch.setattr(api, "_push_bootstrap_attempted", False)
    early_push_config.push.sinks[0].tags_from_env = [
        {"tag": "project", "env": "PROJECT_NAME"},
        {"tag": "experiment_name", "env": "EXPERIMENT_NAME"},
    ]
    OmegaConf.save(early_push_config, os.environ["RL_INSIGHT_CONFIG"])
    if bootstrap_first:
        update_prometheus_config(["127.0.0.1:9000"], "vllm", [{"replica": 0}])
    with (
        caplog.at_level(logging.INFO, logger="rl_insight.push.logging"),
        api.trace_state("early"),
    ):
        pass
    assert "'project': 'configured-project'" in caplog.text
    assert "'experiment_name': 'configured-experiment'" in caplog.text
    caplog.clear()
    client = api._STATE.client
    registry = poller_module.get_active_registry()
    assert registry is not None
    update_prometheus_config(["127.0.0.1:9000"], "vllm", [{"replica": 0}])
    assert api._STATE.enabled is bootstrap_first

    with (
        pytest.warns(RuntimeWarning, match="ignoring re-initialization")
        if bootstrap_first
        else nullcontext()
    ):
        api.init(project="configured-project", experiment_name="configured-experiment")

    assert api._STATE.enabled
    assert api._STATE.client is client
    assert poller_module.get_active_registry() is registry
    assert registry.snapshot()[0].labels == {"replica": "0"}
    with (
        caplog.at_level(logging.INFO, logger="rl_insight.push.logging"),
        api.trace_state("formal"),
    ):
        pass
    assert "'project': 'configured-project'" in caplog.text
    assert "'experiment_name': 'configured-experiment'" in caplog.text
    with pytest.warns(RuntimeWarning, match="ignoring re-initialization"):
        api.init(project="ignored")
    assert api._STATE.client is client
    assert api._STATE.labels["project"] == "configured-project"


def test_finish_closes_client_when_early_trace_publishes_late(
    early_push_config: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    create = api.create_monitor_client
    ready = threading.Event()
    publish = threading.Event()

    def delayed_create(conf=None):
        client = create(conf)
        if conf is None:
            ready.set()
            assert publish.wait(timeout=10)
        return client

    monkeypatch.setattr(api, "create_monitor_client", delayed_create)

    def trace() -> None:
        with api.trace_state("early"):
            pass

    with ThreadPoolExecutor(max_workers=1) as pool:
        traced = pool.submit(trace)
        try:
            assert ready.wait(timeout=10)
            api.init()
            final_client = api._STATE.client
            assert final_client is not None
            thread = final_client._poller._thread
        finally:
            publish.set()
        traced.result(timeout=10)

    try:
        assert api._STATE.client is final_client
        api.finish()
        assert final_client._closed
        assert not thread.is_alive()
        with api.trace_state("next_execution"):
            pass
        assert api._STATE.client is not None
        assert api._STATE.client is not final_client
        assert not api._STATE.client._closed
    finally:
        final_client.close()
