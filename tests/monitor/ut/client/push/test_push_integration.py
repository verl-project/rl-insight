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

import logging
import sys
import types
from collections.abc import Generator
from typing import Any

import pytest
from omegaconf import OmegaConf

from rl_insight import api
from rl_insight.client.push import poller as poller_module
from rl_insight.client.push.sinks.base import PushSink

LOGGING_DRIVER = "rl_insight.client.push.sinks.logging:create_sink"


@pytest.fixture(autouse=True)
def reset_monitor_state() -> Generator[None, None, None]:
    api.finish()
    yield
    api.finish()
    poller_module.set_active_registry(None)


def test_init_push_without_server_url_is_enabled(caplog: pytest.LogCaptureFixture) -> None:
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

    module.create_sink = factory
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
                        {"source": "eng:ttft", "type": "summary_mean_us",
                         "name": "TTFT"}
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
