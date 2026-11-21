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

"""Push worker initialization regressions without importing verl.

这三个用例复现 Push 模式下 worker 先于训练 logger 正式初始化时的问题。

问题 1: Push sink 模式不支持 pre-init attach。worker 的早期 state trace
必须能根据 ``RL_INSIGHT_CONFIG`` 创建 process-local Push client，但不能
触发 formal ``init()``，否则会先用默认身份初始化。

问题 2: 早期 worker report 不能用 default/default 抢占身份。后续训练
logger formal init 后，必须复用已 attach 的 Push client，并让所有事件带
配置好的 project / experiment_name，而不是默认身份。

问题 3: worker 收不到 ``RL_INSIGHT_CONFIG`` 时，不能把“监控禁用”永久
latch。之后 formal init 带来正确 Push 配置时，必须能恢复创建 client。

这里抽出 verl ``RLInsightLogger`` 的最小行为副本，是为了把测试边界限制
在 rl-insight 与 logger 逻辑本身；真实后端如何 push 不在覆盖范围内。
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import pytest
from omegaconf import OmegaConf

from rl_insight import api
from rl_insight.client.push import poller as poller_module
from rl_insight.client.push.sinks.base import PushSink


class RecordingSink(PushSink):
    """Record every push emission; no external backend is involved."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        self.events: list[dict[str, Any]] = []

    def emit_counter(self, name: str, value: float, tags: Mapping[str, str]) -> None:
        self.events.append(
            {
                "kind": "counter",
                "name": self.metric_name(name),
                "value": value,
                "tags": dict(tags),
            }
        )

    def emit_store(self, name: str, value: float, tags: Mapping[str, str]) -> None:
        self.events.append(
            {
                "kind": "store",
                "name": self.metric_name(name),
                "value": value,
                "tags": dict(tags),
            }
        )

    def emit_timer(self, name: str, value_us: float, tags: Mapping[str, str]) -> None:
        self.events.append(
            {
                "kind": "timer",
                "name": self.metric_name(name),
                "value_us": value_us,
                "tags": dict(tags),
            }
        )

    def emit_span(
        self,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Mapping[str, Any],
    ) -> None:
        self.events.append(
            {
                "kind": "span",
                "name": self.metric_name(name),
                "start_time_ns": start_time_ns,
                "end_time_ns": end_time_ns,
                "attributes": dict(attributes),
            }
        )


class MinimalRLInsightLogger:
    """Minimal copy of verl's early-worker initialization behavior."""

    ENABLE_ENV = "VERL_RL_INSIGHT_ENABLE"
    _init_done = False
    _module: Any | None = None

    @classmethod
    def _get_rl_insight(cls) -> Any:
        if cls._module is None:
            import rl_insight

            cls._module = rl_insight
        return cls._module

    @classmethod
    def enabled(cls) -> bool:
        return os.getenv(cls.ENABLE_ENV) == "1"

    @classmethod
    def init(
        cls,
        project_name: str | None = None,
        experiment_name: str | None = None,
        config: Any | None = None,
    ) -> None:
        if not cls.enabled() or cls._init_done:
            return
        rl_insight_config: Any = {}
        if config is not None:
            try:
                rl_insight_config = (
                    config.get("trainer", {}).get("rl_insight", {}) or {}
                )
            except (AttributeError, KeyError, TypeError):
                pass
        cls._get_rl_insight().init(
            project=project_name,
            experiment_name=experiment_name,
            config=rl_insight_config,
        )
        cls._init_done = True

    @classmethod
    def log(cls, data: dict[str, Any], step: int) -> None:
        if not cls.enabled():
            return
        cls._ensure_rl_insight_init()
        module = cls._get_rl_insight()
        for key, value in data.items():
            module.metric_log(str(key), float(value), step=int(step))

    @classmethod
    @contextmanager
    def trace_state(
        cls,
        state_name: str,
        *,
        state_lane_id: str | int | None = None,
        **labels: Any,
    ):
        if not cls.enabled():
            yield
            return
        # 这个调用是复现点：worker 早期上报不应该触发 formal init。
        cls._ensure_rl_insight_init()
        with cls._get_rl_insight().trace_state(
            state_name, state_lane_id=state_lane_id, **labels
        ):
            yield

    @classmethod
    def _ensure_rl_insight_init(cls) -> None:
        if not cls._init_done:
            cls._get_rl_insight().init()
            cls._init_done = True

    @classmethod
    def reset_for_test(cls) -> None:
        cls._init_done = False
        cls._module = None


@pytest.fixture(autouse=True)
def _reset_monitor_state() -> None:
    api.finish()
    poller_module.set_active_registry(None)
    MinimalRLInsightLogger.reset_for_test()
    yield
    api.finish()
    poller_module.set_active_registry(None)
    MinimalRLInsightLogger.reset_for_test()
    sys.modules.pop("fake_worker_push_sink_mod", None)


def _install_fake_driver() -> None:
    module = types.ModuleType("fake_worker_push_sink_mod")

    def create_sink(conf: Any) -> RecordingSink:
        return RecordingSink(str(OmegaConf.select(conf, "prefix") or ""))

    setattr(module, "create_sink", create_sink)
    sys.modules["fake_worker_push_sink_mod"] = module


def _write_push_config(tmp_path: Path) -> Path:
    conf = OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "driver": "fake_worker_push_sink_mod:create_sink",
                        "prefix": "repro",
                        "streams": ["metric", "trace"],
                        "tags_from_env": [
                            {"tag": "project", "env": "PROJECT_NAME"},
                            {"tag": "experiment_name", "env": "EXPERIMENT_NAME"},
                        ],
                    }
                ]
            },
        }
    )
    path = tmp_path / "push.yaml"
    OmegaConf.save(conf, path)
    return path


def _recording_sink() -> RecordingSink:
    client = api._STATE.client
    assert client is not None
    assert client._bound
    sink = client._bound[0].sink
    assert isinstance(sink, RecordingSink)
    return sink


def _prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, with_config: bool
) -> None:
    monkeypatch.setenv(MinimalRLInsightLogger.ENABLE_ENV, "1")
    monkeypatch.setenv("PROJECT_NAME", "configured-project")
    monkeypatch.setenv("EXPERIMENT_NAME", "configured-experiment")
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    monkeypatch.delenv("RL_INSIGHT_SERVER_BACKEND", raising=False)
    if with_config:
        monkeypatch.setenv("RL_INSIGHT_CONFIG", str(_write_push_config(tmp_path)))
    else:
        monkeypatch.delenv("RL_INSIGHT_CONFIG", raising=False)
    _install_fake_driver()


def test_early_push_worker_attaches_without_formal_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Push worker 先收到 metrics/traces，此时训练 logger 尚未 formal init。
    # 预期：只 attach process-local Push client；事件由 sink tags 提供身份。
    _prepare(monkeypatch, tmp_path, with_config=True)

    with MinimalRLInsightLogger.trace_state("early_worker", state_lane_id="vllm-0"):
        pass

    assert MinimalRLInsightLogger._init_done is False
    assert api._STATE.enabled is False
    assert api._STATE.client is not None
    assert api._STATE.labels == {}
    events = _recording_sink().events
    assert len(events) == 1
    assert events[0]["attributes"]["project"] == "configured-project"
    assert events[0]["attributes"]["experiment_name"] == "configured-experiment"


def test_formal_init_reuses_attached_push_client_and_configured_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 防止 default/default 抢占身份：早期 attach 和 formal init 必须复用同一
    # Push client，formal init 只补充 labels，不创建第二个 sink/poller。
    _prepare(monkeypatch, tmp_path, with_config=True)

    with MinimalRLInsightLogger.trace_state("early_worker", state_lane_id="vllm-0"):
        pass
    early_client = api._STATE.client
    early_sink = _recording_sink()
    assert early_client is not None
    assert MinimalRLInsightLogger._init_done is False

    MinimalRLInsightLogger.init(
        project_name="configured-project",
        experiment_name="configured-experiment",
        config={"trainer": {"rl_insight": {"server": {"backend": "push"}}}},
    )
    MinimalRLInsightLogger.log({"loss": 0.25}, step=1)

    assert api._STATE.client is early_client
    assert api._STATE.labels == {
        "project": "configured-project",
        "experiment_name": "configured-experiment",
    }
    events = early_sink.events
    assert len(events) == 2
    assert events[0]["attributes"]["project"] == "configured-project"
    assert events[1]["attributes"]["project"] == "configured-project"
    assert events[1]["attributes"]["experiment_name"] == "configured-experiment"


def test_missing_worker_config_does_not_latch_disabled_and_late_init_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # worker 进程没有收到 RL_INSIGHT_CONFIG 时，早期上报最多保持未初始化，
    # 不能把 disabled 状态永久 latch；formal init 带来 Push 配置后必须恢复。
    _prepare(monkeypatch, tmp_path, with_config=False)

    with MinimalRLInsightLogger.trace_state(
        "worker_without_config", state_lane_id="vllm-0"
    ):
        pass
    assert MinimalRLInsightLogger._init_done is False
    assert api._STATE.client is None

    MinimalRLInsightLogger.init(
        project_name="configured-project",
        experiment_name="configured-experiment",
        config={
            "trainer": {
                "rl_insight": {
                    "server": {"backend": "push"},
                    "push": {
                        "sinks": [
                            {
                                "driver": "fake_worker_push_sink_mod:create_sink",
                                "streams": ["metric", "trace"],
                            }
                        ]
                    },
                }
            }
        },
    )
    assert api._STATE.enabled is True
    assert api._STATE.labels == {
        "project": "configured-project",
        "experiment_name": "configured-experiment",
    }
    MinimalRLInsightLogger.log({"loss": 0.25}, step=1)
    events = _recording_sink().events
    assert len(events) == 1
    assert events[0]["attributes"]["project"] == "configured-project"
    assert events[0]["attributes"]["experiment_name"] == "configured-experiment"
