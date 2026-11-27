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

"""Unit tests for PushMonitorClient event mapping, stream routing, fan-out."""

from __future__ import annotations

import sys
import types
from collections.abc import Generator
from typing import Any

import pytest
from omegaconf import OmegaConf

from rl_insight.client.push import poller as poller_module
from rl_insight.client.push.client import create_push_monitor_client
from rl_insight.utils import prometheus_utils as prom_utils
from rl_insight import api
from rl_insight.client.push.sinks.base import PushSink


class RecordingSink(PushSink):
    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        # Heterogeneous records: metric ops are 4-tuples, spans are 5-tuples.
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def emit_counter(self, name, value, tags):
        self.calls.append(("counter", self.metric_name(name), value, dict(tags)))

    def emit_store(self, name, value, tags):
        self.calls.append(("store", self.metric_name(name), value, dict(tags)))

    def emit_timer(self, name, value_us, tags):
        self.calls.append(("timer", self.metric_name(name), value_us, dict(tags)))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_active() -> Generator[None, None, None]:
    yield
    poller_module.set_active_registry(None)


class FullRecordingSink(RecordingSink):
    """Sink that also overrides the optional histogram/span primitives."""

    def emit_histogram(self, name, value, tags):  # type: ignore[override]
        self.calls.append(("histogram", self.metric_name(name), value, dict(tags)))

    def emit_span(self, name, start_ns, end_ns, attributes):  # type: ignore[override]
        self.calls.append(
            (
                "span",
                self.metric_name(name),
                int(start_ns),
                int(end_ns),
                dict(attributes),
            )
        )


def _install_driver(
    name: str,
    sinks: list[RecordingSink],
    factory: str = "create_sink",
    sink_cls: type[RecordingSink] = RecordingSink,
):
    module = types.ModuleType(name)

    def create_sink(conf: Any) -> RecordingSink:
        prefix = str(OmegaConf.select(conf, "prefix") or "")
        sink = sink_cls(prefix)
        sinks.append(sink)
        return sink

    def create_raising_sink(conf: Any) -> RecordingSink:
        sink = create_sink(conf)

        def boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("sdk down")

        sink.emit_store = boom  # type: ignore[method-assign]
        return sink

    setattr(module, "create_sink", create_sink)
    setattr(module, "create_raising_sink", create_raising_sink)
    sys.modules[name] = module
    return module


def _dual_conf(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TRIAL_ID", "trial-42")
    return OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "name": "alpha",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "prefix": "seed.alphaseed",
                        "streams": ["metric", "trace", "rollout"],
                        "tags_from_env": [
                            {"tag": "trial", "env": ["TRIAL_ID"], "default": "0"}
                        ],
                    },
                    {
                        "name": "xgpt",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "prefix": "xgpt.server.infer",
                        "streams": ["rollout"],
                        "rollout_name_prefix": "",
                    },
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


def test_gauge_routes_to_metric_sink_as_store_with_prefix_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    client = create_push_monitor_client(_dual_conf(monkeypatch))
    assert client is not None

    client.apply_event(
        {"kind": "gauge", "name": "reward/mean", "value": 1.25, "labels": {"step": "3"}}
    )

    alpha, xgpt = sinks
    assert len(alpha.calls) == 1
    op, full_name, value, tags = alpha.calls[0]
    assert (op, full_name, value) == (
        "store",
        "seed.alphaseed.trainer.reward/mean",
        1.25,
    )
    assert tags["trial"] == "trial-42"
    assert tags["step"] == "3"
    assert xgpt.calls == []  # xgpt does not subscribe to the metric stream
    client.close()


def test_metric_mapping_renames_retags_scales_and_whitelists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARNOLD_TRIAL_ID", "303094623")
    sinks: list[RecordingSink] = []
    _install_driver("fake_map_mod", sinks)
    conf = OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "name": "alpha",
                        "driver": "fake_map_mod:create_sink",
                        "prefix": "seed.alphaseed",
                        "streams": ["metric"],
                        "tags_from_env": [
                            {
                                "tag": "arnold_trial_id",
                                "env": ["ARNOLD_TRIAL_ID"],
                                "default": "0",
                            }
                        ],
                        "metric_mapping": [
                            {"source": "training_global_step", "name": "global_step"},
                            {
                                "source": "perf_mfu_actor",
                                "name": "mfu.ratio",
                                "op": "timer",
                                "tags": {"stage": "actor"},
                            },
                            {
                                "source": "timing_s_update_weight",
                                "name": "timing.driver.duration",
                                "op": "timer",
                                "scale": 1_000_000.0,
                                "tags": {"stage": "weight_sync_total"},
                            },
                        ],
                    }
                ]
            },
        }
    )
    client = create_push_monitor_client(conf)
    assert client is not None

    client.apply_event(
        {"kind": "gauge", "name": "training_global_step", "value": 7, "labels": {}}
    )
    client.apply_event(
        {"kind": "gauge", "name": "perf_mfu_actor", "value": 0.42, "labels": {}}
    )
    client.apply_event(
        {
            "kind": "gauge",
            "name": "timing_s_update_weight",
            "value": 0.003,
            "labels": {},
        }
    )
    client.apply_event(
        {"kind": "gauge", "name": "critic_score_mean", "value": 1.0, "labels": {}}
    )

    alpha = sinks[0]
    assert [c[0:2] for c in alpha.calls] == [
        ("store", "seed.alphaseed.global_step"),
        ("timer", "seed.alphaseed.mfu.ratio"),
        ("timer", "seed.alphaseed.timing.driver.duration"),
    ]
    assert alpha.calls[0][2] == 7
    assert alpha.calls[0][3]["arnold_trial_id"] == "303094623"
    assert alpha.calls[1][2] == 0.42
    assert alpha.calls[1][3]["stage"] == "actor"
    assert alpha.calls[2][2] == 3000.0  # seconds -> microseconds
    assert alpha.calls[2][3]["stage"] == "weight_sync_total"
    client.close()


def test_counter_and_histogram_event_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    client = create_push_monitor_client(_dual_conf(monkeypatch))
    assert client is not None

    client.apply_event({"kind": "counter", "name": "steps", "value": 2, "labels": {}})
    client.apply_event({"kind": "histogram", "name": "lat", "value": 900, "labels": {}})

    ops = [(c[0], c[1], c[2]) for c in sinks[0].calls]
    assert ("counter", "seed.alphaseed.trainer.steps", 2) in ops
    assert ("timer", "seed.alphaseed.trainer.lat", 900) in ops
    client.close()


def test_trace_event_converts_ns_to_us(monkeypatch: pytest.MonkeyPatch) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    client = create_push_monitor_client(_dual_conf(monkeypatch))
    assert client is not None

    client.apply_event(
        {
            "kind": "trace",
            "name": "generate",
            "start_time_ns": 1_000_000,
            "end_time_ns": 3_500_000,
            "attributes": {"state_lane_id": "replica_0", "seq": [1, 2]},
        }
    )

    op, full_name, value, tags = sinks[0].calls[0]
    assert (op, full_name, value) == (
        "timer",
        "seed.alphaseed.trace.generate",
        2500.0,
    )
    assert tags["state_lane_id"] == "replica_0"
    assert "seq" not in tags  # non-scalar attributes are dropped
    assert sinks[1].calls == []
    client.close()


def test_histogram_and_trace_keep_native_primitives_when_sink_supports_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks, sink_cls=FullRecordingSink)
    client = create_push_monitor_client(_dual_conf(monkeypatch))
    assert client is not None

    client.apply_event({"kind": "histogram", "name": "lat", "value": 900, "labels": {}})
    client.apply_event(
        {
            "kind": "trace",
            "name": "generate",
            "start_time_ns": 1_000_000,
            "end_time_ns": 3_500_000,
            "attributes": {"state_lane_id": "replica_0", "seq": [1, 2]},
        }
    )

    ops = sinks[0].calls
    assert (
        "histogram",
        "seed.alphaseed.trainer.lat",
        900.0,
        {"trial": "trial-42"},
    ) in ops
    span = next(c for c in ops if c[0] == "span")
    _, full_name, start_ns, end_ns, attrs = span
    assert full_name == "seed.alphaseed.trace.generate"
    assert (start_ns, end_ns) == (1_000_000, 3_500_000)
    # The tracing sink receives raw attributes, including non-scalar sequences.
    assert attrs["state_lane_id"] == "replica_0"
    assert attrs["seq"] == [1, 2]
    assert attrs["trial"] == "trial-42"  # bound sink tags are merged
    client.close()


def test_rollout_emit_fans_out_with_per_sink_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    client = create_push_monitor_client(_dual_conf(monkeypatch))
    assert client is not None

    client.emit_rollout("TTFT", "timer", 100_000.0, {"replica": "0"})

    alpha, xgpt = sinks
    assert alpha.calls[0][:3] == (
        "timer",
        "seed.alphaseed.rollout.TTFT",
        100_000.0,
    )
    assert alpha.calls[0][3]["replica"] == "0"
    assert alpha.calls[0][3]["trial"] == "trial-42"
    assert xgpt.calls[0][:3] == ("timer", "xgpt.server.infer.TTFT", 100_000.0)
    assert xgpt.calls[0][3] == {"replica": "0"}
    client.close()


def test_malformed_sink_config_is_skipped_without_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    conf = OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    # Valid sink must keep working despite the bad neighbour.
                    {
                        "name": "good",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "prefix": "p",
                        "streams": ["metric"],
                    },
                    # Unknown op in metric_mapping -> this sink is dropped.
                    {
                        "name": "bad-op",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "prefix": "p",
                        "streams": ["metric"],
                        "metric_mapping": [
                            {"source": "x", "name": "y", "op": "not_an_op"}
                        ],
                    },
                    # Non-numeric scale -> this sink is dropped.
                    {
                        "name": "bad-scale",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "prefix": "p",
                        "streams": ["metric"],
                        "metric_mapping": [
                            {"source": "x", "name": "y", "scale": "fast"}
                        ],
                    },
                ]
            },
        }
    )

    with caplog.at_level("WARNING", logger="rl_insight.client.push.client"):
        client = create_push_monitor_client(conf)
    assert client is not None  # malformed entries do not disable monitoring

    client.apply_event({"kind": "gauge", "name": "reward", "value": 1, "labels": {}})
    # All three drivers loaded, but only the well-configured one was bound and
    # received the event; the two malformed sinks were skipped with a warning.
    active = [s for s in sinks if s.calls]
    assert len(active) == 1
    assert active[0].calls[0][:2] == ("store", "p.trainer.reward")
    assert any("configured incorrectly" in r.message for r in caplog.records)
    client.close()


def test_all_sinks_misconfigured_disables_push_without_raising() -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    conf = OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "name": "bad",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "metric_mapping": [{"source": "", "name": ""}],
                    }
                ]
            },
        }
    )

    assert create_push_monitor_client(conf) is None
    # The driver loaded one instance, but it was never bound (no emissions).
    assert len(sinks) == 1
    assert sinks[0].calls == []


def test_invalid_rollout_interval_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sinks: list[RecordingSink] = []
    _install_driver("fake_push_sinks_mod", sinks)
    conf = OmegaConf.create(
        {
            "server": {"backend": "push"},
            "push": {
                "sinks": [
                    {
                        "name": "a",
                        "driver": "fake_push_sinks_mod:create_sink",
                        "streams": ["metric"],
                    }
                ],
                "rollout": {"interval_seconds": "soon"},
            },
        }
    )
    with caplog.at_level("WARNING", logger="rl_insight.client.push.client"):
        client = create_push_monitor_client(conf)
    assert client is not None
    assert client._poller is None  # no rollout.metrics rules
    assert any("interval_seconds" in r.message for r in caplog.records)
    client.close()


def test_failing_sink_is_removed_but_others_keep_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good: list[RecordingSink] = []
    bad_mod = types.ModuleType("fake_bad_mod")

    def make_bad(conf: Any) -> RecordingSink:
        sink = RecordingSink("bad")

        def boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("sdk down")

        sink.emit_store = boom  # type: ignore[method-assign]
        return sink

    setattr(bad_mod, "create_sink", make_bad)
    sys.modules["fake_bad_mod"] = bad_mod
    good_mod = _install_driver("fake_good_mod", good)

    conf = OmegaConf.create(
        {
            "push": {
                "sinks": [
                    {
                        "name": "bad",
                        "driver": "fake_bad_mod:create_sink",
                        "prefix": "bad",
                        "streams": ["metric"],
                    },
                    {
                        "name": "good",
                        "driver": "fake_good_mod:create_sink",
                        "prefix": "good",
                        "streams": ["metric"],
                    },
                ]
            }
        }
    )
    client = create_push_monitor_client(conf)
    assert client is not None

    client.apply_event({"kind": "gauge", "name": "x", "value": 1, "labels": {}})
    client.apply_event({"kind": "gauge", "name": "x", "value": 2, "labels": {}})

    assert [c[2] for c in good[0].calls] == [1, 2]
    assert len(client._bound) == 1  # bad sink removed after its first failure
    client.close()
    assert good[0].closed
    sys.modules.pop("fake_bad_mod", None)
    _ = good_mod


def test_factory_returns_none_without_sinks() -> None:
    conf = OmegaConf.create({"server": {"backend": "push"}, "push": {}})
    assert create_push_monitor_client(conf) is None


def test_factory_returns_none_when_all_drivers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf = OmegaConf.create(
        {"push": {"sinks": [{"driver": "missing_module_xyz:factory"}]}}
    )
    assert create_push_monitor_client(conf) is None


def _reset_bootstrap() -> None:
    api.finish()
    poller_module.set_active_registry(None)
    api._push_bootstrap_attempted = False


@pytest.mark.parametrize("identity_from_env", [False, True])
@pytest.mark.parametrize("invalid_sink", [False, True])
def test_register_targets_bootstraps_push_client_in_uninitialized_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    identity_from_env: bool,
    invalid_sink: bool,
) -> None:
    # Simulate a Ray rollout actor process: rl_insight.init() was never called,
    # but RL_INSIGHT_CONFIG points at a push config and the engine registers its
    # scrape targets via update_prometheus_config().
    sinks: list[RecordingSink] = []
    _install_driver("fake_boot_mod", sinks)
    conf_path = tmp_path / "push.yaml"
    conf_path.write_text(
        """
server:
  backend: push
push:
  sinks:
    - name: xgpt
      driver: fake_boot_mod:create_sink
      prefix: xgpt.server.infer
      streams: [rollout]
      rollout_name_prefix: ""
      tags_from_env:
        - {tag: project, env: PROJECT_NAME}
        - {tag: experiment_name, env: EXPERIMENT_NAME}
  rollout:
    interval_seconds: 60
    metrics:
      - {source: "eng:ttft", type: summary_mean_us, name: TTFT}
""",
        encoding="utf-8",
    )
    if invalid_sink:
        conf = OmegaConf.load(conf_path)
        conf.push.sinks.append(
            {
                "driver": "fake_boot_mod:create_sink",
                "streams": ["rollout"],
                "tags_from_env": [
                    {"tag": "host", "default": "worker", "format": "{missing}"}
                ],
            }
        )
        OmegaConf.save(conf, conf_path)
    monkeypatch.setenv("RL_INSIGHT_CONFIG", str(conf_path))
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    monkeypatch.delenv("EXPERIMENT_NAME", raising=False)
    names = {"project": "default", "experiment_name": "default"}
    if identity_from_env:
        monkeypatch.setenv("PROJECT_NAME", "configured-project")
        monkeypatch.setenv("EXPERIMENT_NAME", "configured-experiment")
        names = {
            "project": "configured-project",
            "experiment_name": "configured-experiment",
        }
    _reset_bootstrap()
    try:
        prom_utils.update_prometheus_config(
            ["[fdbd::1]:1", "127.0.0.1:2"],
            "vllm",
            [{"replica": 0}, {"replica": 1}],
        )
        registry = poller_module.get_active_registry()
        assert registry is not None
        snap = {t.address: t.labels for t in registry.snapshot()}
        assert snap == {
            "[fdbd::1]:1": {"replica": "0"},
            "127.0.0.1:2": {"replica": "1"},
        }
        assert len(sinks) == (2 if invalid_sink else 1)
        client = api._STATE.client
        assert client is not None
        assert len(client._bound) == 1
        assert api._STATE.enabled is True
        assert api._STATE.labels == names
        with pytest.warns(RuntimeWarning, match="ignoring re-initialization"):
            api.init(project="ignored-project", experiment_name="ignored-experiment")
        assert api._STATE.client is client
        assert api._STATE.labels == names
        assert poller_module.get_active_registry() is registry
        assert len(sinks) == (2 if invalid_sink else 1)
        api.finish()
        assert sinks[0].closed
        assert poller_module.get_active_registry() is None
    finally:
        _reset_bootstrap()
        sys.modules.pop("fake_boot_mod", None)


def test_register_targets_does_not_bootstrap_for_non_push_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    conf_path = tmp_path / "ray.yaml"
    conf_path.write_text(
        """
server:
  backend: ray
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("RL_INSIGHT_CONFIG", str(conf_path))
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    _reset_bootstrap()
    try:
        prom_utils.update_prometheus_config(["127.0.0.1:1"])
        assert poller_module.get_active_registry() is None
    finally:
        _reset_bootstrap()
