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

"""Unit tests for rollout target registry and the background poller."""

from __future__ import annotations

import time
from typing import Any

from omegaconf import OmegaConf

from rl_insight.client.push import poller as poller_module
from rl_insight.client.push.poller import PrometheusPoller, TargetRegistry

_RULES = OmegaConf.create(
    [
        {"source": "eng:tokens_total", "type": "counter_delta", "name": "tokens"},
        {"source": "eng:running", "type": "gauge", "name": "running"},
    ]
)

_TEXT_1 = """
# TYPE eng:tokens_total counter
eng:tokens_total 10.0
# TYPE eng:running gauge
eng:running 2.0
"""

_TEXT_2 = """
# TYPE eng:tokens_total counter
eng:tokens_total 30.0
# TYPE eng:running gauge
eng:running 5.0
"""


def test_target_registry_should_upsert_and_stringify_labels() -> None:
    registry = TargetRegistry()
    registry.set_targets(["h1:8000", "h2:8000"], [{"replica": 0}, None])
    registry.set_targets(["h1:8000", "h3:8000"], [{"replica": 1}, {"replica": 2}])

    snapshot = {t.address: t.labels for t in registry.snapshot()}
    assert snapshot == {
        "h1:8000": {"replica": "1"},  # latest call wins
        "h2:8000": {},
        "h3:8000": {"replica": "2"},
    }


def _recording_poller(texts):
    emitted: list[tuple[Any, ...]] = []
    seq = iter(texts)

    def fetch(_address: str) -> str:
        return next(seq)

    poller = PrometheusPoller(
        _RULES, interval_seconds=0, emit=lambda *args: emitted.append(tuple(args)),
        fetch=fetch,
    )
    poller.targets.set_targets(["127.0.0.1:8000"], [{"replica": "0"}])
    return poller, emitted


def test_poller_second_scrape_emits_delta_and_merges_target_labels() -> None:
    poller, emitted = _recording_poller([_TEXT_1, _TEXT_2])
    poller.poll_once()
    poller.poll_once()

    by_name = {}
    for name, op, value, tags in emitted:
        by_name.setdefault(name, []).append((op, value, tags))

    # First scrape baselines the counter; gauge emitted each scrape.
    assert by_name["running"][-1] == ("store", 5.0, {"replica": "0"})
    assert by_name["tokens"] == [("counter", 20.0, {"replica": "0"})]


def test_poller_should_swallow_fetch_failure_and_continue() -> None:
    emitted: list[tuple[Any, ...]] = []

    def flaky(address: str) -> str:
        raise ConnectionError("engine not ready")

    poller = PrometheusPoller(
        _RULES, interval_seconds=0, emit=lambda *a: emitted.append(a), fetch=flaky
    )
    poller.targets.set_targets(["127.0.0.1:8000"])
    poller.poll_once()  # must not raise
    assert emitted == []


def test_poller_should_swallow_emitter_failure() -> None:
    def boom(*_args: Any) -> None:
        raise RuntimeError("sink down")

    poller = PrometheusPoller(_RULES, interval_seconds=0, emit=boom, fetch=lambda a: _TEXT_2)
    poller.targets.set_targets(["127.0.0.1:8000"])
    poller.poll_once()  # emitter raises inside; poll must survive


def test_poller_start_stop_should_run_background_scrape() -> None:
    emitted: list[tuple[Any, ...]] = []
    poller = PrometheusPoller(
        _RULES,
        interval_seconds=0.02,
        emit=lambda *a: emitted.append(a),
        fetch=lambda a: _TEXT_2,
    )
    poller.targets.set_targets(["127.0.0.1:8000"])
    poller.start()
    try:
        deadline = time.time() + 2.0
        while not emitted and time.time() < deadline:
            time.sleep(0.02)
        assert emitted, "background poll produced no emissions"
    finally:
        poller.stop()


def test_active_registry_swap() -> None:
    assert poller_module.get_active_registry() is None
    registry = TargetRegistry()
    poller_module.set_active_registry(registry)
    try:
        assert poller_module.get_active_registry() is registry
    finally:
        poller_module.set_active_registry(None)
    assert poller_module.get_active_registry() is None
