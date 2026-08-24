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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_insight import agent_loop


def test_agent_loop_helpers_should_be_public_exports() -> None:
    import rl_insight

    assert callable(rl_insight.agent_loop_session)


@dataclass
class Trajectory:
    chain_id: int | None
    reward_score: float | None
    num_turns: int | None


def test_agent_loop_lane_id_should_use_canonical_format() -> None:
    lane_id = agent_loop.agent_loop_lane_id("experiment-a", 1, 0, 2)

    assert lane_id == "experiment=experiment-a/sample=1/session=0/traj=2"


def test_agent_loop_session_metadata_should_emit_dashboard_hierarchy(
    monkeypatch: Any,
) -> None:
    events: list[tuple[str, float, dict[str, Any]]] = []

    def metric_gauge(name: str, value: float, **labels: Any) -> None:
        events.append((name, value, labels))

    monkeypatch.setattr(agent_loop, "metric_gauge", metric_gauge)

    agent_loop._publish_agent_loop_session(
        experiment_name="experiment-a",
        global_steps=7,
        sample=1,
        session=0,
        trajectories=[Trajectory(chain_id=2, reward_score=0.75, num_turns=3)],
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
    )

    assert [(name, value) for name, value, _ in events] == [
        ("agent_loop_run_info", 1.0),
        ("agent_loop_sample_info", 1.0),
        ("agent_loop_session_info", 1.0),
        ("agent_loop_traj_info", 1.0),
        ("agent_loop_first_turn_unixtime", 1.0),
        ("agent_loop_last_turn_unixtime", 2.0),
    ]
    assert events[3][2]["traj"] == "1"
    assert events[0][2]["experiment_name"] == "experiment-a"
    assert events[0][2]["global_steps"] == "7"
    assert events[3][2]["leaf"] == "sample=1/session=0/traj=1"
    assert events[3][2]["title"] == "Trajectory #1 · reward 0.75 · 3 turns"


def test_agent_loop_session_metadata_should_not_raise_when_gauge_fails(
    monkeypatch: Any,
) -> None:
    def metric_gauge(name: str, value: float, **labels: Any) -> None:
        raise RuntimeError("monitor unavailable")

    monkeypatch.setattr(agent_loop, "metric_gauge", metric_gauge)

    agent_loop._publish_agent_loop_session(
        experiment_name="experiment-a",
        sample=1,
        session=0,
        trajectories=[],
    )


def test_agent_loop_session_should_build_identity_and_finish_span(
    monkeypatch: Any,
) -> None:
    spans: list[dict[str, Any]] = []

    def trace_span(**kwargs):
        spans.append(kwargs)

    monkeypatch.setattr(agent_loop, "trace_span", trace_span)

    session = agent_loop.agent_loop_session(
        project="project-a",
        experiment_name="experiment-a",
        sample=1,
        session=0,
        uid="uid-a",
        global_steps=7,
        session_id="session-a",
    )

    assert session.identity == {
        "project": "project-a",
        "experiment_name": "experiment-a",
        "sample": "1",
        "session": "0",
        "traj": "0",
        "state_lane_id": "experiment=experiment-a/sample=1/session=0/traj=0",
        "uid": "uid-a",
        "global_steps": 7,
        "session_id": "session-a",
    }

    session.finish(
        trajectories=[Trajectory(chain_id=2, reward_score=0.5, num_turns=3)],
        status="success",
        runner_name="task",
        reward_source="reward_info",
        finished=True,
    )

    attributes = spans[0]["attributes"]
    assert spans[0]["name"] == "agent_session"
    assert (
        attributes["state_lane_id"]
        == "experiment=experiment-a/sample=1/session=0/traj=0"
    )
    assert attributes["runner_name"] == "task"
    assert attributes["status"] == "success"
    assert attributes["num_trajectories"] == 1
    assert attributes["reward_source"] == "reward_info"
    assert attributes["finished"] is True
