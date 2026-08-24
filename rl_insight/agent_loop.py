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

"""Agent-loop dashboard protocol and session helpers.

This module owns the metric names, lane IDs, session identity, and hierarchy
labels consumed by the Agent Loop Trajectory dashboard. Training frameworks
should adapt their domain objects to :func:`agent_loop_session` instead of
reimplementing the protocol.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .api import metric_gauge, trace_span

logger = logging.getLogger(__name__)


def _agent_loop_sess_key(sample: Any, session: Any) -> str:
    return f"sample={sample}/session={session}"


def _agent_loop_leaf(sample: Any, session: Any, traj: Any) -> str:
    return f"sample={sample}/session={session}/traj={traj}"


def agent_loop_lane_id(
    experiment_name: Any, sample: Any, session: Any, traj: Any
) -> str:
    """Return the canonical lane ID for one agent-loop trajectory."""
    return f"experiment={experiment_name}/{_agent_loop_sess_key(sample, session)}/traj={traj}"


def _metric_gauge(name: str, value: float, **labels: Any) -> None:
    """Publish one dashboard gauge without breaking the training process."""
    try:
        metric_gauge(
            str(name),
            float(value),
            **{str(key): str(label_value) for key, label_value in labels.items()},
        )
    except Exception:
        logger.exception("[rl-insight] failed to publish agent-loop gauge %s", name)


def _agent_loop_reward(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _publish_agent_loop_session(
    *,
    experiment_name: Any,
    sample: Any,
    session: Any,
    trajectories: list[Any],
    global_steps: Any = None,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> None:
    """Publish dashboard hierarchy gauges for one finalized agent session."""
    experiment_name = str(experiment_name)
    sample = str(sample)
    session = str(session)
    global_steps = "" if global_steps is None else str(global_steps)

    _metric_gauge(
        "agent_loop_run_info",
        1.0,
        experiment_name=experiment_name,
        global_steps=global_steps,
        title=f"Experiment · {experiment_name}",
    )
    _metric_gauge(
        "agent_loop_sample_info",
        1.0,
        experiment_name=experiment_name,
        global_steps=global_steps,
        sample=sample,
        title=f"Sample {sample}",
    )
    sess_key = _agent_loop_sess_key(sample, session)
    _metric_gauge(
        "agent_loop_session_info",
        1.0,
        experiment_name=experiment_name,
        global_steps=global_steps,
        sample=sample,
        session=session,
        sess_key=sess_key,
        title=f"Session {session}",
    )

    for index, traj in enumerate(trajectories):
        chain_id = getattr(traj, "chain_id", None)
        traj_id = str(int(chain_id) - 1) if chain_id is not None else str(index)
        reward = _agent_loop_reward(getattr(traj, "reward_score", 0))
        turns = int(getattr(traj, "num_turns", 0) or 0)
        leaf = _agent_loop_leaf(sample, session, traj_id)
        _metric_gauge(
            "agent_loop_traj_info",
            1.0,
            experiment_name=experiment_name,
            global_steps=global_steps,
            sample=sample,
            session=session,
            traj=traj_id,
            leaf=leaf,
            title=f"Trajectory #{traj_id} · reward {reward:g} · {turns} turns",
        )

    if start_time_ns is not None:
        _metric_gauge(
            "agent_loop_first_turn_unixtime",
            float(start_time_ns) / 1_000_000_000.0,
            experiment_name=experiment_name,
            global_steps=global_steps,
        )
    if end_time_ns is not None:
        _metric_gauge(
            "agent_loop_last_turn_unixtime",
            float(end_time_ns) / 1_000_000_000.0,
            experiment_name=experiment_name,
            global_steps=global_steps,
        )


@dataclass(frozen=True)
class _AgentLoopSession:
    """Completed-span state for one agent-loop session."""

    identity: dict[str, Any]
    start_ns: int

    def finish(
        self,
        *,
        trajectories: list[Any],
        status: str,
        runner_name: str | None = None,
        reward_source: str | None = None,
        finished: bool | None = None,
    ) -> None:
        """Publish dashboard metadata and the session-level span."""
        identity = self.identity
        _publish_agent_loop_session(
            experiment_name=identity["experiment_name"],
            sample=identity["sample"],
            session=identity["session"],
            trajectories=trajectories,
            global_steps=identity.get("global_steps"),
            start_time_ns=self.start_ns,
            end_time_ns=time.time_ns(),
        )
        trace_span(
            name="agent_session",
            start_time_ns=self.start_ns,
            end_time_ns=time.time_ns(),
            attributes={
                **identity,
                "monitor.trace_source": "session",
                "runner_name": runner_name or "",
                "status": status,
                "num_trajectories": len(trajectories),
                "reward_source": reward_source or "",
                "finished": finished if finished is not None else "",
            },
        )


def agent_loop_session(
    *,
    project: Any = None,
    experiment_name: Any,
    sample: Any,
    session: Any,
    traj: Any = 0,
    uid: Any = None,
    global_steps: Any = None,
    session_id: Any = None,
) -> _AgentLoopSession:
    """Create the shared identity and timestamp for one agent-loop session."""
    project = "" if project is None else str(project)
    experiment_name = str(experiment_name)
    sample = str(sample)
    session = str(session)
    traj = str(traj)
    return _AgentLoopSession(
        identity={
            "project": project,
            "experiment_name": experiment_name,
            "sample": sample,
            "session": session,
            "traj": traj,
            "state_lane_id": agent_loop_lane_id(experiment_name, sample, session, traj),
            "uid": "" if uid is None else str(uid),
            "global_steps": "" if global_steps is None else global_steps,
            "session_id": "" if session_id is None else str(session_id),
        },
        start_ns=time.time_ns(),
    )
