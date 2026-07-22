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

"""In-memory trajectory data structures, aligned with uni-agent.

Hierarchy::

    SampleRecord          ← one RL dataset sample
      └── SessionRecord   ← one GatewaySession (rollout attempt)
            └── TrajectoryRecord  ← one chain / trajectory
                  └── Step        ← one model-call + tool-execution cycle
                        └── ToolResult  ← one tool call result

Alignment with uni-agent types::

    ============================ ===========================================
    rl_insight                    uni-agent
    ============================ ===========================================
    ToolResult                    interaction.interaction.ToolResult
                                  (tool_call_id, name, action, observation,
                                   status, execution_time)
    Step                          interaction.interaction.StepOutput
                                  (step_idx, thought, response, tool_results,
                                   done, exit_reason)
    TrajectoryRecord (token)      gateway.session.types.Trajectory
                                  (prompt_ids, response_ids, response_mask,
                                   response_logprobs, routed_experts,
                                   multi_modal_data, reward_score, num_turns)
    TrajectoryRecord (step)       interaction.StepOutput (list)
    SessionRecord                 gateway.session.GatewaySession
    ============================ ===========================================

``SampleRecord`` is the primary entry point. ``SessionRecord`` and
``TrajectoryRecord`` are internal but importable for type annotations.

Lifecycle::

    from rl_insight.experimental import SampleRecord, Step, ToolResult

    sample = SampleRecord.create(uid="task-1", sample_index=0)
    sample.new_trajectory(session_index=0)
    sample.add_step(0, 0, Step(thought="...", tool_results=[...]))
    sample.finish_trajectory(0, 0, "stop")
    sample.set_trajectory_reward(0, 0, 1.0)

    # Load from experiment JSONL
    samples = SampleRecord.load_jsonl("trajectories.jsonl")

    # Serialize (Pydantic native -- includes steps, token data, all fields)
    data = sample.model_dump()
    copy = SampleRecord.model_validate(data)

    # Serialize (flat experiment format -- compact, loses steps)
    flat = sample.get_trajectory(0, 0).to_flat_dict()
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ToolResult & Step
# ---------------------------------------------------------------------------

ToolStatus = Literal["ok", "timeout", "syntax_error", "skipped"]


class ToolResult(BaseModel):
    """One tool-call result inside a step. Mirrors ``uni_agent.interaction.interaction.ToolResult``."""

    tool_call_id: str = ""
    name: str = ""
    action: str = ""
    observation: str = ""
    status: ToolStatus = "ok"
    execution_time: float | None = None


class Step(BaseModel):
    """One model-call + tool-execution cycle. Mirrors ``uni_agent.interaction.interaction.StepOutput``."""

    step_idx: int = 0
    response: str = ""
    thought: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    done: bool = False
    exit_reason: str = ""


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

TrainingStatus = Literal["success", "truncated", "error", "timeout", "aborted"]


class TrajectoryTag(BaseModel):
    """Per-trajectory metadata."""

    status: TrainingStatus = "success"
    finish_reason: str = ""
    global_steps: list[int] = Field(default_factory=list)
    uid: str = ""
    prompt_len: int = 0
    response_len: int = 0
    seq_len: int = 0

    @property
    def is_normal_exit(self) -> bool:
        return self.status == "success" and self.finish_reason not in (
            "",
            "length",
            "max_step_limit",
        )


class SessionTag(BaseModel):
    """Per-session metadata."""

    session_id: str = ""
    status: str = ""


class SampleTag(BaseModel):
    """Per-sample metadata."""

    uid: str = ""
    sample_index: int = 0


# ---------------------------------------------------------------------------
# TrajectoryRecord
# ---------------------------------------------------------------------------


class TrajectoryRecord(BaseModel):
    """A single trajectory: one chain produced by a GatewaySession.

    Token-level fields align with ``gateway.types.Trajectory``.
    Step-level fields align with ``interaction.StepOutput``.
    """

    # -- Identity
    uid: str = ""
    sample_index: int = 0
    session_index: int = 0
    trajectory_index: int = 0

    # -- Token-level (aligned with gateway.types.Trajectory)
    prompt_ids: list[int] | None = None
    response_ids: list[int] | None = None
    response_mask: list[int] | None = None
    response_logprobs: list[float] | None = None
    routed_experts: Any | None = None
    multi_modal_data: dict[str, Any] | None = None

    # -- Step-level (aligned with interaction.StepOutput)
    steps: list[Step] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    execution_time: float | None = None

    # -- Reward
    reward_score: float | None = None
    reward_extra_info: dict[str, Any] = Field(default_factory=dict)

    # -- Summary (auto-computed on add_step / finish)
    prompt_len: int = 0
    response_len: int = 0
    seq_len: int = 0
    num_turns: int = 0

    # -- Meta
    tag: TrajectoryTag = Field(default_factory=TrajectoryTag)
    extra_fields: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        return f"{self.uid}_{self.session_index}_{self.trajectory_index}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exit_reason(self) -> str:
        if not self.steps:
            return ""
        return self.steps[-1].exit_reason

    @property
    def is_completed(self) -> bool:
        if not self.steps:
            return False
        last = self.steps[-1]
        return last.done and last.exit_reason in ("finished", "stop")

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_results) for s in self.steps)

    @property
    def tool_call_success_rate(self) -> float:
        total = self.total_tool_calls
        if total == 0:
            return 1.0
        ok = sum(1 for s in self.steps for tr in s.tool_results if tr.status == "ok")
        return ok / total

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        uid: str,
        sample_index: int = 0,
        session_index: int = 0,
        trajectory_index: int = 0,
        **kwargs,
    ) -> TrajectoryRecord:
        """Create a new empty trajectory with identity fields set."""
        tag = TrajectoryTag(uid=uid)
        return cls(
            uid=uid,
            sample_index=sample_index,
            session_index=session_index,
            trajectory_index=trajectory_index,
            tag=tag,
            **kwargs,
        )

    @classmethod
    def from_interaction_result(
        cls,
        *,
        uid: str,
        sample_index: int = 0,
        session_index: int = 0,
        trajectory_index: int = 0,
        interaction_result: dict[str, Any],
        reward_score: float | None = None,
        reward_extra_info: dict[str, Any] | None = None,
        tag_status: TrainingStatus = "success",
        tag_finish_reason: str = "",
        tag_global_steps: list[int] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> TrajectoryRecord:
        """Build from ``AgentInteraction.run()`` output dict.

        Populates steps from ``interaction_result["trajectory"]`` (list of
        StepOutput) and token data from ``interaction_result["rollout_cache"]``.
        """
        trajectory: list[Any] = interaction_result.get("trajectory", [])
        rollout_cache: dict[str, Any] = interaction_result.get("rollout_cache", {})
        messages: list[dict[str, Any]] = interaction_result.get("messages", [])
        execution_time: float | None = interaction_result.get("execution_time")

        # Convert interaction StepOutput → our Step.
        steps: list[Step] = []
        for s in trajectory:
            tool_results = [
                ToolResult(
                    tool_call_id=tr.tool_call_id if hasattr(tr, "tool_call_id") else "",
                    name=tr.name,
                    action=tr.action,
                    observation=tr.observation,
                    status=tr.status,
                    execution_time=tr.execution_time,
                )
                for tr in (s.tool_results if hasattr(s, "tool_results") else [])
            ]
            steps.append(
                Step(
                    step_idx=s.step_idx if hasattr(s, "step_idx") else 0,
                    response=s.response if hasattr(s, "response") else "",
                    thought=s.thought if hasattr(s, "thought") else "",
                    tool_results=tool_results,
                    done=s.done if hasattr(s, "done") else False,
                    exit_reason=s.exit_reason if hasattr(s, "exit_reason") else "",
                )
            )

        prompt_ids: list[int] | None = rollout_cache.get("prompt_ids")
        response_ids = _safe_int_list(rollout_cache.get("response_ids"))
        response_mask = _safe_int_list(rollout_cache.get("response_mask"))
        response_logprobs = _safe_float_list(rollout_cache.get("response_logprobs"))
        routed_experts = rollout_cache.get("routed_experts")
        multi_modal_data = rollout_cache.get("multi_modal_data")

        prompt_len = len(prompt_ids) if prompt_ids else 0
        response_len = len(response_mask) if response_mask else 0
        num_turns = len(steps)

        tag = TrajectoryTag(
            status=tag_status,
            finish_reason=tag_finish_reason or (steps[-1].exit_reason if steps else ""),
            global_steps=tag_global_steps or [],
            uid=uid,
            prompt_len=prompt_len,
            response_len=response_len,
            seq_len=prompt_len + response_len,
        )

        return cls(
            uid=uid,
            sample_index=sample_index,
            session_index=session_index,
            trajectory_index=trajectory_index,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            steps=steps,
            messages=messages,
            execution_time=execution_time,
            reward_score=reward_score,
            reward_extra_info=dict(reward_extra_info or {}),
            prompt_len=prompt_len,
            response_len=response_len,
            seq_len=prompt_len + response_len,
            num_turns=num_turns,
            tag=tag,
            extra_fields=dict(extra_fields or {}),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def add_step(self, step: Step) -> None:
        """Append a step and auto-increment ``step_idx`` if unset.

        Also updates ``num_turns`` and derives ``exit_reason`` / ``done``
        from the last step.
        """
        if step.step_idx == 0:
            step.step_idx = len(self.steps) + 1
        self.steps.append(step)
        self.num_turns = len(self.steps)

    def set_token_data(
        self,
        *,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        response_mask: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        routed_experts: Any = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> None:
        """Populate token-level fields and recompute summary stats."""
        if prompt_ids is not None:
            self.prompt_ids = prompt_ids
        if response_ids is not None:
            self.response_ids = response_ids
        if response_mask is not None:
            self.response_mask = response_mask
        if response_logprobs is not None:
            self.response_logprobs = response_logprobs
        if routed_experts is not None:
            self.routed_experts = routed_experts
        if multi_modal_data is not None:
            self.multi_modal_data = multi_modal_data

        self.prompt_len = len(self.prompt_ids) if self.prompt_ids else 0
        self.response_len = len(self.response_mask) if self.response_mask else 0
        self.seq_len = self.prompt_len + self.response_len
        self.tag.prompt_len = self.prompt_len
        self.tag.response_len = self.response_len
        self.tag.seq_len = self.seq_len

    def set_reward(
        self, score: float, extra_info: dict[str, Any] | None = None
    ) -> None:
        """Set reward score and optional extra info."""
        self.reward_score = score
        if extra_info is not None:
            self.reward_extra_info = extra_info

    def finish(
        self,
        exit_reason: str = "finished",
        status: TrainingStatus = "success",
    ) -> None:
        """Mark this trajectory as done.

        Args:
            exit_reason: Why the trajectory ended (``finished``, ``length``,
                ``max_step_limit``, ``token_limit``, ...).
            status: High-level outcome.
        """
        # Reflect in the last step if one exists.
        if self.steps and not self.steps[-1].done:
            self.steps[-1].done = True
            if not self.steps[-1].exit_reason:
                self.steps[-1].exit_reason = exit_reason
        self.tag.finish_reason = exit_reason
        self.tag.status = status

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> TrajectoryRecord:
        """Parse from the flat experiment JSONL format (``reward`` key, no steps)."""
        tag_raw = d.get("tag", {})
        tag = TrajectoryTag(**tag_raw) if isinstance(tag_raw, dict) else TrajectoryTag()
        return cls(
            uid=d.get("uid", ""),
            sample_index=d.get("sample_index", 0),
            session_index=d.get("session_index", 0),
            trajectory_index=d.get("trajectory_index", 0),
            prompt_ids=d.get("prompt_ids"),
            response_ids=d.get("response_ids"),
            response_mask=d.get("response_mask"),
            response_logprobs=d.get("response_logprobs"),
            routed_experts=d.get("routed_experts"),
            prompt_len=d.get("prompt_len", 0),
            response_len=d.get("response_len", 0),
            seq_len=d.get("seq_len", 0),
            num_turns=d.get("num_turns", 0),
            reward_score=d.get("reward"),
            reward_extra_info=d.get("reward_extra_info", {}),
            tag=tag,
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize to the flat experiment JSONL format."""
        return {
            "key": self.key,
            "uid": self.uid,
            "sample_index": self.sample_index,
            "session_index": self.session_index,
            "trajectory_index": self.trajectory_index,
            "prompt_len": self.prompt_len,
            "response_len": self.response_len,
            "seq_len": self.seq_len,
            "num_turns": self.num_turns,
            "reward": self.reward_score,
            "reward_extra_info": self.reward_extra_info,
            "tag": self.tag.model_dump(),
        }


# ---------------------------------------------------------------------------
# SessionRecord
# ---------------------------------------------------------------------------


class SessionRecord(BaseModel):
    """One GatewaySession: a single rollout attempt for a sample.

    Trajectories within a session are sequential and cumulative -- later
    trajectories inherit the message history of earlier ones.
    """

    uid: str = ""
    sample_index: int = 0
    session_index: int = 0
    session_id: str = ""
    trajectories: list[TrajectoryRecord] = Field(default_factory=list)
    tag: SessionTag = Field(default_factory=SessionTag)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def total_reward(self) -> float:
        return sum(t.reward_score or 0.0 for t in self.trajectories)

    @property
    def mean_reward(self) -> float:
        if not self.trajectories:
            return 0.0
        return self.total_reward / len(self.trajectories)

    @property
    def total_steps(self) -> int:
        return sum(t.num_turns for t in self.trajectories)

    @property
    def completed_trajectories(self) -> list[TrajectoryRecord]:
        return [t for t in self.trajectories if t.is_completed]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        uid: str,
        sample_index: int = 0,
        session_index: int = 0,
        session_id: str = "",
        tag_status: str = "",
    ) -> SessionRecord:
        """Create a new empty session."""
        return cls(
            uid=uid,
            sample_index=sample_index,
            session_index=session_index,
            session_id=session_id,
            tag=SessionTag(session_id=session_id, status=tag_status),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def new_trajectory(self, **kwargs) -> TrajectoryRecord:
        """Create and append a new trajectory with auto-incremented index.

        ``uid``, ``sample_index``, and ``session_index`` are inherited from
        the session; ``trajectory_index`` is set to ``len(self.trajectories)``.
        """
        traj = TrajectoryRecord.create(
            uid=self.uid,
            sample_index=self.sample_index,
            session_index=self.session_index,
            trajectory_index=kwargs.pop("trajectory_index", len(self.trajectories)),
            **kwargs,
        )
        self.trajectories.append(traj)
        return traj

    def add_trajectory(self, traj: TrajectoryRecord) -> None:
        """Append an externally-created trajectory.

        Syncs identity fields if they are unset on the incoming record.
        """
        if not traj.uid:
            traj.uid = self.uid
        if not traj.sample_index:
            traj.sample_index = self.sample_index
        if not traj.session_index:
            traj.session_index = self.session_index
        if traj.trajectory_index == 0 and not any(
            t.trajectory_index == traj.trajectory_index for t in self.trajectories
        ):
            pass  # keep caller's index
        elif traj.trajectory_index == 0:
            traj.trajectory_index = len(self.trajectories)
        self.trajectories.append(traj)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SampleRecord
# ---------------------------------------------------------------------------


class SampleRecord(BaseModel):
    """One RL dataset sample -- the primary entry point for all operations.

    A sample represents a single problem/task. It contains multiple sessions
    (independent rollout attempts), each containing multiple trajectories
    (sequential chains within a GatewaySession).

    All write operations go through ``SampleRecord`` methods::

        sample = SampleRecord.create(uid="...", sample_index=0)
        sample.new_trajectory(session_index=0)          # auto-creates session
        sample.add_step(0, 0, Step(thought="...", tool_results=[...]))
        sample.finish_trajectory(0, 0, "finished")
        sample.set_trajectory_reward(0, 0, 1.0)

    Query properties::

        sample.num_sessions          # int
        sample.num_trajectories      # int
        sample.mean_reward           # float
        sample.get_trajectory(0, 0)  # TrajectoryRecord | None

    Serialization::

        data = sample.model_dump()                     # full JSON
        copy = SampleRecord.model_validate(data)       # round-trip
        samples = SampleRecord.load_jsonl("trajs.jsonl")  # from flat format
    """

    uid: str = ""
    sample_index: int = 0
    sessions: list[SessionRecord] = Field(default_factory=list)
    tag: SampleTag = Field(default_factory=SampleTag)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def num_trajectories(self) -> int:
        return sum(s.num_trajectories for s in self.sessions)

    @property
    def total_reward(self) -> float:
        return sum(s.total_reward for s in self.sessions)

    @property
    def mean_reward(self) -> float:
        if not self.num_trajectories:
            return 0.0
        return self.total_reward / self.num_trajectories

    @property
    def mean_reward_per_session(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(s.mean_reward for s in self.sessions) / len(self.sessions)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, *, uid: str, sample_index: int = 0, **kwargs) -> SampleRecord:
        """Create a new empty sample."""
        return cls(
            uid=uid,
            sample_index=sample_index,
            tag=SampleTag(uid=uid, sample_index=sample_index),
            **kwargs,
        )

    @classmethod
    def load_jsonl(cls, path: str) -> list[SampleRecord]:
        """Load experiment JSONL and group into Sample → Session → Trajectory."""
        import json
        from collections import defaultdict

        with open(path) as f:
            flat_trajs = [json.loads(line) for line in f]

        samples: dict[str, dict[int, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        sample_indices: dict[str, int] = {}

        for t in flat_trajs:
            uid = t["uid"]
            si = t["session_index"]
            samples[uid][si].append(t)
            if uid not in sample_indices:
                sample_indices[uid] = t.get("sample_index", len(sample_indices))

        result = []
        for uid in sorted(samples.keys(), key=lambda u: sample_indices.get(u, 0)):
            session_dict = samples[uid]
            sessions = []
            for si in sorted(session_dict.keys()):
                trajs = sorted(session_dict[si], key=lambda t: t["trajectory_index"])
                sessions.append(
                    SessionRecord(
                        uid=uid,
                        sample_index=sample_indices[uid],
                        session_index=si,
                        trajectories=[
                            TrajectoryRecord.from_flat_dict(t) for t in trajs
                        ],
                    )
                )
            result.append(
                SampleRecord(
                    uid=uid,
                    sample_index=sample_indices[uid],
                    sessions=sessions,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def new_session(self, **kwargs) -> SessionRecord:
        """Create and append a new session with auto-incremented index.

        Inherits ``uid`` and ``sample_index``.
        """
        session = SessionRecord.create(
            uid=self.uid,
            sample_index=self.sample_index,
            session_index=kwargs.pop("session_index", len(self.sessions)),
            **kwargs,
        )
        self.sessions.append(session)
        return session

    def get_session(self, session_index: int) -> SessionRecord | None:
        """Return the session at the given index, or None."""
        for s in self.sessions:
            if s.session_index == session_index:
                return s
        return None

    def get_or_create_session(self, session_index: int, **kwargs) -> SessionRecord:
        """Return existing session or create a new one at the given index."""
        session = self.get_session(session_index)
        if session is not None:
            return session
        return self.new_session(session_index=session_index, **kwargs)

    def add_session(self, session: SessionRecord) -> None:
        """Append an externally-created session.

        Syncs identity fields on the incoming record.
        """
        if not session.uid:
            session.uid = self.uid
        if not session.sample_index:
            session.sample_index = self.sample_index
        if session.session_index == 0 and not any(
            s.session_index == session.session_index for s in self.sessions
        ):
            pass
        elif session.session_index == 0:
            session.session_index = len(self.sessions)
        self.sessions.append(session)

    # ------------------------------------------------------------------
    # Trajectory operations (delegated)
    # ------------------------------------------------------------------

    def _require_session(self, session_index: int) -> SessionRecord:
        """Look up a session by index; raises if not found."""
        session = self.get_session(session_index)
        if session is None:
            raise KeyError(f"session {session_index} not found in sample {self.uid}")
        return session

    def new_trajectory(self, session_index: int = 0, **kwargs) -> TrajectoryRecord:
        """Create and append a trajectory to the given session.

        If the session does not exist it is created first.
        """
        session = self.get_or_create_session(session_index)
        return session.new_trajectory(**kwargs)

    def get_trajectory(
        self, session_index: int, trajectory_index: int
    ) -> TrajectoryRecord | None:
        """Look up a trajectory by session and trajectory index, or None."""
        session = self.get_session(session_index)
        if session is None:
            return None
        for t in session.trajectories:
            if t.trajectory_index == trajectory_index:
                return t
        return None

    def _require_trajectory(
        self, session_index: int, trajectory_index: int
    ) -> TrajectoryRecord:
        traj = self.get_trajectory(session_index, trajectory_index)
        if traj is None:
            raise KeyError(
                f"trajectory {session_index}/{trajectory_index} not found in sample {self.uid}"
            )
        return traj

    def add_step(self, session_index: int, trajectory_index: int, step: Step) -> None:
        """Append a step to a specific trajectory."""
        self._require_trajectory(session_index, trajectory_index).add_step(step)

    def set_trajectory_reward(
        self,
        session_index: int,
        trajectory_index: int,
        score: float,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """Set reward for a specific trajectory."""
        self._require_trajectory(session_index, trajectory_index).set_reward(
            score, extra_info
        )

    def set_trajectory_token_data(
        self,
        session_index: int,
        trajectory_index: int,
        *,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        response_mask: list[int] | None = None,
        response_logprobs: list[float] | None = None,
        routed_experts: Any = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> None:
        """Set token-level data for a specific trajectory."""
        self._require_trajectory(session_index, trajectory_index).set_token_data(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
        )

    def finish_trajectory(
        self,
        session_index: int,
        trajectory_index: int,
        exit_reason: str = "finished",
        status: TrainingStatus = "success",
    ) -> None:
        """Mark a trajectory as done."""
        self._require_trajectory(session_index, trajectory_index).finish(
            exit_reason, status
        )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int_list(obj: Any) -> list[int] | None:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [int(v) for v in obj]
    return None


def _safe_float_list(obj: Any) -> list[float] | None:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [float(v) for v in obj]
    return None
