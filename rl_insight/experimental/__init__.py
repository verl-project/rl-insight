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

"""Experiment data structures for RL training analysis.

``SampleRecord`` is the primary data model. ``TrajectoryBuilder`` is the
event-driven adapter that builds samples from ``trajectory_begin`` and
``step`` events.

Hierarchy::

    TrajectoryBuilder      ← ingests events
      └── SampleRecord     ← one RL dataset sample
            └── SessionRecord   ← one GatewaySession (rollout attempt)
                  └── TrajectoryRecord  ← one chain / trajectory
                        └── Step        ← one model-call + tool-execution cycle
                              └── ToolResult  ← one tool call result

Quick start::

    from rl_insight.experimental import TrajectoryBuilder

    # Build from events
    builder = TrajectoryBuilder()
    builder.feed({"event": "trajectory_begin", "uid": "...", ...})
    builder.feed({"event": "step", "uid": "...", ...})
    samples = builder.samples

    # Or load a JSONL directly
    builder = TrajectoryBuilder.from_jsonl("events.jsonl")

    # Query the built sample
    sample = builder.get("uid")
    traj = sample.get_trajectory(0, 0)
    traj.num_turns           # int
    traj.total_tool_calls    # int
"""

from rl_insight.experimental.samples import (
    BaseSample,
    FileSampleRecord,
    SampleRecord,
    SampleTag,
    SessionRecord,
    SessionTag,
    Step,
    ToolResult,
    ToolStatus,
    TrajectoryRecord,
    TrajectoryTag,
    TrainingStatus,
)
from rl_insight.experimental.builder import TrajectoryBuilder

__all__ = [
    "BaseSample",
    "TrajectoryBuilder",
    "FileSampleRecord",
    "SampleRecord",
    "SampleTag",
    "SessionRecord",
    "SessionTag",
    "Step",
    "ToolResult",
    "ToolStatus",
    "TrajectoryRecord",
    "TrajectoryTag",
    "TrainingStatus",
]
