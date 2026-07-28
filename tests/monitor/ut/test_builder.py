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

from rl_insight.experimental.builder import TrajectoryBuilder


def test_builder_preserves_finish_reason_for_every_turn():
    builder = TrajectoryBuilder()
    builder.feed(
        {
            "event": "trajectory_begin",
            "uid": "sample-1",
            "sample_index": 0,
            "session_index": 0,
            "trajectory_index": 0,
        }
    )
    builder.feed(
        {
            "event": "step",
            "uid": "sample-1",
            "step_index": 1,
            "finish_reason": "tool_calls",
        }
    )
    builder.feed(
        {
            "event": "step",
            "uid": "sample-1",
            "step_index": 2,
            "finish_reason": "stop",
            "reward": 1.0,
        }
    )

    traj = builder.samples[0].sessions[0].trajectories[0]
    assert [step.exit_reason for step in traj.steps] == ["tool_calls", "stop"]
    assert traj.reward_score == 1.0


def test_builder_does_not_invent_missing_finish_reason():
    builder = TrajectoryBuilder()
    builder.feed(
        {
            "event": "trajectory_begin",
            "uid": "sample-1",
            "sample_index": 0,
            "session_index": 0,
            "trajectory_index": 0,
        }
    )
    builder.feed({"event": "step", "uid": "sample-1", "step_index": 1})

    step = builder.samples[0].sessions[0].trajectories[0].steps[0]
    assert step.exit_reason == ""
