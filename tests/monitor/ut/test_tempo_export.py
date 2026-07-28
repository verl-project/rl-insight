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

from rl_insight.experimental import tempo_export
from rl_insight.experimental.builder import TrajectoryBuilder
from rl_insight.experimental.samples.sample import SampleRecord, Step


def _span(start_s: int, end_s: int) -> dict:
    return {
        "start_time_ns": start_s * 1_000_000_000,
        "end_time_ns": end_s * 1_000_000_000,
    }


def test_compress_span_times_does_not_expand_short_fixture(monkeypatch):
    monkeypatch.setattr(tempo_export.time, "time", lambda: 100.0)

    result = tempo_export.compress_span_times(
        [_span(0, 1), _span(2, 3)],
        window_s=30,
        lag_s=10,
    )

    assert result == [_span(87, 88), _span(89, 90)]


def test_compress_span_times_shrinks_fixture_larger_than_window(monkeypatch):
    monkeypatch.setattr(tempo_export.time, "time", lambda: 100.0)

    result = tempo_export.compress_span_times(
        [_span(0, 20), _span(40, 60)],
        window_s=30,
        lag_s=10,
    )

    assert result == [_span(60, 70), _span(80, 90)]


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
        }
    )

    steps = builder.samples[0].sessions[0].trajectories[0].steps
    assert [step.exit_reason for step in steps] == ["tool_calls", "stop"]


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


def test_tempo_export_marks_missing_nonterminal_reason_unknown():
    sample = SampleRecord.create(uid="sample-1")
    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(0, 0, Step(step_idx=1))
    sample.add_step(0, 0, Step(step_idx=2, exit_reason="stop"))
    sample.finish_trajectory(0, 0, "stop")

    spans = tempo_export.samples_to_span_dicts(
        [sample],
        run_id="run-1",
        clock_start_ns=1_000_000_000,
    )

    assert [span["attributes"]["finish_reason"] for span in spans] == [
        "unknown",
        "stop",
    ]
    assert all("state_name" not in span["attributes"] for span in spans)


def test_tempo_export_uses_content_from_thought_or_response():
    """content prefers thought; falls back to response; no invented fields."""
    sample = SampleRecord.create(uid="sample-1")
    sample.new_trajectory(session_index=0, trajectory_index=0)
    sample.add_step(
        0,
        0,
        Step(step_idx=1, thought="parsed thought", response="raw model output"),
    )
    sample.finish_trajectory(0, 0, "stop")

    spans = tempo_export.samples_to_span_dicts(
        [sample], run_id="run-1", clock_start_ns=1_000_000_000
    )

    attrs = spans[0]["attributes"]
    assert attrs["content"] == "parsed thought"
    assert "thought" not in attrs
    assert "response" not in attrs
    assert "request_context" not in attrs
    assert "prompt_len" not in attrs
