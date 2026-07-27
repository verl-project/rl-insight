#!/usr/bin/env python3

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

"""Generate trajectory events and feed them through TrajectoryBuilder.

This script only knows about ``TrajectoryBuilder``. It emits the two
builder event types (``trajectory_begin`` / ``step``) and never touches
any ``BaseSample`` implementation directly. The storage backend is
decided by the factory passed to ``TrajectoryBuilder`` in ``main()``.

Usage::

    python generate_data.py /path/to/output --samples 8
    python generate_data.py /path/to/output --stream
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path


_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse  # noqa: E402
import random  # noqa: E402
from typing import Any  # noqa: E402

from rl_insight.experimental.builder import TrajectoryBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Scenario templates -- each is a plausible coding task
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "task": "Fix separability_matrix bug in CompoundModel",
        "tools": [
            "Read",
            "Bash",
            "Read",
            "Read",
            "Edit",
            "Bash",
            "Edit",
            "Read",
            "Bash",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Let me start by understanding the bug report.",
            "Running the failing test to see the actual error.",
            "The error points to separability_matrix, reading that function.",
            "Tracing the call chain through _cstack to find the root cause.",
            "Found it -- the matrix placement is off by one row in _cstack.",
            "Running the test again to confirm the fix.",
            "There's a second issue in the edge case handling.",
            "Reading the edge case test fixtures to understand expected behavior.",
            "Adding a guard clause for the empty-matrix case.",
            "Running the full test suite to make sure nothing else broke.",
        ],
    },
    {
        "task": "Optimize database query performance",
        "tools": [
            "Bash",
            "Read",
            "Bash",
            "Read",
            "Edit",
            "Read",
            "Edit",
            "Bash",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "First, let me run EXPLAIN ANALYZE on the slow query.",
            "Reading the query builder code to understand the join strategy.",
            "Running a profiling script to get baseline numbers.",
            "The EXPLAIN shows seq scan on orders table -- 2M rows.",
            "Adding a composite index on (user_id, created_at).",
            "Checking the migration script before applying.",
            "Also need to denormalize the status column to avoid another join.",
            "Running the migration on a staging replica.",
            "Benchmarking: 450ms -> 2.3ms. That's a 200x improvement.",
        ],
    },
    {
        "task": "Add error handling for API timeout",
        "tools": [
            "Read",
            "Bash",
            "Read",
            "Read",
            "Edit",
            "Edit",
            "Read",
            "Bash",
            "Bash",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Finding all places where we call the external payment API.",
            "Checking the current error handling -- looks like bare try/except.",
            "Reading the API client wrapper to understand the timeout config.",
            "Also reading the circuit breaker library we already have in deps.",
            "Adding retry with exponential backoff: 1s, 2s, 4s, max 3 attempts.",
            "Wrapping the retry logic in the circuit breaker with half-open state.",
            "Reviewing the edge cases: what happens on network partition?",
            "Writing integration tests with simulated latency.",
            "Testing timeout scenario: 5s timeout, retries triggered correctly.",
            "Testing circuit breaker: opens after 5 failures, recovers after 30s.",
        ],
    },
    {
        "task": "Refactor authentication middleware",
        "tools": [
            "Read",
            "Read",
            "Bash",
            "Read",
            "Edit",
            "Bash",
            "Edit",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Reading the auth middleware to understand the current structure.",
            "The token validation is 200 lines inline, needs extraction.",
            "Checking test coverage before refactoring -- 78%, not bad.",
            "Reading all callers of the middleware to map dependencies.",
            "Extracting TokenService class: decode, validate, refresh logic.",
            "Running tests to catch regressions early.",
            "Updating the middleware to delegate to TokenService.",
            "Running full test suite -- 2 test failures to fix.",
        ],
    },
    {
        "task": "Fix race condition in task scheduler",
        "tools": [
            "Read",
            "Bash",
            "Read",
            "Read",
            "Edit",
            "Bash",
            "Edit",
            "Bash",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Reading the task scheduler's dispatch loop.",
            "Running stress test with 100 concurrent tasks to reproduce.",
            "Got it: two tasks picked up the same job ID.",
            "Reading the lock acquisition code -- it's after the SELECT.",
            "Moving the SELECT ... FOR UPDATE before the state check.",
            "Stress test: 50 iterations, no duplicate assignments so far.",
            "Also adding optimistic locking as a second safety net.",
            "1000 iterations: zero duplicates. Race condition fixed.",
            "Running integration tests with the full pipeline.",
        ],
    },
    {
        "task": "Implement LRU cache for file reads",
        "tools": [
            "Bash",
            "Read",
            "Bash",
            "Read",
            "Edit",
            "Bash",
            "Read",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Running a trace on file reads during a typical request.",
            "Reading the FileAccess layer -- lots of repeated config reads.",
            "Profiling: 60% of file reads are duplicates, cache would help a lot.",
            "Reading the existing in-memory helpers to reuse patterns.",
            "Implementing LRU with OrderedDict, 1000 entry cap, TTL 60s.",
            "Benchmark baseline: 1200 reads/s.",
            "Reviewing the cache invalidation: file mtime check on read.",
            "Benchmark with cache: 8200 reads/s. ~7x improvement.",
        ],
    },
    {
        "task": "Migrate logging to structured JSON",
        "tools": [
            "Bash",
            "Read",
            "Read",
            "Edit",
            "Edit",
            "Read",
            "Edit",
            "Bash",
            "Read",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Finding all log calls across the codebase -- grep for logger. and print(.",
            "Reading the current logging config in settings.py.",
            "Also reading how request IDs are currently handled (hint: they aren't).",
            "Converting the main logger to structlog: JSONRenderer, add timestamp+level.",
            "Converting 47 log statements to use key=value binding syntax.",
            "Reviewing the output: missing trace_id in sub-calls.",
            "Adding contextvars-based request ID propagation across async boundaries.",
            "Running the app and tailing logs to verify JSON output.",
            "Spot-checking: all log lines parse as valid JSON.",
            "Running log analysis script: 47 log sites, 0 text-format remaining.",
        ],
    },
    {
        "task": "Fix memory leak in WebSocket handler",
        "tools": [
            "Bash",
            "Read",
            "Bash",
            "Read",
            "Edit",
            "Read",
            "Bash",
            "Edit",
            "Bash",
            "finish",
        ],
        "thoughts": [
            "Running memory profiler with 500 concurrent connections.",
            "Reading the WebSocket connection manager class.",
            "After 10 min: memory grows from 80MB to 450MB. Definitely a leak.",
            "Reading the on_close callback -- it removes from dict but not from event bus.",
            "Adding event listener cleanup in the disconnect handler.",
            "Also checking if the heartbeat timer is being cancelled on close.",
            "Running the leak test again: stable at 85MB over 30 minutes.",
            "Adding a guard to prevent double-close from triggering cleanup twice.",
            "Final soak test: 1000 connections, 1 hour, memory flat at 82MB.",
        ],
    },
]

TOOL_OBSERVATIONS = {
    "Bash": [
        "test_separable.py::test_compound PASSED",
        "test_separable.py::test_compound FAILED - assert 4 == 6",
        "main.py:42: error: undefined variable 'result'",
        "Coverage: 87% (was 82%)",
        "Query executed in 2.3ms (was 450ms)",
        "Memory usage: stable at 120MB over 1000 iterations",
        "race_test.py: 0 failures in 1000 iterations",
        "Connection count: 5 active, 0 leaked",
    ],
    "Read": [
        "def _cstack(left, right):\n    noutp = _compute_n_outputs(left, right)\n    ...",
        "class CompoundModel(Model):\n    def __init__(self, op, left, right):\n        ...",
        "SELECT * FROM users WHERE email = ? -- no index on email column",
        "async def authenticate(token: str) -> User:\n    payload = decode(token)\n    ...",
    ],
    "Edit": [
        "Replaced lines 245-250: fixed matrix placement logic",
        "Added composite index: CREATE INDEX idx_email ON users(email)",
        "Added retry logic: max_retries=3, backoff_factor=0.5",
        "Extracted TokenService class: 120 lines added, 80 removed",
    ],
}


def _generate_action(tool_name: str) -> str:
    actions = {
        "Bash": [
            "pytest tests/test_separable.py -xvs",
            "python -m memory_profiler main.py",
            "python -m pytest tests/ --cov",
            "curl -s http://localhost:8080/health",
            "python benchmark.py --iterations 100",
            "git diff HEAD~1 --stat",
        ],
        "Read": [
            "cat src/core.py | head -100",
            "rg 'def authenticate' --type py",
            "git show HEAD:src/models.py",
        ],
        "Edit": [
            "sed -i 's/old_pattern/new_pattern/' src/core.py",
            "str_replace_editor: replace lines 200-210",
        ],
    }
    return (
        random.choice(actions.get(tool_name, ["..."]))
        if tool_name in actions
        else "..."
    )


# ---------------------------------------------------------------------------
# Event builders -- only produce dicts for TrajectoryBuilder.feed()
# ---------------------------------------------------------------------------


def _build_step_event(
    uid: str,
    step_index: int,
    finish_reason: str,
    thought: str,
    tool_results: list[dict[str, Any]],
    is_last: bool,
) -> dict[str, Any]:
    """Build a ``step`` event dict."""
    fr = finish_reason if is_last else "tool_calls"
    return {
        "event": "step",
        "uid": uid,
        "step_index": step_index,
        "finish_reason": fr,
        "thought": thought,
        "tool_results": tool_results,
    }


def build_trajectory_events(
    uid: str,
    sample_index: int,
    session_index: int,
    trajectory_index: int,
    scenario: dict,
    sample_success: bool,
    is_last_in_session: bool,
) -> list[dict[str, Any]]:
    """Build the full event list for one trajectory.

    Returns a list of event dicts: one ``trajectory_begin`` followed
    by N ``step`` events. The caller feeds them to ``builder.feed()``.
    """
    tools = list(scenario["tools"])
    thoughts = list(scenario["thoughts"])

    while len(thoughts) < len(tools):
        thoughts.append("Continuing to work on the task...")

    max_len = min(len(tools), len(thoughts))
    traj_len = random.randint(max(4, max_len // 2), max_len)
    tools = tools[:traj_len]
    thoughts = thoughts[:traj_len]

    if sample_success and (is_last_in_session or random.random() < 0.4):
        reward = 1.0
        finish_reason = "stop"
    else:
        reward = 0.0
        if random.random() < 0.3:
            finish_reason = "length"
        elif random.random() < 0.2:
            finish_reason = "max_step_limit"
        else:
            finish_reason = "stop"
        if tools[-1] == "finish" and reward == 0.0 and random.random() < 0.5:
            tools[-1] = random.choice(["Bash", "Read"])

    events: list[dict[str, Any]] = []

    events.append(
        {
            "event": "trajectory_begin",
            "uid": uid,
            "sample_index": sample_index,
            "session_index": session_index,
            "trajectory_index": trajectory_index,
            "reason": "initial",
        }
    )

    for step_i in range(len(tools)):
        tool_name = tools[step_i]
        thought = thoughts[step_i]
        is_last_step = step_i == len(tools) - 1

        if tool_name == "finish":
            tool_results = [
                {
                    "name": "finish",
                    "action": "submit",
                    "observation": (
                        "Task completed successfully."
                        if reward > 0
                        else "Unable to resolve."
                    ),
                    "status": "ok",
                }
            ]
        else:
            obs = random.choice(TOOL_OBSERVATIONS.get(tool_name, ["..."]))
            tool_results = [
                {
                    "name": tool_name,
                    "action": _generate_action(tool_name),
                    "observation": obs,
                    "status": "ok" if random.random() < 0.9 else "timeout",
                }
            ]

        events.append(
            _build_step_event(
                uid=uid,
                step_index=step_i + 1,
                finish_reason=finish_reason,
                thought=thought,
                tool_results=tool_results,
                is_last=is_last_step,
            )
        )

    return events


# ---------------------------------------------------------------------------
# Core generation -- only interacts with TrajectoryBuilder
# ---------------------------------------------------------------------------


def generate(builder: TrajectoryBuilder, sample_count: int, seed: int) -> None:
    """Feed all trajectory events to *builder* at once (batch mode)."""
    random.seed(seed)

    for si in range(sample_count):
        uid = f"task-{si:04d}-{random.randint(1000, 9999)}"
        scenario = SCENARIOS[si % len(SCENARIOS)]
        sample_success = random.random() < 0.35
        session_count = random.randint(3, 5)

        for sess_i in range(session_count):
            traj_count = random.randint(1, 4)
            for ti in range(traj_count):
                events = build_trajectory_events(
                    uid=uid,
                    sample_index=si,
                    session_index=sess_i,
                    trajectory_index=ti,
                    scenario=scenario,
                    sample_success=sample_success,
                    is_last_in_session=(ti == traj_count - 1),
                )
                for event in events:
                    builder.feed(event)

    print(f"Generated {sample_count} samples.")


def stream(
    builder: TrajectoryBuilder,
    sample_count: int,
    interval: float,
    seed: int,
) -> None:
    """Feed trajectory events incrementally (step by step with sleeps)."""
    import time as _time  # noqa: E402

    random.seed(seed)
    uids: list[str] = []
    for si in range(sample_count):
        uids.append(f"task-{si:04d}-{random.randint(1000, 9999)}")

    sample_success = [random.random() < 0.35 for _ in range(sample_count)]
    session_counts = [random.randint(3, 5) for _ in range(sample_count)]
    session_counts[0] = 2

    # Build per-session trajectory event lists, then interleave round-robin.
    session_queues: list[list[tuple[int, int, int, list[dict[str, Any]]]]] = []
    for si in range(sample_count):
        for sess_i in range(session_counts[si]):
            traj_count = random.randint(1, 4)
            session_events: list[tuple[int, int, int, list[dict[str, Any]]]] = []
            for ti in range(traj_count):
                events = build_trajectory_events(
                    uid=uids[si],
                    sample_index=si,
                    session_index=sess_i,
                    trajectory_index=ti,
                    scenario=SCENARIOS[si % len(SCENARIOS)],
                    sample_success=sample_success[si],
                    is_last_in_session=(ti == traj_count - 1),
                )
                session_events.append((si, sess_i, ti, events))
            session_queues.append(session_events)

    # Interleave: one trajectory's events per round across all sessions
    max_trajs = max(len(q) for q in session_queues)
    event_queue: list[tuple[int, int, int, dict[str, Any]]] = []
    for round_i in range(max_trajs):
        for q in session_queues:
            if round_i < len(q):
                si, sess_i, ti, events = q[round_i]
                for ev in events:
                    event_queue.append((si, sess_i, ti, ev))

    total_trajs = sum(len(q) for q in session_queues)
    print(
        f"Streaming {total_trajs} trajectories across {sample_count} samples "
        f"(~{len(event_queue) * interval:.0f}s)."
    )
    print("Open http://localhost:8080 to watch.\n")

    for idx, (si, sess_i, ti, event) in enumerate(event_queue):
        uid = uids[si]
        builder.feed(event)
        etype = event["event"]
        if etype == "trajectory_begin":
            print(
                f"[event {idx + 1}/{len(event_queue)}] "
                f"{uid[:12]} s={sess_i} t={ti} begin"
            )
        else:
            print(
                f"[event {idx + 1}/{len(event_queue)}] "
                f"{uid[:12]} s={sess_i} t={ti} "
                f"step={event.get('step_index', '?')} "
                f"fr={event.get('finish_reason', '?')}"
            )
        if idx < len(event_queue) - 1:
            _time.sleep(interval)

    print(f"\nDone. {total_trajs} trajectories generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trajectory events through TrajectoryBuilder"
    )
    parser.add_argument("output_dir", help="Output directory for trajectory data")
    parser.add_argument(
        "--samples", type=int, default=12, help="Number of samples (default: 12)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream data incrementally (step by step with sleeps)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do NOT clear output directory before generating",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="Seconds between events in stream mode (default: 0.3)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use in-memory SampleRecord instead of FileSampleRecord",
    )
    args = parser.parse_args()

    if not args.no_clean and _Path(args.output_dir).exists():
        import shutil  # noqa: E402

        shutil.rmtree(args.output_dir)
        print(f"Cleared {args.output_dir}")

    # Build the builder -- the only place that knows about storage backends.
    if args.memory:
        from rl_insight.experimental.samples import SampleRecord  # noqa: E402

        builder = TrajectoryBuilder(
            sample_factory=lambda uid, si: SampleRecord.create(uid=uid, sample_index=si)  # noqa: E731
        )
    else:
        from rl_insight.experimental.samples import FileSampleRecord  # noqa: E402

        builder = TrajectoryBuilder(
            sample_factory=lambda uid, si: FileSampleRecord.create(  # noqa: E731
                args.output_dir, uid=uid, sample_index=si
            )
        )

    if args.stream:
        stream(builder, args.samples, args.interval, args.seed)
    else:
        generate(builder, args.samples, args.seed)

    print("\nStart the viewer:")
    print(f"  python rl_insight/experimental/server.py {args.output_dir} --port 8080")


if __name__ == "__main__":
    main()
