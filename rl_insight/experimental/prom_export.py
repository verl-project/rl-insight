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

"""SampleRecord tree → Prometheus gauges for Agent Loop Repeat.

Metric names, label sets, and value types match DATA_SOURCE_FIELD_COMPARE.md /
REPEAT_POC.md. Visualization reads these gauges; do not rename without updating
the dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

TRAJ_LABELS = ("run_id", "sample", "session", "traj")
UNIT_LABELS = ("run_id", "sample", "session", "traj", "uid")

g_unit = Gauge(
    "agent_loop_unit",
    "Agent Loop traj node presence (Repeat enumeration)",
    UNIT_LABELS,
)
g_turns = Gauge(
    "agent_loop_traj_turns",
    "Agent Loop traj turn count",
    TRAJ_LABELS,
)
g_reward = Gauge(
    "agent_loop_traj_reward",
    "Agent Loop traj reward",
    TRAJ_LABELS,
)
g_success = Gauge(
    "agent_loop_traj_success",
    "Agent Loop traj success (1=reward_score > 0, 0=otherwise)",
    TRAJ_LABELS,
)
g_run_info = Gauge(
    "agent_loop_run_info",
    "Agent Loop run row title",
    ("run_id", "title"),
)
g_sample_info = Gauge(
    "agent_loop_sample_info",
    "Agent Loop sample row title",
    ("run_id", "sample", "title"),
)
g_session_info = Gauge(
    "agent_loop_session_info",
    "Agent Loop session row title",
    # sess_key = sample=<i>/session=<j> — unique Grafana variable value
    # (plain session index collides across samples when parent is All).
    ("run_id", "sample", "session", "sess_key", "title"),
)
g_traj_info = Gauge(
    "agent_loop_traj_info",
    "Agent Loop traj row title",
    # leaf = sample=<i>/session=<j>/traj=<k> — unique traj variable value
    ("run_id", "sample", "session", "traj", "leaf", "title"),
)

_ALL_GAUGES = (
    g_unit,
    g_turns,
    g_reward,
    g_success,
    g_run_info,
    g_sample_info,
    g_session_info,
    g_traj_info,
)


def _coerce_sample(sample: Any) -> Any:
    """Accept SampleRecord or FileSampleRecord / thin wrappers."""
    from rl_insight.experimental.samples.sample import SampleRecord

    if isinstance(sample, SampleRecord):
        return sample
    inner = getattr(sample, "_inner", None)
    if isinstance(inner, SampleRecord):
        return inner
    load = getattr(sample, "load", None)
    if callable(load):
        loaded = load()
        if isinstance(loaded, SampleRecord):
            return loaded
    # In-memory BaseSample that already exposes .sessions
    if hasattr(sample, "sessions") and hasattr(sample, "sample_index"):
        return sample
    raise TypeError(
        f"prom_export expects SampleRecord-like output, got {type(sample)!r}"
    )


def clear_prom() -> None:
    """Drop all label children so removed nodes disappear on the next scrape."""
    for gauge in _ALL_GAUGES:
        gauge.clear()


def traj_turns(traj: Any) -> int:
    n = getattr(traj, "num_turns", None)
    if n is not None and int(n) > 0:
        return int(n)
    steps = getattr(traj, "steps", None) or []
    return len(steps)


def traj_reward(traj: Any) -> float | None:
    score = getattr(traj, "reward_score", None)
    if score is not None:
        return float(score)
    # Missing reward is not equivalent to success/failure. Keep it absent
    # instead of manufacturing a business value from finish_reason.
    return None


def traj_success(traj: Any) -> bool:
    """Match the upstream demo viewer: success iff ``reward_score > 0``."""
    score = getattr(traj, "reward_score", None)
    return float(score or 0.0) > 0.0


def publish_sample_runs(
    runs: Sequence[tuple[str, Iterable[Any]]],
    *,
    clear: bool = True,
) -> int:
    """Write Repeat Prom series from one or more ``(run_id, samples)`` trees.

    Returns number of traj nodes published.
    """
    if clear:
        clear_prom()

    n = 0
    for run_id, raw_samples in runs:
        samples = [_coerce_sample(s) for s in raw_samples]
        n += _publish_one_run(run_id, samples)
    logger.info("prom_export published %s traj nodes across %s runs", n, len(runs))
    return n


def _publish_one_run(run_id: str, samples: list[Any]) -> int:
    run_succ = run_total = 0
    for sample in samples:
        for session in sample.sessions:
            for traj in session.trajectories:
                run_total += 1
                if traj_success(traj):
                    run_succ += 1

    run_title = (
        f"Run · {run_id} · samples {len(samples)} · "
        f"success {run_succ}/{run_total}"
    )
    g_run_info.labels(run_id=run_id, title=run_title).set(1)

    n = 0
    for sample in samples:
        sample_s = str(sample.sample_index)
        s_succ = s_total = s_turns = 0
        for session in sample.sessions:
            for traj in session.trajectories:
                s_total += 1
                s_turns += traj_turns(traj)
                if traj_success(traj):
                    s_succ += 1
        sample_title = (
            f"Sample {sample_s} · success {s_succ}/{s_total} · "
            f"{s_turns} turns · {len(sample.sessions)} sessions"
        )
        g_sample_info.labels(
            run_id=run_id, sample=sample_s, title=sample_title
        ).set(1)

        for session in sample.sessions:
            session_s = str(session.session_index)
            sess_succ = sess_total = sess_turns = 0
            for traj in session.trajectories:
                sess_total += 1
                sess_turns += traj_turns(traj)
                if traj_success(traj):
                    sess_succ += 1
            sess_key = f"sample={sample_s}/session={session_s}"
            session_title = (
                f"Session {session_s} · success {sess_succ}/{sess_total} · "
                f"{sess_turns} turns · {sess_total} trajectories"
            )
            g_session_info.labels(
                run_id=run_id,
                sample=sample_s,
                session=session_s,
                sess_key=sess_key,
                title=session_title,
            ).set(1)

            for traj in session.trajectories:
                traj_s = str(traj.trajectory_index)
                labels = {
                    "run_id": run_id,
                    "sample": sample_s,
                    "session": session_s,
                    "traj": traj_s,
                }
                turns = traj_turns(traj)
                reward = traj_reward(traj)
                ok = traj_success(traj)
                leaf = f"sample={sample_s}/session={session_s}/traj={traj_s}"
                reward_text = "n/a" if reward is None else str(reward)
                traj_title = (
                    f"Trajectory #{traj_s} · reward {reward_text} · {turns} turns"
                )
                uid = getattr(sample, "uid", None) or leaf
                g_unit.labels(
                    run_id=run_id,
                    sample=sample_s,
                    session=session_s,
                    traj=traj_s,
                    uid=str(uid),
                ).set(1)
                g_turns.labels(**labels).set(float(turns))
                if reward is not None:
                    g_reward.labels(**labels).set(reward)
                g_success.labels(**labels).set(1.0 if ok else 0.0)
                g_traj_info.labels(**labels, leaf=leaf, title=traj_title).set(1)
                n += 1
    return n
