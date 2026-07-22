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

"""Deprecated shim: Tempo export lives in ``tempo_export`` (post-hoc mapper).

``GrafanaRecord`` previously side-effect-exported during ``add_step``. That path
is removed so PR#120 Builder/SampleRecord stay a pure producer black box.
"""

from __future__ import annotations

from typing import Any

from rl_insight.experimental.samples.sample import (
    SampleRecord,
    Step,
    TrajectoryRecord,
    TrainingStatus,
)
from rl_insight.experimental.tempo_export import (  # noqa: F401
    SERVICE_NAME_VALUE,
    compress_span_times,
    export_samples_to_tempo,
    flush_span_dicts,
    lane_id,
    new_run_id,
    wait_for_otlp,
)

# Compatibility aliases
configure_exporter = None  # removed; use export_samples_to_tempo()
shutdown_exporter = None
emitted_span_count = lambda: 0  # noqa: E731
pending_span_count = emitted_span_count


def service_name_for(demo_key: str = "", *, base: str = SERVICE_NAME_VALUE) -> str:
    del demo_key
    return base


def demo_key_for(seed: int, samples: int) -> str:
    return f"seed{seed}-n{samples}"


class GrafanaRecord:
    """Thin SampleRecord wrapper (no Tempo side effects). Prefer SampleRecord."""

    def __init__(self, inner: SampleRecord, **_kwargs: Any) -> None:
        self._inner = inner

    @classmethod
    def create(
        cls,
        *,
        uid: str,
        sample_index: int = 0,
        **_kwargs: Any,
    ) -> GrafanaRecord:
        return cls(SampleRecord.create(uid=uid, sample_index=sample_index))

    @property
    def uid(self) -> str:
        return self._inner.uid

    @property
    def sample_index(self) -> int:
        return self._inner.sample_index

    @property
    def sessions(self):
        return self._inner.sessions

    def new_trajectory(self, session_index: int = 0, **kwargs: Any) -> TrajectoryRecord:
        return self._inner.new_trajectory(session_index, **kwargs)

    def get_trajectory(
        self, session_index: int, trajectory_index: int
    ) -> TrajectoryRecord | None:
        return self._inner.get_trajectory(session_index, trajectory_index)

    def add_step(self, session_index: int, trajectory_index: int, step: Step) -> None:
        self._inner.add_step(session_index, trajectory_index, step)

    def finish_trajectory(
        self,
        session_index: int,
        trajectory_index: int,
        exit_reason: str = "finished",
        status: TrainingStatus = "success",
    ) -> None:
        self._inner.finish_trajectory(
            session_index, trajectory_index, exit_reason, status
        )

    def set_trajectory_reward(
        self,
        session_index: int,
        trajectory_index: int,
        score: float,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        self._inner.set_trajectory_reward(
            session_index, trajectory_index, score, extra_info
        )

    def set_trajectory_token_data(self, *args: Any, **kwargs: Any) -> None:
        self._inner.set_trajectory_token_data(*args, **kwargs)

    def flush_pending(self) -> None:
        return None
