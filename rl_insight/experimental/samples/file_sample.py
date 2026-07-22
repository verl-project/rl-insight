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

"""Filesystem-backed sample for distributed trajectory logging.

Directory layout::

    {root_dir}/
      └── {uid}/
          ├── _index.json            # {"sample_index": 0, "trajectories": {"0": [0,1], "1": [0]}}
          ├── t_0_0.json             # TrajectoryRecord (full model_dump)
          ├── t_0_1.json
          ├── t_1_0.json
          └── ...

Design decisions:

- **One file per trajectory**: Each ``t_{si}_{ti}.json`` is a complete,
  self-contained ``TrajectoryRecord`` serialized via ``model_dump()``.
  Writers for different ``(si, ti)`` pairs never touch the same file.
- **Atomic writes**: temp file → fsync → rename. Safe for concurrent
  writers on the same sample, as long as they use disjoint trajectory
  indices.
- **Lightweight index**: ``_index.json`` tracks which trajectories exist
  so reads don't need to scan the directory. Updated atomically on each
  ``new_trajectory()`` call.
- **No in-memory state**: Every ``get_trajectory()`` reads from disk.
  Use ``load()`` to batch-read everything into an in-memory
  ``SampleRecord`` for aggregation or analysis.

Implements ``BaseSample`` (Protocol) -- same trajectory CRUD methods as
``SampleRecord``.

Usage::

    from rl_insight.experimental import FileSampleRecord, Step, ToolResult

    # Create (fails if directory exists)
    sample = FileSampleRecord.create("/data/trajs", uid="task-1", sample_index=0)

    # Open existing
    sample = FileSampleRecord.open("/data/trajs", uid="task-1")

    # Write trajectories
    sample.new_trajectory(session_index=0)
    sample.add_step(0, 0, Step(thought="...", tool_results=[...]))
    sample.finish_trajectory(0, 0, "stop")
    sample.set_trajectory_reward(0, 0, 1.0)

    # Read
    traj = sample.get_trajectory(0, 0)  # reads from disk
    sample.list_trajectories()           # [(0,0), (0,1), ...]

    # Batch load into memory
    mem = sample.load()                  # → SampleRecord
    mem.mean_reward                      # aggregate queries
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


from rl_insight.experimental.samples.sample import (
    SampleRecord,
    Step,
    TrajectoryRecord,
    TrainingStatus,
)


class FileSampleRecord:
    """Filesystem-backed sample: same API as ``SampleRecord``, persisted to disk.

    Each trajectory is stored as an independent JSON file. The index
    (``_index.json``) tracks which trajectories exist so reads don't
    need to scan the directory.
    """

    def __init__(self, root_dir: str | Path, uid: str) -> None:
        self.root_dir = Path(root_dir)
        self.uid = uid
        self._dir = self.root_dir / uid
        self._index_path = self._dir / "_index.json"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        root_dir: str | Path,
        *,
        uid: str,
        sample_index: int = 0,
    ) -> FileSampleRecord:
        """Create a new sample on disk. Raises if the directory already exists."""
        instance = cls(root_dir, uid)
        if instance._dir.exists():
            raise FileExistsError(f"sample directory already exists: {instance._dir}")
        instance._dir.mkdir(parents=True)
        instance._write_index({"sample_index": sample_index, "trajectories": {}})
        return instance

    @classmethod
    def open(cls, root_dir: str | Path, uid: str) -> FileSampleRecord:
        """Open an existing sample directory. Raises if not found."""
        instance = cls(root_dir, uid)
        if not instance._dir.is_dir():
            raise FileNotFoundError(f"sample directory not found: {instance._dir}")
        return instance

    # ------------------------------------------------------------------
    # Session / trajectory management
    # ------------------------------------------------------------------

    @property
    def sample_index(self) -> int:
        return self._read_index().get("sample_index", 0)

    def new_trajectory(self, session_index: int = 0, **kwargs) -> TrajectoryRecord:
        """Create a new empty trajectory on disk.

        ``trajectory_index`` is auto-incremented from existing trajectories
        in the same session.
        """
        index = self._read_index()
        trajs = index.setdefault("trajectories", {})
        si_key = str(session_index)
        existing = trajs.get(si_key, [])
        ti = len(existing)
        if kwargs.get("trajectory_index") is not None:
            ti = kwargs.pop("trajectory_index")

        traj = TrajectoryRecord.create(
            uid=self.uid,
            sample_index=index["sample_index"],
            session_index=session_index,
            trajectory_index=ti,
            **kwargs,
        )
        self._write_traj(session_index, ti, traj)

        existing.append(ti)
        trajs[si_key] = existing
        self._write_index(index)
        return traj

    def get_trajectory(
        self, session_index: int, trajectory_index: int
    ) -> TrajectoryRecord | None:
        """Read a trajectory from disk, or None."""
        path = self._traj_path(session_index, trajectory_index)
        if not path.exists():
            return None
        return self._read_traj(path)

    def add_step(self, session_index: int, trajectory_index: int, step: Step) -> None:
        traj = self._require_traj(session_index, trajectory_index)
        traj.add_step(step)
        self._write_traj(session_index, trajectory_index, traj)

    def finish_trajectory(
        self,
        session_index: int,
        trajectory_index: int,
        exit_reason: str = "finished",
        status: TrainingStatus = "success",
    ) -> None:
        traj = self._require_traj(session_index, trajectory_index)
        traj.finish(exit_reason, status)
        self._write_traj(session_index, trajectory_index, traj)

    def set_trajectory_reward(
        self,
        session_index: int,
        trajectory_index: int,
        score: float,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        traj = self._require_traj(session_index, trajectory_index)
        traj.set_reward(score, extra_info)
        self._write_traj(session_index, trajectory_index, traj)

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
        traj = self._require_traj(session_index, trajectory_index)
        traj.set_token_data(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
        )
        self._write_traj(session_index, trajectory_index, traj)

    def list_trajectories(self) -> list[tuple[int, int]]:
        """Return all ``(session_index, trajectory_index)`` pairs on disk."""
        index = self._read_index()
        result = []
        for si_key, ti_list in index.get("trajectories", {}).items():
            for ti in ti_list:
                result.append((int(si_key), ti))
        result.sort()
        return result

    # ------------------------------------------------------------------
    # Load into memory
    # ------------------------------------------------------------------

    def load(self) -> SampleRecord:
        """Read all trajectories from disk and build an in-memory ``SampleRecord``."""
        index = self._read_index()
        sample = SampleRecord.create(
            uid=self.uid,
            sample_index=index["sample_index"],
        )
        for si_key, ti_list in sorted(index.get("trajectories", {}).items()):
            si = int(si_key)
            session = sample.new_session(session_index=si)
            for ti in sorted(ti_list):
                traj = self._read_traj(self._traj_path(si, ti))
                session.add_trajectory(traj)
        return sample

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _traj_path(self, session_index: int, trajectory_index: int) -> Path:
        return self._dir / f"t_{session_index}_{trajectory_index}.json"

    def _require_traj(
        self, session_index: int, trajectory_index: int
    ) -> TrajectoryRecord:
        traj = self.get_trajectory(session_index, trajectory_index)
        if traj is None:
            raise KeyError(
                f"trajectory {session_index}/{trajectory_index} not found "
                f"in sample {self.uid!r}"
            )
        return traj

    def _read_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"sample_index": 0, "trajectories": {}}
        return json.loads(self._index_path.read_text())

    def _write_index(self, index: dict[str, Any]) -> None:
        self._atomic_write(self._index_path, index)

    def _read_traj(self, path: Path) -> TrajectoryRecord:
        return TrajectoryRecord.model_validate(json.loads(path.read_text()))

    def _write_traj(
        self, session_index: int, trajectory_index: int, traj: TrajectoryRecord
    ) -> None:
        self._atomic_write(
            self._traj_path(session_index, trajectory_index),
            traj.model_dump(),
        )

    @staticmethod
    def _atomic_write(path: Path, data: dict[str, Any]) -> None:
        """Write JSON atomically: temp file → rename."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix="." + path.name,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        )
        try:
            json.dump(data, tmp, ensure_ascii=False, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, str(path))
        except Exception:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
