# Copyright (c) 2025 verl-project authors.
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

import bisect
import csv
import ijson
import json
from loguru import logger
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig

from .parser import BaseClusterParser, register_cluster_parser
from recipe.utils.schema import Constant, DataMap
from recipe.data import DataEnum


@register_cluster_parser("memory")
class MemoryClusterParser(BaseClusterParser):
    """Ascend NPU memory allocation parser.

    Parses ``operator_memory.csv`` and ``trace_view.json`` from Ascend Profiler
    output to produce per-rank memory event records (``MemoryEventRow``).

    Since no built-in visualizer exists for memory data yet, the primary
    consumption path is the ``pd.DataFrame`` returned by ``run()``.  Callers
    should inspect the DataFrame columns directly or export to CSV / JSON for
    external analysis.

    Output DataFrame columns (one row per memory allocation / deallocation):

    =============== ====== ===================================================
    Column          Type   Description
    =============== ====== ===================================================
    name            str    Operator name from ``operator_memory.csv``
                          (e.g. ``aten::empty``, ``aten::matmul``).
                          Covers **all** operator types that appear in the
                          CSV, not only ``cpu_op`` — communication ops and
                          kernel ops are also included.
    role            str    RL role inferred from directory structure or
                          ``profiler_metadata.json`` (e.g.
                          ``actor_update``, ``actor_compute_log_prob``).
    rank_id         int    Rank identifier extracted from
                          ``profiler_info_<rank_id>.json``.
    call_stack      str    Full Python call stack associated with the
                          operator, sourced from ``trace_view.json``
                          events where ``cat=="cpu_op"`` and ``args``
                          contains ``"Call stack"``.  Multiple frames are
                          separated by ``";\\r\\n"``.  **Empty string** when
                          no matching ``cpu_op`` event is found (common for
                          communication / kernel ops which lack call-stack
                          data in the trace).
    call_stack_top  str    First line of the call stack — the user-code
                          entry point (e.g. ``fsdp2.py(112): train_batch``).
                          **Empty string** when ``call_stack`` is empty.
    size_kb         float  Memory size in KB.  **Positive** = allocation,
                          **negative** = deallocation.  Mirrors the
                          ``Size(KB)`` column in ``operator_memory.csv``.
    start_time_ms   float Timestamp of the allocation / deallocation
                          in **milliseconds** (converted from microseconds
                          in the CSV).  Named ``start_time_ms`` to align
                          with the key expected by
                          ``BaseClusterParser.reducer_func``.
    duration_ms     float  How long the memory block stayed allocated, in
                          **milliseconds**.  ``0.0`` when the memory has
                          not been released yet (``Duration(us)`` empty in
                          CSV).
    total_allocated_mb float Cumulative allocated memory (MB) at the
                          moment of this allocation, as reported by the
                          Ascend memory tracker.
    total_reserved_mb  float Cumulative reserved memory (MB) at the
                          moment of this allocation.
    total_active_mb    float Cumulative active memory (MB) at the
                          moment of this allocation.
    device_type     str    Device type string (e.g. ``NPU:0``).
    =============== ====== ===================================================

    Call-stack matching strategy
    ----------------------------

    ``operator_memory.csv`` records **what** memory was allocated and **when**,
    but not **why** (i.e. which Python call stack triggered it).  The parser
    recovers call stacks by correlating each CSV row with ``trace_view.json``:

    1. Build an in-memory index from ``trace_view.json``: keep only events
       where ``cat == "cpu_op"`` **and** ``args`` contains ``"Call stack"``;
       group by ``name``, sort each group by ``ts`` ascending.
    2. For each CSV row, look up its ``Name`` in the index and binary-search
       (``bisect_right``) for the entry whose ``ts`` is the largest value
       **≤** the row's ``Allocation Time``.  This works because ``ts`` is the
       operator start time and ``Allocation Time`` is the moment the operator
       internally calls a memory allocator, so ``Allocation Time ≥ ts``.
    3. Rows whose ``Name`` has no ``cpu_op`` entry in the index (e.g.
       communication ops, kernel ops) receive empty ``call_stack`` and
       ``call_stack_top``.

    Large-file handling
    -------------------

    ``trace_view.json`` can reach hundreds of MB.  The parser uses ``ijson``
    for streaming iteration (``ijson.items(f, "item")``) to avoid loading
    the entire file into memory at once.

    Data scope
    ----------

    The output includes **all** rows from ``operator_memory.csv``, regardless
    of operator category.  Only the call-stack enrichment is limited to
    ``cpu_op`` events because that is the only category that carries Python
    call-stack information in Ascend Profiler output.
    """

    input_type: DataEnum = DataEnum.ASCEND_MEMORY

    def __init__(self, params: Union[DictConfig, dict]) -> None:
        super().__init__(params)

    def parse_analysis_data(
        self, profiler_data_path: str, rank_id: int, role: str
    ) -> list[dict[str, Any]]:
        """Parse memory profiling data for a single rank.

        Reads ``operator_memory.csv`` for memory allocation records and
        ``trace_view.json`` for call-stack enrichment.  Returns a list of
        dicts with keys defined by ``MemoryEventRow`` — see class docstring
        for full column descriptions.

        Args:
            profiler_data_path: Path to the ``ASCEND_PROFILER_OUTPUT``
                directory containing both ``operator_memory.csv`` and
                ``trace_view.json``.
            rank_id: Rank identifier (for data attribution).
            role: RL role name (e.g. ``actor_update``).

        Returns:
            list[dict[str, Any]]: One entry per CSV row.  Empty list when
                either required file is missing or contains no data.
        """
        if not profiler_data_path:
            logger.warning(f"Rank {rank_id}: profiler_data_path is empty")
            return []

        trace_view_path = os.path.join(profiler_data_path, "trace_view.json")
        operator_memory_path = os.path.join(profiler_data_path, "operator_memory.csv")

        if not os.path.exists(trace_view_path):
            logger.warning(
                f"Rank {rank_id}: trace_view.json not found at {trace_view_path}"
            )
            return []

        if not os.path.exists(operator_memory_path):
            logger.warning(
                f"Rank {rank_id}: operator_memory.csv not found at {operator_memory_path}"
            )
            return []

        call_stack_index = self._build_call_stack_index(trace_view_path)

        results = self._parse_operator_memory(
            operator_memory_path, call_stack_index, rank_id, role
        )

        logger.info(f"Rank {rank_id} Role {role}: parsed {len(results)} memory events")
        return results

    def _build_call_stack_index(self, trace_view_path: str) -> dict:
        """Build a call-stack lookup index from ``trace_view.json``.

        Streams through the JSON file with ``ijson`` and keeps only events
        where ``cat == "cpu_op"`` **and** ``args`` contains a ``"Call stack"``
        key.  Events are grouped by ``name`` and sorted by ``ts`` within each
        group so that :meth:`_match_call_stack` can binary-search them.

        The returned index pre-computes a ``ts_list`` alongside ``entries``
        so that :meth:`_match_call_stack` can call ``bisect_right`` directly
        without rebuilding the list on every invocation.

        Args:
            trace_view_path: Path to the ``trace_view.json`` file.

        Returns:
            dict mapping ``name (str)`` → ``dict`` with keys:

            - ``"entries"`` (``list[dict]``): each dict has ``ts`` (float,
              µs), ``dur`` (float, µs), ``call_stack`` (str), sorted by
              ``ts`` ascending.
            - ``"ts_list"`` (``list[float]``): pre-extracted ``ts`` values
              in the same order as ``entries``, for O(log n) binary search.

            Returns an empty dict when no qualifying events are found.
        """
        raw_index: dict[str, list] = {}

        # Use ijson for streaming iteration to avoid loading the entire JSON
        # into memory at once; trace_view.json can reach hundreds of MB
        with open(trace_view_path, "rb") as f:
            for event in ijson.items(f, "item"):
                # Only cpu_op events carry call stacks; other categories
                # (e.g. communication, kernel) do not
                if event.get("cat") != "cpu_op":
                    continue

                # Only events whose args contain "Call stack" have stack info;
                # some cpu_op events may lack this field and must be skipped
                args = event.get("args", {})
                if not isinstance(args, dict) or "Call stack" not in args:
                    continue

                name = event.get("name", "")
                # ts is a string in JSON (e.g. "1755143611835441.990"); convert to float
                ts = float(event["ts"])
                dur = float(event.get("dur", 0))
                call_stack = args["Call stack"]

                # Group by name; each invocation of the same operator produces
                # a separate entry
                if name not in raw_index:
                    raw_index[name] = []
                raw_index[name].append({"ts": ts, "dur": dur, "call_stack": call_stack})

        # Sort each group by ts ascending and pre-compute ts_list so that
        # _match_call_stack can binary-search in O(log n) without rebuilding
        # the list per call.
        index: dict[str, dict] = {}
        for name, entries in raw_index.items():
            entries.sort(key=lambda x: x["ts"])
            index[name] = {
                "entries": entries,
                "ts_list": [e["ts"] for e in entries],
            }

        return index

    def _parse_operator_memory(
        self,
        csv_path: str,
        call_stack_index: dict,
        rank_id: int,
        role: str,
    ) -> list[dict[str, Any]]:
        """Parse ``operator_memory.csv`` and enrich each row with call stacks.

        Iterates over every row in the CSV (both allocations and deallocations)
        and attempts to attach a Python call stack by looking up the operator
        ``Name`` in *call_stack_index*.  Rows whose ``Name`` has no matching
        ``cpu_op`` entry receive empty ``call_stack`` / ``call_stack_top``.

        Time values from the CSV (microseconds) are converted to milliseconds
        to stay consistent with the rest of the framework.

        Args:
            csv_path: Path to ``operator_memory.csv``.
            call_stack_index: Index built by :meth:`_build_call_stack_index`.
            rank_id: Rank identifier for data attribution.
            role: RL role name for data attribution.

        Returns:
            list[dict[str, Any]]: One dict per CSV row.  See class docstring
                for field semantics.
        """
        results: list[dict[str, Any]] = []
        us_to_ms = Constant.US_TO_MS

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Size(KB): positive = allocation, negative = deallocation;
                # both are retained
                size_kb = float(row["Size(KB)"].strip())

                # Allocation Time(us): the CSV field may have trailing tabs; strip
                allocation_time_us = float(row["Allocation Time(us)"].strip())

                # Call-stack matching: look up the operator name in the index
                # and find the closest entry with ts <= allocation_time;
                # returns empty strings when no match is found
                call_stack, call_stack_top = self._match_call_stack(
                    row["Name"].strip(), allocation_time_us, call_stack_index
                )

                # Duration(us) may be empty (memory not yet released);
                # default duration_ms to 0.0 in that case
                duration_us = row.get("Duration(us)", "").strip()
                duration_ms = float(duration_us) / us_to_ms if duration_us else 0.0

                # Build memory event row; all times are converted to milliseconds
                results.append(
                    {
                        "name": row["Name"].strip(),
                        "role": role,
                        "rank_id": rank_id,
                        "call_stack": call_stack,
                        "call_stack_top": call_stack_top,
                        "size_kb": size_kb,
                        "start_time_ms": allocation_time_us / us_to_ms,
                        "duration_ms": duration_ms,
                        "total_allocated_mb": float(
                            row["Allocation Total Allocated(MB)"].strip() or 0
                        ),
                        "total_reserved_mb": float(
                            row["Allocation Total Reserved(MB)"].strip() or 0
                        ),
                        "total_active_mb": float(
                            row["Allocation Total Active(MB)"].strip() or 0
                        ),
                        "device_type": row["Device Type"].strip(),
                    }
                )

        return results

    def _match_call_stack(
        self,
        name: str,
        allocation_time_us: float,
        call_stack_index: dict,
    ) -> tuple[str, str]:
        """Look up the call stack for a memory record by operator name and time.

        Uses binary search (``bisect_right``) on the ``ts`` list of the
        matching ``name`` group to find the entry whose ``ts`` is the largest
        value **≤** *allocation_time_us*.  This is correct because ``ts`` is
        the operator start time and ``Allocation Time`` is the moment the
        operator internally triggers a memory allocation, so
        ``Allocation Time ≥ ts``.

        Args:
            name: Operator name from ``operator_memory.csv`` (e.g.
                ``aten::empty``).
            allocation_time_us: Allocation timestamp in microseconds.
            call_stack_index: Index built by :meth:`_build_call_stack_index`.

        Returns:
            A tuple ``(call_stack, call_stack_top)``.  Both are empty strings
                when *name* is absent from the index or all indexed ``ts``
                values are greater than *allocation_time_us*.
        """
        if name not in call_stack_index:
            return "", ""

        group = call_stack_index[name]
        entries = group["entries"]
        ts_list = group["ts_list"]

        # Binary search: find the closest entry with ts <= allocation_time.
        # bisect_right returns the first position greater than allocation_time;
        # subtracting 1 gives the last position <= allocation_time.
        idx = bisect.bisect_right(ts_list, allocation_time_us) - 1
        if idx < 0:
            # All ts values are greater than allocation_time; no match
            return "", ""

        entry = entries[idx]
        call_stack = entry["call_stack"]
        # Call stack frames are separated by ";\r\n"; take the first line as
        # the top-level entry point
        call_stack_top = call_stack.split(";\r\n")[0] if call_stack else ""
        return call_stack, call_stack_top

    def allocate_prof_data(self, input_path: str) -> list[DataMap]:
        ascend_pt_dirs = []
        for root, dirs, _ in os.walk(input_path):
            for dir_name in dirs:
                if dir_name.endswith(Constant.ASCEND_PROFILER_SUFFIX):
                    path = os.path.join(root, dir_name)
                    ascend_pt_dirs.append(
                        {"role": Path(path).parent.name, "path": path}
                    )
        data_map = self._get_data_map(ascend_pt_dirs)
        data_maps = self._get_rank_path_with_role(data_map)
        return data_maps

    def _get_profiler_data_path(self, rank_id, data_path):
        return os.path.join(data_path, Constant.ASCEND_PROFILER_OUTPUT)

    def _get_rank_path_with_role(self, data_map) -> list[DataMap]:
        if self._rank_list != "all":
            logger.error("RL analysis currently only supports processing all ranks")
            return []

        rank_ids_with_role = list(data_map.keys())
        data_paths: list[DataMap] = []
        for task_role, rank_id in rank_ids_with_role:
            rank_path_list = data_map[(task_role, rank_id)]
            profiler_data_path_list = [
                self._get_profiler_data_path(rank_id, rank_path)
                for rank_path in rank_path_list
            ]
            for profiler_data_path in profiler_data_path_list:
                data_path_dict: DataMap = {
                    Constant.RANK_ID: rank_id,
                    Constant.ROLE: task_role,
                    Constant.PROFILER_DATA_PATH: "",
                    "step": None,
                }

                if os.path.exists(profiler_data_path):
                    data_path_dict[Constant.PROFILER_DATA_PATH] = profiler_data_path
                    data_paths.append(data_path_dict)
                else:
                    logger.warning(
                        f"Profiler data dir not found, rank id: {rank_id}, data path: {profiler_data_path}."
                    )
        return data_paths

    def _get_data_map(self, path_list) -> dict[tuple[str, int], list[str]]:
        data_map: dict[tuple[str, int], list[str]] = {}
        rank_id_map = defaultdict(list)
        for path_info in path_list:
            role = path_info.get("role")
            dir_name = path_info.get("path")
            rank_id = self._get_rank_id(dir_name)
            task_role = self._get_task_role(dir_name)
            if task_role is None:
                task_role = role
            if rank_id < 0:
                logger.error(f"direct:{dir_name} fail to get rankid or rankid invalid.")
                continue
            # For RL Analysis
            rank_id_map[(task_role, rank_id)].append(dir_name)
        try:
            for map_key, dir_list in rank_id_map.items():
                dir_list.sort(key=self._extract_timestamp_key)
                data_map[map_key] = dir_list
        except Exception as e:
            raise RuntimeError("Found invalid directory name!") from e
        return data_map

    @staticmethod
    def _extract_timestamp_key(path_value: str) -> str:
        """Extract the timestamp-like segment from an Ascend profiler directory name.

        Ascend profiler directories follow the convention
        ``<date>_<time>_ascend_pt`` (e.g. ``20250101_120000_ascend_pt``).
        The sort key is the ``<date>_<time>`` portion so that directories
        are ordered chronologically.

        Args:
            path_value: Full path to the Ascend profiler directory.

        Returns:
            A string suitable for chronological sorting.
        """
        dir_name = Path(path_value).name
        parts = dir_name.split("_")
        if len(parts) >= 4:
            return "_".join(parts[-4:-2])
        if len(parts) >= 3:
            return parts[-3]
        return dir_name

    def _get_rank_id(self, dir_name: str) -> int:
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name.startswith(
                Constant.ASCEND_PROFILER_INFO_HEAD
            ) and file_name.endswith(Constant.JSON_EXTENSION):
                rank_id_str = file_name[
                    len(Constant.ASCEND_PROFILER_INFO_HEAD) : -1
                    * len(Constant.JSON_EXTENSION)
                ]
                try:
                    rank_id = int(rank_id_str)
                except ValueError:
                    rank_id = -1
                return rank_id
        return -1

    def _get_task_role(self, dir_name: str):
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name == Constant.ASCEND_PROFILER_METADATA_JSON:
                with open(os.path.join(dir_name, file_name), encoding="utf-8") as f:
                    config = json.load(f)
                task_role = config.get("role")
                if task_role:
                    return task_role
        return None
