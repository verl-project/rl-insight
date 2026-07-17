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

import json
import os
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from recipe.config.utils import get_config_value
from recipe.data import DataEnum

from .visualizer import (
    BaseVisualizer,
    register_cluster_visualizer,
)


@register_cluster_visualizer("memory_html")
class MemoryVisualizer(BaseVisualizer):
    """HTML memory allocation timeline; interactive Gantt chart of memory events.

    Displays memory allocations over time for each (role, rank), color-coded
    by operator name.  Includes hover details for memory size and cumulative
    memory statistics (allocated / reserved / active).
    """

    input_type: DataEnum = DataEnum.MEMORY_SUMMARY

    # ── Rendering constants ────────────────────────────────────────────
    _MAX_TIMELINE_POINTS = 2000  # max points in Chart1 memory line
    _HOVER_TOP_N = 10  # Chart1 hover shows top-N by size
    _KB_TO_MB = 1.0 / 1024.0  # KB → MB conversion factor

    def __init__(self, config: Union[DictConfig, dict]):
        super().__init__(config)
        self.output_path = get_config_value(config, "output.path", None)

    def run(self, data):
        return self.generate_memory_timeline(data)

    def generate_memory_timeline(self, data):
        """Generate an interactive memory timeline HTML visualization.

        Creates a self-contained HTML file with two synchronized charts:
        1. Memory usage timeline — total allocated memory over time.
        2. Operator Gantt chart — horizontal bars showing each operator's
           memory allocation duration.

        Includes interactive controls for time-range selection and operator
        count filtering.

        Each (role, rank_id) group produces its own set of HTML files.

        Args:
            data: Preprocessed DataFrame with columns: name, size_kb,
                  start_time_ms, duration_ms, total_allocated_mb,
                  total_reserved_mb, total_active_mb, device_type,
                  call_stack_top, role, rank_id.  Only positive allocations
                  (size_kb > 0).
        """
        logger.info(
            f"Starting memory timeline generation: "
            f"{len(data) if data is not None else 0} input records"
        )

        if data is None or data.empty:
            logger.info("No memory allocations found — nothing to visualize.")
            return None

        # Filter to only positive allocations (skip releases)
        data = data[data["size_kb"] > 0]

        if data.empty:
            logger.info("No positive memory allocations found — nothing to visualize.")
            return None

        logger.info(f"Filtered to {len(data)} positive allocation events")

        # Group by (role, rank_id) and generate one HTML per rank
        if "role" in data.columns and "rank_id" in data.columns:
            data = data.copy()
            data["role"] = data["role"].fillna("unknown")
            data["rank_id"] = data["rank_id"].fillna(-1).astype(int)
            groups = data.groupby(["role", "rank_id"])
        else:
            groups = [(None, data)]

        first_output = None
        for group_key, group_data in groups:
            if isinstance(group_key, tuple):
                role, rank_id = group_key
                logger.info(
                    f"Generating memory timeline for role={role}, rank_id={rank_id} "
                    f"({len(group_data)} events)"
                )
            else:
                role, rank_id = None, None
                logger.info(f"Generating memory timeline ({len(group_data)} events)")

            result = self._generate_single_timeline(group_data, role, rank_id)
            if first_output is None and result is not None:
                first_output = result

        return first_output

    def _generate_single_timeline(self, data: pd.DataFrame, role, rank_id):
        """Generate memory timeline HTML for a single (role, rank_id) group."""
        data = data.sort_values("start_time_ms").copy()

        t_max_abs = float((data["start_time_ms"] + data["duration_ms"]).max())
        data["duration_ms"] = np.where(
            data["duration_ms"] == 0,
            t_max_abs - data["start_time_ms"],
            data["duration_ms"],
        )
        data["end_time_ms"] = data["start_time_ms"] + data["duration_ms"]
        data["size_mb"] = data["size_kb"] * self._KB_TO_MB

        if role is not None and rank_id is not None:
            rank_prefix = f"{role}_rank{rank_id}"
        else:
            rank_prefix = None

        # ── Global time range ─────────────────────────────────────────
        t_min_abs = float(data["start_time_ms"].min())
        t_max_abs = float(data["end_time_ms"].max())
        logger.info(
            f"Time range: {t_min_abs:.0f} – {t_max_abs:.0f} ms "
            f"(duration: {(t_max_abs - t_min_abs) / 1000:.2f} s)"
        )

        # ── Chart 1 data: Memory usage timeline ───────────────────────
        # Vectorized: build start(+size) and end(-size) event columns
        rel_starts = (data["start_time_ms"] - t_min_abs).round(2)
        rel_ends = (data["end_time_ms"] - t_min_abs).round(2)
        sizes = data["size_kb"].astype(float)

        times = np.concatenate([rel_starts, rel_ends])
        deltas = np.concatenate([sizes, -sizes])

        events_df = pd.DataFrame({"time": times, "delta_kb": deltas})
        events_df = events_df.groupby("time", as_index=False)["delta_kb"].sum()
        events_df = events_df.sort_values("time")
        events_df["total_mb"] = events_df["delta_kb"].cumsum() * self._KB_TO_MB

        # Downsample memory timeline if too many points
        orig_timeline_points = len(events_df)
        if len(events_df) > self._MAX_TIMELINE_POINTS:
            events_df = events_df.iloc[
                np.linspace(0, len(events_df) - 1, self._MAX_TIMELINE_POINTS, dtype=int)
            ]
            logger.info(
                f"Timeline downsampled: {orig_timeline_points} → "
                f"{len(events_df)} points"
            )

        memory_timeline = [
            {"time": float(row["time"]), "total_mb": round(float(row["total_mb"]), 4)}
            for _, row in events_df.iterrows()
        ]

        # ── Chart 2 data: Operator Gantt ──────────────────────────────
        # Split parallel arrays: compact JSON, no nested keys
        gantt_name_ids = []  # indices into op_names
        gantt_starts = []
        gantt_durations = []
        gantt_sizes = []
        total_alloc_arr = []
        call_stack_pool: list[str] = []
        call_stack_pool_map = {}
        call_stack_idx_arr = []
        op_names: list[str] = []
        name_to_id = {}

        # Pre-extract columns to avoid iterrows() per-row Series overhead.
        # Numeric fields: vectorized round() on full columns before .tolist().
        _start_col = (data["start_time_ms"] - t_min_abs).round(2).tolist()
        _dur_col = data["duration_ms"].round(2).tolist()
        _size_col = data["size_kb"].round(2).tolist()
        _alloc_col = data["total_allocated_mb"].round(2).tolist()
        _name_col = data["name"].tolist()
        _cs_col = data.get("call_stack", pd.Series([""] * len(data))).tolist()

        for i in range(len(data)):
            op_name = _name_col[i]
            if op_name not in name_to_id:
                name_to_id[op_name] = len(op_names)
                op_names.append(op_name)
            gantt_name_ids.append(name_to_id[op_name])
            gantt_starts.append(_start_col[i])
            gantt_durations.append(_dur_col[i])
            gantt_sizes.append(_size_col[i])
            total_alloc_arr.append(_alloc_col[i])
            cs = _cs_col[i]
            if not cs or (isinstance(cs, float) and pd.isna(cs)):
                call_stack_idx_arr.append(-1)
            else:
                cs = str(cs)
                if cs not in call_stack_pool_map:
                    call_stack_pool_map[cs] = len(call_stack_pool)
                    call_stack_pool.append(cs)
                call_stack_idx_arr.append(call_stack_pool_map[cs])

        total_bar_count = len(gantt_name_ids)
        logger.info(
            f"Built {total_bar_count} bar entries across "
            f"{len(op_names)} unique operators"
        )

        # ── Build Chart1 data (full timeline) ─────────────────────────
        tl_xy, tl_active = self._build_chart1_data(
            memory_timeline,
            gantt_name_ids,
            gantt_starts,
            gantt_durations,
            gantt_sizes,
            op_names,
        )

        # Color map for operators
        color_palette = [
            "#4e79a7",
            "#f28e8b",
            "#59a14f",
            "#b07aa1",
            "#9c755f",
            "#76b7b2",
            "#edc948",
            "#bab0ab",
            "#8cd17d",
            "#ff9da7",
            "#e15759",
            "#86bcb6",
            "#b6992d",
            "#d37295",
            "#a0cbe8",
            "#ffbe7d",
            "#b07aa1",
            "#d4a6c8",
            "#8c564b",
            "#c49c94",
        ]
        op_color_map = {}
        for i, op_name in enumerate(op_names):
            op_color_map[op_name] = color_palette[i % len(color_palette)]

        output_dir = self.output_path or "."
        if output_dir.endswith(".html"):
            output_dir = os.path.dirname(output_dir) or "."
        os.makedirs(output_dir, exist_ok=True)

        # Build a single HTML + detail_data.js containing all events.
        # (Previously split into up to 20 time segments for performance;
        #  rendering is now fast enough to keep everything in one file.)
        html, detail_js = self._build_memory_html(
            t_offset=t_min_abs,
            tl_xy=tl_xy,
            tl_active=tl_active,
            gantt_name_ids=gantt_name_ids,
            gantt_starts=gantt_starts,
            gantt_durations=gantt_durations,
            gantt_sizes=gantt_sizes,
            total_alloc_arr=total_alloc_arr,
            call_stack_pool=call_stack_pool,
            call_stack_idx_arr=call_stack_idx_arr,
            op_names=op_names,
            op_color_map=op_color_map,
            total_bar_count=total_bar_count,
            rank_prefix=rank_prefix,
        )

        if rank_prefix:
            data_path = os.path.join(output_dir, f"detail_data_{rank_prefix}.js")
            html_path = os.path.join(output_dir, f"memory_timeline_{rank_prefix}.html")
        else:
            data_path = os.path.join(output_dir, "detail_data.js")
            html_path = os.path.join(output_dir, "memory_timeline.html")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(detail_js)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(
            f"  {os.path.basename(html_path)} "
            f"({len(html) / 1024:.0f} KB HTML + "
            f"{len(detail_js) / 1024:.0f} KB data) "
            f"— {total_bar_count} events"
        )

        # Summary
        logger.info(
            f"Memory timeline generation complete: "
            f"{total_bar_count} events, "
            f"{len(op_names)} operators → {output_dir}"
        )
        if rank_prefix:
            return os.path.join(output_dir, f"memory_timeline_{rank_prefix}.html")
        return os.path.join(output_dir, "memory_timeline.html")

    @staticmethod
    def _build_chart1_data(
        memory_timeline,
        gantt_name_ids,
        gantt_starts,
        gantt_durations,
        gantt_sizes,
        op_names,
    ):
        """Build Chart1 (memory timeline) data from all bars.

        Returns (tl_xy, tl_active) — shared across all segments.
        """
        all_intervals = []
        for i in range(len(gantt_name_ids)):
            all_intervals.append(
                (
                    gantt_starts[i],
                    gantt_starts[i] + gantt_durations[i],
                    op_names[gantt_name_ids[i]],
                    gantt_sizes[i],
                )
            )
        all_intervals.sort(key=lambda x: x[0])

        tl_xy = []
        tl_active = []
        if memory_timeline and all_intervals:
            interval_idx = 0
            n_intervals = len(all_intervals)

            # Min-heap keyed by (end_time, start_time) for efficient expiry.
            # Elements: (end_time, start_time, op_name, size)
            import heapq

            active_heap: list[tuple] = []
            active_count = 0

            for point in memory_timeline:
                t = point["time"]

                # Add intervals that start at or before t
                while (
                    interval_idx < n_intervals and all_intervals[interval_idx][0] <= t
                ):
                    interval = all_intervals[interval_idx]
                    # Heap key is (end_time, start_time) so earliest-ending
                    # intervals surface first.
                    heapq.heappush(
                        active_heap,
                        (interval[1], interval[0], interval[2], interval[3]),
                    )
                    active_count += 1
                    interval_idx += 1

                # Remove intervals that have ended (end_time <= t)
                while active_heap and active_heap[0][0] <= t:
                    heapq.heappop(active_heap)
                    active_count -= 1

                tl_xy.append([round(t, 2), round(point["total_mb"], 2)])
                if active_heap:
                    # heapq.nlargest avoids a full sort when N << K.
                    # Key on size (index 3 of the heap tuple).
                    top_n = heapq.nlargest(
                        MemoryVisualizer._HOVER_TOP_N, active_heap, key=lambda x: x[3]
                    )
                    tl_active.append(
                        [
                            active_count,
                            [[op_name, round(sz, 1)] for _, _, op_name, sz in top_n],
                        ]
                    )
                else:
                    # always append，to keep tl_active align with tl_xy
                    tl_active.append([0, []])
        return tl_xy, tl_active

    def _build_memory_html(
        self,
        t_offset,
        tl_xy,
        tl_active,
        gantt_name_ids,
        gantt_starts,
        gantt_durations,
        gantt_sizes,
        total_alloc_arr,
        call_stack_pool,
        call_stack_idx_arr,
        op_names,
        op_color_map,
        total_bar_count,
        rank_prefix=None,
    ):
        """Build a single HTML + detail_data.js containing all events.

        Chart1 (tl_xy, tl_active) is the full memory timeline.
        Chart2 arrays contain every allocation event.
        """
        compact_opts = {"separators": (",", ":"), "ensure_ascii": False}
        to_json = json.dumps

        # All data goes into detail_data.js — HTML is a pure template (~3 KB)
        detail_lines = [
            "var TL_XY = " + to_json(tl_xy, **compact_opts) + ";",
            "var GANTT_IDS = " + to_json(gantt_name_ids, **compact_opts) + ";",
            "var GANTT_STARTS = " + to_json(gantt_starts, **compact_opts) + ";",
            "var GANTT_DURS = " + to_json(gantt_durations, **compact_opts) + ";",
            "var OP_NAMES = " + to_json(op_names, **compact_opts) + ";",
            "var T_OFFSET = " + to_json(t_offset) + ";",
            "var TOTAL_OP_COUNT = " + str(total_bar_count) + ";",
            "var COLOR_MAP = " + to_json(op_color_map, **compact_opts) + ";",
            "var GANTT_SIZES = " + to_json(gantt_sizes, **compact_opts) + ";",
            "var TOTAL_ALLOC = " + to_json(total_alloc_arr, **compact_opts) + ";",
            "var TL_ACTIVE = " + to_json(tl_active, **compact_opts) + ";",
            "var CS_POOL = " + to_json(call_stack_pool, **compact_opts) + ";",
            "var CS_IDX = " + to_json(call_stack_idx_arr, **compact_opts) + ";",
        ]
        detail_js = "\n".join(detail_lines)

        # Read HTML template and inject the data file reference
        template_path = os.path.join(os.path.dirname(__file__), "memory_template.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()

        data_filename = (
            f"detail_data_{rank_prefix}.js" if rank_prefix else "detail_data.js"
        )
        html = html.replace("__DATA_FILE__", data_filename)

        return html, detail_js
