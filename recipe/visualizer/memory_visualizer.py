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
    _MAX_SEGMENTS = 20  # upper bound on segment file count
    _TARGET_BARS_PER_SEGMENT = 5000  # target bar count before splitting
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

        Args:
            data: Preprocessed DataFrame with columns: name, size_kb,
                  start_time_ms, duration_ms, total_allocated_mb,
                  total_reserved_mb, total_active_mb, device_type,
                  call_stack_top.  Only positive allocations (size_kb > 0).
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

        # Compute end times and size in MB
        data["end_time_ms"] = data["start_time_ms"] + data["duration_ms"]
        data["size_mb"] = data["size_kb"] * self._KB_TO_MB

        # ── Global time range ─────────────────────────────────────────
        # data is pre-sorted by start_time_ms
        t_min_abs = float(data["start_time_ms"].iloc[0])
        t_max_abs = float((data["start_time_ms"] + data["duration_ms"]).max())
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
        call_stack_pool = []
        call_stack_pool_map = {}
        call_stack_idx_arr = []
        op_names = []
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

        # ── Split into time segments ──────────────────────────────────
        num_segments = max(
            1,
            min(
                self._MAX_SEGMENTS,
                int(np.ceil(total_bar_count / self._TARGET_BARS_PER_SEGMENT)),
            ),
        )
        logger.info(
            f"Splitting into {num_segments} time segment(s) (max {self._MAX_SEGMENTS})"
        )

        t_rel_min = gantt_starts[0]
        t_rel_max = max(s + d for s, d in zip(gantt_starts, gantt_durations))
        seg_width = (t_rel_max - t_rel_min) / num_segments

        # Build Chart1 data once (full timeline, shared across segments)
        tl_xy, tl_active = self._build_chart1_data(
            memory_timeline,
            gantt_name_ids,
            gantt_starts,
            gantt_durations,
            gantt_sizes,
            op_names,
        )

        # Color map (shared across segments)
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

        segments = []
        # Pre-compute all segment boundaries for the segment-map feature
        all_segments_info = []
        for seg_idx in range(num_segments):
            seg_start = t_rel_min + seg_idx * seg_width
            seg_end = t_rel_min + (seg_idx + 1) * seg_width
            if seg_idx == num_segments - 1:
                seg_end = t_rel_max + 1  # include all remaining

            seg_label = f"{seg_start + t_min_abs:.0f} – {seg_end + t_min_abs:.0f} ms"
            all_segments_info.append(
                (seg_idx, round(seg_start, 2), round(seg_end, 2), seg_label)
            )

        # Segment map for JS: [idx, rel_start, rel_end] — shared by all segments
        seg_data = [[si, rs, re] for si, rs, re, _ in all_segments_info]

        for seg_idx in range(num_segments):
            seg_start = all_segments_info[seg_idx][1]
            seg_end = all_segments_info[seg_idx][2]
            seg_label = all_segments_info[seg_idx][3]

            # Filter bar indices that overlap with this time segment
            bar_indices = [
                i
                for i in range(len(gantt_starts))
                if gantt_starts[i] + gantt_durations[i] > seg_start
                and gantt_starts[i] < seg_end
            ]
            if not bar_indices:
                logger.info(f"  Segment {seg_idx + 1}/{num_segments} empty — skipped")
                continue  # skip empty segments

            seg_gantt_name_ids = [gantt_name_ids[i] for i in bar_indices]
            seg_gantt_starts = [gantt_starts[i] for i in bar_indices]
            seg_gantt_durations = [gantt_durations[i] for i in bar_indices]
            seg_gantt_sizes = [gantt_sizes[i] for i in bar_indices]
            seg_total_alloc = [total_alloc_arr[i] for i in bar_indices]
            seg_call_stack_idx_arr = [call_stack_idx_arr[i] for i in bar_indices]

            segments.append((seg_idx, seg_label, len(bar_indices)))

            # Build per-segment HTML + detail_data.js
            html, detail_js = self._build_memory_html(
                t_offset=t_min_abs,
                tl_xy=tl_xy,
                tl_active=tl_active,
                gantt_name_ids=seg_gantt_name_ids,
                gantt_starts=seg_gantt_starts,
                gantt_durations=seg_gantt_durations,
                gantt_sizes=seg_gantt_sizes,
                total_alloc_arr=seg_total_alloc,
                call_stack_pool=call_stack_pool,
                call_stack_idx_arr=seg_call_stack_idx_arr,
                op_names=op_names,
                op_color_map=op_color_map,
                total_bar_count=len(seg_gantt_name_ids),
                global_bar_count=total_bar_count,
                seg_idx=seg_idx,
                seg_label=seg_label,
                num_segments=num_segments,
                seg_data=seg_data,
                t_rel_max=t_rel_max,
            )

            data_path = os.path.join(output_dir, f"detail_data_{seg_idx:02d}.js")
            html_path = os.path.join(output_dir, f"memory_timeline_{seg_idx:02d}.html")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(detail_js)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info(
                f"  [{seg_idx + 1}/{num_segments}] "
                f"memory_timeline_{seg_idx:02d}.html "
                f"({len(html) / 1024:.0f} KB HTML + "
                f"{len(detail_js) / 1024:.0f} KB data) "
                f"— {len(bar_indices)} events, "
                f"{seg_label}"
            )

        # Summary
        logger.info(
            f"Memory timeline generation complete: "
            f"{len(segments)} segment(s), {total_bar_count} events, "
            f"{len(op_names)} operators → {output_dir}"
        )
        return os.path.join(output_dir, "memory_timeline_00.html")

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
            active_intervals = []

            for point in memory_timeline:
                t = point["time"]

                # Add intervals that start at or before t
                while (
                    interval_idx < n_intervals and all_intervals[interval_idx][0] <= t
                ):
                    active_intervals.append(all_intervals[interval_idx])
                    interval_idx += 1

                # Remove intervals that have ended
                active_intervals = [item for item in active_intervals if item[1] > t]

                tl_xy.append([round(t, 2), round(point["total_mb"], 2)])
                if active_intervals:
                    sorted_active = sorted(active_intervals, key=lambda x: -x[3])
                    top_n = sorted_active[: MemoryVisualizer._HOVER_TOP_N]
                    tl_active.append(
                        [
                            len(active_intervals),
                            [[op_name, round(sz, 1)] for _, _, op_name, sz in top_n],
                        ]
                    )
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
        global_bar_count,
        seg_idx,
        seg_label,
        num_segments,
        seg_data,
        t_rel_max,
    ):
        """Build HTML + detail_data.js for one time segment.

        Chart1 (tl_xy, tl_active) is global — same across all segments.
        Chart2 arrays are per-segment filtered.
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
            # Segment info for hint feature: [idx, rel_start, rel_end]
            "var SEGMENTS = " + to_json(seg_data, **compact_opts) + ";",
            "var T_REL_MAX = " + to_json(t_rel_max) + ";",
            "var SEG_INDEX = " + str(seg_idx) + ";",
        ]
        detail_js = "\n".join(detail_lines)

        # Read HTML template and inject segment navigation
        template_path = os.path.join(os.path.dirname(__file__), "memory_template.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Inject segment navigation and data file reference
        data_filename = f"detail_data_{seg_idx:02d}.js"
        nav_html = self._build_segment_nav(
            seg_idx, seg_label, num_segments, global_bar_count
        )
        html = html.replace("__SEGMENT_NAV__", nav_html)
        html = html.replace("__SEGMENT_LABEL__", seg_label)
        html = html.replace("__DATA_FILE__", data_filename)

        return html, detail_js

    @staticmethod
    def _build_segment_nav(seg_idx, seg_label, num_segments, global_bar_count):
        """Build segment navigation HTML snippet."""
        parts = [
            '<div class="control-group" style="border-left:2px solid #eee;'
            'padding-left:16px;gap:4px">'
        ]
        if seg_idx > 0:
            parts.append(
                f'<a href="memory_timeline_{seg_idx - 1:02d}.html" '
                f'style="text-decoration:none;color:#4e79a7;font-size:13px;'
                f'padding:4px 8px;border:1px solid #4e79a7;border-radius:4px"'
                f">← Prev</a>"
            )
        parts.append(
            f'<span style="font-size:12px;color:#888;margin:0 6px">'
            f"Seg {seg_idx + 1}/{num_segments}: {seg_label}</span>"
        )
        if seg_idx < num_segments - 1:
            parts.append(
                f'<a href="memory_timeline_{seg_idx + 1:02d}.html" '
                f'style="text-decoration:none;color:#4e79a7;font-size:13px;'
                f'padding:4px 8px;border:1px solid #4e79a7;border-radius:4px"'
                f">Next →</a>"
            )
        parts.append(f'<span class="unit">({global_bar_count} total events)</span>')
        parts.append("</div>")
        return "".join(parts)
