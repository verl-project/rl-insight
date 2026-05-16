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

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image, ImageDraw

from rl_insight.visualizer.visualizer import BaseVisualizer, register_cluster_visualizer
from rl_insight.data import DataEnum


@register_cluster_visualizer("gmm_heatmap")
class GmmVisualizer(BaseVisualizer):
    input_type = DataEnum.GMM_SUMMARY

    @staticmethod
    def _resolve_output_path(output_cfg) -> Path:
        """
        Resolve output path robustly.
        - Existing directory -> append default file name.
        - Path without suffix (e.g., 'output/gmm') -> treat as directory.
        - Path with suffix (e.g., 'a/b/c.png') -> treat as explicit file path.
        """
        output = Path(output_cfg)
        if output.is_dir() or output.suffix == "":
            output = output / "gmm_heatmap.png"
        return output

    @staticmethod
    def _load_signature(stage_data: pd.DataFrame) -> np.ndarray:
        """Build deterministic load signature vector for one stage."""
        return stage_data.sort_values("expert_index")["load"].to_numpy(dtype=np.float64)

    def run(self, data):
        """Run GMM heatmap visualization from parsed data."""
        output_cfg = self.config.get(
            "output_path", "./output/gmm_group_list_heatmap.png"
        )
        output = self._resolve_output_path(output_cfg)
        gmm_per_layer = int(self.config.get("gmm_per_layer", 3))

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("No GMM data provided")

        logger.info(f"GmmVisualizer received DataFrame with {len(data)} rows")
        logger.info(f"DataFrame columns: {list(data.columns)}")
        logger.info("Visualizer consumes parser-filtered GMM summary data.")

        # For actor_update, filter out backward/recompute data by detecting
        # consecutive identical expert loads.
        #
        # In MoE models, each layer has 3 GMM calls (gate_proj, up_proj, down_proj)
        # that share the same expert routing, so their group_list values are identical.
        # Detection: when a run of consecutive identical loads exceeds 3,
        # we've entered the backward phase. Truncate from that point.
        # This works regardless of whether gradient recomputation is enabled:
        #   - With recomputation: forward runs of 3, then a run >3 triggers cutoff
        #   - Without recomputation: forward runs of 3, then a run of 3+3=6 triggers cutoff
        if "actor_update" in data["role"].unique():
            grouped = data.groupby(["step", "role", "rank_id"])
            filtered_data = []
            for (step_val, role_val, rank_val), group in grouped:
                if role_val != "actor_update":
                    filtered_data.append(group)
                    continue

                sorted_group = group.sort_values("stage")
                unique_stages = sorted(sorted_group["stage"].unique())
                stage_loads = {}
                for stage in unique_stages:
                    stage_data = sorted_group[sorted_group["stage"] == stage]
                    stage_loads[stage] = self._load_signature(stage_data)

                forward_stages = []
                prev_load = None
                consecutive = 0
                backward_detected = False
                for stage in unique_stages:
                    if backward_detected:
                        break
                    load = stage_loads[stage]
                    if prev_load is not None and np.array_equal(load, prev_load):
                        consecutive += 1
                    else:
                        prev_load = load
                        consecutive = 1

                    if consecutive <= gmm_per_layer:
                        forward_stages.append(stage)
                    else:
                        backward_detected = True

                filtered_group = sorted_group[
                    sorted_group["stage"].isin(forward_stages)
                ]
                filtered_data.append(filtered_group)
                logger.info(
                    f"For actor_update (step={step_val}, rank={rank_val}): "
                    f"kept {len(forward_stages)} forward stages out of {len(unique_stages)} total "
                    f"(backward detected={backward_detected}, gmm_per_layer={gmm_per_layer})"
                )

            if not filtered_data:
                raise ValueError("No data left after filtering")
            data = pd.concat(filtered_data)
            logger.info(
                f"After filtering actor_update forward-only data, now {len(data)} rows"
            )

        mat, rec_list, boundaries = self._build_matrix_from_data(data)
        logger.info(f"Built matrix with shape {mat.shape}")
        segments = self._segment_labels(rec_list, boundaries)
        self._plot_heatmap(mat, segments, output)
        return str(output)

    def _build_matrix_from_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, List[dict], List[int]]:
        """Build a matrix from the parsed data."""
        # Group data by step, role, rank_id, stage.
        # First sort the data to ensure consistent ordering.
        sorted_data = data.sort_values(["step", "role", "rank_id", "stage"])
        grouped = sorted_data.groupby(["step", "role", "rank_id", "stage"])

        # Get unique steps, roles, ranks and stages.
        steps = sorted(data["step"].unique())
        roles = sorted(data["role"].unique())
        ranks = sorted(data["rank_id"].unique())
        stages = sorted(data["stage"].unique())
        max_expert = int(data["expert_index"].max())
        logger.info(f"Steps: {steps}")
        logger.info(f"Roles: {roles}")
        logger.info(f"Ranks: {ranks}")
        logger.info(f"Stages: {stages}")
        logger.info(f"Max expert index: {max_expert}")

        vecs = []
        rec_list = []

        # Track layer mapping per (step, role, rank) group.
        current_group = None
        seen_vectors: dict[tuple[Any, ...], int] = {}
        layer_counter = 0

        for name, group in grouped:
            step, role, rank, stage_idx = name
            logger.info(
                f"Processing step: {step}, role: {role}, rank: {rank}, stage: {stage_idx}"
            )
            # Check if we're in a new (step, role, rank) group.
            new_group = (step, role, rank)
            if new_group != current_group:
                # Reset layer counter and seen vectors for the new group.
                current_group = new_group
                seen_vectors.clear()
                layer_counter = 0
                logger.info(
                    f"New group detected: {new_group}, resetting layer counter to 0"
                )

            # Create a vector for this step, role, rank and stage.
            vec = np.full(max_expert + 1, np.nan, dtype=np.float64)
            for _, row in group.iterrows():
                vec[int(row["expert_index"])] = row["load"]

            # Convert vector to tuple for hashing, replacing NaN to keep comparisons stable.
            vec_tuple = tuple(v if not np.isnan(v) else -1 for v in vec)
            if vec_tuple not in seen_vectors:
                # New layer.
                seen_vectors[vec_tuple] = layer_counter
                layer_idx = layer_counter
                layer_counter += 1
            else:
                # Duplicate layer.
                layer_idx = seen_vectors[vec_tuple]

            vecs.append(vec)
            rec_list.append(
                {
                    "step": step,
                    "role": role,
                    "rank_id": rank,
                    "stage": stage_idx,
                    "op_index": stage_idx,  # Original op index
                    "layer_idx": layer_idx,  # Mapped layer index.
                }
            )

        if not vecs:
            raise ValueError("No data available to build matrix")

        mat = np.stack(vecs, axis=1)  # [n_experts, n_time]
        logger.info(f"Matrix shape: {mat.shape}")

        # Boundaries: split when training step, RL role, or rank changes.
        # Each rec_list column is one (step, role, rank_id, stage) snapshot;
        # grouping by step/role/rank for segments, while keeping stage for individual columns.
        boundaries = [0]
        if rec_list:
            cur_key = (
                rec_list[0]["step"],
                rec_list[0]["role"],
                rec_list[0]["rank_id"],
            )
            for j, rec in enumerate(rec_list[1:], start=1):
                new_key = (rec["step"], rec["role"], rec["rank_id"])
                if new_key != cur_key:
                    boundaries.append(j)
                    cur_key = new_key
        boundaries.append(mat.shape[1])
        logger.info(f"Boundaries (step/role/rank): {boundaries}")
        return mat, rec_list, boundaries

    def _segment_labels(
        self, rec_list: List[dict], boundaries: List[int]
    ) -> List[Tuple[int, int, int, str, int]]:
        """Generate segment labels: (x0, x1, step, role, rank_id)."""
        segments = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if a >= b:
                continue
            rec = rec_list[a]
            segments.append((a, b, rec["step"], rec["role"], rec["rank_id"]))
        logger.info(f"Segments: {segments}")
        return segments

    def _plot_heatmap(
        self,
        mat: np.ndarray,
        segments: List[Tuple[int, int, int, str, int]],
        out_path: Path,
    ) -> None:
        """Plot the heatmap."""
        n_exp, n_time = mat.shape
        cell_w = 28
        cell_h = 28
        left_bar_w = 120
        pad = 24

        finite_vals = mat[np.isfinite(mat)]
        vmin = float(finite_vals.min()) if finite_vals.size else 0.0
        vmax = float(finite_vals.max()) if finite_vals.size else 1.0
        scale = vmax - vmin if vmax > vmin else 1.0

        img_w = pad * 2 + left_bar_w + n_exp * cell_w
        img_h = pad * 2 + n_time * cell_h
        image = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(image)

        # Segment bar: one color per (step, role, rank), shown on left side.
        segment_colors = [
            self._viridis_rgb(i / max(1, len(segments) - 1))
            for i in range(len(segments))
        ]

        for idx, (a, b, _step, _role, _rank) in enumerate(segments):
            y0 = pad + a * cell_h
            y1 = pad + b * cell_h
            draw.rectangle(
                [pad, y0, pad + left_bar_w - 1, y1 - 1],
                fill=segment_colors[idx],
            )

        # Main heatmap is rendered as a stable bitmap to avoid backend-specific crashes.
        for t in range(n_time):
            for e in range(n_exp):
                value = mat[e, t]
                if np.isnan(value):
                    color = (235, 235, 235)
                else:
                    color = self._viridis_rgb((float(value) - vmin) / scale)
                x0 = pad + left_bar_w + e * cell_w
                y0 = pad + t * cell_h
                draw.rectangle(
                    [x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                    fill=color,
                    outline=(255, 255, 255),
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

    @staticmethod
    def _viridis_rgb(x: float) -> tuple[int, int, int]:
        anchors = [
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ]
        x = min(1.0, max(0.0, x))
        pos = x * (len(anchors) - 1)
        left = int(pos)
        right = min(left + 1, len(anchors) - 1)
        frac = pos - left
        c0, c1 = anchors[left], anchors[right]
        return tuple(int(c0[i] + (c1[i] - c0[i]) * frac) for i in range(3))
