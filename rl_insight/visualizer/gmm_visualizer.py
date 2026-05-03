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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

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
        is_dir_semantics = output.is_dir() or output.suffix == ""
        if is_dir_semantics:
            output = output / "gmm_heatmap.png"
        return output

    @staticmethod
    def _load_signature(stage_data: pd.DataFrame) -> np.ndarray:
        """Build deterministic load signature vector for one stage."""
        return stage_data.sort_values("expert_index")["load"].to_numpy(dtype=np.float64)

    def run(self, data):
        """Run GMM heatmap visualization from parsed data."""
        # Extract parameters from config
        output_cfg = self.config.get(
            "output_path", "./output/gmm_group_list_heatmap.png"
        )
        output = self._resolve_output_path(output_cfg)
        dpi = self.config.get("dpi", 150)
        cmap = self.config.get("cmap", "viridis")
        gmm_per_layer = int(self.config.get("gmm_per_layer", 3))

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(data).__name__}")

        logger.info(f"GmmVisualizer received DataFrame with {len(data)} rows")
        logger.info(f"DataFrame columns: {list(data.columns)}")

        if data.empty:
            raise ValueError("No GMM data provided")
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
        is_actor_update = "actor_update" in data["role"].unique()
        if is_actor_update:
            grouped = data.groupby(["step", "role", "rank_id"])
            filtered_data = []
            for name, group in grouped:
                step_val, role_val, rank_val = name
                if role_val == "actor_update":
                    sorted_group = group.sort_values("stage")
                    unique_stages = sorted(sorted_group["stage"].unique())

                    # Build load signature for each stage
                    stage_loads = {}
                    for stage in unique_stages:
                        stage_data = sorted_group[sorted_group["stage"] == stage]
                        load_sig = self._load_signature(stage_data)
                        stage_loads[stage] = load_sig

                    # Scan forward: keep stages until a run exceeds gmm_per_layer.
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
                else:
                    filtered_data.append(group)

            if filtered_data:
                data = pd.concat(filtered_data)
                logger.info(
                    f"After filtering actor_update forward-only data, now {len(data)} rows"
                )
            else:
                logger.warning("No data left after filtering")
                raise ValueError("No data left after filtering")

        # Build matrix
        mat, rec_list, boundaries = self._build_matrix_from_data(data)
        logger.info(f"Built matrix with shape {mat.shape}")

        segments = self._segment_labels(rec_list, boundaries)

        # Generate title
        unique_ranks = sorted(data["rank_id"].unique())
        if len(unique_ranks) == 1:
            rank_str = f" rank={unique_ranks[0]}"
        else:
            rank_str = f" ranks={len(unique_ranks)}"
        title = f"GMM expert load (group_list){rank_str} — {len(rec_list)} snapshots, {mat.shape[0]} experts"

        # Plot heatmap
        self._plot_heatmap(mat, rec_list, segments, title, output, dpi, cmap)

        return str(output)

    def _build_matrix_from_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, List[dict], List[int]]:
        """Build a matrix from the parsed data."""
        # Group data by step, role, rank_id, stage
        # First sort the data to ensure consistent ordering
        sorted_data = data.sort_values(["step", "role", "rank_id", "stage"])
        grouped = sorted_data.groupby(["step", "role", "rank_id", "stage"])

        # Get unique steps, roles, ranks and stages
        steps = sorted(data["step"].unique())
        roles = sorted(data["role"].unique())
        ranks = sorted(data["rank_id"].unique())
        stages = sorted(data["stage"].unique())
        max_expert = data["expert_index"].max()

        logger.info(f"Steps: {steps}")
        logger.info(f"Roles: {roles}")
        logger.info(f"Ranks: {ranks}")
        logger.info(f"Stages: {stages}")
        logger.info(f"Max expert index: {max_expert}")

        # Build matrix and detect duplicate stages
        vecs = []
        rec_list = []

        # Track layer mapping per (step, role, rank) group
        current_group = None
        seen_vectors: dict[tuple[Any, ...], int] = {}
        layer_counter = 0

        for name, group in grouped:
            step, role, rank, stage_idx = name
            logger.info(
                f"Processing step: {step}, role: {role}, rank: {rank}, stage: {stage_idx}"
            )

            # Check if we're in a new (step, role, rank) group
            new_group = (step, role, rank)
            if new_group != current_group:
                # Reset layer counter and seen vectors for new group
                current_group = new_group
                seen_vectors.clear()
                layer_counter = 0
                logger.info(
                    f"New group detected: {new_group}, resetting layer counter to 0"
                )

            # Create a vector for this step, role, rank and stage
            vec = np.full(max_expert + 1, np.nan, dtype=np.float64)
            for _, row in group.iterrows():
                expert_idx = row["expert_index"]
                vec[expert_idx] = row["load"]

            # Convert vector to tuple for hashing (handle NaN values)
            vec_tuple = tuple(v if not np.isnan(v) else -1 for v in vec)

            # Check if this vector has been seen before in current group
            if vec_tuple not in seen_vectors:
                # New layer
                seen_vectors[vec_tuple] = layer_counter
                layer_idx = layer_counter
                layer_counter += 1
            else:
                # Duplicate layer
                layer_idx = seen_vectors[vec_tuple]

            vecs.append(vec)
            rec_list.append(
                {
                    "step": step,
                    "role": role,
                    "rank_id": rank,
                    "stage": stage_idx,
                    "op_index": stage_idx,  # Original op index
                    "layer_idx": layer_idx,  # Mapped layer index
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
        rec_list: List[dict],
        segments: List[Tuple[int, int, int, str, int]],
        title: str,
        out_path: Path,
        dpi: int,
        cmap: str,
    ) -> None:
        """Plot the heatmap."""
        n_exp, n_time = mat.shape
        # Keep figure size readable when segment/time dimension is large.
        # Use sub-linear growth for height to avoid overly tall and narrow figures.
        fig_w = min(32, max(10, n_exp * 0.18))
        fig_h = min(22, max(8, 6 + np.sqrt(max(n_time, 1)) * 0.9))
        fig = plt.figure(figsize=(fig_w + 2.8, fig_h))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.16, 1], wspace=0.05)
        ax_bar = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])

        # Main heatmap is transposed to put experts on X axis.
        # mat: [n_experts, n_time] -> heatmap_data: [n_time, n_experts]
        heatmap_data = mat.T
        ax_bar.set_ylim(-0.5, n_time - 0.5)
        ax.set_ylim(-0.5, n_time - 0.5)
        ax.set_xlim(-0.5, n_exp - 0.5)

        # Segment bar: one color per (step, role, rank), shown on left side.
        # Use viridis colormap for consistency with heatmap
        palette = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        for i, (a, b, step, role, rank_id) in enumerate(segments):
            color = palette[i]
            ax_bar.axhspan(
                a - 0.5, b - 0.5, facecolor=color, alpha=0.55, edgecolor="none"
            )

        # Add separator lines between segments
        for a, b, step, role, rank_id in segments:
            if a > 0:
                ax_bar.axhline(a - 0.5, color="white", linewidth=0.8, alpha=0.7)
        # Add last separator line at the end
        if n_time > 0:
            ax_bar.axhline(n_time - 0.5, color="white", linewidth=0.8, alpha=0.7)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        ax_bar.set_title(
            "Row: layerK (K = merged layer index)\nstep · role · rank",
            fontsize=10,
            pad=8,
        )
        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xlabel("Expert index")
        ax.set_title(title)

        # Horizontal lines at every segment boundary (includes step / role / rank changes)
        for a, b, step, role, rank_id in segments:
            ax.axhline(a - 0.5, color="white", linewidth=0.8, alpha=0.7)
        ax.axhline(n_time - 0.5, color="white", linewidth=0.8, alpha=0.7)

        # Y axis: mark each layer only once
        layer_positions = []
        layer_labels = []
        if n_time > 0:
            current_layer = rec_list[0]["layer_idx"]
            layer_positions.append(0)
            layer_labels.append(f"layer{current_layer}")

            for j in range(1, n_time):
                if rec_list[j]["layer_idx"] != current_layer:
                    current_layer = rec_list[j]["layer_idx"]
                    layer_positions.append(j)
                    layer_labels.append(f"layer{current_layer}")

        # Add the last position if needed
        if n_time > 0 and layer_positions[-1] != n_time - 1:
            layer_positions.append(n_time - 1)
            layer_labels.append(f"layer{rec_list[-1]['layer_idx']}")

        # Downsample layer ticks when snapshots are too many.
        max_layer_labels = 40
        if len(layer_positions) > max_layer_labels:
            sel_idx = np.linspace(
                0, len(layer_positions) - 1, max_layer_labels, dtype=int
            )
            layer_positions = [layer_positions[i] for i in sel_idx]
            layer_labels = [layer_labels[i] for i in sel_idx]

        ax.set_yticks(layer_positions)
        ax.set_yticklabels(layer_labels, fontsize=6)
        ax.set_ylabel("")

        x_stride = max(1, n_exp // 40)
        ax.set_xticks(list(range(0, n_exp, x_stride)))

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Tokens per expert (group_list)")

        def _seg_legend_label(s: Tuple[int, int, int, str, int]) -> str:
            _, _, st, rl, rk = s
            rshort = (rl[:14] + "…") if len(str(rl)) > 14 else str(rl)
            return f"st{st} · {rshort} · r{rk}"

        # Render step/role/rank directly inside segment blocks (centered).
        if segments:
            for i, (a, b, step, role, rank_id) in enumerate(segments):
                label = _seg_legend_label((a, b, step, role, rank_id))
                seg_h = max(1.0, b - a)
                # Adaptive label size based on segment height.
                font_size = min(11.5, max(5.5, 4.8 + 0.45 * seg_h))
                ax_bar.text(
                    0.5,
                    a + (b - a - 1) / 2,
                    label,
                    fontsize=font_size,
                    va="center",
                    ha="center",
                    rotation=0,
                    color="black",
                    clip_on=True,
                )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
