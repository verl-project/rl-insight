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

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger

from rl_insight.visualizer.visualizer import BaseVisualizer, register_cluster_visualizer
from rl_insight.data import DataEnum


# Match verl precision_debugger stage ordering where possible; unknown stages last.
_STAGE_ORDER = [
    "actor_compute_log_prob",
    "ref_compute_log_prob",
    "actor_update",
    "compute_values",
    "critic_update",
    "compute_rm_score",
]


@dataclass(frozen=True)
class GroupListRecord:
    path: Path
    step: int
    stage: str
    inner_step: int
    rank: int
    op_index: int

    @property
    def sort_key(self) -> tuple:
        stage_rank = _STAGE_ORDER.index(self.stage) if self.stage in _STAGE_ORDER else len(_STAGE_ORDER)
        return (self.step, stage_rank, self.stage, self.op_index, self.rank, self.inner_step)


@register_cluster_visualizer("gmm_heatmap")
class GmmVisualizer(BaseVisualizer):
    input_type = DataEnum.SUMMARY_EVENT
    
    def run(self, data):
        """Run GMM heatmap visualization from parsed data."""
        # Extract parameters from config
        output = Path(self.config.get("output", "./output/gmm_group_list_heatmap.png"))
        # If output is a directory, append default filename
        if output.is_dir():
            output = output / "gmm_heatmap.png"
        dpi = self.config.get("dpi", 150)
        cmap = self.config.get("cmap", "viridis")
        rank = self.config.get("rank", None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(data).__name__}")
        
        logger.info(f"GmmVisualizer received DataFrame with {len(data)} rows")
        logger.info(f"DataFrame columns: {list(data.columns)}")
        
        if data.empty:
            raise ValueError("No GMM data provided")
        
        # Filter by rank if specified
        if rank is not None:
            data = data[data['rank_id'] == rank]
            logger.info(f"Filtered data to rank {rank}, now {len(data)} rows")
        
        # Filter by step if specified
        step = self.config.get('step', None)
        if step is not None:
            data = data[data['step'] == step]
            logger.info(f"Filtered data to step {step}, now {len(data)} rows")
        
        # Filter by role if specified
        role = self.config.get('role', None)
        if role is not None:
            data = data[data['role'] == role]
            logger.info(f"Filtered data to role {role}, now {len(data)} rows")
        
        # Build matrix
        mat, rec_list, boundaries = self._build_matrix_from_data(data)
        logger.info(f"Built matrix with shape {mat.shape}")
        
        segments = self._segment_labels(rec_list, boundaries)
        
        # Generate title
        rank_str = f" rank={rank}" if rank is not None else " all ranks"
        title = f"GMM expert load (group_list){rank_str} — {len(rec_list)} snapshots, {mat.shape[0]} experts"
        
        # Plot heatmap
        self._plot_heatmap(mat, rec_list, segments, title, output, dpi, cmap)
        
        return str(output)
    
    def _build_matrix_from_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[dict], List[int]]:
        """Build a matrix from the parsed data."""
        # Group data by step, role, rank_id, stage
        # First sort the data to ensure consistent ordering
        sorted_data = data.sort_values(['step', 'role', 'rank_id', 'stage'])
        grouped = sorted_data.groupby(['step', 'role', 'rank_id', 'stage'])
        
        # Get unique steps, roles, ranks and stages
        steps = sorted(data['step'].unique())
        roles = sorted(data['role'].unique())
        ranks = sorted(data['rank_id'].unique())
        stages = sorted(data['stage'].unique())
        max_expert = data['expert_index'].max()
        
        logger.info(f"Steps: {steps}")
        logger.info(f"Roles: {roles}")
        logger.info(f"Ranks: {ranks}")
        logger.info(f"Stages: {stages}")
        logger.info(f"Max expert index: {max_expert}")
        
        # Build matrix
        vecs = []
        rec_list = []
        
        for name, group in grouped:
            step, role, rank, stage_idx = name
            logger.info(f"Processing step: {step}, role: {role}, rank: {rank}, stage: {stage_idx}")
            
            # Create a vector for this step, role, rank and stage
            vec = np.full(max_expert + 1, np.nan, dtype=np.float64)
            for _, row in group.iterrows():
                expert_idx = row['expert_index']
                vec[expert_idx] = row['load']
            
            vecs.append(vec)
            rec_list.append({
                'step': step,
                'role': role,
                'rank_id': rank,
                'stage': stage_idx,
                'op_index': stage_idx  # Use stage as op_index
            })
        
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
    
    def _segment_labels(self, rec_list: List[dict], boundaries: List[int]) -> List[Tuple[int, int, int, str, int]]:
        """Generate segment labels: (x0, x1, step, role, rank_id)."""
        segments = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if a >= b:
                continue
            rec = rec_list[a]
            segments.append(
                (a, b, rec["step"], rec["role"], rec["rank_id"])
            )
        logger.info(f"Segments: {segments}")
        return segments
    
    def _plot_heatmap(
        self,
        mat: np.ndarray,
        rec_list: List[dict],
        segments: List[Tuple[int, int, int, str, int, int]],
        title: str,
        out_path: Path,
        dpi: int,
        cmap: str,
    ) -> None:
        """Plot the heatmap."""
        n_exp, n_time = mat.shape
        fig_h = min(22, max(6, n_exp * 0.12))
        fig_w = min(48, max(10, n_time * 0.04))
        fig = plt.figure(figsize=(fig_w, fig_h + 1.2))
        # Use tight layout to ensure consistent width
        fig.tight_layout()
        # Create gridspec with equal width for both subplots
        gs = fig.add_gridspec(2, 1, height_ratios=[0.12, 1], hspace=0.05)
        ax_bar = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        # Ensure both axes have the same x-axis limits
        ax_bar.set_xlim(-0.5, n_time - 0.5)
        ax.set_xlim(-0.5, n_time - 0.5)

        # Span bar: one color per (step, role, rank); stage (op index) is shown in x-axis labels.
        # Use viridis colormap for consistency with heatmap
        palette = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        for i, (a, b, step, role, rank_id) in enumerate(segments):
            color = palette[i]
            ax_bar.axvspan(a - 0.5, b - 0.5, facecolor=color, alpha=0.55, edgecolor="none")
        
        # Add vertical separator lines between segments
        for i, (a, b, step, role, rank_id) in enumerate(segments):
            if a > 0:
                ax_bar.axvline(a - 0.5, color="white", linewidth=0.8, alpha=0.7)
        # Add last separator line at the end
        if n_time > 0:
            ax_bar.axvline(n_time - 0.5, color="white", linewidth=0.8, alpha=0.7)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])
        ax_bar.set_title(
            "Segments: training step · RL role · rank",
            fontsize=10,
        )

        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest", origin="upper")
        ax.set_ylabel("Expert index")
        ax.set_title(title)

        # Vertical lines at every segment boundary (includes step / role / rank changes)
        for a, b, step, role, rank_id in segments:
            ax.axvline(a - 0.5, color="white", linewidth=0.8, alpha=0.7)
        ax.axvline(n_time - 0.5, color="white", linewidth=0.8, alpha=0.7)

        # X axis: one tick per column — only GMM op index (stage) for simplicity
        col_centers = np.arange(n_time, dtype=float)
        if n_time <= 60:
            ax.set_xticks(col_centers)
            labels = [
                f"op{rec_list[j]['stage']}"
                for j in range(n_time)
            ]
            ax.set_xticklabels(labels, fontsize=6, rotation=90 if n_time > 24 else 0)
        else:
            ax.set_xticks(col_centers[:: max(1, n_time // 40)])
            ax.set_xticklabels(
                [
                    f"op{rec_list[j]['stage']}"
                    for j in range(0, n_time, max(1, n_time // 40))
                ],
                fontsize=6,
                rotation=45,
            )
        ax.set_xlabel("Column: opK (K = npu_grouped_matmul index in forward / MoE-GMM call site)")

        y_stride = max(1, n_exp // 40)
        ax.set_yticks(list(range(0, n_exp, y_stride)))

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Tokens per expert (group_list)")

        def _seg_legend_label(s: Tuple[int, int, int, str, int]) -> str:
            _, _, st, rl, rk = s
            rshort = (rl[:14] + "…") if len(str(rl)) > 14 else str(rl)
            return f"st{st} · {rshort} · r{rk}"

        handles = [
            mpatches.Patch(color=palette[i % len(palette)], label=_seg_legend_label(s))
            for i, s in enumerate(segments)
        ]
        max_legend = 128
        if len(handles) <= max_legend:
            ax_bar.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=6,
            title="step · role · rank",
        )
        else:
            ax_bar.text(
                1.01,
                0.5,
                f"{len(segments)} segments\n(legend omitted >{max_legend};\nsee x-axis labels)",
                transform=ax_bar.transAxes,
                fontsize=7,
                va="center",
            )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)