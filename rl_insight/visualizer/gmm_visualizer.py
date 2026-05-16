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
from PIL import Image, ImageDraw, ImageFont

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
        self._plot_heatmap(mat, rec_list, segments, output)
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
        rec_list: List[dict],
        segments: List[Tuple[int, int, int, str, int]],
        out_path: Path,
    ) -> None:
        """Plot the heatmap."""
        n_exp, n_time = mat.shape
        layout = self._compute_layout(n_exp, n_time)
        pad = layout["pad"]
        title_h = layout["title_h"]
        left_bar_w = layout["left_bar_w"]
        layer_axis_w = layout["layer_axis_w"]
        colorbar_gap = layout["colorbar_gap"]
        colorbar_w = layout["colorbar_w"]
        heatmap_w = layout["heatmap_w"]
        heatmap_h = layout["heatmap_h"]
        img_w = layout["img_w"]
        img_h = layout["img_h"]
        title = self._build_title(rec_list, n_exp)

        finite_vals = mat[np.isfinite(mat)]
        vmin = float(finite_vals.min()) if finite_vals.size else 0.0
        vmax = float(finite_vals.max()) if finite_vals.size else 1.0
        scale = vmax - vmin if vmax > vmin else 1.0

        image = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        heatmap_x0 = pad + left_bar_w + layer_axis_w
        heatmap_y0 = pad + title_h
        heatmap_x1 = heatmap_x0 + heatmap_w
        heatmap_y1 = heatmap_y0 + heatmap_h
        colorbar_x0 = heatmap_x1 + colorbar_gap
        colorbar_x1 = colorbar_x0 + colorbar_w

        draw.text((pad, pad), title, fill="black", font=title_font)
        draw.text((pad, pad + 16), "step | role | rank", fill="black", font=font)
        draw.text(
            (heatmap_x0 + max(0, heatmap_w // 2 - 35), heatmap_y1 + 22),
            "Expert index",
            fill="black",
            font=font,
        )
        self._draw_rotated_text(
            image,
            (pad + left_bar_w + 8, heatmap_y0 + max(0, heatmap_h // 2 - 28)),
            "Layer index",
            font,
            "black",
        )

        # Segment bar: one color per (step, role, rank), shown on left side.
        segment_colors = [
            self._viridis_rgb(i / max(1, len(segments) - 1))
            for i in range(len(segments))
        ]

        for idx, segment in enumerate(segments):
            a, b, _step, _role, _rank = segment
            y0 = self._scaled_position(a, n_time, heatmap_h, heatmap_y0)
            y1 = self._scaled_position(b, n_time, heatmap_h, heatmap_y0)
            if y1 <= y0:
                y1 = min(heatmap_y1, y0 + 1)
            draw.rectangle(
                [pad, y0, pad + left_bar_w - 1, y1 - 1],
                fill=segment_colors[idx],
            )
            label = self._segment_legend_label(segment)
            label = self._fit_text(draw, label, left_bar_w - 10, font)
            if label and (y1 - y0) >= 12:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_y = y0 + max(0, (y1 - y0 - (text_bbox[3] - text_bbox[1])) // 2)
                draw.text((pad + 4, text_y), label, fill="black", font=font)

        draw.rectangle(
            [heatmap_x0 - 1, heatmap_y0 - 1, heatmap_x1, heatmap_y1],
            outline=(200, 200, 200),
        )

        layer_ticks = self._layer_ticks(rec_list)
        for pos, label in layer_ticks:
            y = self._scaled_position(pos, n_time, heatmap_h, heatmap_y0)
            draw.line([(heatmap_x0 - 6, y), (heatmap_x0 - 1, y)], fill="black", width=1)
            draw.text((pad + left_bar_w + 14, max(heatmap_y0, y - 6)), label, fill="black", font=font)

        # Main heatmap is rendered as a stable bitmap to avoid backend-specific crashes.
        heatmap_rgb = self._heatmap_rgb(mat, vmin, scale)
        heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
        if heatmap_image.size != (heatmap_w, heatmap_h):
            heatmap_image = heatmap_image.resize(
                (heatmap_w, heatmap_h), resample=Image.Resampling.NEAREST
            )
        image.paste(heatmap_image, (heatmap_x0, heatmap_y0))

        self._draw_expert_ticks(draw, font, heatmap_x0, heatmap_y1, heatmap_w, n_exp)
        self._draw_colorbar(
            draw,
            font,
            colorbar_x0,
            colorbar_x1,
            heatmap_y0,
            heatmap_y1,
            vmin,
            vmax,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

    def _compute_layout(self, n_exp: int, n_time: int) -> dict[str, int]:
        pad = 24
        title_h = 46
        bottom_h = 58
        left_bar_w = 150
        layer_axis_w = 62
        colorbar_gap = 16
        colorbar_w = 44
        target_cell_w = int(self.config.get("cell_width", 28))
        target_cell_h = int(self.config.get("cell_height", 28))
        max_img_w = int(self.config.get("max_image_width", 4096))
        max_img_h = int(self.config.get("max_image_height", 8192))

        available_w = max(
            1,
            max_img_w
            - (pad * 2 + left_bar_w + layer_axis_w + colorbar_gap + colorbar_w),
        )
        available_h = max(1, max_img_h - (pad * 2 + title_h + bottom_h))

        heatmap_w = min(available_w, max(1, n_exp * target_cell_w))
        heatmap_h = min(available_h, max(1, n_time * target_cell_h))

        img_w = pad * 2 + left_bar_w + layer_axis_w + heatmap_w + colorbar_gap + colorbar_w
        img_h = pad * 2 + title_h + heatmap_h + bottom_h
        return {
            "pad": pad,
            "title_h": title_h,
            "bottom_h": bottom_h,
            "left_bar_w": left_bar_w,
            "layer_axis_w": layer_axis_w,
            "colorbar_gap": colorbar_gap,
            "colorbar_w": colorbar_w,
            "heatmap_w": heatmap_w,
            "heatmap_h": heatmap_h,
            "img_w": img_w,
            "img_h": img_h,
        }

    @staticmethod
    def _scaled_position(index: int, total: int, extent: int, offset: int) -> int:
        if total <= 0:
            return offset
        return offset + int(round(index * extent / total))

    @staticmethod
    def _segment_legend_label(segment: Tuple[int, int, int, str, int]) -> str:
        _, _, step, role, rank_id = segment
        return f"st{step} | {role} | r{rank_id}"

    @staticmethod
    def _build_title(rec_list: List[dict], n_exp: int) -> str:
        ranks = sorted({rec["rank_id"] for rec in rec_list})
        snapshots = len(rec_list)
        if len(ranks) == 1:
            rank_text = f"rank={ranks[0]}"
        else:
            rank_text = f"ranks={len(ranks)}"
        return f"GMM expert load ({rank_text}, {snapshots} snapshots, {n_exp} experts)"

    @staticmethod
    def _fit_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        font: ImageFont.ImageFont,
    ) -> str:
        if draw.textlength(text, font=font) <= max_width:
            return text
        suffix = "..."
        trimmed = text
        while trimmed and draw.textlength(trimmed + suffix, font=font) > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + suffix) if trimmed else ""

    @staticmethod
    def _draw_rotated_text(
        image: Image.Image,
        position: tuple[int, int],
        text: str,
        font: ImageFont.ImageFont,
        fill: str | tuple[int, int, int],
    ) -> None:
        tmp = Image.new("RGBA", (160, 32), (255, 255, 255, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((0, 0), text, fill=fill, font=font)
        rotated = tmp.rotate(90, expand=True)
        image.paste(rotated, position, rotated)

    def _heatmap_rgb(
        self,
        mat: np.ndarray,
        vmin: float,
        scale: float,
    ) -> np.ndarray:
        heatmap = mat.T
        rgb = np.full((heatmap.shape[0], heatmap.shape[1], 3), 235, dtype=np.uint8)
        finite_mask = np.isfinite(heatmap)
        if np.any(finite_mask):
            normalized = np.clip((heatmap[finite_mask] - vmin) / scale, 0.0, 1.0)
            palette_idx = np.rint(normalized * 255).astype(np.uint8)
            palette = self._viridis_palette()
            rgb[finite_mask] = palette[palette_idx]
        return rgb

    @staticmethod
    def _layer_ticks(rec_list: List[dict]) -> List[Tuple[int, str]]:
        if not rec_list:
            return []

        positions = [0]
        labels = [f"layer{rec_list[0]['layer_idx']}"]
        current_layer = rec_list[0]["layer_idx"]
        for idx, rec in enumerate(rec_list[1:], start=1):
            if rec["layer_idx"] != current_layer:
                current_layer = rec["layer_idx"]
                positions.append(idx)
                labels.append(f"layer{current_layer}")

        if positions[-1] != len(rec_list) - 1:
            positions.append(len(rec_list) - 1)
            labels.append(f"layer{rec_list[-1]['layer_idx']}")

        max_labels = 40
        if len(positions) > max_labels:
            selected = np.linspace(0, len(positions) - 1, max_labels, dtype=int)
            positions = [positions[idx] for idx in selected]
            labels = [labels[idx] for idx in selected]
        return list(zip(positions, labels))

    def _draw_expert_ticks(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        heatmap_x0: int,
        heatmap_y1: int,
        heatmap_w: int,
        n_exp: int,
    ) -> None:
        if n_exp <= 0:
            return

        tick_count = min(6, n_exp)
        tick_indices = np.linspace(0, n_exp - 1, tick_count, dtype=int)
        seen: set[int] = set()
        for expert_idx in tick_indices:
            if int(expert_idx) in seen:
                continue
            seen.add(int(expert_idx))
            x = heatmap_x0 + int(round((int(expert_idx) + 0.5) * heatmap_w / n_exp))
            draw.line([(x, heatmap_y1), (x, heatmap_y1 + 5)], fill="black", width=1)
            label = str(int(expert_idx))
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) // 2, heatmap_y1 + 8), label, fill="black", font=font)

    def _draw_colorbar(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        vmin: float,
        vmax: float,
    ) -> None:
        palette = self._viridis_palette()
        height = max(1, y1 - y0)
        for offset in range(height):
            idx = min(255, max(0, int(round((1 - offset / max(1, height - 1)) * 255))))
            color = tuple(int(v) for v in palette[idx])
            draw.line([(x0, y0 + offset), (x1, y0 + offset)], fill=color, width=1)

        draw.rectangle([x0, y0, x1, y1], outline=(120, 120, 120))
        draw.text((x0 - 2, max(0, y0 - 18)), f"{vmax:.2f}", fill="black", font=font)
        draw.text((x0 - 2, y1 + 4), f"{vmin:.2f}", fill="black", font=font)
        draw.text((x0 - 6, max(0, y0 - 34)), "Load", fill="black", font=font)

    @classmethod
    def _viridis_palette(cls) -> np.ndarray:
        return np.array([cls._viridis_rgb(i / 255.0) for i in range(256)], dtype=np.uint8)

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
