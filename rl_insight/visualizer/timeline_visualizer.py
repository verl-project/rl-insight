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

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_image

from rl_insight.data import DataEnum
from rl_insight.utils.schema import FigureConfig

from .visualizer import (
    BaseVisualizer,
    register_cluster_visualizer,
)


@register_cluster_visualizer("html")
class RLTimelineVisualizer(BaseVisualizer):
    """HTML / chart timeline; ``vis_type`` in config selects behavior inside ``run``."""

    COLOR_PALETTE = [
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
    ]

    input_type: DataEnum = DataEnum.SUMMARY_EVENT

    def __init__(self, config: dict):
        super().__init__(config)
        self.output_path = config.get("output_path", None)

    def run(self, data):
        return self.generate_rl_timeline(data)

    def generate_rl_timeline(
        self,
        input_data: pd.DataFrame,
        output_dir: str | None = None,
        output_filename: str = "rl_timeline.html",
        title_prefix: str = "RL Timeline",
    ) -> go.Figure:
        """
        Generate an RL event timeline Gantt chart with interactive Y-axis sorting by Rank ID.

        Args:
            input_data: A pandas DataFrame containing events_summary data.
                        DataFrame should have columns: role, domain, rank_id, start_time_ms, end_time_ms
            output_dir: Directory to save the HTML file (defaults to ``output_path`` from config)
            output_filename: Name of the output HTML file
            title_prefix: Prefix for the chart title
        """
        out_dir: str = output_dir or self.output_path or "output"
        df, t0 = self.load_and_preprocess(input_data)
        df = self.merge_short_events(df)
        df = self.downsample_if_needed(df)
        y_mappings, y_axis_spacing = self.build_y_mappings(df)
        traces = self.build_traces(df, y_mappings["default"])
        cfg = FigureConfig(
            title_prefix=title_prefix,
            t0=t0,
            y_mappings=y_mappings,
            y_axis_spacing=y_axis_spacing,
        )
        fig = self.assemble_figure(traces, df, cfg)
        self.save_html(fig, out_dir, output_filename)
        return fig

    def load_and_preprocess(
        self, input_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        """
        Load and preprocess data from a pandas DataFrame.

        Args:
            input_data: A pandas DataFrame containing events_summary data

        Returns:
            Tuple of (preprocessed DataFrame, t0 offset)
        """
        if input_data is None:
            raise ValueError(f"input_data: {input_data} is None!")

        df = input_data.copy()

        df.rename(
            columns={
                "role": "Role",
                "name": "Name",
                "rank_id": "Rank ID",
                "start_time_ms": "Start",
                "end_time_ms": "Finish",
            },
            inplace=True,
            errors="ignore",
        )

        required = ["Role", "Name", "Rank ID", "Start", "Finish"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")

        df = df.dropna(subset=required).copy()
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
        df["Finish"] = pd.to_numeric(df["Finish"], errors="coerce")
        df["Rank ID"] = pd.to_numeric(df["Rank ID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Start", "Finish", "Rank ID"])
        df = df[df["Finish"] > df["Start"]].copy()
        df["Duration"] = df["Finish"] - df["Start"]

        if df.empty:
            return df, 0.0

        t0 = df["Start"].min()
        df["Start"] -= t0
        df["Finish"] -= t0
        df["Duration"] = df["Finish"] - df["Start"]
        return df, t0

    def merge_short_events(
        self, df: pd.DataFrame, threshold_ms: float = 10.0
    ) -> pd.DataFrame:
        def _merge_group(g: pd.DataFrame) -> pd.DataFrame:
            short = g[g["Duration"] < threshold_ms]
            long = g[g["Duration"] >= threshold_ms]

            role, rank_id, name = g.name
            long["Role"] = role
            long["Rank ID"] = rank_id
            long["Name"] = name
            if short.empty:
                return long
            merged = pd.DataFrame(
                [
                    {
                        "Start": short["Start"].min(),
                        "Finish": short["Finish"].max(),
                        "Role": role,
                        "Rank ID": rank_id,
                        "Name": name,
                        "Duration": short["Finish"].max() - short["Start"].min(),
                    }
                ]
            )
            return pd.concat([long, merged], ignore_index=True)

        return (
            df.groupby(["Role", "Rank ID", "Name"], group_keys=False)
            .apply(_merge_group)
            .reset_index(drop=True)
        )

    def downsample_if_needed(
        self,
        df: pd.DataFrame,
        max_records: int = 5000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        if len(df) <= max_records:
            return df
        n_domains = df["Name"].nunique()
        samples_per_domain = max_records // max(1, n_domains)

        def _sample_domain(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= samples_per_domain:
                return g
            return g.sample(n=samples_per_domain, random_state=random_state)

        return (
            df.groupby("Name", group_keys=False)
            .apply(_sample_domain)
            .reset_index(drop=True)
        )

    def build_y_mappings(self, df: pd.DataFrame):
        df["Y_Label"] = df["Role"] + " - Rank " + df["Rank ID"].astype(str)
        unique_y_labels = df["Y_Label"].unique()

        def _extract_rank(label: str):
            try:
                return int(label.split(" - Rank ")[-1])
            except Exception:
                return float("inf")

        y_axis_spacing = max(60, min(100, 800 // max(1, len(unique_y_labels))))
        bar_height = y_axis_spacing * 0.8

        y_labels_default = unique_y_labels
        mapping_default = {
            label: i * y_axis_spacing for i, label in enumerate(y_labels_default)
        }
        df["Y_default"] = df["Y_Label"].map(mapping_default)

        y_labels_by_rank = sorted(unique_y_labels, key=lambda x: (_extract_rank(x), x))
        mapping_by_rank = {
            label: i * y_axis_spacing for i, label in enumerate(y_labels_by_rank)
        }
        df["Y_by_rank"] = df["Y_Label"].map(mapping_by_rank)

        return {
            "default": mapping_default,
            "by_rank": mapping_by_rank,
            "bar_height": bar_height,
        }, y_axis_spacing

    def build_traces(self, df: pd.DataFrame, y_mapping: dict):
        unique_domains = df["Name"].unique()
        color_map = {
            dom: self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]
            for i, dom in enumerate(unique_domains)
        }
        bar_height = y_mapping.get("bar_height", 48)

        traces = []
        for domain in unique_domains:
            dom_df = df[df["Name"] == domain]
            trace = go.Bar(
                base=dom_df["Start"],
                x=dom_df["Duration"],
                y=dom_df["Y_default"],
                orientation="h",
                name=domain,
                marker_color=color_map[domain],
                width=bar_height,
                hovertemplate=(
                    "<b>%{data.name}</b><br>"
                    "Start: %{base:.3f} ms<br>"
                    "End: %{customdata[1]:.3f} ms<br>"
                    "Duration: %{x:.3f} ms<br>"
                    "Rank: %{customdata[0]}<extra></extra>"
                ),
                customdata=np.column_stack([dom_df["Y_Label"], dom_df["Finish"]]),
                showlegend=True,
                textposition="none",
            )
            traces.append(trace)
        return traces

    def assemble_figure(
        self, traces: list[go.Bar], df: pd.DataFrame, cfg: FigureConfig
    ) -> go.Figure:
        max_time = df["Finish"].max()
        unique_y_labels = sorted(df["Y_Label"].unique())

        h = max(
            cfg.chart_height_min,
            min(len(unique_y_labels) * cfg.y_axis_spacing, cfg.chart_height_max),
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"{cfg.title_prefix} (Relative Time, Origin = {cfg.t0:.3f} ms)",
            xaxis_title="Time (ms, Relative)",
            yaxis_title="Module - Rank",
            xaxis=dict(
                range=[0, max_time * (1 + cfg.xaxis_max_pad_ratio)],
                tickformat=".1f",
                nticks=cfg.nticks,
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(cfg.y_mappings["default"].values()),
                ticktext=list(cfg.y_mappings["default"].keys()),
                autorange="reversed",
            ),
            barmode="overlay",
            height=h,
            hovermode="closest",
            legend_title="Event Type",
            margin=dict(
                l=cfg.margin_left,
                r=cfg.margin_right,
                t=cfg.margin_top,
                b=cfg.margin_bottom,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"hovermode": "closest"}],
                            label="Hover: Current Only",
                            method="relayout",
                        ),
                        dict(
                            args=[{"hovermode": "x unified"}],
                            label="Hover: All Ranks",
                            method="relayout",
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.7,
                    xanchor="left",
                    y=1.07,
                    yanchor="top",
                ),
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[
                                {
                                    "y": [
                                        df[df["Name"] == t.name]["Y_default"].tolist()
                                        for t in traces
                                    ]
                                },
                                {
                                    "yaxis.tickvals": list(
                                        cfg.y_mappings["default"].values()
                                    ),
                                    "yaxis.ticktext": list(
                                        cfg.y_mappings["default"].keys()
                                    ),
                                },
                            ],
                            label="Sort: Default",
                            method="update",
                        ),
                        dict(
                            args=[
                                {
                                    "y": [
                                        df[df["Name"] == t.name]["Y_by_rank"].tolist()
                                        for t in traces
                                    ]
                                },
                                {
                                    "yaxis.tickvals": list(
                                        cfg.y_mappings["by_rank"].values()
                                    ),
                                    "yaxis.ticktext": list(
                                        cfg.y_mappings["by_rank"].keys()
                                    ),
                                },
                            ],
                            label="Sort: By Rank ID",
                            method="update",
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.85,
                    xanchor="left",
                    y=1.07,
                    yanchor="top",
                ),
            ],
        )
        return fig

    def save_html(self, fig: go.Figure, output_dir: str, output_filename: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        fig.write_html(
            out_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        )


@register_cluster_visualizer("png")
class RLTimelinePNGVisualizer(BaseVisualizer):
    COLOR_PALETTE = [
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
    ]

    input_type: DataEnum = DataEnum.SUMMARY_EVENT

    def __init__(self, config: dict):
        super().__init__(config)
        self.output_path = config.get("output_path", None)
        self.width = config.get("width", 2000)
        self.scale = config.get("scale", 2)

    def run(self, data):
        return self.generate_rl_timeline_png(data)

    def generate_rl_timeline_png(
        self,
        input_data: pd.DataFrame,
        output_dir: str | None = None,
        output_filename: str = "rl_timeline.png",
    ):
        out_dir = output_dir or self.output_path or "output"

        df, t0 = self.load_and_preprocess(input_data)
        df = self.merge_short_events(df)
        df = self.downsample_if_needed(df)
        y_mappings, y_axis_spacing = self.build_y_mappings(df)
        traces = self.build_traces(df, y_mappings)

        fig = self.assemble_static_figure(traces, df, t0, y_mappings, y_axis_spacing)
        self.save_png(fig, out_dir, output_filename)
        return fig

    def load_and_preprocess(
        self, input_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        if input_data is None or input_data.empty:
            raise ValueError("input_data is None or empty!")

        df = input_data.copy()
        df = df.rename(
            columns={
                "role": "Role",
                "name": "Name",
                "rank_id": "Rank ID",
                "start_time_ms": "Start",
                "end_time_ms": "Finish",
            },
            errors="ignore",
        )

        required = ["Role", "Name", "Rank ID", "Start", "Finish"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")

        df = df.dropna(subset=required)
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
        df["Finish"] = pd.to_numeric(df["Finish"], errors="coerce")
        df["Rank ID"] = pd.to_numeric(df["Rank ID"], errors="coerce").astype("Int64")
        df = df[(df["Finish"] > df["Start"]) & (df["Rank ID"].notna())]

        df["Duration"] = df["Finish"] - df["Start"]

        if df.empty:
            return df, 0.0

        t0 = df["Start"].min()
        df["Start_rel"] = df["Start"] - t0
        df["End_rel"] = df["Finish"] - t0
        df = df.sort_values(by=["Rank ID", "Start_rel"]).reset_index(drop=True)
        return df, t0

    def merge_short_events(
        self,
        df: pd.DataFrame,
        duration_threshold_ms: float = 8.0,
        gap_threshold_ms: float = 2.0,
    ) -> pd.DataFrame:
        def merge_group(rows):
            row = rows.iloc[0].copy()
            row["Start"] = rows["Start"].min()
            row["Finish"] = rows["Finish"].max()
            row["Duration"] = row["Finish"] - row["Start"]
            row["Start_rel"] = rows["Start_rel"].min()
            row["End_rel"] = rows["End_rel"].max()
            return row

        def process_group(g):
            if len(g) <= 1:
                return g

            g = g.sort_values("Start_rel").reset_index(drop=True)
            groups = []
            current = [g.iloc[0]]

            for i in range(1, len(g)):
                curr_row = g.iloc[i]
                last = current[-1]

                if (
                    curr_row["Duration"] <= duration_threshold_ms
                    and curr_row["Start_rel"] - last["End_rel"] <= gap_threshold_ms
                ):
                    current.append(curr_row)
                else:
                    groups.append(pd.concat(current, axis=1).T)
                    current = [curr_row]

            if current:
                groups.append(pd.concat(current, axis=1).T)

            return pd.DataFrame([merge_group(grp) for grp in groups])

        result_groups = []
        for _, group in df.groupby(["Role", "Rank ID", "Name"]):
            processed = process_group(group)
            result_groups.append(processed)

        if result_groups:
            return pd.concat(result_groups, ignore_index=True)
        else:
            return df

    def downsample_if_needed(
        self, df: pd.DataFrame, max_points: int = 3000
    ) -> pd.DataFrame:
        if len(df) <= max_points:
            return df
        n_tasks = df["Name"].nunique()
        n_per_task = max(10, max_points // max(1, n_tasks))

        def sample_task(g):
            if len(g) <= n_per_task:
                return g
            return g.nlargest(n_per_task, "Duration").sort_values("Start_rel")

        sampled_groups = []
        for _, group in df.groupby("Name"):
            sampled = sample_task(group)
            sampled_groups.append(sampled)

        if sampled_groups:
            return pd.concat(sampled_groups, ignore_index=True)
        else:
            return df

    def build_y_mappings(self, df: pd.DataFrame) -> tuple[dict, int]:
        df["y_label"] = df["Role"] + " - Rank " + df["Rank ID"].astype(str)
        unique_labels = (
            df[["y_label", "Rank ID"]]
            .drop_duplicates()
            .sort_values(["Rank ID", "y_label"])["y_label"]
            .tolist()
        )

        y_step = 50
        y_pos = {label: idx * y_step for idx, label in enumerate(unique_labels)}
        df["y_pos"] = df["y_label"].map(y_pos)

        bar_height = y_step * 0.42
        y_map = {
            "positions": y_pos,
            "bar_height": bar_height,
            "labels": unique_labels,
        }
        return y_map, y_step

    def build_traces(self, df: pd.DataFrame, y_mappings: dict) -> list[go.Bar]:
        tasks = sorted(df["Name"].unique())
        color_map = {
            t: self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]
            for i, t in enumerate(tasks)
        }

        traces = []
        for task in tasks:
            sub = df[df["Name"] == task]
            traces.append(
                go.Bar(
                    name=task,
                    base=sub["Start_rel"],
                    x=sub["Duration"],
                    y=sub["y_pos"],
                    orientation="h",
                    marker=dict(
                        color=color_map[task],
                        line=dict(color="white", width=0.8),
                    ),
                    width=y_mappings["bar_height"],
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        return traces

    def assemble_static_figure(self, traces, df, t0, y_mappings, y_step):
        max_t = df["End_rel"].max() * 1.05
        n_ranks = len(y_mappings["labels"])

        fig = go.Figure(traces)
        fig.update_layout(
            title=dict(
                text=f"(Relative Time, Origin =  {t0:.1f} ms)",
                font=dict(size=24, family="Arial"),
                x=0.5,
            ),
            width=self.width,
            height=max(900, n_ranks * y_step + 200),
            xaxis=dict(
                title=dict(text="Time (ms)", font=dict(size=18)),
                tickfont=dict(size=14),
                range=[0, max_t],
                tickformat=".0f",
                showgrid=True,
                gridcolor="#EAEAEA",
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="Module - Rank", font=dict(size=18)),
                tickfont=dict(size=13),
                tickvals=list(y_mappings["positions"].values()),
                ticktext=list(y_mappings["positions"].keys()),
                autorange="reversed",
                showgrid=False,
                zeroline=False,
            ),
            legend=dict(
                title=dict(text="Task Type", font=dict(size=12)),
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=240, r=60, t=120, b=80),
            barmode="overlay",
            hovermode=False,
        )
        return fig

    def save_png(self, fig: go.Figure, out_dir, fname):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        img = to_image(fig, format="png", scale=self.scale, width=self.width)
        with open(path, "wb") as f:
            f.write(img)
