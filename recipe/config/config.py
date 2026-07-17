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

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class InputConfig:
    """Input data configuration."""

    path: str = MISSING  # Path to profiling data (required)
    rank_list: str = "all"  # Rank id list, e.g. '0,1,2' or 'all'


@dataclass
class OutputConfig:
    """Output configuration."""

    path: str = "output"  # Output directory path


@dataclass
class TimelineParserConfig:
    """Timeline parser configuration."""

    type: Optional[str] = None  # mstx | torch | nvtx


@dataclass
class TimelineVisualizerConfig:
    """Timeline visualizer configuration."""

    type: str = "html"  # html | png
    width: int = 2000  # Image width in pixels (png only)
    scale: int = 2  # Image scale factor (png only)


@dataclass
class TimelineConfig:
    """Timeline configuration."""

    parser: TimelineParserConfig = field(default_factory=TimelineParserConfig)
    visualizer: TimelineVisualizerConfig = field(
        default_factory=TimelineVisualizerConfig
    )


@dataclass
class HeatmapParserConfig:
    """Heatmap parser configuration."""

    type: Optional[str] = None  # gmm
    step: Optional[str] = None  # Step filter, e.g. '1' or '1,2'
    role: Optional[str] = None  # Role filter


@dataclass
class HeatmapVisualizerConfig:
    """Heatmap visualizer configuration."""

    type: str = "gmm_heatmap"  # gmm_heatmap
    dpi: int = 200  # DPI for heatmap PNG output
    cmap: str = "viridis"  # Matplotlib colormap name
    gmm_per_layer: int = 3  # Grouped matmul count per MoE layer


@dataclass
class HeatmapConfig:
    """Heatmap configuration."""

    parser: HeatmapParserConfig = field(default_factory=HeatmapParserConfig)
    visualizer: HeatmapVisualizerConfig = field(default_factory=HeatmapVisualizerConfig)


@dataclass
class MemoryParserConfig:
    """Memory parser configuration."""

    type: Optional[str] = None  # memory
    step: Optional[str] = None  # Step filter, e.g. '1' or '1,2'
    role: Optional[str] = None  # Role filter


@dataclass
class MemoryVisualizerConfig:
    """Memory visualizer configuration."""

    type: str = "memory_html"  # memory_html


@dataclass
class MemoryConfig:
    """Memory configuration."""

    parser: MemoryParserConfig = field(default_factory=MemoryParserConfig)
    visualizer: MemoryVisualizerConfig = field(default_factory=MemoryVisualizerConfig)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    type: str = "OfflineInsightPipeline"  # OfflineInsightPipeline


@dataclass
class AppConfig:
    """RL Insight configuration."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
