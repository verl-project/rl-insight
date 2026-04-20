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

import argparse

from .timeline_visualizer import RLTimelineVisualizer
from .timeline_visualizer import RLTimelinePNGVisualizer
from .visualizer import (
    BaseVisualizer,
    get_cluster_visualizer_cls,
)
from .gmm_visualizer import GmmVisualizer


def register_visualizer_specific_args(arg_parser: argparse.ArgumentParser) -> None:
    """Register optional visualizer CLI flags (additive). Safe for html timeline; extras are ignored."""
    heatmap_group = arg_parser.add_argument_group("GMM heatmap parameters")
    heatmap_group.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for heatmap PNG output",
    )
    heatmap_group.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name",
    )
    heatmap_group.add_argument(
        "--gmm-per-layer",
        type=int,
        default=3,
        help="Expected grouped_matmul count per MoE layer in forward pass",
    )


__all__ = [
    "BaseVisualizer",
    "get_cluster_visualizer_cls",
    "register_visualizer_specific_args",
    "RLTimelineVisualizer",
    "RLTimelinePNGVisualizer",
    "GmmVisualizer",
]
