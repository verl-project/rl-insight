# Copyright (c) 2026 verl-project authors.
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

import numpy as np
import pandas as pd
from PIL import Image

from rl_insight.visualizer.gmm_visualizer import GmmVisualizer


def _build_gmm_dataframe(
    num_steps: int = 1,
    num_stages: int = 16,
    num_experts: int = 32,
) -> pd.DataFrame:
    rows = []
    for step in range(num_steps):
        for stage in range(num_stages):
            for expert_index in range(num_experts):
                rows.append(
                    {
                        "role": "actor_update",
                        "rank_id": 0,
                        "step": step,
                        "stage": stage,
                        "expert_index": expert_index,
                        "load": float((stage + expert_index) % 11),
                    }
                )
    return pd.DataFrame(rows)


def test_gmm_visualizer_caps_large_output_and_renders_metadata(tmp_path):
    output_dir = tmp_path / "gmm_output"
    data = _build_gmm_dataframe(num_stages=160, num_experts=96)
    visualizer = GmmVisualizer(
        {
            "output_path": str(output_dir),
            "max_image_width": 720,
            "max_image_height": 720,
            "gmm_per_layer": 1,
        }
    )

    output_path = visualizer.run(data)

    with Image.open(output_path) as image:
        pixels = np.asarray(image)

    assert pixels.shape[1] <= 720
    assert pixels.shape[0] <= 720

    top_strip = pixels[:50]
    bottom_strip = pixels[-70:]
    right_strip = pixels[:, -80:]
    center_strip = pixels[:, 120:220]

    assert np.any(np.any(top_strip != 255, axis=2))
    assert np.any(np.any(bottom_strip != 255, axis=2))
    assert np.any(np.any(right_strip != 255, axis=2))
    assert np.any(np.any(center_strip != 255, axis=2))
