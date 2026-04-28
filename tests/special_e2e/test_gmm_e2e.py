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

import sys
import torch
from rl_insight.main import main


def _write_group_list(path, loads):
    path.parent.mkdir(parents=True, exist_ok=True)
    # group_list.pt is a 1-D tensor: [num_expert], each value is expert load.
    torch.save(torch.tensor(loads, dtype=torch.float32), path)


def test_gmm_e2e_with_input_path(monkeypatch, tmp_path):
    input_dir = tmp_path / "gmm_input"
    output_dir = tmp_path / "gmm_output"

    base = input_dir / "rank0" / "step_1" / "actor_update" / "dump_tensor_data"
    _write_group_list(
        base / "npu_grouped_matmul.0.forward.kwargs.group_list.pt",
        [10.0, 3.0, 7.0, 2.0],
    )
    _write_group_list(
        base / "npu_grouped_matmul.1.forward.kwargs.group_list.pt",
        [6.0, 5.0, 3.0, 8.0],
    )

    test_args = [
        "main.py",
        f"--input-path={input_dir}",
        f"--output-path={output_dir}",
        "--profiler-type=gmm",
        "--input-type=gmm_data",
        "--vis-type=gmm_heatmap",
        "--step=1",
        "--role=actor_update",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    output_file = output_dir / "gmm_heatmap.png"
    assert output_file.exists()
