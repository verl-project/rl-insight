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
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from recipe.main import main


def test_gmm_e2e_with_repo_sample_data(monkeypatch, tmp_path):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]

    input_dir = project_root / "data" / "recipe" / "gmm_data"
    output_dir = tmp_path / "gmm_output"

    assert input_dir.is_dir(), f"Sample GMM data missing: {input_dir}"

    test_args = [
        "main.py",
        f"input.path={input_dir}",
        f"output.path={output_dir}",
        "heatmap.visualizer.type=gmm_heatmap",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    output_file = output_dir / "gmm_heatmap.png"
    assert output_file.exists()
