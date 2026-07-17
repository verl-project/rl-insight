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
from recipe.main import main


def test_memory_e2e_with_sample_data(monkeypatch, tmp_path):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]

    input_dir = project_root / "data" / "recipe" / "memory_data"
    output_dir = tmp_path / "memory_output"

    assert input_dir.is_dir(), f"Sample memory data missing: {input_dir}"

    test_args = [
        "main.py",
        f"input.path={input_dir}",
        f"output.path={output_dir}",
        "memory.parser.type=memory",
        "memory.visualizer.type=memory_html",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    # data uses role="memory_data", rank_id=0 — single HTML (no segment split).
    html_file = output_dir / "memory_timeline_memory_data_rank0.html"
    data_file = output_dir / "detail_data_memory_data_rank0.js"
    assert html_file.exists(), f"Missing: {html_file}"
    assert data_file.exists(), f"Missing: {data_file}"
