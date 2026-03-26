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

import sys
from pathlib import Path
from rl_insight.main import main


def test_mstx_e2e_with_input_path(monkeypatch, tmp_path):
    # Get the root directory of the project
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    # Get the input data path
    input_dir = project_root / "data" / "mstx_data"
    output_dir = tmp_path / "mstx_output"

    # Ensure the input directory exists
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"

    # Set command line parameters
    test_args = [
        "main.py",
        f"--input-path={input_dir}",
        f"--output-path={output_dir}",
        "--profiler-type=mstx",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    # Verify output file
    output_file = output_dir / "rl_timeline.html"
    assert output_file.exists()
