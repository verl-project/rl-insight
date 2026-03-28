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
import sys
import argparse
from loguru import logger

from rl_insight.utils.schema import Constant


def main():
    arg_parser = argparse.ArgumentParser(description="Run mstx offline analysis")
    arg_parser.add_argument("path", help="Path to profiling data")
    args = arg_parser.parse_args()

    path = args.path
    if not mstx_preprocessing(path):
        sys.exit(1)


def mstx_preprocessing(path: str) -> bool:
    analyse = None
    export_type = None
    all_successful = True
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        parsed_output_dir = None
        # Check current level first.
        direct_output_dir = os.path.join(dir_path, Constant.ASCEND_PROFILER_OUTPUT)
        if os.path.isdir(direct_output_dir):
            parsed_output_dir = direct_output_dir
        else:
            # Also check one level deeper, e.g. */*_ascend_pt/ASCEND_PROFILER_OUTPUT.
            for sub_name in os.listdir(dir_path):
                sub_path = os.path.join(dir_path, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                nested_output_dir = os.path.join(
                    sub_path, Constant.ASCEND_PROFILER_OUTPUT
                )
                if os.path.isdir(nested_output_dir):
                    parsed_output_dir = nested_output_dir
                    break

        if parsed_output_dir is not None:
            logger.info(
                f"Found existing parsed output at {parsed_output_dir}, skip offline analysis."
            )
            continue

        try:
            if analyse is None or export_type is None:
                # Lazy import to keep top-level imports minimal.
                import torch_npu
                from torch_npu.profiler.profiler import analyse as torch_npu_analyse

                analyse = torch_npu_analyse
                export_type = torch_npu.profiler.ExportType.Text
            logger.info(f"Analyzing {dir_path}...")
            analyse(dir_path, export_type=export_type)
        except Exception as exc:
            logger.error(f"Offline analysis failed for {dir_path}: {exc}")
            all_successful = False
    return all_successful


if __name__ == "__main__":
    main()
