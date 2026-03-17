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
import logging
import torch_npu
from torch_npu.profiler.profiler import analyse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    arg_parser = argparse.ArgumentParser(description="Run mstx offline analysis")
    arg_parser.add_argument("path", help="Path to profiling data")
    args = arg_parser.parse_args()

    path = args.path
    if not mstx_preprocessing(path):
        sys.exit(1)


def mstx_preprocessing(path: str) -> bool:
    all_successful = True
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        try:
            logger.info(f"Analyzing {dir_path}...")
            analyse(dir_path, export_type=torch_npu.profiler.ExportType.Text)
        except Exception as exc:
            logger.error(f"Offline analysis failed for {dir_path}: {exc}")
            all_successful = False
    return all_successful


if __name__ == "__main__":
    main()
