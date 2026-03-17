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
from .pipeline.offline_insight_pipeline import OfflineInsightPipeline

SUPPORTED_PIPELINE_TYPES = {"OfflineInsightPipeline": OfflineInsightPipeline}


def run_pipeline(config, pipeline_class=None):
    if pipeline_class is None:
        raise ValueError("A pipeline class must be provided.")

    runner = pipeline_class(config)
    runner.run()


def main():
    arg_parser = argparse.ArgumentParser(description="Cluster scheduling visualization")
    arg_parser.add_argument(
        "--input-path", default="test", help="Raw path of profiling data"
    )
    arg_parser.add_argument(
        "--profiler-type", default="mstx", help="Profiler type, supported mstx/nvtx"
    )
    arg_parser.add_argument("--output-path", default="test", help="Output path")
    arg_parser.add_argument(
        "--vis-type", default="html", help="Visualization type, supported html"
    )
    arg_parser.add_argument("--rank-list", type=str, help="Rank id list", default="all")
    arg_parser.add_argument(
        "--pipeline-type",
        type=str,
        help="Tool pipeline type",
        default="OfflineInsightPipeline",
    )
    config = arg_parser.parse_args()

    # Validate pipeline type
    if config.pipeline_type not in SUPPORTED_PIPELINE_TYPES:
        supported_types = ", ".join(SUPPORTED_PIPELINE_TYPES.keys())
        raise ValueError(
            f"Unsupported pipeline type: {config.pipeline_type}. Supported types are: {supported_types}"
        )

    # Run the pipeline
    pipeline_class = SUPPORTED_PIPELINE_TYPES[config.pipeline_type]
    run_pipeline(config, pipeline_class)


if __name__ == "__main__":
    main()
