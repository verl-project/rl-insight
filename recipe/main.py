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

from omegaconf import DictConfig

from .config import ConfigLoader
from .pipeline.offline_insight_pipeline import OfflineInsightPipeline

SUPPORTED_PIPELINE_TYPES = {"OfflineInsightPipeline": OfflineInsightPipeline}


def run_pipeline(config: DictConfig, pipeline_class=None):
    if pipeline_class is None:
        raise ValueError("A pipeline class must be provided.")

    runner = pipeline_class(config)
    runner.run()


def validate_config(cfg: DictConfig) -> None:
    if cfg.input.path is None:
        raise ValueError("input.path is required")

    if cfg.pipeline.type not in SUPPORTED_PIPELINE_TYPES:
        supported_types = ", ".join(SUPPORTED_PIPELINE_TYPES.keys())
        raise ValueError(
            f"Unsupported pipeline type: {cfg.pipeline.type}. "
            f"Supported types are: {supported_types}"
        )


def main():
    cfg = ConfigLoader.load_from_cli()
    validate_config(cfg)
    pipeline_class = SUPPORTED_PIPELINE_TYPES[cfg.pipeline.type]
    run_pipeline(cfg, pipeline_class)


if __name__ == "__main__":
    main()
