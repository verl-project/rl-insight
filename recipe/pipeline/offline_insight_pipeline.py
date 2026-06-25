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

from recipe.data import DataChecker
from recipe.parser import get_cluster_parser_cls
from recipe.visualizer import get_cluster_visualizer_cls


class OfflineInsightPipeline:
    def __init__(self, config: DictConfig):
        self.config = config

        timeline_parser_type = config.timeline.parser.type
        if timeline_parser_type is not None:
            parser_cls = get_cluster_parser_cls(timeline_parser_type)
            visualizer_cls = get_cluster_visualizer_cls(config.timeline.visualizer.type)
        else:
            parser_cls = get_cluster_parser_cls(config.heatmap.parser.type)
            visualizer_cls = get_cluster_visualizer_cls(config.heatmap.visualizer.type)

        self.parser = parser_cls(self.config)

        self.visualizer = visualizer_cls(self.config)

    def run(self):
        DataChecker(self.parser.input_type, self.config.input.path).run()

        output_data = self.parser.run(self.config.input.path)

        DataChecker(self.visualizer.input_type, output_data).run()

        self.visualizer.run(output_data)
