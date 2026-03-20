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

from rl_insight.data import DataChecker, DataEnum
from rl_insight.parser import get_cluster_parser_cls
from rl_insight.utils.schema import Constant
from rl_insight.visualizer.visualizer import RLTimelineVisualizer


class OfflineInsightPipeline:
    def __init__(self, config):
        self.config = config

        # init data
        self.input_data_type = DataEnum(self.config.input_type)

        # parser related
        parser_config = self._prepare_parser_config()
        parser_cls = get_cluster_parser_cls(self.config.profiler_type)
        self.parser = parser_cls(parser_config)

        # visualizer related
        visualizer_config = self._prepare_visualizer_config()
        self.visualizer = RLTimelineVisualizer(visualizer_config)

    def _prepare_parser_config(self):
        return {
            Constant.RANK_LIST: self.config.rank_list,
        }

    def _prepare_visualizer_config(self):
        return {
            "output_path": self.config.output_path,
            "vis_type": self.config.vis_type,
        }

    def run(self):
        if self.input_data_type != self.parser.input_type:
            raise ValueError(
                f"Input data type {self.input_data_type} does not match parser input type {self.parser.input_type}"
            )
        # validate input data
        DataChecker(self.input_data_type, self.config.input_path).run()
        
        output_data = self.parser.run(self.config.input_path)
        
        # validate output data
        DataChecker(self.visualizer.input_type, output_data).run()

        self.visualizer.run(output_data)
